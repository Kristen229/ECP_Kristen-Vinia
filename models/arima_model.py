import pandas as pd
import numpy as np
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_arima(df, target_index):
    """
    Entraîne un modèle ARIMA avec auto-optimisation
    
    Args:
        df: DataFrame avec colonnes 'Date' et target_index
        target_index: Nom de la colonne cible
    
    Returns:
        Modèle ARIMA entraîné
    """
    # Préparer série temporelle
    series = df.set_index('Date')[target_index].asfreq('MS')
    
    # Auto-optimisation ARIMA
    model = pm.auto_arima(
        series,
        d=1,  # Ordre de différenciation
        start_p=0, max_p=5,
        start_q=0, max_q=5,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore',
        trace=False
    )
    
    return model

def predict_arima(model, n_periods):
    """
    Génère des prévisions avec ARIMA
    
    Args:
        model: Modèle ARIMA entraîné
        n_periods: Nombre de périodes à prédire
    
    Returns:
        DataFrame avec colonnes: Date, Forecast, Lower_CI, Upper_CI
    """
    # Générer prévisions
    forecast_result = model.predict(n_periods=n_periods, return_conf_int=True)
    forecast_values = forecast_result[0]
    conf_int = forecast_result[1]
    
    # Créer dates futures
    last_date = model.data.dates[-1] if hasattr(model.data, 'dates') else pd.Timestamp.now()
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=n_periods,
        freq='MS'
    )
    
    # Créer DataFrame résultat
    result_df = pd.DataFrame({
        'Date': future_dates,
        'Forecast': forecast_values,
        'Lower_CI': conf_int[:, 0],
        'Upper_CI': conf_int[:, 1]
    })
    
    return result_df

def evaluate_arima(model, df, target_index):
    """
    Évalue les performances du modèle ARIMA
    
    Args:
        model: Modèle entraîné
        df: DataFrame complet
        target_index: Nom de la colonne cible
    
    Returns:
        Dictionnaire avec métriques (mae, rmse, mape)
    """
    # Split train/test
    train_size = int(len(df) * 0.85)
    train_series = df[target_index].iloc[:train_size]
    test_series = df[target_index].iloc[train_size:]
    
    # Entraîner sur train
    model_eval = pm.auto_arima(
        train_series,
        d=1,
        start_p=0, max_p=5,
        start_q=0, max_q=5,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore'
    )
    
    # Prédire sur test
    forecast = model_eval.predict(n_periods=len(test_series))
    
    # Calculer métriques
    actual = test_series.values
    mae = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape
    }