import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_prophet(df, target_index):
    """
    Entraîne un modèle Prophet
    
    Args:
        df: DataFrame avec colonnes 'Date' et target_index
        target_index: Nom de la colonne cible
    
    Returns:
        Modèle Prophet entraîné
    """
    # Préparer données pour Prophet (format ds, y)
    df_prophet = df[['Date', target_index]].copy()
    df_prophet.columns = ['ds', 'y']
    
    # Créer et entraîner le modèle
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.05
    )
    
    model.fit(df_prophet)
    
    return model

def predict_prophet(model, n_periods):
    """
    Génère des prévisions avec Prophet
    
    Args:
        model: Modèle Prophet entraîné
        n_periods: Nombre de périodes à prédire
    
    Returns:
        DataFrame avec colonnes: Date, Forecast, Lower_CI, Upper_CI
    """
    # Créer dataframe futur
    future = model.make_future_dataframe(
        periods=n_periods,
        freq='MS',
        include_history=False
    )
    
    # Générer prévisions
    forecast = model.predict(future)
    
    # Extraire résultats
    result_df = pd.DataFrame({
        'Date': forecast['ds'],
        'Forecast': forecast['yhat'],
        'Lower_CI': forecast['yhat_lower'],
        'Upper_CI': forecast['yhat_upper']
    })
    
    return result_df

def evaluate_prophet(model, df, target_index):
    """
    Évalue les performances du modèle Prophet
    
    Args:
        model: Modèle entraîné
        df: DataFrame complet
        target_index: Nom de la colonne cible
    
    Returns:
        Dictionnaire avec métriques (mae, rmse, mape)
    """
    # Split train/test
    train_size = int(len(df) * 0.85)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    
    # Préparer pour Prophet
    df_prophet_train = train_df[['Date', target_index]].copy()
    df_prophet_train.columns = ['ds', 'y']
    
    df_prophet_test = test_df[['Date', target_index]].copy()
    df_prophet_test.columns = ['ds', 'y']
    
    # Entraîner sur train
    model_eval = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )
    model_eval.fit(df_prophet_train)
    
    # Prédire sur test
    future_test = model_eval.make_future_dataframe(
        periods=len(test_df),
        freq='MS'
    )
    forecast = model_eval.predict(future_test)
    pred_test = forecast.tail(len(test_df))['yhat'].values
    
    # Calculer métriques
    actual = df_prophet_test['y'].values
    mae = mean_absolute_error(actual, pred_test)
    rmse = np.sqrt(mean_squared_error(actual, pred_test))
    mape = np.mean(np.abs((actual - pred_test) / actual)) * 100
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape
    }