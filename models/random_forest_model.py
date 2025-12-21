# ==================== models/random_forest_model.py ====================
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np


def train_rf(df, target_index):
    """Entraîne Random Forest"""
    """ feature_cols = [col for col in df.columns if col not in ['Date', target_index]]
    X = df[feature_cols]
    y = df[target_index]
    
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    return model """

    # Garder uniquement les colonnes numériques
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if target_index not in numeric_cols:
        raise ValueError(f"{target_index} n'est pas une colonne numérique")

    feature_cols = [col for col in numeric_cols if col != target_index]

    X = df[feature_cols]
    y = df[target_index]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)
    return model

def predict_rf(model, df,target_index, n_periods):
    """Génère prévisions avec Random Forest"""
    """ last_date = df['Date'].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=n_periods, freq='MS')
    
    # Utiliser dernière observation pour features
    last_obs = df.iloc[-1:].copy()
    predictions = []
    
    for i, date in enumerate(future_dates):
        # Mettre à jour features temporelles
        last_obs['Year'] = date.year
        last_obs['Month'] = date.month
        last_obs['Quarter'] = date.quarter
        last_obs['Month_sin'] = np.sin(2 * np.pi * date.month / 12)
        last_obs['Month_cos'] = np.cos(2 * np.pi * date.month / 12)
        
        # Prédire
        feature_cols = [col for col in last_obs.columns if col != 'Date']
        pred = model.predict(last_obs[feature_cols])[0]
        predictions.append(pred)
    
    return pd.DataFrame({'Date': future_dates, 'Forecast': predictions}) """
    last_date = df['Date'].max()
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=n_periods,
        freq='MS'
    )

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    feature_cols = [col for col in numeric_cols if col != target_index]

    last_obs = df.iloc[-1:].copy()
    predictions = []

    for date in future_dates:
        last_obs['Year'] = date.year
        last_obs['Month'] = date.month
        last_obs['Quarter'] = date.quarter
        last_obs['Month_sin'] = np.sin(2 * np.pi * date.month / 12)
        last_obs['Month_cos'] = np.cos(2 * np.pi * date.month / 12)

        pred = model.predict(last_obs[feature_cols])[0]
        predictions.append(pred)

    return pd.DataFrame({
        'Date': future_dates,
        'Forecast': predictions
    })


def evaluate_rf(model, df, target_index):
    """Évalue Random Forest"""
    train_size = int(len(df) * 0.85)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    feature_cols = [col for col in df.columns if col not in ['Date', target_index]]
    X_test = test_df[feature_cols]
    y_test = test_df[target_index]
    
    pred = model.predict(X_test)
    
    return {
        'mae': mean_absolute_error(y_test, pred),
        'rmse': np.sqrt(mean_squared_error(y_test, pred)),
        'mape': np.mean(np.abs((y_test - pred) / y_test)) * 100
    }