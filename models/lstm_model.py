# ==================== models/lstm_model.py ====================
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

def create_sequences(data, target_col, seq_length=12):
    """Crée séquences pour LSTM"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[target_col].iloc[i:i+seq_length].values)
        y.append(data[target_col].iloc[i+seq_length])
    return np.array(X), np.array(y)

def train_lstm(df, target_index, seq_length=12):
    """Entraîne un modèle LSTM"""
    # Normaliser
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[target_index] = scaler.fit_transform(df[[target_index]])
    
    # Créer séquences
    X, y = create_sequences(df_scaled, target_index, seq_length)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    # Créer modèle
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Entraîner
    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    model.fit(X, y, epochs=50, batch_size=32, verbose=0, callbacks=[early_stop])
    
    return model, scaler


def predict_lstm(model, scaler, df, target_index, n_periods, seq_length=12):
    # Dernière séquence (1 seule colonne)
    last_sequence = df[target_index].iloc[-seq_length:].values.reshape(-1, 1)

    # Scaling
    last_sequence_scaled = scaler.transform(last_sequence)
    current_seq = last_sequence_scaled.reshape((1, seq_length, 1))

    future_pred = []

    for _ in range(n_periods):
        next_pred = model.predict(current_seq, verbose=0)[0, 0]
        future_pred.append(next_pred)

        current_seq = np.roll(current_seq, -1, axis=1)
        current_seq[0, -1, 0] = next_pred

    # Inverse scaling
    future_pred = scaler.inverse_transform(
        np.array(future_pred).reshape(-1, 1)
    ).flatten()

    # Dates futures
    last_date = df['Date'].max()
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=n_periods,
        freq='MS'
    )

    return pd.DataFrame({
        'Date': future_dates,
        'Forecast': future_pred
    })




def evaluate_lstm(model, scaler, df, target_index):
    """Évalue LSTM"""
    train_size = int(len(df) * 0.85)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    # Normaliser
    train_scaled = train_df.copy()
    train_scaled[target_index] = scaler.fit_transform(train_df[[target_index]])
    test_scaled = test_df.copy()
    test_scaled[target_index] = scaler.transform(test_df[[target_index]])
    
    # Créer séquences
    X_test, y_test = create_sequences(test_scaled, target_index, 12)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    # Prédire
    pred_scaled = model.predict(X_test, verbose=0)
    pred = scaler.inverse_transform(pred_scaled).flatten()
    actual = test_df[target_index].iloc[12:].values
    
    return {
        'mae': mean_absolute_error(actual, pred),
        'rmse': np.sqrt(mean_squared_error(actual, pred)),
        'mape': np.mean(np.abs((actual - pred) / actual)) * 100
    }


