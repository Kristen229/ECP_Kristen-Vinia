import pandas as pd
import numpy as np

def load_and_prepare_data(df, target_index):
    """
    Prépare les données pour l'entraînement des modèles
    
    Args:
        df: DataFrame avec colonnes Date et target_index
        target_index: Nom de la colonne cible (ex: 'FIP')
    
    Returns:
        DataFrame préparé avec features temporelles et lag features
    """
    df = df.copy()
    
    # S'assurer que Date est en datetime
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
    
    # S'assurer que la colonne Date existe
    if 'Date' not in df.columns:
        if 'date' in df.columns:
            df['Date'] = pd.to_datetime(df['date'], errors='coerce')
        else:
            if 'Date' not in df.columns:
                raise ValueError("La colonne 'Date' est manquante dans le dataframe")

            """ st.error("La colonne 'Date' est manquante dans le dataframe") """
            """ return None """

    
    # Trier par date
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Features temporelles
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    
    # Lag features (variables retardées)
    for lag in [1, 2, 3, 6, 12]:
        df[f'{target_index}_lag_{lag}'] = df[target_index].shift(lag)
    
    # Moyennes mobiles
    df[f'{target_index}_MA3'] = df[target_index].rolling(window=3).mean()
    df[f'{target_index}_MA6'] = df[target_index].rolling(window=6).mean()
    df[f'{target_index}_MA12'] = df[target_index].rolling(window=12).mean()
    
    # Supprimer les NaN dus aux lags
    df = df.dropna().reset_index(drop=True)
    
    return df

def split_train_test(df, test_size=0.15):
    """
    Divise les données en ensemble d'entraînement et de test
    
    Args:
        df: DataFrame à diviser
        test_size: Proportion du test set (entre 0 et 1)
    
    Returns:
        train_df, test_df
    """
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    return train_df, test_df

def create_future_dates(last_date, n_periods, freq='MS'):
    """
    Crée un array de dates futures
    
    Args:
        last_date: Dernière date connue
        n_periods: Nombre de périodes à générer
        freq: Fréquence ('MS' pour début de mois, 'D' pour jour)
    
    Returns:
        Array de dates
    """
    """     return pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=n_periods,
        freq=freq
    ) """

    return pd.date_range(
    start=last_date + pd.DateOffset(months=1),
    periods=n_periods,
    freq=freq
)
