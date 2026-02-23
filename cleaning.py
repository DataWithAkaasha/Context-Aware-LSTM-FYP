import pandas as pd

def clean_timeseries_data(data_path, label_path, save_data_path, save_label_path):
    """Clean raw time series data and save cleaned CSVs"""
    df = pd.read_csv(data_path)
    labels = pd.read_csv(label_path)

    # Drop duplicates & invalid timestamps
    df = df.drop_duplicates()
    df = df.dropna(subset=['timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])

    # Remove constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    df.drop(columns=constant_cols, inplace=True)

    # Drop category column if exists
    if 'category' in df.columns:
        df.drop(columns=['category'], inplace=True)

    # Separate target
    if 'y' in df.columns:
        y = df['y'].copy()
        df.drop(columns=['y'], inplace=True)
    else:
        y = labels['y'].copy()

    # Context features
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek

    # Rolling features
    rolling_features = ['aimp', 'amud', 'adbr', 'adfl', 'arnd', 'asin1', 'asin2', 'bed1']
    for col in rolling_features:
        if col in df.columns:
            df[f'{col}_rolling_mean'] = df[col].rolling(window=30, min_periods=1).mean().shift(1)

    df.fillna(0, inplace=True)

    # Drop non-numeric columns except timestamp
    non_numeric_cols = [c for c in df.select_dtypes(exclude=['number']).columns if c != 'timestamp']
    df.drop(columns=non_numeric_cols, inplace=True)

    # Save
    df.to_csv(save_data_path, index=False)
    y.to_csv(save_label_path, index=False)
    print("✅ Cleaned data saved!")