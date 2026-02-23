import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_cleaned_data(data_path, labels_path):
    data = pd.read_csv(data_path)
    labels = pd.read_csv(labels_path)
    if 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
        data = data.set_index('timestamp')
    data['y'] = labels['y'].values
    return data

def basic_eda(data, input_features, external_features, telemetry_features):
    all_features = input_features + external_features + telemetry_features

    # Daily anomaly plot
    if isinstance(data.index, pd.DatetimeIndex):
        plt.figure(figsize=(12,6))
        daily_anomalies = data['y'].resample('D').sum()
        daily_anomalies.plot(kind='bar')
        plt.title(f'Daily Anomaly Count: {data["y"].sum()}')
        plt.xlabel('Date'); plt.ylabel('Count'); plt.tight_layout(); plt.show()

    # Feature distributions (first 4)
    plt.figure(figsize=(14,8))
    for i, col in enumerate(all_features[:4],1):
        plt.subplot(2,2,i)
        sns.histplot(data[col], kde=True, bins=50)
        anomalies = data[data['y']==1]
        for val in anomalies[col].sample(min(20, len(anomalies))):
            plt.axvline(x=val, color='red', alpha=0.3, linestyle='--')
        plt.title(f'Distribution: {col}'); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.show()

def prepare_lstm_features(data, input_features, external_features, telemetry_features):
    all_features = input_features + external_features + telemetry_features
    lstm_features = []

    # Lag and diff features
    for col in all_features:
        data[f'{col}_lag1'] = data[col].shift(1)
        data[f'{col}_diff1'] = data[col] - data[col].shift(1)
        lstm_features.extend([f'{col}_lag1', f'{col}_diff1'])

    # Time features
    if isinstance(data.index, pd.DatetimeIndex):
        data['hour'] = data.index.hour
        data['dayofweek'] = data.index.dayofweek
        data['hour_sin'] = np.sin(2*np.pi*data['hour']/24)
        data['hour_cos'] = np.cos(2*np.pi*data['hour']/24)
        data['dow_sin'] = np.sin(2*np.pi*data['dayofweek']/7)
        data['dow_cos'] = np.cos(2*np.pi*data['dayofweek']/7)
        lstm_features.extend(['hour_sin','hour_cos','dow_sin','dow_cos'])

    data.fillna(method='bfill', inplace=True)
    return data, lstm_features

def save_lstm_ready(data, lstm_features, save_path):
    data[lstm_features + ['y']].to_csv(save_path, index=False)
    print(f"LSTM-ready data saved to {save_path}")


from eda_lstm import load_cleaned_data, basic_eda, prepare_lstm_features, save_lstm_ready

input_data = '../data/cldT_data.csv'
input_labels = '../data/yT_labs.csv'
output_data = '../data/CATS_LSTM_ready.csv'

input_features = ['aimp', 'amud', 'adbr', 'adfl']
external_features = ['arnd', 'asin1', 'asin2']
telemetry_features = ['bed1', 'bed2', 'bfo1', 'bfo2', 'bso1', 'bso2', 'bso3', 'ced1', 'cfo1', 'cso1']

data = load_cleaned_data(input_data, input_labels)
basic_eda(data, input_features, external_features, telemetry_features)
data, lstm_features = prepare_lstm_features(data, input_features, external_features, telemetry_features)
save_lstm_ready(data, lstm_features, output_data)