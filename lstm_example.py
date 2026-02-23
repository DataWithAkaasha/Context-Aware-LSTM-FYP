from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate

def create_example_lstm(seq_len, n_features, context_size):
    seq_input = Input(shape=(seq_len, n_features))
    x = LSTM(64)(seq_input)
    context_input = Input(shape=(context_size,))
    context_layer = Dense(16, activation='relu')(context_input)
    merged = Concatenate()([x, context_layer])
    output = Dense(1, activation='sigmoid')(merged)
    return Model([seq_input, context_input], output)

# Example usage with dummy data
if __name__ == "__main__":
    import numpy as np
    X_dummy = np.random.rand(10, 30, 5)  # 10 samples, 30 timesteps, 5 features
    context_dummy = np.random.rand(10, 2)
    y_dummy = np.random.randint(0,2,10)

    model = create_example_lstm(30, 5, 2)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit([X_dummy, context_dummy], y_dummy, epochs=1)