from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, Input, Conv1D, MaxPooling1D, BatchNormalization
import config
from tensorflow.keras import Input

def create_mlp(optimizer='adam'):
    
    model = Sequential([
        Input(shape=config.NN_INPUT_SHAPE),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(config.NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_cnn(optimizer='adam'):
    """
    Uma arquitetura de CNN inspirada em padrões como VGG.
    """
    model = Sequential([
        Input(shape=config.NN_INPUT_SHAPE),
        Reshape((config.NN_INPUT_SHAPE[0], 1)),

        # --- Bloco Convolucional 1 ---
        Conv1D(filters=64, kernel_size=5, activation='relu', padding='same'),
        Conv1D(filters=64, kernel_size=5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),

        # --- Bloco Convolucional 2 ---
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # --- Parte Densa (Classificador) ---
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(config.NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
