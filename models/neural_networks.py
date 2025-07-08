from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization
import config
from tensorflow.keras import Input

def create_mlp(meta):
    
    model = Sequential([
        Input(shape=(config.NN_INPUT_SHAPE[0],)),
        Reshape((config.NN_INPUT_SHAPE[0],)),  
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(config.NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_cnn(meta):
    
    model = Sequential([
        
        Input(shape=(config.NN_INPUT_SHAPE[0],)),
        
        Reshape(config.NN_INPUT_SHAPE),
        
        Conv1D(filters=64, kernel_size=5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(config.NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
