import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, Dropout

class GestureRecognizer:
    def __init__(self, num_classes):
        self.model = self.build_model(num_classes)
        
    def build_model(self, num_classes):
        model = Sequential()
        model.add(Conv1D(64, 3, activation='relu', input_shape=(30, 1662)))
        model.add(Conv1D(128, 3, activation='relu'))
        model.add(Dropout(0.5))
        model.add(LSTM(256, return_sequences=True))
        model.add(LSTM(128))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, X_train, y_train, epochs=50):
        self.model.fit(X_train, y_train, epochs=epochs, validation_split=0.2)

    def save_model(self, path):
        self.model.save(path)