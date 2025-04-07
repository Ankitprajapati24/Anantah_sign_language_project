import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Flatten

model = Sequential([
    # Spatial feature extraction (CNN)
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(30, 63)),  # 21 landmarks * 3 (x,y,z) * 2 hands
    Dropout(0.2),
    # Temporal modeling (LSTM)
    LSTM(128, return_sequences=True),
    LSTM(64),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')  # num_classes = number of gestures
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])