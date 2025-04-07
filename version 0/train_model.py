import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # Simple classifier
import joblib  # To save the model

gestures = ['open_hand', 'fist', 'peace']

# Load data
X = []
y = []
for idx, gesture in enumerate(gestures):
    files = os.listdir(f'data/{gesture}')
    for file in files:
        data = np.load(f'data/{gesture}/{file}')
        X.append(data.flatten())  # Flatten 21 landmarks (x,y,z) into 63 values
        y.append(idx)

X = np.array(X)
y = np.array(y)

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a simple SVM classifier
model = SVC()
model.fit(X_train, y_train)

# Test accuracy
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save the model
joblib.dump(model, 'gesture_model.pkl')