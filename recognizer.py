import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load the trained model
model = joblib.load('gesture_model.pkl')
gestures = ['open_hand', 'fist', 'peace']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break
    
    # Detect hands
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if results.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks.extend([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        
        # Flatten and predict
        input_data = np.array(landmarks).flatten().reshape(1, -1)
        prediction = model.predict(input_data)
        gesture_name = gestures[prediction[0]]
        
        # Display the result
        cv2.putText(frame, gesture_name, (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Gesture Recognizer', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()