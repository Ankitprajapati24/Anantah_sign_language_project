import cv2
import mediapipe as mp
import os
import numpy as np

# Create folders if they don't exist
gestures = ['open_hand', 'fist', 'peace']
for gesture in gestures:
    os.makedirs(f'data/{gesture}', exist_ok=True)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

counter = 0  # To count saved images
gesture_index = 0  # Current gesture (0: open_hand, 1: fist, 2: peace)

while True:
    success, frame = cap.read()
    if not success:
        break
    
    # Show instructions
    cv2.putText(frame, f"Current Gesture: {gestures[gesture_index]}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "Press 's' to save, 'n' for next gesture", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Detect hands
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Capture key press (MOVED OUTSIDE THE HAND DETECTION BLOCK)
    key = cv2.waitKey(1)  # <--- This line was missing/fix here
    
    if results.multi_hand_landmarks:
        # Save landmarks when 's' is pressed
        if key == ord('s'):
            landmarks = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks[:2]:
                    landmarks.extend([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            if len(landmarks) < 42:
                landmarks += [[0.0, 0.0, 0.0]] * (42 - len(landmarks))
                landmarks = landmarks[:42] 
            
            # Save as .npy file
            np.save(f'data/{gestures[gesture_index]}/{counter}.npy', landmarks)
            counter += 1
            print(f"Saved {counter} samples for {gestures[gesture_index]}")
        
        # Switch gesture when 'n' is pressed
        elif key == ord('n'):
            gesture_index = (gesture_index + 1) % 3
            counter = 0
            print(f"Switched to {gestures[gesture_index]}")
    
    cv2.imshow('Data Collector', frame)
    
    # Quit when 'q' is pressed (check outside hand detection)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()