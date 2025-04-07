import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmarks (x, y, z) for each of 21 points
            landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
    cv2.imshow('Sign Language Capture', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()