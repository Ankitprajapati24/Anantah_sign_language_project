# main.py

import cv2
import mediapipe as mp
import pyttsx3
from finger_utils import get_finger_states, detect_sign

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.9)
mp_drawing = mp.solutions.drawing_utils

# Initialize TTS
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)
last_spoken = ""

def speak_if_new(text):
    global last_spoken
    if text != last_spoken and text != "Unknown" and text != "No Hand Detected":
        engine.say(text)
        engine.runAndWait()
        last_spoken = text

# Start webcam
cap = cv2.VideoCapture(1)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    sign_text = "No Hand Detected"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            finger_states = get_finger_states(hand_landmarks.landmark)
            sign_text = detect_sign(finger_states)
            speak_if_new(sign_text)

    # Show sign label
    cv2.putText(image, f'Sign: {sign_text}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow('Gesture to Speech Translator', image)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
