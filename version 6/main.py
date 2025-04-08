import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

def get_finger_states(landmarks):
    """
    Returns list indicating if each finger is open (1) or closed (0)
    Order: Thumb, Index, Middle, Ring, Pinky
    """
    finger_states = []

    # Thumb (check x-axis instead of y)
    if landmarks[4].x < landmarks[3].x:
        finger_states.append(1)
    else:
        finger_states.append(0)

    # Fingers: tip.y < pip.y means finger is up
    tips_ids = [8, 12, 16, 20]
    pip_ids = [6, 10, 14, 18]

    for tip, pip in zip(tips_ids, pip_ids):
        if landmarks[tip].y < landmarks[pip].y:
            finger_states.append(1)
        else:
            finger_states.append(0)

    return finger_states

def detect_sign(finger_states):
    """
    Map finger states to gestures
    """
    thumb, index, middle, ring, pinky = finger_states

    if finger_states == [0, 0, 0, 0, 0]:
        return "Fist 👊"
    elif finger_states == [1, 1, 1, 1, 1]:
        return "Open Palm ✋"
    elif finger_states == [1, 0, 0, 0, 0]:
        return "Thumbs Up 👍"
    elif finger_states == [0, 1, 1, 0, 0]:
        return "Victory ✌"
    elif finger_states == [0, 1, 0, 0, 0]:
        return "Pointing ☝"
    else:
        return "Unknown"

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

    # Show sign label
    cv2.putText(image, f'Sign: {sign_text}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow('Manual Sign Detection', image)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()