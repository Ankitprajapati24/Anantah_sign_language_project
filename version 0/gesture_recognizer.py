import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Download the pre-trained gesture model
model_path = "gesture_recognizer.task"  # Will auto-download
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)


# Open webcam
cap = cv2.VideoCapture(1)

with mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6
) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # Convert to RGB and process
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect hands
        hand_results = hands.process(rgb_frame)
        
        if hand_results.multi_hand_landmarks:
            # Recognize gestures
            gesture_result = recognizer.recognize(mp_image)
            
            if gesture_result.gestures:
                top_gesture = gesture_result.gestures[0][0]
                gesture_name = top_gesture.category_name
                confidence = top_gesture.score
                
                # Display result if confidence > 70%
                if confidence > 0.7:
                    cv2.putText(frame, f"{gesture_name} ({confidence:.2f})", (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Gesture Recognizer', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()