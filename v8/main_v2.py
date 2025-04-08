import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import pickle
import time
import pyttsx3
import mediapipe as mp
import threading
from threading import Lock
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from collections import deque

class IndianSignLanguageTranslator:
    def __init__(self):
        # Optimized MediaPipe configuration
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Single hand detection
            min_detection_confidence=0.6,
            min_tracking_confidence=0.4
        )
        
        # Text-to-speech engine with queue management
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 160)
        self.tts_lock = Lock()
        self.is_speaking = False
        
        # Model configuration
        self.model = None
        self.labels = []
        self.data = []
        
        # Gesture stabilization
        self.prediction_history = deque(maxlen=7)  # Reduced buffer size
        self.last_spoken_time = time.time()
        self.cooldown = 2.5  # Increased cooldown
        self.last_gesture = None
        
        # File paths
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)

        # ISL dictionary and patterns (keep existing)

    def extract_features(self, frame):
        # Resize frame for faster processing
        frame = cv2.resize(frame, (640, 480))
        features = []
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Hand landmarks only
        hands_results = self.hands.process(frame_rgb)
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                # Track only key landmarks (thumb, index, middle tips)
                key_indices = [4, 8, 12, 16, 20]
                for idx in key_indices:
                    landmark = hand_landmarks.landmark[idx]
                    features.extend([landmark.x, landmark.y, landmark.z])
        else:
            features.extend([0.0] * (5 * 3))  # 5 key points * 3 coordinates

        # Pad features to fixed size
        return self.validate_feature_length(features), frame

    def validate_feature_length(self, features):
        expected_length = 192
        return features[:expected_length] + [0.0]*(expected_length - len(features))

    def run_recognition(self):
        if not self.model and not self.load_model():
            return

        cap = cv2.VideoCapture(0)
        frame_skip = 3  # Process every 3rd frame
        frame_counter = 0
        
        try:
            while cap.isOpened():
                success, frame = cap.read()
                frame_counter += 1
                if not success or frame_counter % frame_skip != 0:
                    continue

                # Feature extraction
                features, processed_frame = self.extract_features(frame)
                
                # Prediction
                gesture = self.predict_gesture(features)
                if gesture:
                    self.prediction_history.append(gesture)
                    
                    # Stabilize prediction
                    if len(self.prediction_history) >= 5:
                        majority_gesture = max(set(self.prediction_history), 
                                            key=self.prediction_history.count)
                        
                        current_time = time.time()
                        if (majority_gesture != self.last_gesture or 
                            current_time - self.last_spoken_time > self.cooldown):
                            
                            self.display_and_speak(processed_frame, majority_gesture)
                            self.last_gesture = majority_gesture
                            self.last_spoken_time = current_time
                            self.prediction_history.clear()

                cv2.imshow("Recognition", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

    def display_and_speak(self, frame, gesture):
        translated = self.isl_dictionary.get(gesture, gesture)
        cv2.putText(frame, f"Gesture: {translated}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if not self.is_speaking:
            self.is_speaking = True
            threading.Thread(target=self.safe_speak, args=(gesture,)).start()

    def safe_speak(self, text):
        with self.tts_lock:
            try:
                self.engine.stop()
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"TTS error: {e}")
            finally:
                self.is_speaking = False

    def run_recognition(self):
        if not self.model and not self.load_model():
            return
        
        frame_counter = 0

        cap = None
        try:
            for camera_idx in [0, 1]:
                cap = cv2.VideoCapture(camera_idx)
                if cap.isOpened():
                    print(f"Using camera index {camera_idx}")
                    break
            else:
                print("Camera error")
                return

            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                features, frame = self.extract_features(frame)

                if features and len(features) == 192:
                    gesture = self.predict_gesture(features)
                    if gesture:
                        translated = self.isl_dictionary.get(gesture, gesture)
                        
                        # Display gesture
                        cv2.putText(frame, f"Gesture: {translated}", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Speak gesture
                        current_time = time.time()
                        if current_time - self.last_prediction_time > self.prediction_cooldown:
                            self.speak_text(gesture)
                            self.last_prediction_time = current_time

                cv2.imshow("Recognition", frame)
                if cv2.waitKey(1) == ord('q'):
                    break

        finally:
            if cap and cap.isOpened():
                self.safe_capture_release(cap)
            cv2.destroyAllWindows()

def main():
    translator = IndianSignLanguageTranslator()
    
    while True:
        print("\n===== INDIAN SIGN LANGUAGE TRANSLATOR =====")
        print("1. Collect training data")
        print("2. Train model")
        print("3. Run real-time recognition")
        print("4. Exit")
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == '1':
            translator.collect_data()
        elif choice == '2':
            translator.train_model()
        elif choice == '3':
            translator.run_recognition()
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()