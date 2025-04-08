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
        # MediaPipe configuration
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.4
        )
        
        # Text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 160)
        self.tts_lock = Lock()
        self.is_speaking = False
        
        # Model and data
        self.model = None
        self.labels = []
        self.data = []
        
        # Prediction stabilization
        self.prediction_history = deque(maxlen=7)
        self.last_spoken_time = time.time()
        self.cooldown = 2.5
        self.last_gesture = None
        
        # File paths
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)

        # ISL dictionary
        self.isl_dictionary = {
            "hello": "नमस्ते (Namaste)",
            "thank_you": "धन्यवाद (Dhanyavaad)",
            "yes": "हां (Haan)",
            "no": "नहीं (Nahi)",
            "help": "मदद (Madad)",
            "water": "पानी (Paani)",
            "food": "खाना (Khaana)",
            "family": "परिवार (Parivaar)",
            "friend": "दोस्त (Dost)",
            "school": "स्कूल (School)",
        }

    def validate_feature_length(self, features):
        expected_length = 192
        return features[:expected_length] + [0.0]*(expected_length - len(features))

    def extract_features(self, frame):
        frame = cv2.resize(frame, (640, 480))
        features = []
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Hand landmarks processing
        hands_results = self.hands.process(frame_rgb)
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                key_indices = [4, 8, 12, 16, 20]
                for idx in key_indices:
                    landmark = hand_landmarks.landmark[idx]
                    features.extend([landmark.x, landmark.y, landmark.z])
        else:
            features.extend([0.0] * (5 * 3))

        return self.validate_feature_length(features), frame

    def safe_capture_release(self, cap):
        try:
            cap.release()
        except Exception as e:
            print(f"Error releasing camera: {e}")

    def collect_data(self):
        print("\n===== DATA COLLECTION MODE =====")
        cap = None
        try:
            for camera_idx in [0, 1]:
                cap = cv2.VideoCapture(camera_idx)
                if cap.isOpened():
                    print(f"Using camera index {camera_idx}")
                    break
            else:
                print("Error: Could not open webcam")
                return

            collecting = False
            current_label = ""
            start_time = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Frame capture error")
                    break

                frame = cv2.flip(frame, 1)
                features, frame = self.extract_features(frame)

                if collecting:
                    elapsed = time.time() - start_time
                    if elapsed >= 3.0:
                        collecting = False
                        print(f"Finished collecting {current_label}")
                        continue

                    self.data.append(features)
                    self.labels.append(current_label)
                    progress = min(elapsed / 3.0, 1.0)
                    cv2.putText(frame, f"Collecting {current_label}: {progress:.0%}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Press 'c' to collect, 'q' to quit", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                cv2.imshow("Data Collection", frame)
                key = cv2.waitKey(1)

                if key == ord('q'):
                    break
                elif key == ord('c') and not collecting:
                    current_label = input("Enter gesture name: ")
                    collecting = True
                    start_time = time.time()
                    print(f"Collecting: {current_label}")

        finally:
            if cap and cap.isOpened():
                self.safe_capture_release(cap)
            cv2.destroyAllWindows()

            if self.data:
                with open('data/isl_gesture_data.pkl', 'wb') as f:
                    pickle.dump((self.data, self.labels), f)
                print(f"Saved {len(self.data)} samples")

    def load_data(self):
        try:
            with open('data/isl_gesture_data.pkl', 'rb') as f:
                self.data, self.labels = pickle.load(f)
            print(f"Loaded {len(self.data)} samples")
            return True
        except FileNotFoundError:
            print("No data found!")
            return False

    def train_model(self):
        if not self.data and not self.load_data():
            return False

        X = np.array(self.data)
        y = np.array(self.labels)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model = SGDClassifier(loss='log_loss', penalty='l2', max_iter=1000)
        self.model.fit(X_train, y_train)
        
        accuracy = self.model.score(X_test, y_test)
        print(f"Model trained! Accuracy: {accuracy:.2f}")
        
        with open('models/isl_gesture_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        return True

    def load_model(self):
        try:
            with open('models/isl_gesture_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            print("Model loaded successfully")
            return True
        except FileNotFoundError:
            print("No trained model found!")
            return False

    def predict_gesture(self, features):
        if not self.model or len(features) != 192:
            return None
            
        try:
            return self.model.predict([features])[0]
        except Exception as e:
            print(f"Prediction error: {e}")
            return None

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

        cap = cv2.VideoCapture(0)
        frame_skip = 3
        frame_counter = 0
        
        try:
            while cap.isOpened():
                success, frame = cap.read()
                frame_counter += 1
                if not success or frame_counter % frame_skip != 0:
                    continue

                features, processed_frame = self.extract_features(frame)
                
                gesture = self.predict_gesture(features)
                if gesture:
                    self.prediction_history.append(gesture)
                    
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