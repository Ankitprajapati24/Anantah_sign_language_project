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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from collections import deque
from concurrent.futures import ThreadPoolExecutor

class IndianSignLanguageTranslator:
    def __init__(self):
        # Initialize MediaPipe solutions
        self.hand_indices = [0,4,5,8,9,12,13,16,17,20]
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.4
        )
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.4
        )
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.4
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.tts_lock = Lock()
        
        # Model and data handling
        self.model = None
        self.labels = []
        self.data = []
        
        # Create required directories
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        # Prediction state management
        self.last_prediction_time = time.time()
        self.prediction_cooldown = 2.0
        self.prediction_history = deque(maxlen=15)
        self.current_gesture = ""
        
        # Sequence handling
        self.gesture_sequence = []
        self.last_sequence_time = time.time()
        self.sequence_timeout = 5.0
        
        # Motion tracking
        self.motion_history = deque(maxlen=20)
        
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
        
        self.sentence_patterns = {
            ("hello", "friend"): "Hello my friend",
            ("help", "water"): "I need water please",
            ("thank_you", "food"): "Thank you for the food",
            ("school", "yes"): "Yes, I'm going to school",
            ("family", "friend"): "My family and friends"
        }

    def validate_feature_length(self, features):
        expected_length = 192
        if len(features) < expected_length:
            features += [0.0] * (expected_length - len(features))
        elif len(features) > expected_length:
            features = features[:expected_length]
        return features

    def extract_features(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (640, 480)) 
        features = []
        drawn = {"left_hand": False, "right_hand": False, "face": False, "pose": False}

        # Process hands
        hands_results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if hands_results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

                handedness = "Unknown"
                if hands_results.multi_handedness and idx < len(hands_results.multi_handedness):
                    handedness = hands_results.multi_handedness[idx].classification[0].label
                
                hand_type = f"{handedness.lower()}_hand"
                if hand_type in drawn:
                    drawn[hand_type] = True

                for landmark in hand_landmarks.landmark:
                    features.extend([landmark.x, landmark.y, landmark.z])

        # Pad missing hands
        if not drawn["left_hand"]:
            features.extend([0.0] * (21 * 3))
        if not drawn["right_hand"]:
            features.extend([0.0] * (21 * 3))

        # Process face
        try:
            face_results = self.face_mesh.process(frame_rgb)
            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0]
                self.mp_drawing.draw_landmarks(
                    frame, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                )
                key_face_indices = [0, 17, 61, 291, 199, 263, 33, 133, 362, 263, 78, 308]
                for idx in key_face_indices:
                    landmark = face_landmarks.landmark[idx]
                    features.extend([landmark.x, landmark.y, landmark.z])
                drawn["face"] = True
        except Exception as e:
            print(f"Face processing error: {e}")

        if not drawn["face"]:
            features.extend([0.0] * (12 * 3))

        # Process pose
        pose_results = self.pose.process(frame_rgb)
        if pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            upper_body_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24]
            for idx in upper_body_indices:
                landmark = pose_results.pose_landmarks.landmark[idx]
                features.extend([landmark.x, landmark.y, landmark.z])
            drawn["pose"] = True
        
        if not drawn["pose"]:
            features.extend([0.0] * (9 * 3))

        # Motion features
        self.motion_history.append(features[:126])
        if len(self.motion_history) >= 10:
            current = np.array(self.motion_history[-1])
            previous = np.array(self.motion_history[-10])
            if current.size > 0 and previous.size > 0:
                velocity = current - previous
                features.extend([np.mean(np.abs(velocity)), np.max(np.abs(velocity)), np.std(velocity)])
            else:
                features.extend([0.0, 0.0, 0.0])
        else:
            features.extend([0.0, 0.0, 0.0])

        features = self.validate_feature_length(features)
        return features, frame

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

    def speak_text(self, text):
        def speak_task():
            with self.tts_lock:
                try:
                    self.engine.say(text)
                    self.engine.runAndWait()
                except Exception as e:
                    print(f"TTS error: {e}")

        threading.Thread(target=speak_task, daemon=True).start()

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
                frame_counter += 1
                if frame_counter % 2 != 0:  # Process every 2nd frame
                    continue

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