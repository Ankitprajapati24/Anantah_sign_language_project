import os
import cv2
import numpy as np
import pickle
import time
import pyttsx3
import mediapipe as mp
import threading
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from collections import deque

class IndianSignLanguageTranslator:
    def __init__(self):
        # Initialize MediaPipe solutions
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Track up to 2 hands
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        
        # Initialize model and data variables
        self.model = None
        self.labels = []
        self.data = []
        
        # Create directories for saving data and models
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        # Variables for prediction smoothing and cooldown
        self.last_prediction_time = time.time()
        self.prediction_cooldown = 2.0  # seconds
        self.prediction_history = deque(maxlen=15)
        self.current_gesture = None
        
        # Sequence detection for continuous gestures/sentences
        self.gesture_sequence = []
        self.last_sequence_time = time.time()
        self.sequence_timeout = 5.0  # seconds before considering a sequence complete
        
        # Motion tracking for dynamic gestures
        self.motion_history = deque(maxlen=20)  # Store recent landmark positions
        self.motion_features = []  # Store extracted motion features
        
        # Dictionary of common Indian Sign Language phrases
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
        
        # Some predefined sentence patterns (for demonstration)
        self.sentence_patterns = {
            ("hello", "friend"): "Hello my friend",
            ("help", "water"): "I need water please",
            ("thank_you", "food"): "Thank you for the food",
            ("school", "yes"): "Yes, I'm going to school",
            ("family", "friend"): "My family and friends"
        }
    
    def extract_features(self, frame):
        """Extract hand, face and pose landmarks from a frame"""
        # Convert the frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process hands
        hands_results = self.hands.process(frame_rgb)
        
        # Process face
        face_results = self.face_mesh.process(frame_rgb)
        
        # Process pose
        pose_results = self.pose.process(frame_rgb)
        
        # Initialize features
        features = []
        
        # Dictionary to track if we've drawn each type of landmark
        drawn = {"left_hand": False, "right_hand": False, "face": False, "pose": False}
        
        # Extract hand landmarks (up to 2 hands)
        if hands_results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Determine if this is a left or right hand
                if idx < len(hands_results.multi_handedness):
                    handedness = hands_results.multi_handedness[idx].classification[0].label
                    drawn[f"{handedness.lower()}_hand"] = True
                    
                    # Add handedness label to frame
                    cv2.putText(
                        frame,
                        f"{handedness} Hand",
                        (int(hand_landmarks.landmark[0].x * frame.shape[1]), 
                         int(hand_landmarks.landmark[0].y * frame.shape[0] - 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1
                    )
                
                # Extract landmark coordinates and flatten them
                for landmark in hand_landmarks.landmark:
                    features.extend([landmark.x, landmark.y, landmark.z])
        
        # If we didn't detect both hands, pad the features with zeros
        if not drawn["left_hand"]:
            features.extend([0.0] * (21 * 3))  # 21 landmarks * 3 coordinates
        if not drawn["right_hand"]:
            features.extend([0.0] * (21 * 3))  # 21 landmarks * 3 coordinates
            
        # Extract key facial landmarks (simplified - just using key points)
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            drawn["face"] = True
            
            # Draw face landmarks
            self.mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            
            # Key facial points (eyes, eyebrows, nose, mouth)
            key_face_indices = [0, 17, 61, 291, 199, 263, 33, 133, 362, 263, 78, 308]
            for idx in key_face_indices:
                landmark = face_landmarks.landmark[idx]
                features.extend([landmark.x, landmark.y, landmark.z])
        else:
            # If no face detected, pad with zeros
            features.extend([0.0] * (12 * 3))  # 12 key facial points * 3 coordinates
        
        # Extract key pose landmarks
        if pose_results.pose_landmarks:
            pose_landmarks = pose_results.pose_landmarks
            drawn["pose"] = True
            
            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(
                frame,
                pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style() 
            )
            
            # Use upper body landmarks for sign language
            upper_body_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24]
            for idx in upper_body_indices:
                landmark = pose_landmarks.landmark[idx]
                features.extend([landmark.x, landmark.y, landmark.z])
        else:
            # If no pose detected, pad with zeros
            features.extend([0.0] * (9 * 3))  # 9 upper body pose points * 3 coordinates
        
        # Extract motion features
        self.motion_history.append(features[:126])  # Store only hand features for motion
        
        # If we have enough history, calculate motion features
        if len(self.motion_history) >= 10:
            # Calculate velocity between current and previous frames
            current = np.array(self.motion_history[-1])
            previous = np.array(self.motion_history[-10])
            
            if len(current) > 0 and len(previous) > 0:
                velocity = current - previous
                # Add some velocity statistics to our features
                features.extend([
                    np.mean(np.abs(velocity)),
                    np.max(np.abs(velocity)),
                    np.std(velocity)
                ])
            else:
                features.extend([0.0, 0.0, 0.0])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        return features, frame
    
    def collect_data(self):
        """Collect training data for different gestures"""
        print("\n===== DATA COLLECTION MODE =====")
        print("Press 'c' to start collecting a new gesture")
        print("Press 'm' to collect a moving gesture (demonstrate the gesture for 3 seconds)")
        print("Press 'q' to quit data collection mode\n")
        
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        collecting = False
        collecting_moving = False
        current_label = None
        frames_collected = 0
        start_time = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Flip the frame horizontally for a more intuitive view
            frame = cv2.flip(frame, 1)
            
            features, frame = self.extract_features(frame)
            
            if collecting and features:
                if collecting_moving:
                    # For moving gestures, collect for 3 seconds
                    elapsed_time = time.time() - start_time
                    progress = min(elapsed_time / 3.0, 1.0)
                    frames_collected = int(progress * 100)
                    
                    # Display collection progress
                    cv2.putText(
                        frame, 
                        f"Collecting moving gesture: {current_label} - {frames_collected}%", 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (0, 255, 0), 
                        2
                    )
                    
                    # Add motion features
                    self.data.append(features)
                    self.labels.append(current_label)
                    
                    if elapsed_time >= 3.0:
                        print(f"Collected moving gesture '{current_label}'")
                        collecting = False
                        collecting_moving = False
                else:
                    # Static gesture collection
                    self.data.append(features)
                    self.labels.append(current_label)
                    frames_collected += 1
                    
                    # Display collection progress
                    cv2.putText(
                        frame, 
                        f"Collecting: {current_label} - {frames_collected}/100", 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (0, 255, 0), 
                        2
                    )
                    
                    if frames_collected >= 100:  # Collect 100 frames per gesture
                        print(f"Collected 100 frames for '{current_label}'")
                        collecting = False
                        frames_collected = 0
            
            elif not collecting:
                cv2.putText(
                    frame, 
                    "Press 'c' for static gesture, 'm' for moving gesture", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (255, 0, 0), 
                    2
                )
            
            cv2.imshow("Data Collection", frame)
            key = cv2.waitKey(1)
            
            if key == ord('q'):
                break
            elif key == ord('c') and not collecting:
                current_label = input("Enter the name of the gesture (e.g., hello, yes, no): ")
                collecting = True
                collecting_moving = False
                frames_collected = 0
                print(f"Starting to collect data for static gesture '{current_label}'. Make the gesture...")
            elif key == ord('m') and not collecting:
                current_label = input("Enter the name of the moving gesture: ")
                collecting = True
                collecting_moving = True
                start_time = time.time()
                print(f"Starting to collect data for moving gesture '{current_label}'. Perform the gesture for 3 seconds...")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Save collected data
        if self.data and self.labels:
            with open('data/isl_gesture_data.pkl', 'wb') as f:
                pickle.dump((self.data, self.labels), f)
            print(f"Data saved: {len(self.labels)} frames for {len(set(self.labels))} gestures")
        else:
            print("No data collected")
    
    def load_data(self):
        """Load previously collected data"""
        try:
            with open('data/isl_gesture_data.pkl', 'rb') as f:
                self.data, self.labels = pickle.load(f)
            print(f"Data loaded: {len(self.labels)} frames for {len(set(self.labels))} gestures")
            return True
        except FileNotFoundError:
            print("No saved data found. Please collect data first.")
            return False
        
    def train_model(self):
        """Train a RandomForest classifier on the collected data"""
        if not self.data or not self.labels:
            if not self.load_data():
                print("Cannot train: No data available")
                return False
        
        # Handle cases where some frames didn't capture all features
        valid_data = []
        valid_labels = []
        
        # Check the expected feature length from the first valid sample
        expected_length = None
        for features in self.data:
            if features:
                expected_length = len(features)
                break
        
        if expected_length is None:
            print("Error: No valid features found in data")
            return False
        
        print(f"Expected feature vector length: {expected_length}")
        
        for features, label in zip(self.data, self.labels):
            if len(features) == expected_length:
                valid_data.append(features)
                valid_labels.append(label)
        
        if not valid_data:
            print("Error: No valid data samples found")
            return False
        
        print(f"Using {len(valid_data)} valid samples out of {len(self.data)} total samples")
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            valid_data, valid_labels, test_size=0.2, random_state=42
        )
        
        # Train a RandomForest classifier
        print("Training model...")
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        accuracy = self.model.score(X_test, y_test)
        print(f"Model trained with accuracy: {accuracy:.2f}")
        
        # Save the trained model
        with open('models/isl_gesture_model.pkl', 'wb') as f:
            pickle.dump((self.model, list(set(valid_labels))), f)
        print("Model saved to models/isl_gesture_model.pkl")
        
        return True
    
    def load_model(self):
        """Load a trained model"""
        try:
            with open('models/isl_gesture_model.pkl', 'rb') as f:
                self.model, unique_labels = pickle.load(f)
            print(f"Model loaded. Recognizes gestures: {', '.join(unique_labels)}")
            return True
        except FileNotFoundError:
            print("No saved model found. Please train a model first.")
            return False
    
    def predict_gesture(self, features):
        """Predict gesture from features using the trained model"""
        if not self.model:
            return None
            
        if not features:
            return None
            
        # Check if feature vector has expected length
        expected_length = None
        for feature_vector in self.data:
            if feature_vector:
                expected_length = len(feature_vector)
                break
                
        if expected_length is None or len(features) != expected_length:
            return None
            
        prediction = self.model.predict([features])[0]
        
        # Add to prediction history for smoothing
        self.prediction_history.append(prediction)
        
        # Count occurrences of each prediction in history
        prediction_counts = {}
        for pred in self.prediction_history:
            if pred in prediction_counts:
                prediction_counts[pred] += 1
            else:
                prediction_counts[pred] = 1
        
        # Find the most common prediction
        most_common = max(prediction_counts.items(), key=lambda x: x[1])
        
        # Only consider it valid if it appears in at least 60% of recent frames
        if most_common[1] >= len(self.prediction_history) * 0.6:
            return most_common[0]
        else:
            return None
    
    def speak_text(self, text):
        """Speak text using text-to-speech in a separate thread"""
        def speak_thread():
            self.engine.say(text)
            self.engine.runAndWait()
        
        # Run in a separate thread to avoid blocking the main thread
        threading.Thread(target=speak_thread).start()
    
    def process_gesture_sequence(self, gesture):
        """Process a sequence of gestures to form sentences"""
        current_time = time.time()
        
        # If this is a new gesture or enough time has passed since the last one
        if (not self.gesture_sequence or 
            gesture != self.gesture_sequence[-1] or
            current_time - self.last_sequence_time > self.sequence_timeout):
            
            # If enough time has passed since the last gesture, consider it a new sequence
            if current_time - self.last_sequence_time > self.sequence_timeout and self.gesture_sequence:
                self.gesture_sequence = []
            
            # Add the new gesture to the sequence
            if gesture not in self.gesture_sequence:
                self.gesture_sequence.append(gesture)
            
            # Update the timestamp
            self.last_sequence_time = current_time
            
            # Check if this sequence forms a known sentence pattern
            sentence = None
            
            # Try all combinations of gestures in the sequence
            for i in range(len(self.gesture_sequence)):
                for j in range(i+1, len(self.gesture_sequence)+1):
                    sub_sequence = tuple(self.gesture_sequence[i:j])
                    if len(sub_sequence) >= 2 and sub_sequence in self.sentence_patterns:
                        sentence = self.sentence_patterns[sub_sequence]
            
            # If we found a sentence pattern, speak it
            if sentence and gesture == self.gesture_sequence[-1]:
                # Only speak if enough time has passed since the last prediction
                if current_time - self.last_prediction_time > self.prediction_cooldown:
                    self.speak_text(sentence)
                    self.last_prediction_time = current_time
                return sentence
        
        return None
    
    def translate_gesture(self, gesture):
        """Translate gesture to ISL meaning with Hindi and English"""
        if gesture in self.isl_dictionary:
            return self.isl_dictionary[gesture]
        return gesture
    
    def run_recognition(self):
        """Run real-time gesture recognition"""
        if not self.model and not self.load_model():
            print("Cannot run recognition: No model available")
            return
        
        print("\n===== RECOGNITION MODE =====")
        print("Press 'q' to quit")
        print("Press 'c' to clear gesture sequence\n")
        
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # For displaying gesture sequence on the frame
        sequence_text = ""
        sentence_text = ""
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Flip the frame horizontally for a more intuitive view
            frame = cv2.flip(frame, 1)
            
            # Extract features and update the frame with landmarks
            features, frame = self.extract_features(frame)
            
            if features:
                gesture = self.predict_gesture(features)
                
                if gesture:
                    # Get translated gesture
                    translated_gesture = self.translate_gesture(gesture)
                    
                    # Try to form a sentence from gesture sequence
                    sentence = self.process_gesture_sequence(gesture)
                    if sentence:
                        sentence_text = f"Sentence: {sentence}"
                    
                    # Update sequence text for display
                    if self.gesture_sequence:
                        sequence_text = "Sequence: " + " → ".join([self.translate_gesture(g) for g in self.gesture_sequence])
                    
                    # Display the recognized gesture
                    cv2.putText(
                        frame, 
                        f"Gesture: {translated_gesture}", 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (0, 255, 0), 
                        2
                    )
                    
                    # Display the current sequence
                    cv2.putText(
                        frame, 
                        sequence_text, 
                        (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (255, 255, 0), 
                        2
                    )
                    
                    # Display detected sentence if any
                    if sentence_text:
                        cv2.putText(
                            frame, 
                            sentence_text, 
                            (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, 
                            (0, 255, 255), 
                            2
                        )
                    
                    # Speak single gesture with cooldown
                    current_time = time.time()
                    if gesture != self.current_gesture or current_time - self.last_prediction_time > self.prediction_cooldown:
                        # Only speak the individual gesture if it's not part of a recently spoken sentence
                        if not sentence:
                            self.speak_text(gesture)
                        self.last_prediction_time = current_time
                        self.current_gesture = gesture
            else:
                cv2.putText(
                    frame, 
                    "No gesture detected", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 0, 255), 
                    2
                )
            
            # Display instructions
            cv2.putText(
                frame,
                "Press 'q' to quit, 'c' to clear sequence",
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )
            
            cv2.imshow("Indian Sign Language Recognition", frame)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Clear the current gesture sequence
                self.gesture_sequence = []
                sequence_text = ""
                sentence_text = ""
                print("Gesture sequence cleared")
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function to run the Indian Sign Language Translator"""
    translator = IndianSignLanguageTranslator()
    
    while True:
        print("\n===== INDIAN SIGN LANGUAGE TRANSLATOR =====")
        print("1. Collect training data")
        print("2. Train model")
        print("3. Run real-time recognition")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            translator.collect_data()
        elif choice == '2':
            translator.train_model()
        elif choice == '3':
            translator.run_recognition()
        elif choice == '4':
            print("Exiting program...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()