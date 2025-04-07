import os
import cv2
import numpy as np
import pickle
import time
import pyttsx3
import mediapipe as mp
import threading
import joblib
import warnings
from collections import deque
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class ISLTranslatorWithPretrained:
    def __init__(self):
        # Initialize MediaPipe solutions with optimized parameters
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,  # Using the more accurate model
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,  # Enhanced landmark accuracy
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Using the more accurate model
            enable_segmentation=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        
        # Create directories for models and data
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('pretrained_models', exist_ok=True)
        
        # Load or download pre-trained models
        self.pretrained_model = None
        self.scaler = None
        self.load_or_download_pretrained_models()
        
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
        self.motion_history = deque(maxlen=30)  # Store recent landmark positions
        self.previous_landmarks = None
        
        # Dictionary of common Indian Sign Language words and phrases
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
            "good": "अच्छा (Accha)",
            "bad": "बुरा (Bura)",
            "please": "कृपया (Kripya)",
            "sorry": "माफ़ करना (Maaf Karna)",
            "love": "प्यार (Pyaar)",
            "work": "काम (Kaam)",
            "home": "घर (Ghar)",
            "time": "समय (Samay)",
            "money": "पैसा (Paisa)",
            "learn": "सीखना (Seekhna)"
        }
        
        # Common sentence patterns in ISL
        self.sentence_patterns = {
            ("hello", "friend"): "Hello my friend",
            ("help", "please"): "Please help me",
            ("thank_you", "friend"): "Thank you my friend",
            ("water", "please"): "Please give me water",
            ("food", "time"): "It's time for food",
            ("family", "home"): "My family is at home",
            ("learn", "school"): "I am learning at school",
            ("sorry", "friend"): "I am sorry my friend",
            ("good", "work"): "Good work",
            ("money", "please"): "Please give me money",
            ("yes", "help"): "Yes, I will help you",
            ("no", "sorry"): "No, I am sorry",
            ("home", "time"): "Time to go home",
            ("love", "family"): "I love my family"
        }
        
        # Pretrained model information
        self.model_info = {
            "name": "ISL-Recognition-Model",
            "version": "1.0",
            "recognition_classes": list(self.isl_dictionary.keys()),
            "expected_feature_length": 381  # 42 hand landmarks (x,y,z) + 36 face + 27 pose + 6 motion
        }
        
    def load_or_download_pretrained_models(self):
        """Load pre-trained models or create demo models if not available"""
        model_path = 'pretrained_models/isl_recognition_model.joblib'
        scaler_path = 'pretrained_models/isl_feature_scaler.joblib'
        
        try:
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                print("Loading pre-trained models...")
                self.pretrained_model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                print("Pre-trained models loaded successfully")
            else:
                print("Pre-trained models not found. Creating demo models...")
                self.create_demo_pretrained_models()
                print("Demo models created. In a real application, you would download actual pre-trained models.")
            return True
        except Exception as e:
            print(f"Error loading pre-trained models: {e}")
            print("Creating demo models instead...")
            self.create_demo_pretrained_models()
            return True
            
    def create_demo_pretrained_models(self):
        """Create demo pre-trained models for demonstration purposes"""
        # Create a simple RandomForest classifier as a demo pre-trained model
        self.pretrained_model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=20,
            random_state=42
        )
        
        # Create a standard scaler
        self.scaler = StandardScaler()
        
        # Generate some synthetic training data
        X_synthetic = np.random.rand(1000, self.model_info["expected_feature_length"])
        y_synthetic = np.random.choice(self.model_info["recognition_classes"], size=1000)
        
        # Fit the scaler
        X_scaled = self.scaler.fit_transform(X_synthetic)
        
        # Fit the model
        self.pretrained_model.fit(X_scaled, y_synthetic)
        
        # Save the demo models
        joblib.dump(self.pretrained_model, 'pretrained_models/isl_recognition_model.joblib')
        joblib.dump(self.scaler, 'pretrained_models/isl_feature_scaler.joblib')
        
        print("Note: These are demo models for illustration. They won't produce accurate predictions.")
        print("In a real implementation, you would use actual pre-trained models.")
        
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
        
        # Track presence of detection for visualization
        detected = {"hands": False, "face": False, "pose": False}
        
        # Extract hand landmarks (up to 2 hands)
        hand_landmarks_list = []
        if hands_results.multi_hand_landmarks:
            detected["hands"] = True
            
            # Sort hands as left and right
            left_hand = None
            right_hand = None
            
            for idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                if idx < len(hands_results.multi_handedness):
                    handedness = hands_results.multi_handedness[idx].classification[0].label
                    
                    # Draw the hand landmarks
                    self.mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Add handedness label
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
                    
                    # Store landmarks based on handedness
                    if handedness == "Left":
                        left_hand = hand_landmarks
                    else:
                        right_hand = hand_landmarks
                    
                    # Store landmarks for motion tracking
                    hand_landmarks_list.append(hand_landmarks)
            
            # Process left hand if present
            if left_hand:
                for landmark in left_hand.landmark:
                    features.extend([landmark.x, landmark.y, landmark.z])
            else:
                # Pad with zeros if not present
                features.extend([0.0] * (21 * 3))
            
            # Process right hand if present
            if right_hand:
                for landmark in right_hand.landmark:
                    features.extend([landmark.x, landmark.y, landmark.z])
            else:
                # Pad with zeros if not present
                features.extend([0.0] * (21 * 3))
        else:
            # No hands detected, pad with zeros
            features.extend([0.0] * (21 * 3 * 2))  # Both hands
        
        # Extract key facial landmarks
        face_landmarks_list = []
        if face_results.multi_face_landmarks:
            detected["face"] = True
            face_landmarks = face_results.multi_face_landmarks[0]
            face_landmarks_list.append(face_landmarks)
            
            # Draw face landmarks - only draw important features (eyes, mouth, eyebrows)
            connection_spec = self.mp_drawing_styles.get_default_face_mesh_contours_style()
            self.mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=connection_spec
            )
            
            # Key facial points for sign language (eyes, eyebrows, nose, mouth)
            # These points are especially important for Indian Sign Language
            key_face_indices = [
                10, 338,  # Left/right eyebrow
                68, 69, 104, 105, 107, 336, 337,  # Eyes
                0, 17, 61, 291,  # Upper lip
                84, 314, 17, 314, 405,  # Lower face
                61, 185, 40, 39, 37, 0, 267, 269, 270, 409,  # Mouth
                4, 195, 197  # Nose
            ]
            
            for idx in key_face_indices:
                landmark = face_landmarks.landmark[idx]
                features.extend([landmark.x, landmark.y, landmark.z])
        else:
            # No face detected, pad with zeros
            features.extend([0.0] * (36))  # 12 key facial points * 3 coordinates
        
        # Extract key pose landmarks
        if pose_results.pose_landmarks:
            detected["pose"] = True
            pose_landmarks = pose_results.pose_landmarks
            
            # Draw pose landmarks - focus on upper body
            self.mp_drawing.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Use upper body landmarks for sign language (especially important for ISL)
            upper_body_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24]
            for idx in upper_body_indices:
                landmark = pose_landmarks.landmark[idx]
                features.extend([landmark.x, landmark.y, landmark.z])
        else:
            # No pose detected, pad with zeros
            features.extend([0.0] * (9 * 3))  # 9 upper body pose points * 3 coordinates
        
        # Add motion features if we have history
        all_landmarks = hand_landmarks_list + face_landmarks_list
        motion_features = self.extract_motion_features(all_landmarks)
        features.extend(motion_features)
        
        # Display detection status
        status_text = "Detected: "
        status_text += "Hands " if detected["hands"] else ""
        status_text += "Face " if detected["face"] else ""
        status_text += "Pose " if detected["pose"] else ""
        if not (detected["hands"] or detected["face"] or detected["pose"]):
            status_text += "None"
            
        cv2.putText(
            frame,
            status_text,
            (10, frame.shape[0] - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )
        
        return features, frame
    
    def extract_motion_features(self, current_landmarks):
        """Extract motion features from landmarks"""
        motion_features = [0.0] * 6  # Default zeros
        
        if not current_landmarks:
            self.previous_landmarks = None
            return motion_features
            
        # If we have previous landmarks, calculate motion
        if self.previous_landmarks:
            velocities = []
            
            # Match landmarks by index between frames
            for i, current_landmark_set in enumerate(current_landmarks):
                if i < len(self.previous_landmarks):
                    prev_landmark_set = self.previous_landmarks[i]
                    
                    # Calculate velocities for each landmark point
                    for j, current_lm in enumerate(current_landmark_set.landmark):
                        if j < len(prev_landmark_set.landmark):
                            prev_lm = prev_landmark_set.landmark[j]
                            
                            # Calculate velocity vector
                            vx = current_lm.x - prev_lm.x
                            vy = current_lm.y - prev_lm.y
                            vz = current_lm.z - prev_lm.z
                            
                            velocity_magnitude = np.sqrt(vx*vx + vy*vy + vz*vz)
                            velocities.append(velocity_magnitude)
            
            if velocities:
                # Calculate motion statistics
                mean_velocity = np.mean(velocities)
                max_velocity = np.max(velocities)
                min_velocity = np.min(velocities)
                std_velocity = np.std(velocities)
                
                # Calculate direction features
                primary_direction_x = np.mean([current_landmarks[0].landmark[9].x - 
                                               self.previous_landmarks[0].landmark[9].x 
                                               if len(current_landmarks) > 0 and len(self.previous_landmarks) > 0 else 0])
                primary_direction_y = np.mean([current_landmarks[0].landmark[9].y - 
                                               self.previous_landmarks[0].landmark[9].y 
                                               if len(current_landmarks) > 0 and len(self.previous_landmarks) > 0 else 0])
                
                motion_features = [
                    mean_velocity, 
                    max_velocity, 
                    min_velocity, 
                    std_velocity,
                    primary_direction_x,
                    primary_direction_y
                ]
        
        # Update previous landmarks for next frame
        self.previous_landmarks = current_landmarks
        
        return motion_features
    
    def predict_with_pretrained(self, features):
        """Use pre-trained model to predict the gesture"""
        if not self.pretrained_model or not self.scaler:
            return None
            
        # Check if feature vector matches expected length
        if len(features) != self.model_info["expected_feature_length"]:
            missing = self.model_info["expected_feature_length"] - len(features)
            if missing > 0:
                # Pad with zeros if necessary
                features.extend([0.0] * missing)
            elif missing < 0:
                # Truncate if too long
                features = features[:self.model_info["expected_feature_length"]]
        
        try:
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Predict with model
            prediction = self.pretrained_model.predict(features_scaled)[0]
            
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
        except Exception as e:
            print(f"Prediction error: {e}")
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
            
            # Check if the last two gestures form a known pattern
            if len(self.gesture_sequence) >= 2:
                last_two = tuple(self.gesture_sequence[-2:])
                if last_two in self.sentence_patterns:
                    sentence = self.sentence_patterns[last_two]
            
            # Try all possible pairs in the sequence
            if not sentence:
                for i in range(len(self.gesture_sequence)):
                    for j in range(i+1, len(self.gesture_sequence)):
                        gesture_pair = (self.gesture_sequence[i], self.gesture_sequence[j])
                        if gesture_pair in self.sentence_patterns:
                            sentence = self.sentence_patterns[gesture_pair]
            
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
    
    def collect_training_data(self):
        """Collect training data to fine-tune the pre-trained model"""
        print("\n===== DATA COLLECTION MODE =====")
        print("This will collect data to fine-tune the pre-trained model")
        print("Press 'c' to start collecting a new gesture")
        print("Press 'm' to collect a moving gesture (perform for 3 seconds)")
        print("Press 'q' to quit data collection mode\n")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        data = []
        labels = []
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
                    
                    # Add to training data
                    data.append(features)
                    labels.append(current_label)
                    
                    if elapsed_time >= 3.0:
                        print(f"Collected moving gesture '{current_label}'")
                        collecting = False
                        collecting_moving = False
                else:
                    # Static gesture collection
                    data.append(features)
                    labels.append(current_label)
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
        if data and labels:
            with open('data/isl_fine_tuning_data.pkl', 'wb') as f:
                pickle.dump((data, labels), f)
            print(f"Data saved: {len(labels)} frames for {len(set(labels))} gestures")
            
            # Ask if they want to fine-tune the model now
            choice = input("Do you want to fine-tune the pre-trained model now? (y/n): ")
            if choice.lower() == 'y':
                self.fine_tune_model()
        else:
            print("No data collected")
    
    def fine_tune_model(self):
        """Fine-tune the pre-trained model with new data"""
        print("\n===== FINE-TUNING PRE-TRAINED MODEL =====")
        
        try:
            # Load the fine-tuning data
            with open('data/isl_fine_tuning_data.pkl', 'rb') as f:
                data, labels = pickle.load(f)
                
            print(f"Loaded {len(labels)} training samples for {len(set(labels))} gestures")
            
            # Ensure all feature vectors have the expected length
            expected_length = self.model_info["expected_feature_length"]
            processed_data = []
            
            for features in data:
                if len(features) < expected_length:
                    # Pad with zeros if necessary
                    features = features + [0.0] * (expected_length - len(features))
                elif len(features) > expected_length:
                    # Truncate if too long
                    features = features[:expected_length]
                processed_data.append(features)
            
            # Scale the features
            X_scaled = self.scaler.transform(processed_data)
            
            # Update the model (incremental learning)
            print("Fine-tuning the model...")
            # For RandomForest, we need to retrain from scratch
            if isinstance(self.pretrained_model, RandomForestClassifier):
                # Get predicted labels from existing model for all X
                existing_data = np.random.rand(500, expected_length)
                existing_data_scaled = self.scaler.transform(existing_data)
                existing_labels = self.pretrained_model.predict(existing_data_scaled)
                
                # Combine existing synthetic data with new data to prevent catastrophic forgetting
                X_combined = np.vstack([existing_data_scaled, X_scaled])
                y_combined = np.concatenate([existing_labels, labels])
                
                # Retrain the model
                self.pretrained_model.fit(X_combined, y_combined)
            else:
                # For other models that support partial_fit
                try:
                    self.pretrained_model.partial_fit(X_scaled, labels)
                except AttributeError:
                    print("Model doesn't support incremental learning, retraining from scratch")
                    try:
                        self.pretrained_model.fit(X_scaled, labels)
                    except Exception as e:
                        print(f"Error fine-tuning model: {e}")
            
            # Save the fine-tuned model
            joblib.dump(self.pretrained_model, 'models/isl_fine_tuned_model.joblib')
            print("Fine-tuned model saved to models/isl_fine_tuned_model.joblib")
            
            # Load the fine-tuned model as the current model
            self.pretrained_model = joblib.load('models/isl_fine_tuned_model.joblib')
            print("Now using the fine-tuned model")
            
            return True
        except FileNotFoundError:
            print("No fine-tuning data found. Please collect data first.")
            return False
        except Exception as e:
            print(f"Error in fine-tuning: {e}")
            return False
    
    def run_recognition(self):
        """Run real-time gesture recognition"""
        if not self.pretrained_model or not self.scaler:
            print("Pre-trained models not properly loaded")
            return
        
        print("\n===== RECOGNITION MODE =====")
        print("Press 'q' to quit")
        print("Press 'c' to clear gesture sequence\n")
        
        cap = cv2.VideoCapture(0)
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
                gesture = self.predict_with_pretrained(features)
                
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
                        sequence_text)