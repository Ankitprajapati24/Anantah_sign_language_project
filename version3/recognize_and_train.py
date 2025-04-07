# Enhanced Sign Language Communication System
# This system detects and translates sign language into speech and text in real-time
# It uses MediaPipe to track body pose, hands, and face for comprehensive gesture recognition

import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3
import time
import os
from sklearn.ensemble import RandomForestClassifier
from collections import deque
import threading

class EnhancedSignLanguageCommunicator:
    def __init__(self):
        # Initialize MediaPipe solutions
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize holistic detection (body, face, and hands)
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,  # Balance between speed and accuracy
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        
        # Initialize models
        self.word_model = None
        self.sentence_model = None
        self.word_labels = []
        self.sentence_labels = []
        
        # Model file paths
        self.word_model_file = 'data/sign_word_model.pkl'
        self.word_labels_file = 'data/sign_word_labels.pkl'
        self.sentence_model_file = 'data/sign_sentence_model.pkl'
        self.sentence_labels_file = 'data/sign_sentence_labels.pkl'
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Gesture sequence tracking
        self.gesture_history = deque(maxlen=30)  # Store recent gestures for sentence formation
        self.word_cooldown = 1.5  # seconds between word recognitions
        self.sentence_cooldown = 4.0  # seconds between sentence recognitions
        self.last_word_time = 0
        self.last_sentence_time = 0
        self.last_spoken_word = ""
        self.last_spoken_sentence = ""
        
        # Motion tracking
        self.position_history = deque(maxlen=15)  # Track positions for motion analysis
        
        # TTS thread safety
        self.tts_lock = threading.Lock()
        
        # Load pre-existing models if available
        self.load_models()
    
    def extract_features(self, frame):
        """Extract holistic features (body pose, face, hands) from a frame using MediaPipe."""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe holistic
        results = self.holistic.process(frame_rgb)
        
        # Draw landmarks on frame for visualization
        self.draw_landmarks(frame, results)
        
        # Initialize feature vector
        features = []
        detection_status = False
        
        # 1. Extract pose landmarks (33 landmarks x 3 coordinates = 99 values)
        if results.pose_landmarks:
            pose_features = []
            for landmark in results.pose_landmarks.landmark:
                pose_features.extend([landmark.x, landmark.y, landmark.z])
            features.extend(pose_features)
            detection_status = True
        else:
            # Fill with zeros if no pose detected
            features.extend([0.0] * (33 * 3))
        
        # 2. Extract left hand landmarks (21 landmarks x 3 coordinates = 63 values)
        if results.left_hand_landmarks:
            left_hand_features = []
            for landmark in results.left_hand_landmarks.landmark:
                left_hand_features.extend([landmark.x, landmark.y, landmark.z])
            features.extend(left_hand_features)
            detection_status = True
        else:
            # Fill with zeros if no left hand detected
            features.extend([0.0] * (21 * 3))
        
        # 3. Extract right hand landmarks (21 landmarks x 3 coordinates = 63 values)
        if results.right_hand_landmarks:
            right_hand_features = []
            for landmark in results.right_hand_landmarks.landmark:
                right_hand_features.extend([landmark.x, landmark.y, landmark.z])
            features.extend(right_hand_features)
            detection_status = True
        else:
            # Fill with zeros if no right hand detected
            features.extend([0.0] * (21 * 3))
        
        # 4. Extract key facial landmarks (focus on key points that matter for sign language)
        if results.face_landmarks:
            # Select key facial points (eyes, eyebrows, nose, mouth - ~20 landmarks)
            key_face_indices = [0, 17, 61, 78, 80, 13, 14, 191, 80, 81, 82, 13, 312, 311, 310, 415, 324, 308]
            face_features = []
            for idx in key_face_indices:
                landmark = results.face_landmarks.landmark[idx]
                face_features.extend([landmark.x, landmark.y, landmark.z])
            features.extend(face_features)
            detection_status = True
        else:
            # Fill with zeros if no face detected
            features.extend([0.0] * (18 * 3))
        
        # 5. Extract motion features
        # Track right wrist position for motion trajectory
        if results.right_hand_landmarks:
            # Use wrist landmark (index 0) for tracking motion
            wrist = results.right_hand_landmarks.landmark[0]
            self.position_history.append((wrist.x, wrist.y))
        elif results.pose_landmarks:
            # Fallback to using right wrist from pose if hand landmarks aren't detected
            wrist = results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.RIGHT_WRIST]
            self.position_history.append((wrist.x, wrist.y))
            
        # Calculate motion features if we have enough history
        motion_features = self.calculate_motion_features()
        features.extend(motion_features)
        
        return features, detection_status
    
    def calculate_motion_features(self):
        """Calculate motion features from position history."""
        if len(self.position_history) < 5:
            return [0.0] * 6  # Return zeros if not enough history
            
        # Calculate velocity (from last 2 positions)
        p1 = self.position_history[-2]
        p2 = self.position_history[-1]
        dx_recent = p2[0] - p1[0]
        dy_recent = p2[1] - p1[1]
        
        # Calculate overall direction (from oldest to newest in buffer)
        p_old = self.position_history[0]
        p_new = self.position_history[-1]
        dx_overall = p_new[0] - p_old[0]
        dy_overall = p_new[1] - p_old[1]
        
        # Calculate acceleration (change in velocity)
        if len(self.position_history) >= 3:
            p0 = self.position_history[-3]
            prev_dx = p1[0] - p0[0]
            prev_dy = p1[1] - p0[1]
            accel_x = dx_recent - prev_dx
            accel_y = dy_recent - prev_dy
        else:
            accel_x, accel_y = 0.0, 0.0
            
        return [dx_recent, dy_recent, dx_overall, dy_overall, accel_x, accel_y]
    
    def draw_landmarks(self, frame, results):
        """Draw the detected landmarks on the frame."""
        # Draw pose landmarks
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Draw hand landmarks
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        # Draw face landmarks (simplified)
        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.face_landmarks,
                self.mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
            )
    
    def collect_training_data(self, mode='word'):
        """Collect training data for gestures - either words or full sentences."""
        cap = cv2.VideoCapture(0)
        
        training_data = []
        labels_data = []
        items = []
        
        mode_name = "words" if mode == 'word' else "sentences"
        print(f"\n=== Collecting training data for {mode_name} ===")
        
        while True:
            # Ask for the name of the gesture
            item_name = input(f"Enter the {mode} to record (or 'done' to finish): ")
            if item_name.lower() == 'done':
                break
            items.append(item_name)
            
            print(f"Prepare to show the sign for '{item_name}'...")
            if mode == 'word':
                print("Press 's' to start recording samples (aim for at least 30 samples)")
            else:
                print("Press 's' to start recording. Perform the full sentence gesture sequence.")
            print("Press 'q' when done with this item")
            
            recording = False
            samples_count = 0
            sequence_features = [] if mode == 'sentence' else None
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Mirror the frame horizontally for a more intuitive interaction
                frame = cv2.flip(frame, 1)
                
                # Extract features
                features, detected = self.extract_features(frame)
                
                # Display status
                if mode == 'word':
                    status_text = "Ready. Press 's' to start recording" if not recording else f"Recording: {samples_count} samples"
                else:
                    status_text = "Ready. Press 's' to start recording sequence" if not recording else f"Recording sequence... (press 'q' when done)"
                
                cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Item: {item_name}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Show the frame
                cv2.imshow('Data Collection', frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                # Start recording with 's'
                if key == ord('s'):
                    recording = True
                    if mode == 'sentence':
                        sequence_features = []
                    print("Recording started...")
                
                # If recording and features detected, save the sample
                if recording and detected:
                    if mode == 'word':
                        training_data.append(features)
                        labels_data.append(item_name)
                        samples_count += 1
                        time.sleep(0.1)  # Brief pause between samples
                    else:  # sentence mode
                        sequence_features.append(features)
                
                # Quit this item recording with 'q'
                if key == ord('q'):
                    if mode == 'word':
                        print(f"Recorded {samples_count} samples for '{item_name}'")
                    else:  # sentence mode
                        if sequence_features:
                            # For sentences, we calculate aggregate features from the sequence
                            # Here we take mean, min, max, and variance of features over time
                            seq_array = np.array(sequence_features)
                            mean_features = np.mean(seq_array, axis=0)
                            min_features = np.min(seq_array, axis=0)
                            max_features = np.max(seq_array, axis=0)
                            var_features = np.var(seq_array, axis=0)
                            
                            # Combine these into a single feature vector
                            combined_features = np.concatenate([mean_features, min_features, max_features, var_features])
                            
                            training_data.append(combined_features)
                            labels_data.append(item_name)
                            print(f"Recorded sequence for sentence '{item_name}'")
                        else:
                            print("No sequence data recorded")
                    break
            
        cap.release()
        cv2.destroyAllWindows()
        
        # Train the model with collected data
        if training_data and labels_data:
            self.train_model(training_data, labels_data, mode)
            print(f"Model trained with {mode_name}: {items}")
        else:
            print(f"No training data collected for {mode_name}")
    
    def train_model(self, X, y, mode='word'):
        """Train the RandomForest classifier with the collected data."""
        print(f"Training {mode} model...")
        
        # Convert to numpy arrays
        X = np.array(X)
        
        # Create and train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Save the model based on mode
        if mode == 'word':
            self.word_model = model
            self.word_labels = list(set(y))
            self.save_model(mode)
        else:  # sentence mode
            self.sentence_model = model
            self.sentence_labels = list(set(y))
            self.save_model(mode)
        
        print(f"{mode.capitalize()} model training completed and saved")
    
    def save_model(self, mode='word'):
        """Save the trained model to a file."""
        if mode == 'word' and self.word_model is not None:
            with open(self.word_model_file, 'wb') as f:
                pickle.dump(self.word_model, f)
            
            with open(self.word_labels_file, 'wb') as f:
                pickle.dump(self.word_labels, f)
            
            print(f"Word model saved to {self.word_model_file}")
        
        elif mode == 'sentence' and self.sentence_model is not None:
            with open(self.sentence_model_file, 'wb') as f:
                pickle.dump(self.sentence_model, f)
            
            with open(self.sentence_labels_file, 'wb') as f:
                pickle.dump(self.sentence_labels, f)
            
            print(f"Sentence model saved to {self.sentence_model_file}")
    
    def load_models(self):
        """Load previously trained models from files."""
        word_model_loaded = False
        sentence_model_loaded = False
        
        # Try to load word model
        try:
            with open(self.word_model_file, 'rb') as f:
                self.word_model = pickle.load(f)
            
            with open(self.word_labels_file, 'rb') as f:
                self.word_labels = pickle.load(f)
            
            print(f"Word model loaded. Available words: {self.word_labels}")
            word_model_loaded = True
        except (FileNotFoundError, EOFError):
            print("No existing word model found.")
        
        # Try to load sentence model
        try:
            with open(self.sentence_model_file, 'rb') as f:
                self.sentence_model = pickle.load(f)
            
            with open(self.sentence_labels_file, 'rb') as f:
                self.sentence_labels = pickle.load(f)
            
            print(f"Sentence model loaded. Available sentences: {self.sentence_labels}")
            sentence_model_loaded = True
        except (FileNotFoundError, EOFError):
            print("No existing sentence model found.")
        
        return word_model_loaded or sentence_model_loaded
    
    def speak_text(self, text):
        """Thread-safe function to speak text."""
        def _speak():
            with self.tts_lock:
                self.engine.say(text)
                self.engine.runAndWait()
        
        # Start in a separate thread to avoid blocking the main loop
        threading.Thread(target=_speak).start()
    
    def recognize_communication(self):
        """Recognize both words and sentences in real-time from webcam feed."""
        if self.word_model is None and self.sentence_model is None:
            print("No models available. Please train at least one model first.")
            return
        
        cap = cv2.VideoCapture(0)
        
        # For word recognition
        word_buffer = deque(maxlen=10)
        
        # For sentence recognition
        sequence_buffer = deque(maxlen=30)  # Store features for sentence detection
        current_time = time.time()
        sequence_start_time = current_time
        
        # For displaying recognized text
        recognized_words = []
        current_sentence = ""
        
        print("Starting sign language communication system...")
        print("Press 'q' to quit, 'c' to clear text")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Mirror the frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Extract features
            features, detected = self.extract_features(frame)
            
            current_time = time.time()
            
            # Create a blank area at the bottom for displaying text
            text_area = np.ones((200, frame.shape[1], 3), dtype=np.uint8) * 255
            frame_with_text = np.vstack([frame, text_area])
            
            if detected:
                # Word recognition (more immediate)
                if self.word_model is not None:
                    word_buffer.append(features)
                    
                    if len(word_buffer) == word_buffer.maxlen:
                        # Use the latest features for word prediction
                        word_prediction = self.word_model.predict([features])[0]
                        word_confidence = np.max(self.word_model.predict_proba([features]))
                        
                        # Only consider prediction if confidence is high enough
                        if word_confidence > 0.7:  # Threshold for word confidence
                            if (word_prediction != self.last_spoken_word or 
                                current_time - self.last_word_time >= self.word_cooldown):
                                
                                # Update last prediction
                                self.last_spoken_word = word_prediction
                                self.last_word_time = current_time
                                
                                # Add to recognized words list
                                recognized_words.append(word_prediction)
                                if len(recognized_words) > 10:  # Keep only the last 10 words
                                    recognized_words.pop(0)
                                
                                # Speak the word
                                self.speak_text(word_prediction)
                                
                                # Update current sentence
                                if len(recognized_words) > 0:
                                    current_sentence = " ".join(recognized_words)
                
                # Sentence recognition (needs sequence data)
                if self.sentence_model is not None:
                    # Add features to sequence buffer
                    sequence_buffer.append(features)
                    
                    # Check if enough time has passed to analyze for a sentence
                    if (len(sequence_buffer) == sequence_buffer.maxlen and 
                        current_time - sequence_start_time >= 3.0 and  # Minimum 3 seconds for a sentence
                        current_time - self.last_sentence_time >= self.sentence_cooldown):
                        
                        # Convert buffer to numpy array
                        seq_array = np.array(list(sequence_buffer))
                        
                        # Calculate sequence features
                        mean_features = np.mean(seq_array, axis=0)
                        min_features = np.min(seq_array, axis=0)
                        max_features = np.max(seq_array, axis=0)
                        var_features = np.var(seq_array, axis=0)
                        
                        # Combine into single feature vector
                        combined_features = np.concatenate([mean_features, min_features, max_features, var_features])
                        
                        # Make prediction
                        sentence_prediction = self.sentence_model.predict([combined_features])[0]
                        sentence_confidence = np.max(self.sentence_model.predict_proba([combined_features]))
                        
                        # Only consider prediction if confidence is high
                        if sentence_confidence > 0.65:  # Lower threshold for sentences
                            self.last_spoken_sentence = sentence_prediction
                            self.last_sentence_time = current_time
                            
                            # Reset sequence buffer and timer
                            sequence_buffer.clear()
                            sequence_start_time = current_time
                            
                            # Speak the sentence
                            self.speak_text(sentence_prediction)
                            
                            # Update current sentence (override word recognition)
                            current_sentence = sentence_prediction
                            recognized_words = []  # Clear recognized words
            
            # Display the current text
            cv2.putText(frame_with_text, "Recognized Communication:", (10, frame.shape[0] + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Word-wrap the current sentence text
            y_offset = frame.shape[0] + 70
            words = current_sentence.split()
            line = ""
            for word in words:
                test_line = line + word + " "
                # Check if adding this word would exceed the width
                if cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0] > frame.shape[1] - 20:
                    # Draw the current line and start a new one
                    cv2.putText(frame_with_text, line, (10, y_offset), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    y_offset += 40
                    line = word + " "
                else:
                    line = test_line
            
            # Draw any remaining text
            if line:
                cv2.putText(frame_with_text, line, (10, y_offset), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display mode information
            model_info = []
            if self.word_model is not None:
                model_info.append("Word recognition active")
            if self.sentence_model is not None:
                model_info.append("Sentence recognition active")
            
            mode_text = " & ".join(model_info)
            cv2.putText(frame_with_text, mode_text, (10, frame.shape[0] + 150), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Display the frame
            cv2.imshow('Sign Language Communication System', frame_with_text)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Exit on 'q' key
            if key == ord('q'):
                break
                
            # Clear text on 'c' key
            if key == ord('c'):
                recognized_words = []
                current_sentence = ""
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function to run the Enhanced Sign Language Communication System."""
    print("=== Enhanced Sign Language Communication System ===")
    print("This system helps translate sign language into speech and text")
    
    communicator = EnhancedSignLanguageCommunicator()
    
    while True:
        print("\nChoose an option:")
        print("1. Collect training data for individual words")
        print("2. Collect training data for complete sentences")
        print("3. Start sign language communication")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            communicator.collect_training_data(mode='word')
        elif choice == '2':
            communicator.collect_training_data(mode='sentence')
        elif choice == '3':
            communicator.recognize_communication()
        elif choice == '4':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()