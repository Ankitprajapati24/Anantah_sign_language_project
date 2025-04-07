# Enhanced Indian Sign Language Recognition System
# This system detects and translates complex Indian Sign Language expressions in real-time
# It tracks body pose, face expressions, and both hands to recognize words and sentences

import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3
import time
import os
from sklearn.ensemble import RandomForestClassifier
from collections import deque

class EnhancedISLRecognizer:
    def __init__(self):
        # Initialize MediaPipe solutions
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize holistic tracking (captures hands, face, and pose in one model)
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,  # Use highest accuracy model
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        
        # Initialize classifier
        self.model = None
        self.labels = []
        self.model_file = 'data/enhanced_isl_model.pkl'
        self.labels_file = 'data/enhanced_isl_labels.pkl'
        
        # Sequence recognition settings
        self.sequence_length = 30  # Number of frames to capture for a sequence
        self.sequence_buffer = deque(maxlen=self.sequence_length)
        
        # Cooldown settings for prediction
        self.last_prediction = ""
        self.last_prediction_time = 0
        self.cooldown = 2  # seconds
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Motion tracking variables
        self.motion_history = {
            'left_wrist': deque(maxlen=30),
            'right_wrist': deque(maxlen=30),
            'pose_landmarks': deque(maxlen=30)
        }
        
        # Dictionary to store common phrases and sentences
        self.common_expressions = {
            'thank_you': "Thank you",
            'welcome': "You're welcome",
            'hello_teacher': "Hello teacher",
            'hello_student': "Hello student",
            'good_morning': "Good morning",
            'good_afternoon': "Good afternoon",
            'good_evening': "Good evening",
            'how_are_you': "How are you?",
            'fine': "I'm fine",
            'yes': "Yes",
            'no': "No",
            'please': "Please",
            'sorry': "Sorry",
            'excuse_me': "Excuse me"
        }
        
        # Try to load pre-existing model if available
        self.load_model()
    
    def extract_features(self, frame):
        """
        Extract comprehensive features including hands, face, and body pose.
        Includes temporal features for gesture dynamics.
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe holistic
        results = self.holistic.process(frame_rgb)
        
        features = []
        detected = False
        
        # Draw landmarks on frame for visual feedback
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.face_landmarks,
                self.mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
            )
        
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
        
        # Extract pose landmarks (body position)
        pose_features = []
        if results.pose_landmarks:
            detected = True
            
            # Store landmarks for motion tracking
            landmarks_array = []
            for landmark in results.pose_landmarks.landmark:
                landmarks_array.append((landmark.x, landmark.y, landmark.z))
            
            self.motion_history['pose_landmarks'].append(landmarks_array)
            
            # Extract upper body landmarks (more relevant for sign language)
            upper_body_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24]  # Key points including shoulders, elbows, wrists
            
            for idx in upper_body_indices:
                landmark = results.pose_landmarks.landmark[idx]
                pose_features.extend([landmark.x, landmark.y, landmark.z])
        else:
            # No pose detected, fill with zeros
            pose_features = [0.0] * (9 * 3)  # 9 landmarks with x,y,z coordinates
        
        features.extend(pose_features)
        
        # Extract hand landmarks
        left_hand_features = []
        if results.left_hand_landmarks:
            detected = True
            
            # Track left wrist for motion
            wrist = results.left_hand_landmarks.landmark[0]  # Wrist landmark
            self.motion_history['left_wrist'].append((wrist.x, wrist.y, wrist.z))
            
            # Extract all landmarks
            for landmark in results.left_hand_landmarks.landmark:
                left_hand_features.extend([landmark.x, landmark.y, landmark.z])
        else:
            # No left hand detected, fill with zeros
            left_hand_features = [0.0] * (21 * 3)  # 21 landmarks with x,y,z coordinates
            
        features.extend(left_hand_features)
        
        right_hand_features = []
        if results.right_hand_landmarks:
            detected = True
            
            # Track right wrist for motion
            wrist = results.right_hand_landmarks.landmark[0]  # Wrist landmark
            self.motion_history['right_wrist'].append((wrist.x, wrist.y, wrist.z))
            
            # Extract all landmarks
            for landmark in results.right_hand_landmarks.landmark:
                right_hand_features.extend([landmark.x, landmark.y, landmark.z])
        else:
            # No right hand detected, fill with zeros
            right_hand_features = [0.0] * (21 * 3)  # 21 landmarks with x,y,z coordinates
            
        features.extend(right_hand_features)
        
        # Extract facial expression features
        face_features = []
        if results.face_landmarks:
            detected = True
            
            # Focus on key facial points important for ISL (eyes, eyebrows, mouth)
            # Select key landmarks for emotions and expressions
            key_face_indices = [
                0,    # Nose
                61, 291,  # Inner lips
                13, 14,   # Lips corners
                78, 308,  # Eyebrows
                159, 386, # Eyes
            ]
            
            for idx in key_face_indices:
                landmark = results.face_landmarks.landmark[idx]
                face_features.extend([landmark.x, landmark.y, landmark.z])
        else:
            # No face detected, fill with zeros
            face_features = [0.0] * (9 * 3)  # 9 key facial landmarks with x,y,z coordinates
            
        features.extend(face_features)
        
        # Calculate motion features
        motion_features = self.calculate_motion_features()
        features.extend(motion_features)
        
        return features, detected
    
    def calculate_motion_features(self):
        """Calculate features that represent motion over time."""
        motion_features = []
        
        # Calculate left wrist movement if we have enough history
        if len(self.motion_history['left_wrist']) >= 10:
            # Calculate average velocity over last 10 frames
            start_pos = self.motion_history['left_wrist'][-10]
            end_pos = self.motion_history['left_wrist'][-1]
            
            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]
            dz = end_pos[2] - start_pos[2]
            
            # Calculate direction changes (acceleration)
            direction_changes = 0
            for i in range(len(self.motion_history['left_wrist']) - 2):
                p1 = self.motion_history['left_wrist'][i]
                p2 = self.motion_history['left_wrist'][i + 1]
                p3 = self.motion_history['left_wrist'][i + 2]
                
                # Calculate velocity vectors
                v1 = (p2[0] - p1[0], p2[1] - p1[1])
                v2 = (p3[0] - p2[0], p3[1] - p2[1])
                
                # Dot product to detect direction change
                dot_product = v1[0]*v2[0] + v1[1]*v2[1]
                if dot_product < 0:  # Direction change
                    direction_changes += 1
            
            motion_features.extend([dx, dy, dz, direction_changes])
        else:
            motion_features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Same for right wrist
        if len(self.motion_history['right_wrist']) >= 10:
            start_pos = self.motion_history['right_wrist'][-10]
            end_pos = self.motion_history['right_wrist'][-1]
            
            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]
            dz = end_pos[2] - start_pos[2]
            
            direction_changes = 0
            for i in range(len(self.motion_history['right_wrist']) - 2):
                p1 = self.motion_history['right_wrist'][i]
                p2 = self.motion_history['right_wrist'][i + 1]
                p3 = self.motion_history['right_wrist'][i + 2]
                
                v1 = (p2[0] - p1[0], p2[1] - p1[1])
                v2 = (p3[0] - p2[0], p3[1] - p2[1])
                
                dot_product = v1[0]*v2[0] + v1[1]*v2[1]
                if dot_product < 0:
                    direction_changes += 1
            
            motion_features.extend([dx, dy, dz, direction_changes])
        else:
            motion_features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Add feature for interaction between hands (distance between wrists)
        if (len(self.motion_history['left_wrist']) > 0 and 
            len(self.motion_history['right_wrist']) > 0):
            left = self.motion_history['left_wrist'][-1]
            right = self.motion_history['right_wrist'][-1]
            
            # Distance between hands
            distance = np.sqrt((left[0] - right[0])**2 + 
                              (left[1] - right[1])**2 + 
                              (left[2] - right[2])**2)
            
            # Are hands moving towards or away from each other?
            hands_converging = 0.0
            if (len(self.motion_history['left_wrist']) >= 5 and 
                len(self.motion_history['right_wrist']) >= 5):
                prev_left = self.motion_history['left_wrist'][-5]
                prev_right = self.motion_history['right_wrist'][-5]
                
                prev_distance = np.sqrt((prev_left[0] - prev_right[0])**2 + 
                                      (prev_left[1] - prev_right[1])**2 + 
                                      (prev_left[2] - prev_right[2])**2)
                
                hands_converging = prev_distance - distance
            
            motion_features.extend([distance, hands_converging])
        else:
            motion_features.extend([0.0, 0.0])
        
        # Detect overall body movement
        body_movement = 0.0
        if len(self.motion_history['pose_landmarks']) >= 5:
            movement_sum = 0
            for i in range(len(self.motion_history['pose_landmarks'][-1])):
                if i < len(self.motion_history['pose_landmarks'][-5]):
                    p1 = self.motion_history['pose_landmarks'][-5][i]
                    p2 = self.motion_history['pose_landmarks'][-1][i]
                    
                    # Calculate displacement
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    dz = p2[2] - p1[2]
                    
                    movement_sum += np.sqrt(dx**2 + dy**2 + dz**2)
            
            body_movement = movement_sum / len(self.motion_history['pose_landmarks'][-1])
        
        motion_features.append(body_movement)
        
        return motion_features
    
    def collect_training_data(self):
        """
        Collect training data for different gestures and expressions.
        Captures sequences for moving gestures and phrases.
        """
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        training_data = []
        labels_data = []
        
        # List common ISL expressions to train
        print("\nCommon ISL expressions you might want to train:")
        for key, value in self.common_expressions.items():
            print(f"- {value}")
        print("You can also add your own custom expressions")
        
        gestures = []
        while True:
            # Ask for the name of the gesture/expression
            expression = input("\nEnter the expression to record (or 'done' to finish): ")
            if expression.lower() == 'done':
                break
            gestures.append(expression)
            
            print(f"\nPrepare to show the '{expression}' expression/gesture...")
            print("For moving gestures, we'll record a sequence of frames")
            print("Press 's' to start recording a sequence")
            print("Press 'q' when done with this expression")
            
            recording = False
            sequences_count = 0
            
            # For sequence collection
            current_sequence = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Mirror the frame horizontally for intuitive interaction
                frame = cv2.flip(frame, 1)
                
                # Extract features
                features, detected = self.extract_features(frame)
                
                # Display status
                if not recording:
                    status_text = "Ready. Press 's' to start recording"
                    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    # Show frame count in sequence
                    frame_count = len(current_sequence)
                    status_text = f"Recording sequence: {frame_count}/{self.sequence_length} frames"
                    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    if detected:
                        # Add features to current sequence
                        current_sequence.append(features)
                        
                        # If sequence complete, add to training data
                        if len(current_sequence) >= self.sequence_length:
                            # Average all frames to create one feature vector
                            avg_features = np.mean(current_sequence, axis=0).tolist()
                            training_data.append(avg_features)
                            labels_data.append(expression)
                            sequences_count += 1
                            
                            print(f"Sequence {sequences_count} recorded")
                            current_sequence = []  # Reset for next sequence
                            recording = False  # Auto-stop after sequence
                
                # Show sequences count
                count_text = f"Recorded sequences: {sequences_count}"
                cv2.putText(frame, count_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show the frame
                cv2.imshow('Training Data Collection', frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                # Start recording with 's'
                if key == ord('s') and not recording:
                    recording = True
                    current_sequence = []  # Clear any previous sequence data
                    print("Recording sequence... Make the complete gesture/expression")
                
                # Quit this expression recording with 'q'
                if key == ord('q'):
                    print(f"Recorded {sequences_count} sequences for '{expression}'")
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Train the model with collected data
        if training_data and labels_data:
            self.train_model(training_data, labels_data)
            print(f"Model trained with expressions: {gestures}")
        else:
            print("No training data collected")
    
    def train_model(self, X, y):
        """Train the RandomForest classifier with the collected data."""
        print("Training model...")
        # Use more trees for better accuracy with complex gestures
        self.model = RandomForestClassifier(n_estimators=200, max_depth=30, random_state=42)
        self.model.fit(X, y)
        self.labels = list(set(y))
        
        # Save the trained model
        self.save_model()
        print("Model training completed and saved")
    
    def save_model(self):
        """Save the trained model to a file."""
        if self.model is not None:
            with open(self.model_file, 'wb') as f:
                pickle.dump(self.model, f)
            
            with open(self.labels_file, 'wb') as f:
                pickle.dump(self.labels, f)
            
            print(f"Model saved to {self.model_file}")
    
    def load_model(self):
        """Load a previously trained model from file."""
        try:
            with open(self.model_file, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(self.labels_file, 'rb') as f:
                self.labels = pickle.load(f)
            
            print(f"Model loaded from {self.model_file}")
            print(f"Available expressions: {self.labels}")
            return True
        except (FileNotFoundError, EOFError):
            print("No existing model found. You will need to train one.")
            return False
    
    def recognize_gestures(self):
        """
        Recognize expressions in real-time from webcam feed.
        Uses temporal information for dynamic gesture recognition.
        """
        if self.model is None:
            print("No model available. Please train a model first.")
            return
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # For sequence recognition
        current_sequence = []
        prediction_history = deque(maxlen=10)
        
        print("Starting expression recognition...")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Mirror the frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Extract features
            features, detected = self.extract_features(frame)
            
            if detected:
                # Add to current sequence buffer
                self.sequence_buffer.append(features)
                
                # Only make prediction when we have enough frames
                if len(self.sequence_buffer) >= self.sequence_buffer.maxlen:
                    # Average features across sequence
                    avg_features = np.mean(list(self.sequence_buffer), axis=0).tolist()
                    
                    # Make prediction
                    prediction = self.model.predict([avg_features])[0]
                    confidence = np.max(self.model.predict_proba([avg_features]))
                    
                    # Add to history for smoothing
                    if confidence > 0.4:  # Only consider predictions with decent confidence
                        prediction_history.append(prediction)
                    
                    # Only consider prediction if it appears frequently in history
                    if len(prediction_history) >= 5:
                        # Count occurrences of each prediction
                        prediction_counts = {}
                        for p in prediction_history:
                            prediction_counts[p] = prediction_counts.get(p, 0) + 1
                        
                        # Find the most common prediction
                        if prediction_counts:
                            most_common = max(prediction_counts.items(), key=lambda x: x[1])
                            stable_prediction = most_common[0]
                            
                            # Only use prediction if it appears in majority of recent frames
                            if most_common[1] >= 3:  # at least 3 out of 10 frames
                                # Check cooldown
                                current_time = time.time()
                                if (stable_prediction != self.last_prediction or 
                                    current_time - self.last_prediction_time >= self.cooldown):
                                    
                                    # Update last prediction
                                    self.last_prediction = stable_prediction
                                    self.last_prediction_time = current_time
                                    
                                    # Speak the prediction
                                    self.engine.say(stable_prediction)
                                    self.engine.runAndWait()
                                
                                # Display the prediction and confidence on frame
                                display_text = stable_prediction
                                confidence_text = f"Confidence: {confidence:.2f}"
                                
                                # Make text background for better readability
                                text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                                cv2.rectangle(frame, (40, 30), (50 + text_size[0], 65), (0, 0, 0), -1)
                                
                                cv2.putText(frame, display_text, (50, 50), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                cv2.putText(frame, confidence_text, (50, 80), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display guide text
            cv2.putText(frame, "Press 'q' to quit", (frame.shape[1] - 150, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display the frame
            cv2.imshow('Indian Sign Language Expression Recognition', frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def use_pretrained_model(self):
        """
        Load a pre-trained model for common ISL expressions.
        This is a placeholder for a future feature - in a real implementation,
        this would load a model trained on a large ISL dataset.
        """
        print("This feature would load a pre-trained model trained on a comprehensive ISL dataset.")
        print("For a production system, this would require:")
        print("1. A large dataset of Indian Sign Language gestures and expressions")
        print("2. Professional ISL interpreters to validate the data")
        print("3. Training on more sophisticated models like CNN+LSTM networks")
        
        choice = input("\nDo you want to proceed with training your own custom model? (y/n): ")
        if choice.lower() == 'y':
            self.collect_training_data()
        else:
            print("Returning to main menu.")

def main():
    """Main function to run the Enhanced Indian Sign Language Recognition System."""
    print("\n===== Enhanced Indian Sign Language Recognition System =====")
    print("This system helps facilitate communication through Indian Sign Language")
    print("It can recognize words, phrases, and sentences with moving gestures")
    
    recognizer = EnhancedISLRecognizer()
    
    while True:
        print("\nChoose an option:")
        print("1. Collect training data (train the system on your own gestures)")
        print("2. Recognize ISL expressions (requires trained model)")
        print("3. About pre-trained models (information)")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            recognizer.collect_training_data()
        elif choice == '2':
            recognizer.recognize_gestures()
        elif choice == '3':
            recognizer.use_pretrained_model()
        elif choice == '4':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()