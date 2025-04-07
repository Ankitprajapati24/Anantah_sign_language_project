# Indian Sign Language Recognition System
# This system detects and translates Indian Sign Language gestures in real-time
# It uses MediaPipe to track hands and face, scikit-learn for gesture recognition

import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3
import time
import os
from sklearn.ensemble import RandomForestClassifier
from collections import deque

class IndianSignLanguageRecognizer:
    def __init__(self):
        # Initialize MediaPipe solutions
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize hand tracking (detect both hands)
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Detect both hands
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize face mesh detection
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        
        # Initialize classifier
        self.model = None
        self.labels = []
        self.model_file = 'data/isl_gesture_model.pkl'
        self.labels_file = 'data/isl_gesture_labels.pkl'
        
        # Cooldown settings for prediction
        self.last_prediction = ""
        self.last_prediction_time = 0
        self.cooldown = 2  # seconds
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Motion tracking variables
        self.motion_history = deque(maxlen=10)  # Track last 10 frames of hand positions
        
        # Try to load pre-existing model if available
        self.load_model()
    
    def extract_features(self, frame):
        """Extract hand landmarks and face landmarks from a frame using MediaPipe."""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe hands and face
        hands_results = self.hands.process(frame_rgb)
        face_results = self.face_mesh.process(frame_rgb)
        
        features = []
        hands_detected = False
        face_detected = False
        
        # Extract hand landmarks if detected
        if hands_results.multi_hand_landmarks:
            hands_detected = True
            
            # Draw hand landmarks on frame
            for hand_landmarks in hands_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
            
            # Extract coordinates from all detected hands (up to 2)
            hand_coords = []
            for hand_landmarks in hands_results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    hand_coords.extend([landmark.x, landmark.y, landmark.z])
                
                # If only one hand is detected, append zeros for the second hand
                if len(hands_results.multi_hand_landmarks) == 1:
                    hand_coords.extend([0.0] * (21 * 3))  # 21 landmarks with x,y,z coordinates
            
            features.extend(hand_coords)
            
            # Track hand motion
            if len(hands_results.multi_hand_landmarks) > 0:
                # Get position of index finger tip (landmark 8)
                idx_tip = hands_results.multi_hand_landmarks[0].landmark[8]
                self.motion_history.append((idx_tip.x, idx_tip.y))
        else:
            # No hands detected, fill with zeros
            features.extend([0.0] * (2 * 21 * 3))  # 2 hands, 21 landmarks per hand, 3 coordinates per landmark
        
        # Extract face landmarks if detected
        if face_results.multi_face_landmarks:
            face_detected = True
            
            # Draw face landmarks
            for face_landmarks in face_results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                )
            
            # Extract facial expression features (focus on key points like eyes, mouth)
            # For simplicity, we'll use a subset of landmarks
            key_face_indices = [0, 17, 61, 78, 80, 13, 14, 291, 312, 311, 402, 415]  # Key facial points
            
            face_coords = []
            for idx in key_face_indices:
                landmark = face_results.multi_face_landmarks[0].landmark[idx]
                face_coords.extend([landmark.x, landmark.y, landmark.z])
            
            features.extend(face_coords)
        else:
            # No face detected, fill with zeros
            features.extend([0.0] * (12 * 3))  # 12 key face landmarks, 3 coordinates per landmark
        
        # Calculate motion features if we have enough history
        motion_features = []
        if len(self.motion_history) >= 2:
            # Calculate velocity between last two frames
            p1 = self.motion_history[-2]
            p2 = self.motion_history[-1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            motion_features = [dx, dy]
        else:
            motion_features = [0.0, 0.0]
        
        features.extend(motion_features)
        
        return features, hands_detected or face_detected
    
    def collect_training_data(self):
        """Collect training data for different gestures."""
        cap = cv2.VideoCapture(1)
        
        training_data = []
        labels_data = []
        
        gestures = []
        while True:
            # Ask for the name of the gesture
            gesture_name = input("Enter the name of the gesture to record (or 'done' to finish): ")
            if gesture_name.lower() == 'done':
                break
            gestures.append(gesture_name)
            
            print(f"Prepare to show the '{gesture_name}' gesture...")
            print("Press 's' to start recording samples (aim for at least 30 samples)")
            print("Press 'q' when done with this gesture")
            
            recording = False
            samples_count = 0
            
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
                status_text = "Ready. Press 's' to start recording" if not recording else f"Recording: {samples_count} samples"
                cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Show the frame
                cv2.imshow('Data Collection', frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                # Start recording with 's'
                if key == ord('s'):
                    recording = True
                    print("Recording started. Make the gesture consistently...")
                
                # If recording and features detected, save the sample
                if recording and detected:
                    training_data.append(features)
                    labels_data.append(gesture_name)
                    samples_count += 1
                    time.sleep(0.1)  # Brief pause between samples
                
                # Quit this gesture recording with 'q'
                if key == ord('q'):
                    print(f"Recorded {samples_count} samples for '{gesture_name}'")
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Train the model with collected data
        if training_data and labels_data:
            self.train_model(training_data, labels_data)
            print(f"Model trained with gestures: {gestures}")
        else:
            print("No training data collected")
    
    def train_model(self, X, y):
        """Train the RandomForest classifier with the collected data."""
        print("Training model...")
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
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
            print(f"Available gestures: {self.labels}")
            return True
        except (FileNotFoundError, EOFError):
            print("No existing model found. You will need to train one.")
            return False
    
    def recognize_gestures(self):
        """Recognize gestures in real-time from webcam feed."""
        if self.model is None:
            print("No model available. Please train a model first.")
            return
        
        cap = cv2.VideoCapture(1)
        
        # For sequence detection
        prediction_history = deque(maxlen=5)
        
        print("Starting gesture recognition...")
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
            
            if detected and self.model is not None:
                # Make prediction
                prediction = self.model.predict([features])[0]
                confidence = np.max(self.model.predict_proba([features]))
                
                # Add to history for smoothing
                prediction_history.append(prediction)
                
                # Only consider prediction if it appears frequently in history
                if len(prediction_history) == prediction_history.maxlen:
                    # Count occurrences of each prediction
                    prediction_counts = {}
                    for p in prediction_history:
                        prediction_counts[p] = prediction_counts.get(p, 0) + 1
                    
                    # Find the most common prediction
                    most_common = max(prediction_counts.items(), key=lambda x: x[1])
                    stable_prediction = most_common[0]
                    
                    # Only use prediction if it appears in majority of frames
                    if most_common[1] >= 3:  # at least 3 out of 5 frames
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
                        
                        # Display the prediction on frame
                        confidence_text = f"Confidence: {confidence:.2f}"
                        cv2.putText(frame, stable_prediction, (50, 50), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                        cv2.putText(frame, confidence_text, (50, 100), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Indian Sign Language Recognition', frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function to run the Sign Language Recognition System."""
    print("=== Indian Sign Language Recognition System ===")
    print("This system helps detect and translate Indian Sign Language gestures")
    
    recognizer = IndianSignLanguageRecognizer()
    
    while True:
        print("\nChoose an option:")
        print("1. Collect training data")
        print("2. Recognize gestures (requires trained model)")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            recognizer.collect_training_data()
        elif choice == '2':
            recognizer.recognize_gestures()
        elif choice == '3':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()