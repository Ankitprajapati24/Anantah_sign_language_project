import os
import cv2
import numpy as np
import pickle
import time
import pyttsx3
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from collections import deque

class SignLanguageTranslator:
    def __init__(self):
        # Initialize MediaPipe hands model
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
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
        self.prediction_history = deque(maxlen=10)
        self.current_gesture = None
        
    def extract_hand_landmarks(self, frame):
        """Extract hand landmarks from a frame using MediaPipe"""
        # Convert the frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        landmarks = []
        if results.multi_hand_landmarks:
            # Process the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw the landmarks on the frame
            self.mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                self.mp_hands.HAND_CONNECTIONS
            )
            
            # Extract landmark coordinates and flatten them
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        return landmarks, frame
    
    def collect_data(self):
        """Collect training data for different gestures"""
        print("\n===== DATA COLLECTION MODE =====")
        print("Press 'c' to start collecting a new gesture")
        print("Press 'q' to quit data collection mode\n")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        collecting = False
        current_label = None
        frames_collected = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Flip the frame horizontally for a more intuitive view
            frame = cv2.flip(frame, 1)
            
            landmarks, frame = self.extract_hand_landmarks(frame)
            
            if collecting and landmarks:
                self.data.append(landmarks)
                self.labels.append(current_label)
                frames_collected += 1
                
                # Display collection progress
                cv2.putText(
                    frame, 
                    f"Collecting: {current_label} - {frames_collected}/100", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
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
                    "Press 'c' to collect a new gesture", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
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
                frames_collected = 0
                print(f"Starting to collect data for '{current_label}'. Make the gesture...")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Save collected data
        if self.data and self.labels:
            with open('data/gesture_data.pkl', 'wb') as f:
                pickle.dump((self.data, self.labels), f)
            print(f"Data saved: {len(self.labels)} frames for {len(set(self.labels))} gestures")
        else:
            print("No data collected")
    
    def load_data(self):
        """Load previously collected data"""
        try:
            with open('data/gesture_data.pkl', 'rb') as f:
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
        
        # Handle cases where some frames didn't capture landmarks
        valid_data = []
        valid_labels = []
        
        # Ensure all feature vectors have the same length (63 for 21 landmarks with x,y,z)
        expected_length = 63
        
        for features, label in zip(self.data, self.labels):
            if len(features) == expected_length:
                valid_data.append(features)
                valid_labels.append(label)
        
        if not valid_data:
            print("Error: No valid data samples found")
            return False
        
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
        with open('models/gesture_model.pkl', 'wb') as f:
            pickle.dump((self.model, list(set(valid_labels))), f)
        print("Model saved to models/gesture_model.pkl")
        
        return True
    
    def load_model(self):
        """Load a trained model"""
        try:
            with open('models/gesture_model.pkl', 'rb') as f:
                self.model, unique_labels = pickle.load(f)
            print(f"Model loaded. Recognizes gestures: {', '.join(unique_labels)}")
            return True
        except FileNotFoundError:
            print("No saved model found. Please train a model first.")
            return False
    
    def predict_gesture(self, landmarks):
        """Predict gesture from landmarks using the trained model"""
        if not self.model:
            return None
            
        if not landmarks or len(landmarks) != 63:  # 21 landmarks * 3 (x, y, z)
            return None
            
        prediction = self.model.predict([landmarks])[0]
        
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
    
    def speak_gesture(self, gesture):
        """Speak the recognized gesture using text-to-speech"""
        # Check if enough time has passed since the last prediction
        current_time = time.time()
        
        # Only speak if it's a new gesture or enough time has passed
        if (gesture != self.current_gesture or 
                current_time - self.last_prediction_time > self.prediction_cooldown):
            self.engine.say(gesture)
            self.engine.runAndWait()
            self.last_prediction_time = current_time
            self.current_gesture = gesture
    
    def run_recognition(self):
        """Run real-time gesture recognition"""
        if not self.model and not self.load_model():
            print("Cannot run recognition: No model available")
            return
        
        print("\n===== RECOGNITION MODE =====")
        print("Press 'q' to quit\n")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Flip the frame horizontally for a more intuitive view
            frame = cv2.flip(frame, 1)
            
            landmarks, frame = self.extract_hand_landmarks(frame)
            
            if landmarks:
                gesture = self.predict_gesture(landmarks)
                
                if gesture:
                    # Display the recognized gesture
                    cv2.putText(
                        frame, 
                        f"Gesture: {gesture}", 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 255, 0), 
                        2
                    )
                    
                    # Speak the gesture
                    self.speak_gesture(gesture)
            else:
                cv2.putText(
                    frame, 
                    "No hand detected", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 0, 255), 
                    2
                )
            
            cv2.imshow("Sign Language Recognition", frame)
            
            if cv2.waitKey(1) == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function to run the Sign Language Translator"""
    translator = SignLanguageTranslator()
    
    while True:
        print("\n===== SIGN LANGUAGE TRANSLATOR =====")
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