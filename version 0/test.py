import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import torch
import torchvision
import torchaudio
import sklearn
import matplotlib.pyplot as plt
import gtts
import speech_recognition as sr
import serial
from flask import Flask
from fastapi import FastAPI
import uvicorn
import pygame
import os
import time

def test_opencv():
    """Test OpenCV by opening the camera."""
    print("\n🎥 Testing OpenCV...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ OpenCV failed to access the webcam.")
        return
    print("✅ OpenCV is working! Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('OpenCV Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def test_mediapipe():
    """Test Mediapipe hand tracking."""
    print("\n🖐️ Testing Mediapipe Hand Tracking...")
    mp_hands = mp.solutions.hands.Hands()
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        cv2.imshow('Mediapipe Hand Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Mediapipe is working!")

def test_numpy():
    """Test NumPy operations."""
    print("\n🧮 Testing NumPy...")
    arr = np.array([1, 2, 3, 4, 5])
    print(f"✅ NumPy is working! Array: {arr}")

def test_pandas():
    """Test Pandas by creating a DataFrame."""
    print("\n📊 Testing Pandas...")
    df = pd.DataFrame({"Name": ["Alice", "Bob"], "Age": [25, 30]})
    print("✅ Pandas is working!\n", df)

def test_tensorflow():
    """Test TensorFlow/Keras."""
    print("\n🧠 Testing TensorFlow/Keras...")
    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
    print(f"✅ TensorFlow is working! Model: {model.summary()}")

def test_pytorch():
    """Test PyTorch tensor operations."""
    print("\n🔥 Testing PyTorch...")
    tensor = torch.tensor([1.0, 2.0, 3.0])
    print(f"✅ PyTorch is working! Tensor: {tensor}")

def test_sklearn():
    """Test Scikit-learn with a simple ML model."""
    print("\n📈 Testing Scikit-learn...")
    from sklearn.linear_model import LinearRegression
    model = LinearRegression().fit([[1], [2]], [2, 4])
    print(f"✅ Scikit-learn is working! Prediction for input [3]: {model.predict([[3]])}")

def test_matplotlib():
    """Test Matplotlib by displaying a plot."""
    print("\n📉 Testing Matplotlib...")
    plt.plot([1, 2, 3], [4, 5, 6])
    plt.title("Matplotlib Test")
    plt.show()
    print("✅ Matplotlib is working!")

def test_gtts():
    """Test gTTS by generating speech and playing it."""
    print("\n🔊 Testing gTTS...")
    tts = gtts.gTTS("Hello, this is a test of gTTS!")
    tts.save("test.mp3")
    pygame.mixer.init()
    pygame.mixer.music.load("test.mp3")
    pygame.mixer.music.play()
    print("✅ gTTS is working! Playing audio...")
    while pygame.mixer.music.get_busy():
        time.sleep(1)

def test_speech_recognition():
    """Test SpeechRecognition by listing microphones."""
    print("\n🎙️ Testing SpeechRecognition...")
    mics = sr.Microphone.list_microphone_names()
    print(f"✅ SpeechRecognition is working! Available microphones: {mics}")

def test_pyaudio():
    """Check PyAudio installation."""
    print("\n🎤 Testing PyAudio...")
    try:
        import pyaudio
        print("✅ PyAudio is installed!")
    except ImportError:
        print("❌ PyAudio not installed. Install with: pip install pyaudio")

def test_flask():
    """Run a simple Flask app."""
    print("\n🌐 Testing Flask...")
    app = Flask(__name__)

    @app.route('/')
    def home():
        return "✅ Flask is working!"

    app.run(port=5000, debug=False, use_reloader=False)

def test_fastapi():
    """Run a simple FastAPI app."""
    print("\n🚀 Testing FastAPI...")
    app = FastAPI()

    @app.get("/")
    def read_root():
        return {"message": "✅ FastAPI is working!"}

    uvicorn.run(app, host="127.0.0.1", port=8000)

def test_uvicorn():
    """Check Uvicorn installation."""
    print("\n⚡ Testing Uvicorn...")
    print("✅ Uvicorn is installed! Run FastAPI with: uvicorn main:app --reload")

def test_serial():
    """List available serial ports."""
    print("\n🔌 Testing PySerial...")
    try:
        ports = serial.tools.list_ports.comports()
        port_list = [port.device for port in ports]
        print(f"✅ Serial Ports: {port_list}")
    except Exception as e:
        print(f"❌ PySerial failed: {e}")

def main():
    """Run all tests."""
    print("\n🔧 Running full setup test...\n")
    
    test_opencv()
    test_mediapipe()
    test_numpy()
    test_pandas()
    test_tensorflow()
    test_pytorch()
    test_sklearn()
    test_matplotlib()
    test_gtts()
    test_speech_recognition()
    test_pyaudio()
    test_serial()
    test_flask()
    test_fastapi()
    test_uvicorn()

    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    main()
