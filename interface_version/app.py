import av
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import mediapipe as mp
from gesture_config import GESTURE_MAP

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

class HandGestureProcessor(VideoProcessorBase):
    def get_finger_states(self, landmarks):
        finger_states = []
        # Thumb
        if landmarks[4].x < landmarks[3].x:
            finger_states.append(1)
        else:
            finger_states.append(0)
        # Other fingers
        tips_ids = [8, 12, 16, 20]
        pip_ids = [6, 10, 14, 18]
        for tip, pip in zip(tips_ids, pip_ids):
            finger_states.append(1 if landmarks[tip].y < landmarks[pip].y else 0)
        return finger_states

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                finger_states = self.get_finger_states(hand_landmarks.landmark)
                gesture = GESTURE_MAP.get(tuple(finger_states), "Unknown")
                
                # Display gesture text
                cv2.putText(img, gesture, (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit UI
st.title("👋 Real-time Hand Gesture Recognition")
st.markdown("""
    ## Sign Language Detection App
    Show your hand to the webcam and see recognized gestures!
    ### Supported Gestures:
    - 👊 Fist
    - ✋ Open Palm
    - 👍 Thumbs Up
    - ✌ Victory
    - ☝ Pointing
""")

webrtc_ctx = webrtc_streamer(
    key="hand-gesture",
    video_processor_factory=HandGestureProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

if not webrtc_ctx.state.playing:
    st.info("Please allow camera access to start detection...")