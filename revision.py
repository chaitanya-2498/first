import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

mp_hands = mp.solutions.hands

class HandDrawingProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(min_detection_confidence=0.7,
                                    min_tracking_confidence=0.7)
        self.prev_x = 0
        self.prev_y = 0
        self.canvas = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        if self.canvas is None:
            self.canvas = np.zeros_like(img)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                h, w, _ = img.shape
                index_tip = hand_landmarks.landmark[8]
                x = int(index_tip.x * w)
                y = int(index_tip.y * h)

                cv2.circle(img, (x, y), 8, (0, 255, 0), -1)

                if self.prev_x == 0 and self.prev_y == 0:
                    self.prev_x, self.prev_y = x, y

                cv2.line(self.canvas, (self.prev_x, self.prev_y), (x, y), (255, 0, 0), 5)

                self.prev_x, self.prev_y = x, y
        else:
            self.prev_x, self.prev_y = 0, 0

        combined = cv2.add(img, self.canvas)
        return combined

st.title("🖐 Air Writing Web App")

rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="example",
    video_processor_factory=HandDrawingProcessor,
    rtc_configuration=rtc_configuration
)
