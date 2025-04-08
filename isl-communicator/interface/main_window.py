from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QTimer
import sys
import cv2
import numpy as np

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi('interface/main_window.ui', self)
        
        # Initialize components
        self.video_label = self.findChild(QtWidgets.QLabel, 'videoLabel')
        self.subtitle_label = self.findChild(QtWidgets.QLabel, 'subtitleLabel')
        self.start_btn = self.findChild(QtWidgets.QPushButton, 'startButton')
        
        # Setup video capture
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Initialize ISL components
        self.landmarker = ISLLandmarker()
        self.model = tf.keras.models.load_model('models/gesture_model.h5')
        self.sequence = []
        
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            landmarks = self.landmarker.extract_landmarks(frame)
            self.sequence.append(landmarks)
            
            if len(self.sequence) == 30:
                prediction = self.model.predict(np.expand_dims(self.sequence, axis=0))
                self.show_prediction(prediction)
                self.sequence = []
            
            # Display frame
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            self.video_label.setPixmap(QtGui.QPixmap.fromImage(qt_image))
            
    def show_prediction(self, prediction):
        gesture = GESTURES[np.argmax(prediction)]
        self.subtitle_label.setText(gesture)
        self.text_to_speech(gesture)
        
    def text_to_speech(self, text):
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        
    def start_stop_capture(self):
        if not self.timer.isActive():
            self.timer.start(20)
            self.start_btn.setText('Stop')
        else:
            self.timer.stop()
            self.start_btn.setText('Start')

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())