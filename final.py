import sys
import cv2
from deepface import DeepFace
from openai import OpenAI
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout,
    QHBoxLayout, QPushButton, QTextEdit
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

# ✅ OpenRouter Client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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"
)
client.headers = {
    "HTTP-Referer": "http://localhost",
    "X-Title": "Mood Rewriter App",
}


class EmotionUI(QWidget):
    def __init__(self):
        super().__init__()

        self.detected_emotion = None
        self.current_frame = None

        self.setWindowTitle("Mood Rewriter 💭")
        self.setStyleSheet("background-color: #F7EFFF;")

        # ✅ Camera Setup
        self.video = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        self.create_ui()
        self.start_camera()

    def create_ui(self):
        layout = QVBoxLayout()

        # ✅ Camera Preview Box
        self.camera_label = QLabel("Camera Loading…")
        self.camera_label.setFixedSize(650, 480)
        self.camera_label.setStyleSheet("""
            border: 3px solid #A78BFA;
            border-radius: 12px;
            background-color: #EDE9FE;
            color: #2D1B69;
            font-size: 20px;
        """)
        self.camera_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.camera_label)

        # ✅ User Input
        self.inputText = QTextEdit()
        self.inputText.setPlaceholderText("Type your message here…")
        self.inputText.setStyleSheet("""
            font-size: 18px;
            padding: 12px;
            background: #F5F3FF;
            color: #2D1B69;
            border: 2px solid #C4B5FD;
            border-radius: 10px;
        """)
        layout.addWidget(self.inputText)

        # ✅ Buttons
        btnLayout = QHBoxLayout()

        self.capture_btn = QPushButton("Capture Emotion 📸")
        self.capture_btn.setStyleSheet(self.btn_styles("#E4B1F9"))
        self.capture_btn.clicked.connect(self.capture_emotion)
        btnLayout.addWidget(self.capture_btn)

        self.generate_btn = QPushButton("Rewrite ✨")
        self.generate_btn.setStyleSheet(self.btn_styles("#BDE0FE"))
        self.generate_btn.clicked.connect(self.rewrite_text)
        btnLayout.addWidget(self.generate_btn)

        layout.addLayout(btnLayout)

        # ✅ Output Text
        self.outputText = QTextEdit()
        self.outputText.setReadOnly(True)
        self.outputText.setStyleSheet("""
            background-color: #FFF5FE;
            font-size: 18px;
            border: 2px solid #A78BFA;
            border-radius: 10px;
            padding: 12px;
            color: #2D1B69;
        """)
        layout.addWidget(self.outputText)

        self.setLayout(layout)

    def btn_styles(self, color):
        return f"""
            background-color: {color};
            border: none;
            padding: 12px;
            font-size: 16px;
            font-weight: bold;
            border-radius: 20px;
        """

    def start_camera(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.video.read()
        if not ret:
            return

        self.current_frame = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qImg = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(qImg))

    def capture_emotion(self):
        if self.current_frame is None:
            return

        gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            self.outputText.setText("😕 No face detected. Try again!")
            return

        x, y, w, h = faces[0]
        face = self.current_frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        try:
            result = DeepFace.analyze(face_rgb, actions=["emotion"], enforce_detection=False)
            self.detected_emotion = result[0]["dominant_emotion"]
            self.outputText.setText(f"✅ Emotion Detected: {self.detected_emotion}")
        except:
            self.outputText.setText("Error detecting emotion 😣")

    def rewrite_text(self):
        if not self.detected_emotion:
            self.outputText.setText("📸 First capture your emotion!")
            return

        user_text = self.inputText.toPlainText()
        if not user_text.strip():
            self.outputText.setText("✍️ Enter text to rewrite!")
            return

        prompt = f"Rewrite this text with a {self.detected_emotion} tone:\n{user_text}"

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Rewrite text naturally with emotional tone"},
                {"role": "user", "content": prompt}
            ]
        )

        rewritten = response.choices[0].message.content.strip()
        self.outputText.setText(rewritten)

    def closeEvent(self, event):
        if self.video.isOpened():
            self.video.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = EmotionUI()
    win.show()
    sys.exit(app.exec_())
