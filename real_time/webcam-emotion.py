import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

class RealTimeEmotion:
    def __init__(self, model_path, img_height, img_width, sequence_length, classes):
        self.model = load_model(model_path)
        self.img_height = img_height
        self.img_width = img_width
        self.sequence_length = sequence_length
        self.classes = classes
        self.frame_queue = deque(maxlen=self.sequence_length)

    def preprocess_frame(self, frame):
        """
        Preprocess a single video frame.
        """
        frame_resized = cv2.resize(frame, (self.img_height, self.img_width))
        return frame_resized / 255.0

    def predict_emotion(self, frame):
        """
        Predict emotion for the current frame sequence.
        """
        self.frame_queue.append(self.preprocess_frame(frame))

        if len(self.frame_queue) == self.sequence_length:
            sequence = np.array(self.frame_queue)
            prediction = self.model.predict(np.expand_dims(sequence, axis=0))
            emotion = self.classes[np.argmax(prediction)]
            return emotion
        return None

    def run(self):
        """
        Run real-time emotion detection using webcam.
        """
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            emotion = self.predict_emotion(frame)

            if emotion:
                cv2.putText(frame, emotion, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imshow('Real-Time Emotion Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
