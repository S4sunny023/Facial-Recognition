import tkinter as tk
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image, ImageTk

class EmotionDetectorApp:
    def __init__(self, root, model_path, cascade_path):
        self.root = root
        self.root.title("Real-time Emotion Detector")

        # Load the pre-trained model
        self.classifier = load_model(model_path)
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

        # Load the face cascade classifier
        self.face_classifier = cv2.CascadeClassifier(cascade_path)

        # Initialize the video capture
        self.cap = cv2.VideoCapture(0)

        # Create a canvas to display video feed
        self.canvas = tk.Canvas(self.root, width=800, height=600)
        self.canvas.pack()

        # Start the emotion detection loop
        self.detect_emotions()

    def detect_emotions(self):
        _, frame = self.cap.read()
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally for natural viewing

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = self.face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract the region of interest (ROI) for the face
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                # Preprocess the ROI for emotion prediction
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # Perform emotion prediction
                prediction = self.classifier.predict(roi)[0]
                label = self.emotion_labels[prediction.argmax()]

                # Display the predicted emotion label on the frame
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            else:
                cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the processed frame in the Tkinter GUI
        self.display_frame(frame)

        # Schedule the next frame capture and emotion detection
        self.root.after(10, self.detect_emotions)

    def display_frame(self, frame):
        # Convert the OpenCV frame to RGB format for displaying in Tkinter
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)

        # Resize the image while preserving aspect ratio
        target_width, target_height = 800, 600
        width, height = img.size
        aspect_ratio = width / height

        if width > target_width or height > target_height:
            if aspect_ratio > 1:
                new_width = target_width
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = target_height
                new_width = int(new_height * aspect_ratio)
            img = img.resize((new_width, new_height), Image.BICUBIC)

        img_tk = ImageTk.PhotoImage(image=img)

        # Update the canvas with the new frame
        if hasattr(self, 'canvas_image'):
            self.canvas.delete(self.canvas_image)
        self.canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.img_tk = img_tk  # Keep a reference to prevent garbage collection

    def __del__(self):
        # Release the video capture and destroy OpenCV windows when the application exits
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()

def main():
    # Specify the paths to the pre-trained model and face cascade classifier
    model_path = 'model.h5'
    cascade_path = 'haarcascade_frontalface_default.xml'

    # Create and start the Tkinter application
    root = tk.Tk()
    app = EmotionDetectorApp(root, model_path, cascade_path)
    root.mainloop()

if __name__ == "__main__":
    main()
