import tkinter as tk
import cv2
from PIL import Image, ImageTk, ImageDraw, ImageFont
import face_recognition
import numpy as np
import tempfile
import os
import time
from threading import Thread
import imutils

import pickle
class WebCamApp:
    def __init__(self, window, all_face_encodings, hairstyle_suggestions, background_image_path):
        self.window = window
        self.window.title("Webcam App")

        # Load reference images and compute average encoding for each face shape
        self.known_faces = np.array(list(all_face_encodings["encodings"]))
        self.known_names = list(all_face_encodings["names"])

        # Hairstyle suggestions
        self.hairstyle_suggestions = hairstyle_suggestions

        # Load and set the background image
        self.background_image = Image.open(background_image_path)
        self.background_photo = ImageTk.PhotoImage(self.background_image)

        # Start button to initiate detection
        self.start_button = tk.Button(self.window, text="Start Detector", command=self.start_webcam, height=5, width=125)
        self.start_button.pack(pady=10, padx=10)

        # Print button
        self.print_button = tk.Button(self.window, text="Print Image", command=self.print_image, height=5, width=125)
        self.print_button.pack(pady=10, padx=10)

        # Label for hairstyle suggestions
        self.suggestion_label = tk.Label(self.window, text="", font=("Helvetica", 14), fg="black")
        self.suggestion_label.pack(pady=10)

        # Canvas for webcam
        self.canvas = tk.Canvas(self.window, width=self.background_image.width, height=self.background_image.height)
        self.canvas.pack()

        # Initialize variables for capturing
        self.cap = None
        self.latest_frame = None
        self.start_time = None

        #Multithreading
        self.camera_thread = Thread(target=self.update_frame, args=())
        self.process_thread = Thread(target=self.process_frame, args=())
        self.camera_thread.daemon = True
        self.process_thread.daemon = True

        # Caffe
        self.net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

        # Variable for detection
        self.detected_shape = None


    def start_webcam(self):
        # Start webcam capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Cannot open webcam")
            return
        self.camera_thread.start()
        self.process_thread.start()
        
        

    def update_frame(self):
        # Capture frame-by-frame from webcam
        while True:
            ret, frame = self.cap.read()
            frame = imutils.resize(frame, width=400)
            if ret:
                # Convert frame to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.latest_frame = rgb_frame  # Store the last frame

                # grab the frame dimensions and convert it to a blob
                (h, w) = self.latest_frame.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                    (300, 300), (104.0, 177.0, 123.0))
            
                # pass the blob through the network and obtain the detections and
                # predictions
                self.net.setInput(blob)
                detections = self.net.forward()
                if detections.any():
                    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
            
                    # draw the bounding box of the face along with the associated
                    # probability
                    text = self.detected_shape
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(self.latest_frame, (startX, startY), (endX, endY),
                        (0, 0, 255), 2)
                    cv2.putText(self.latest_frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

                # Display frame on canvas
                img = Image.fromarray(self.latest_frame)
                photo = ImageTk.PhotoImage(image=img)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                self.photo = photo  # Keep reference to avoid garbage collection
            

    def process_frame(self):
        while True:
            if self.latest_frame is not None:
                #Detect faces
                face_locations = face_recognition.face_locations(self.latest_frame, model="hog")  # Try 'hog' if 'cnn' fails
                face_encodings = face_recognition.face_encodings(self.latest_frame, face_locations)

                if face_encodings:
                    face_encoding = face_encodings[0]

                    #Compare with known face shapes
                    distances = face_recognition.face_distance(self.known_faces, face_encoding)
                    best_match_index = np.argmin(distances)

                    # Apply a threshold for the closest match
                    threshold = 0.75
                    if distances[best_match_index] <= threshold:
                        name = self.known_names[best_match_index]
                    else:
                        name = "Unknown"
                    name = self.known_names[best_match_index]
                    self.detected_shape = name
                    print("Best match:", name, "with distance:", distances[best_match_index])  # Debug print
                    
                    time.sleep(1)
            


    def print_image(self):
        if self.latest_frame is not None:
            pil_image = Image.fromarray(self.latest_frame)  
            draw = ImageDraw.Draw(pil_image)
            face_shape = self.suggestion_label.cget("text").split(": ")[0] if self.suggestion_label.cget("text") else "Unknown"
            hairstyles = self.suggestion_label.cget("text").split(": ")[1] if ":" in self.suggestion_label.cget("text") else "No suggestions"
            text_to_draw = f"Face Shape: {face_shape}\nSuggested Hairstyles: {hairstyles}"

            font = ImageFont.load_default()
            draw.text((10, 10), text_to_draw, fill="green", font=font)

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            pil_image.save(temp_file.name)
            os.startfile(temp_file.name)

# Main reference folder for face shape images
reference_image_folder =  os.getcwd() + "/reference_images"

# Suggested hairstyles 
hairstyle_suggestions = {
    "diamond face shape": ["Style1", "Style2", "Style3"],
    "heart face shape": ["Style1", "Style2", "Style3"],
    "oblong face shape": ["Style1", "Style2", "Style3"],
    "oval face shape": ["Style1", "Style2", "Style3"],
    "round face shape": ["Style1", "Style2", "Style3"],
    "square face shape": ["Style1", "Style2", "Style3"]
}

# Background image path
background_image_path = os.getcwd() + "/background2.jpg"


# Load face encodings
with open('dataset_face_shape.dat', 'rb') as f:
	all_face_encodings = pickle.load(f)

# Initialize the app
root = tk.Tk()
root.geometry("1100x800")
app = WebCamApp(root,  all_face_encodings, hairstyle_suggestions, background_image_path)
root.mainloop()
