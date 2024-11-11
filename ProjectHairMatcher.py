import tkinter as tk
import cv2
from PIL import Image, ImageTk, ImageDraw, ImageFont
import face_recognition
import numpy as np
import os
import time
from threading import Thread
import imutils

import pickle
class WebCamApp:
    def __init__(self, window, all_face_encodings, hairstyle_suggestions):
        self.window = window
        self.window.title("Hair Matcher")

        # Load reference images and compute average encoding for each face shape
        self.known_faces = np.array(list(all_face_encodings["encodings"]))
        self.known_names = list(all_face_encodings["names"])

        # Hairstyle suggestions
        self.hairstyle_suggestions = hairstyle_suggestions
        self.logo_image = ImageTk.PhotoImage(Image.open(os.getcwd()+"\\logo.png").resize((300, 300)))
        self.haircut_image = None

        # Canvas for webcam
        self.canvas = tk.Canvas(self.window,width=300, bd=0, highlightthickness=0)
        self.canvas.pack(padx=0, pady=0)

        
        
        # Label for hairstyle suggestions
        self.suggestion_label = tk.Label(self.window, text="Haircut Matcher", font=("Helvetica", 18), fg="black")
        self.suggestion_label.pack()

        self.image_label = tk.Label(root, image=self.logo_image)
        self.image_label.pack()
        
        
        self.button_frame = tk.Frame(root, width=600, height=100)
        self.prev_button = tk.Button(self.button_frame, text="<", width=20, command=lambda: self.change_suggestion_preview(value=-1))
        self.prev_button.grid(row=0, column=0)
        self.next_button = tk.Button(self.button_frame, text=">", width=20, command=lambda: self.change_suggestion_preview(value=1))
        self.next_button.grid(row=0, column=1)
        self.button_frame.pack()

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
        self.suggestions = []
        self.preview_index = 0

        # Init
        self.start_webcam()


    def start_webcam(self):
        # Start webcam capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Cannot open webcam")
            return
        self.camera_thread.start()
        self.process_thread.start()
        
    def suggest_haircut(self):
        try:
            if self.detected_shape:
                self.suggestions = self.hairstyle_suggestions[self.detected_shape]
                self.haircut_image = ImageTk.PhotoImage(Image.open(os.getcwd()+f"\\Haircuts\\{self.detected_shape}\\{self.hairstyle_suggestions[self.detected_shape][self.preview_index]}.jpg").resize((300, 300)))
                self.image_label.config(image=self.haircut_image)
                self.suggestion_label.config(text=self.hairstyle_suggestions[self.detected_shape][self.preview_index])
            else:
                self.image_label.config(image=self.logo_image)
                self.suggestion_label.config(text="Haircut Matcher")
                self.suggestions = None
        except KeyError:
            self.image_label.config(image=self.logo_image)
            self.suggestion_label.config(text="Haircut Matcher")
            self.suggestions = None
            self.detected_shape = None

    def change_suggestion_preview(self, value):
        if self.suggestions:
            self.preview_index += value
            if self.preview_index > (len(self.suggestions) - 1) or self.preview_index < 0:
                if self.preview_index > (len(self.suggestions) - 1):
                    self.preview_index = 0
                else:
                    self.preview_index = len(self.suggestions) - 1
            self.haircut_image = ImageTk.PhotoImage(Image.open(os.getcwd()+f"\\Haircuts\\{self.detected_shape}\\{self.hairstyle_suggestions[self.detected_shape][self.preview_index]}.jpg").resize((300, 300)))
            self.image_label.config(image=self.haircut_image)
            self.suggestion_label.config(text=self.hairstyle_suggestions[self.detected_shape][self.preview_index])
        else:
            print("NO FACE DETECTED!")

    def update_frame(self):
        # Capture frame-by-frame from webcam
        while True:
            ret, frame = self.cap.read()
            frame = imutils.resize(frame, width=300)
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


                    best_matched_indexes = [i for (i, b) in enumerate(distances) if b <= 0.65]
                    counts = {}

                    # loop over the matched indexes and maintain a count for
                    # each recognized face face
                    for i in best_matched_indexes:
                        name = self.known_names[i]
                        counts[name] = counts.get(name, 0) + 1

                    # determine the recognized face with the largest number
                    # of votes (note: in the event of an unlikely tie Python
                    # will select first entry in the dictionary)
                    name = max(counts, key=counts.get, default=None)
                    self.detected_shape = name

                    print(counts)
                    print("Best match:", name, "with distance:", distances[best_match_index])  # Debug print
                else:
                    name = None
                    self.detected_shape = None
                
                self.suggest_haircut()
            time.sleep(1)

# Suggested hairstyles 
hairstyle_suggestions = {
    "Heart": ["Undercut", "Slick Back", "Fringe"],
    "Oblong": ["Crew Cut", "Tapered Sides", "Textured Crop"],
    "Oval": ["Taper Fade", "Side Part", "Caesar Cut"],
    "Round": ["Textured Top", "High Fade", "Angular Fringe"],
    "Square": ["Buzz Cut", "French Crop", "Pompadour"]
}


# Load face encodings
with open('dataset_face_shape.dat', 'rb') as f:
	all_face_encodings = pickle.load(f)

# Initialize the app
root = tk.Tk()
root.resizable(width=False, height=False)
root.geometry("500x700")
app = WebCamApp(root,  all_face_encodings, hairstyle_suggestions)
root.mainloop()