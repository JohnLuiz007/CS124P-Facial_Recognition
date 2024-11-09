import tkinter as tk
import cv2
from PIL import Image, ImageTk, ImageDraw, ImageFont
import face_recognition
import numpy as np
import tempfile
import os
import time

class WebCamApp:
    def __init__(self, window, reference_image_folder, hairstyle_suggestions, background_image_path):
        self.window = window
        self.window.title("Webcam App")

        # Load reference images and compute average encoding for each face shape
        self.known_faces = []
        self.known_names = []

        # Iterate through each subfolder in the reference image folder
        for face_shape in os.listdir(reference_image_folder):
            shape_folder_path = os.path.join(reference_image_folder, face_shape)
            if os.path.isdir(shape_folder_path):  # Check if it's a directory
                for image_file in os.listdir(shape_folder_path):
                    image_path = os.path.join(shape_folder_path, image_file)
                    image = face_recognition.load_image_file(image_path)
                    encoding = face_recognition.face_encodings(image)
                    if encoding:
                        self.known_faces.append(encoding[0])
                        self.known_names.append(face_shape)  # Use folder name as face shape label

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

    def start_webcam(self):
        # Start webcam capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Cannot open webcam")
            return
        self.start_time = time.time()
        self.update_frame()

    def update_frame(self):
        # Capture frame-by-frame from webcam
        ret, frame = self.cap.read()
        if ret:
            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.latest_frame = rgb_frame  # Store the last frame

            # Display frame on canvas
            img = Image.fromarray(rgb_frame)
            photo = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.background_photo)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.photo = photo  # Keep reference to avoid garbage collection

        # Check if 5 seconds have passed
        if time.time() - self.start_time >= 5:
            # Stop capturing and process the last frame (determines what frame would be used for processing)
            self.process_last_frame()
        else:
            # Scheduler (lower digit = faster frames)
            self.window.after(10, self.update_frame)

    def process_last_frame(self):
        if self.latest_frame is not None:
            # Detect faces
            face_locations = face_recognition.face_locations(self.latest_frame, model="hog")  # Try 'hog' if 'cnn' fails
            print("Face locations:", face_locations)  # Debug print
            face_encodings = face_recognition.face_encodings(self.latest_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Compare with known face shapes
                distances = face_recognition.face_distance(self.known_faces, face_encoding)
                print("Distances:", distances)  # Debug print
                best_match_index = np.argmin(distances)

                # Apply a threshold for the closest match
                threshold = 0.7
                if distances[best_match_index] < threshold:
                    name = self.known_names[best_match_index]
                else:
                    name = "Unknown"
                print("Best match:", name, "with distance:", distances[best_match_index])  # Debug print

                # Display results on the frame
                img_pil = Image.fromarray(self.latest_frame)
                draw = ImageDraw.Draw(img_pil)
                draw.rectangle(((left, top), (right, bottom)), outline="green", width=3)
                draw.text((left, top - 20), name, fill="green")
                self.latest_frame = np.array(img_pil)

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

    def __del__(self):
        if self.cap is not None:
            self.cap.release()

# Main reference folder for face shape images
reference_image_folder = "C:/Users/Chris/OneDrive/Desktop/projectHDv1.0.2/reference_images"

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
background_image_path = "C:/Users/Chris/OneDrive/Desktop/projectHD/background2.jpg"

# Initialize the app
root = tk.Tk()
root.geometry("1100x800")
app = WebCamApp(root, reference_image_folder, hairstyle_suggestions, background_image_path)
root.mainloop()
