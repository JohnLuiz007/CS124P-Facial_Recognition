import tkinter as tk
import cv2
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
import tempfile
import os
import time
import mediapipe as mp

class WebCamApp:
    def __init__(self, window, reference_image_folder, hairstyle_suggestions, background_image_path):
        self.window = window
        self.window.title("Webcam App")

        # Initialize mediapipe face mesh model
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh()

        # Initialize variables for capturing
        self.cap = None
        self.latest_frame = None
        self.start_time = None

        # Load reference images and compute average encoding for each face shape
        self.known_shapes = {
            "diamond": [],
            "heart": [],
            "oblong": [],
            "oval": [],
            "round": [],
            "square": []
        }

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
            # Convert BGR frame to RGB for face mesh processing
            rgb_frame = cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2RGB)
            # Process frame with face mesh model
            results = self.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Extract facial landmarks
                    landmarks = face_landmarks.landmark
                    # Example: Compute geometric properties like the ratio of jaw to forehead
                    # Calculate the distances or ratios between relevant landmarks to classify face shapes
                    
                    # Use simple ratios or distances (this is a basic example)
                    distance_cheekbone_to_chin = np.linalg.norm(np.array([landmarks[234].x, landmarks[234].y]) - np.array([landmarks[454].x, landmarks[454].y]))
                    distance_forehead = np.linalg.norm(np.array([landmarks[10].x, landmarks[10].y]) - np.array([landmarks[152].x, landmarks[152].y]))

                    # Use these metrics to classify the face shape (basic thresholding)
                    if distance_cheekbone_to_chin > distance_forehead:
                        face_shape = "round"
                    else:
                        face_shape = "oval"

                    # Display results on the frame
                    self.suggestion_label.config(text=f"Face Shape: {face_shape}\nSuggested Hairstyles: {', '.join(self.hairstyle_suggestions[face_shape])}")

                    # Draw face landmarks for visualization (optional)
                    img_pil = Image.fromarray(self.latest_frame)
                    draw = ImageDraw.Draw(img_pil)
                    for landmark in landmarks:
                        x = int(landmark.x * img_pil.width)
                        y = int(landmark.y * img_pil.height)
                        draw.ellipse((x-2, y-2, x+2, y+2), fill="red")

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
    "diamond": ["Style1", "Style2", "Style3"],
    "heart": ["Style1", "Style2", "Style3"],
    "oblong": ["Style1", "Style2", "Style3"],
    "oval": ["Style1", "Style2", "Style3"],
    "round": ["Style1", "Style2", "Style3"],
    "square": ["Style1", "Style2", "Style3"]
}

# Background image path
background_image_path = "C:/Users/Chris/OneDrive/Desktop/projectHD/background2.jpg"

# Initialize the app
root = tk.Tk()
root.geometry("1100x800")
app = WebCamApp(root, reference_image_folder, hairstyle_suggestions, background_image_path)
root.mainloop()
