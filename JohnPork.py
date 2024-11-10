import tkinter as tk
import cv2
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
import time
import mediapipe as mp
import os

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

        # Hairstyle suggestions
        self.hairstyle_suggestions = hairstyle_suggestions

        # Load and set bg image
        self.background_image = Image.open(background_image_path)
        self.background_photo = ImageTk.PhotoImage(self.background_image)

        # Reference images and averages
        self.reference_image_folder = reference_image_folder
        self.face_shape_averages = self.compute_averages()

        # Start button to initiate detection
        self.start_button = tk.Button(self.window, text="Start Detector", command=self.start_webcam, height=5, width=125)
        self.start_button.pack(pady=10, padx=10)

        # Print button
        self.print_button = tk.Button(self.window, text="Print Image", command=self.print_image, height=5, width=125)
        self.print_button.pack(pady=10, padx=10)

        # Labels for suggestions and calculations
        self.suggestion_label = tk.Label(self.window, text="", font=("Helvetica", 14), fg="black")
        self.suggestion_label.pack(pady=10)
        self.calculation_label = tk.Label(self.window, text="", font=("Helvetica", 10), fg="black")
        self.calculation_label.pack(pady=5)

        # Canvas for webcam
        self.canvas = tk.Canvas(self.window, width=self.background_image.width, height=self.background_image.height)
        self.canvas.pack()


        #computes averages of landmarks in images folder
    def compute_averages(self):
        face_shape_averages = {}

        for shape in os.listdir(self.reference_image_folder):
            folder_path = os.path.join(self.reference_image_folder, shape)
            landmarks_list = []

            for image_file in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_file)
                landmarks = self.compute_landmarks(image_path)
                if landmarks is not None:
                    landmarks_list.append(landmarks)

            if landmarks_list:
                face_shape_averages[shape] = np.mean(landmarks_list, axis=0)

        return face_shape_averages


    #averages the points of the landmarks in images folder
    def compute_landmarks(self, image_path):
        image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            return np.array([[lm.x, lm.y] for lm in results.multi_face_landmarks[0].landmark])
        return None



    def start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Cannot open webcam")
            return
        self.start_time = time.time()
        self.update_frame()



    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.latest_frame = rgb_frame  # Store the last frame

            # Display frame on canvas
            img = Image.fromarray(rgb_frame)
            photo = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.background_photo)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.photo = photo  # Keep reference to avoid garbage collection

        if time.time() - self.start_time >= 5:
            self.process_last_frame()
        else:
            self.window.after(10, self.update_frame)


    #main logic, gets avg of landmarks in imgs folder then compares with current landmark result from webcam
    def process_last_frame(self):
        if self.latest_frame is not None:
            rgb_frame = cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = np.array([[lm.x, lm.y] for lm in face_landmarks.landmark])

                    # Find closest face shape based on average landmarks
                    closest_shape, min_distance = None, float('inf')
                    for shape, avg_landmarks in self.face_shape_averages.items():
                        distance = np.linalg.norm(landmarks - avg_landmarks)
                        if distance < min_distance:
                            min_distance = distance
                            closest_shape = shape

                # Display closest shape and suggestions
                self.suggestion_label.config(
                    text=f"Face Shape: {closest_shape}\nSuggested Hairstyles: {', '.join(self.hairstyle_suggestions.get(closest_shape, []))}"
                )

                

                # Draw face landmarks for visualization
                img_pil = Image.fromarray(self.latest_frame)
                draw = ImageDraw.Draw(img_pil)
                for landmark in landmarks:
                    # Since landmark is a 2D array, landmark[0] = x, landmark[1] = y
                    x = int(landmark[0] * img_pil.width)
                    y = int(landmark[1] * img_pil.height)
                    draw.ellipse((x-2, y-2, x+2, y+2), fill="red")

                self.latest_frame = np.array(img_pil)




    #prints image and result + points
    def print_image(self):
        if self.latest_frame is not None:
            pil_image = Image.fromarray(self.latest_frame)
            draw = ImageDraw.Draw(pil_image)
        
            label_text = self.suggestion_label.cget("text")
        
            # Check if text contains ": " 
            if ": " in label_text:
                face_shape = label_text.split(": ")[1].split("\n")[0]
                hairstyles = label_text.split(": ")[2] if ":" in label_text else "No suggestions"
            else:
                face_shape = "Unknown"
                hairstyles = "No suggestions"
        
            text_to_draw = f"Face Shape: {face_shape}\nSuggested Hairstyles: {hairstyles}"

            font = ImageFont.load_default()
            draw.text((10, 10), text_to_draw, fill="green", font=font)

            pil_image.show()



    def __del__(self):
        if self.cap is not None:
            self.cap.release()



# Hairstyle suggestions
hairstyle_suggestions = {
    "diamond": ["Style1", "Style2", "Style3"],
    "heart": ["Style1", "Style2", "Style3"],
    "oblong": ["Style1", "Style2", "Style3"],
    "oval": ["Style1", "Style2", "Style3"],
    "round": ["Style1", "Style2", "Style3"],
    "square": ["Style1", "Style2", "Style3"]
}

background_image_path = os.getcwd() + "/background2.jpg"

root = tk.Tk()
root.geometry("1100x800")
app = WebCamApp(root, os.getcwd() + "/reference_images", hairstyle_suggestions, background_image_path)
root.mainloop()
