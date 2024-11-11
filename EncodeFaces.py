import face_recognition
import pickle
import kagglehub
import os
import cv2
data_set = {"names": [], "encodings":[]}
path = kagglehub.dataset_download("niten19/face-shape-dataset")
# Iterate through each subfolder in the reference image folder
for face_shape in os.listdir(path+"\FaceShape Dataset\\testing_set"):
    for image_file in os.listdir(path+"\FaceShape Dataset\\testing_set\\"+face_shape):
        image_path = os.path.join(path+"\FaceShape Dataset\\testing_set\\"+face_shape, image_file)
        image = face_recognition.load_image_file(image_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encoding = face_recognition.face_encodings(rgb)
        if encoding:
            print(image_path + ", encoded")
            data_set["encodings"].append(encoding[0])
            data_set["names"].append(face_shape)

with open('dataset_face_shape.dat', 'wb') as f:
    pickle.dump(data_set, f)
    print("pickle file stored")