import face_recognition
import pickle
import kagglehub
import os
all_face_encodings = {}
path = kagglehub.dataset_download("niten19/face-shape-dataset")
# Iterate through each subfolder in the reference image folder
for face_shape in os.listdir(path+"\FaceShape Dataset\\testing_set"):
    for image_file in os.listdir(path+"\FaceShape Dataset\\testing_set\\"+face_shape):
            image_path = os.path.join(path+"\FaceShape Dataset\\testing_set\\"+face_shape, image_file)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                print(image_path + ", encoded")
                all_face_encodings[face_shape] = encoding[0]
with open('dataset_face_shape.dat', 'wb') as f:
    pickle.dump(all_face_encodings, f)
    print("pickle file stored")