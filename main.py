from fastapi import FastAPI, File, UploadFile
from pathlib import Path
import face_recognition
import dlib
import cv2
import os
import numpy as np

app = FastAPI()

# Directory to store uploaded files
UPLOAD_DIR = "./uploads"

# Create the upload directory if it doesn't exist
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

# Load the face detection model from dlib
face_detector = dlib.get_frontal_face_detector()

# Load the facial landmarks predictor from dlib
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to detect faces in an image using dlib
def detect_faces(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    face_encodings = []
    for face in faces:
        landmarks = predictor(gray, face)
        top, right, bottom, left = face.top(), face.right(), face.bottom(), face.left()
        face_encoding = face_recognition.face_encodings(image, [(top, right, bottom, left)])
        if face_encoding:
            face_encodings.append(face_encoding[0])
    return face_encodings


# Function to perform face recognition and calculate accuracy percentage
def perform_face_recognition(video_frames, aadhar_face_encodings, pan_face_encodings):
    video_face_encodings = []
    for frame in video_frames:
        # Detect faces in the frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        video_face_encodings.extend(face_encodings)

    # Convert face encodings to NumPy arrays
    aadhar_face_encodings_np = np.array(aadhar_face_encodings)
    pan_face_encodings_np = np.array(pan_face_encodings)
    video_face_encodings_np = np.array(video_face_encodings)

    # Calculate face distances
    aadhar_distances = face_recognition.face_distance(video_face_encodings_np, aadhar_face_encodings_np)
    pan_distances = face_recognition.face_distance(video_face_encodings_np, pan_face_encodings_np)

    # Calculate accuracy percentages based on the minimum face distances
    aadhar_accuracy = (1 - np.min(aadhar_distances)) * 100
    pan_accuracy = (1 - np.min(pan_distances)) * 100

    return aadhar_accuracy, pan_accuracy

# FastAPI endpoint to upload files and perform face recognition
@app.post("/process_user_data/{user_id}")
async def process_user_data(user_id: int, video_file: UploadFile = File(...), aadhar_file: UploadFile = File(...), pan_file: UploadFile = File(...)):
    # Save uploaded files to the upload directory
    video_path = os.path.join(UPLOAD_DIR, video_file.filename)
    aadhar_path = os.path.join(UPLOAD_DIR, aadhar_file.filename)
    pan_path = os.path.join(UPLOAD_DIR, pan_file.filename)
    
    with open(video_path, "wb") as video_file_content:
        video_file_content.write(video_file.file.read())
    
    with open(aadhar_path, "wb") as aadhar_file_content:
        aadhar_file_content.write(aadhar_file.file.read())
    
    with open(pan_path, "wb") as pan_file_content:
        pan_file_content.write(pan_file.file.read())

    # Detect faces in Aadhar and Pan card images
    aadhar_face_encodings = detect_faces(aadhar_path)
    pan_face_encodings = detect_faces(pan_path)

    # Load video frames using OpenCV
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)

    # Calculate accuracy percentages based on the face recognition results
    aadhar_accuracy, pan_accuracy = perform_face_recognition(frames, aadhar_face_encodings, pan_face_encodings)

    # Return accuracy percentages in the response
    return {"aadhar_accuracy": aadhar_accuracy, "pan_accuracy": pan_accuracy}



