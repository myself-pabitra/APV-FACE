from fastapi import FastAPI, File, UploadFile, HTTPException
from pathlib import Path
import dlib
import face_recognition
import cv2
import os
import numpy as np
from sklearn.metrics import pairwise_distances

app = FastAPI()

# Directory to store uploaded files
UPLOAD_DIR = "./uploads"
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

# Load the face detection model from dlib with adjusted parameters
face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")



# Function to preprocess an image to enhance its quality

# def preprocess_image(image):

#     # Resize the image to a standard size for consistent face detection
#     # resized_image = cv2.resize(image, (640, 480))

#     # Convert the resized image to grayscale
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

#     # Apply histogram equalization to improve contrast
#     equalized_image = cv2.equalizeHist(gray_image)

#     # Convert the equalized image back to BGR format
#     equalized_bgr_image = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)
#     return equalized_bgr_image


def preprocess_image(image):
    # Convert the image to BGR format
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Adjust bilateral filter parameters for noise reduction
    filtered_image = cv2.bilateralFilter(bgr_image, d=9, sigmaColor=75, sigmaSpace=75)
    
    return filtered_image




# Function to detect faces in a preprocessed image using dlib

def detect_faces_and_get_landmarks(image):
    # Convert BGR image to RGB format
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Find face locations and landmarks in the image
    face_landmarks_list = face_recognition.face_landmarks(rgb_image)

    # Extract landmark sets for each face
    landmarks = []
    for face_landmarks in face_landmarks_list:
        # Convert the dictionary of landmarks to a list of (x, y) points
        landmark_points = [
            (x, y) for feature, points in face_landmarks.items() for (x, y) in points]
        landmarks.append(landmark_points)

    return landmarks


def compute_face_encodings(image, landmarks):
    # Convert BGR image to RGB format
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Compute face encodings for each face
    face_encodings = []
    for landmark in landmarks:
        top, right, bottom, left = (
            min(point[1] for point in landmark),
            max(point[0] for point in landmark),
            max(point[1] for point in landmark),
            min(point[0] for point in landmark)
        )
        face_encoding = face_recognition.face_encodings(
            rgb_image, [(top, right, bottom, left)])
        if face_encoding:
            face_encodings.append(face_encoding[0])

        # print(f"Number of faces detected: {len(landmarks)}")
        # print(f"Landmarks for the first face: {landmarks[0]}")
        # print(f"Face encodings for the first face: {face_encodings[0] if face_encodings else 'No encoding found'}")

    # Convert the list of face encodings to a 2D numpy array
    return np.array(face_encodings)


def perform_face_recognition(frames, aadhar_face_encodings, pan_face_encodings):

    # Initialize lists to store accuracy for Aadhar and Pan card
    aadhar_accuracies = []
    pan_accuracies = []

    # Iterate through frames and calculate distances
    for frame in frames:
        # Detect faces in the frame
        frame_landmarks = detect_faces_and_get_landmarks(frame)
        frame_face_encodings = compute_face_encodings(frame, frame_landmarks)


        print("Length frame face encoding :" , len(frame_face_encodings))
        print("Length Aadhar face encoding :" , len(aadhar_face_encodings))
        print("Length Pan face encoding :" , len(pan_face_encodings))


        # Check if the input arrays have the correct shape
        if len(frame_face_encodings) == 0 or len(aadhar_face_encodings) == 0 or len(pan_face_encodings) == 0:
            # Handle the case where there are no face encodings found
            aadhar_accuracies.append(0)
            pan_accuracies.append(0)
            continue

        # Calculate pairwise distances between frames and Aadhar face encodings
        aadhar_distances = pairwise_distances(
            frame_face_encodings, aadhar_face_encodings, metric='euclidean')
        # Calculate average distance for Aadhar card
        avg_aadhar_distance = np.mean(np.min(aadhar_distances, axis=1))
        print("avg_aadhar_distance :",avg_aadhar_distance)

        # Calculate pairwise distances between frames and Pan card face encodings
        pan_distances = pairwise_distances(
            frame_face_encodings, pan_face_encodings, metric='euclidean')
        # Calculate average distance for Pan card
        avg_pan_distance = np.mean(np.min(pan_distances, axis=1))
        print("avg_pan_distance : ", avg_pan_distance)

        # Set a threshold for face recognition accuracy (you can adjust this threshold)
        accuracy_threshold = 0.6

        # Check if the average distance is below the accuracy threshold for Aadhar card
        if avg_aadhar_distance < accuracy_threshold:
            aadhar_accuracies.append(1)  # 1 indicates recognition
        else:
            aadhar_accuracies.append(0)  # 0 indicates non-recognition

        # Check if the average distance is below the accuracy threshold for Pan card
        if avg_pan_distance < accuracy_threshold:
            pan_accuracies.append(1)  # 1 indicates recognition
        else:
            pan_accuracies.append(0)  # 0 indicates non-recognition

    # Calculate accuracy percentages for Aadhar and Pan card recognition
    aadhar_accuracy = (sum(aadhar_accuracies) / len(aadhar_accuracies)) * 100
    pan_accuracy = (sum(pan_accuracies) / len(pan_accuracies)) * 100

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

    # Preprocess Aadhar and Pan images for enhanced face detection
    aadhar_image = preprocess_image(cv2.imread(aadhar_path))
    pan_image = preprocess_image(cv2.imread(pan_path))

    # Detect faces in Aadhar and Pan card images after preprocessing

    aadhar_landmarks = detect_faces_and_get_landmarks(aadhar_image)
    pan_landmarks = detect_faces_and_get_landmarks(aadhar_image)

    aadhar_face_encodings = compute_face_encodings(
        aadhar_image, aadhar_landmarks)
    pan_face_encodings = compute_face_encodings(pan_image, pan_landmarks)


    # Load video frames using OpenCV
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_landmarks_list = []
    frame_face_encodings_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Preprocess video frames for enhanced face detection
        preprocessed_frame = preprocess_image(frame)
        # preprocessed_frame = frame
        frames.append(preprocessed_frame)

        frame_landmarks = detect_faces_and_get_landmarks(preprocessed_frame)
        frame_face_encodings = compute_face_encodings(preprocessed_frame, frame_landmarks)

        frame_landmarks_list.append(frame_landmarks)
        frame_face_encodings_list.append(frame_face_encodings)

    # Iterate through the frames and print information about detected faces
    # for frame_number, (frame_landmarks, frame_face_encodings) in enumerate(zip(frame_landmarks_list, frame_face_encodings_list)):
    #     print(f"Frame {frame_number}: Number of faces detected: {len(frame_landmarks)}")
    #     print(f"Frame {frame_number}: Landmarks for the first face: {frame_landmarks[0] if frame_landmarks else 'No landmarks found'}")
    #     print(f"Frame {frame_number}: Face encodings for the first face: {frame_face_encodings[0] if frame_face_encodings else 'No encoding found'}")

    #     if len(frame_face_encodings) > 0:
    #         print(f"Frame {frame_number}: Face encodings for the first face: {frame_face_encodings[0]}")
    #     else:
    #         print(f"Frame {frame_number}: No encoding found")

    # Calculate accuracy percentages based on the face recognition results
    aadhar_accuracy, pan_accuracy = perform_face_recognition(
        frames, aadhar_face_encodings, pan_face_encodings)

    # Return accuracy percentages in the response
    return {"aadhar_accuracy": aadhar_accuracy, "pan_accuracy": pan_accuracy}