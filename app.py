import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

st.set_page_config(page_title="Eye State Detection", layout="centered")

st.title("👀 Eye Open / Close Detection")
st.write("Upload a face image to detect eye state")

# Initialize MediaPipe FaceMesh safely
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

uploaded_file = st.file_uploader(
    "Upload an image (jpg / png)", type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    results = face_mesh.process(img)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = img.shape

        left_eye = np.array([
            [int(face_landmarks.landmark[i].x * w),
             int(face_landmarks.landmark[i].y * h)]
            for i in LEFT_EYE
        ])

        right_eye = np.array([
            [int(face_landmarks.landmark[i].x * w),
             int(face_landmarks.landmark[i].y * h)]
            for i in RIGHT_EYE
        ])

        ear = (eye_aspect_ratio(left_eye) +
               eye_aspect_ratio(right_eye)) / 2

        THRESHOLD = 0.25

        if ear < THRESHOLD:
            st.error("😴 Eyes Closed")
        else:
            st.success("👀 Eyes Open")

    else:
        st.warning("No face detected. Please upload a clear face image.")
