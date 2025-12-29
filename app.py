import streamlit as st
import numpy as np
import pandas as pd
import cv2
from scipy.signal import spectrogram
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("eeg_eye_state_image_cnn.h5")

st.title("🧠 EEG Eye State Detection")
st.write("Upload EEG data to predict Eye State")

# Upload CSV
uploaded_file = st.file_uploader("Upload EEG CSV file", type=["csv"])

def eeg_to_image(signal):
    f, t, Sxx = spectrogram(signal, fs=128)
    Sxx = np.log(Sxx + 1e-10)
    Sxx = cv2.resize(Sxx, (128,128))
    return Sxx.reshape(1,128,128,1)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview", df.head())

    X = df.drop('eyeDetection', axis=1).values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    sample = X[0]
    img = eeg_to_image(sample)

    prediction = model.predict(img)
    label = np.argmax(prediction)

    if label == 0:
        st.success("👀 Eye State: OPEN")
    else:
        st.error("😴 Eye State: CLOSED")
