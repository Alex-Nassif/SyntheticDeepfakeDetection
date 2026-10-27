import random as rd
import time
from pathlib import Path

import streamlit as st

from preprocessing import preprocess_image, ImagePreprocessingError

TARGET_CONFIDENCE = 75
IMAGE_TYPES = {".jpg", ".jpeg", ".png"}
VIDEO_TYPES = {".mp4", ".mov"}
AUDIO_TYPES = {".wav", ".mp3"}


# TODO: replace with the real model
def predict(model_input):
    time.sleep(rd.randint(1, 3))  # simulates processing time
    return {"label": rd.choice(["real", "fake"]), "confidence": rd.randint(60, 100)}


def show_result(result):
    if result["confidence"] < TARGET_CONFIDENCE:
        st.write("### :orange[The media is inconclusive.]")
    elif result["label"] == "real":
        st.write("### :green[The media is likely to be real.]")
    else:
        st.write("### :red[The media is likely to be fake.]")
    st.write(f"Confidence: {result['confidence']}%")


st.title("Deepfake Detection System")
st.subheader("Upload media for analysis")
uploaded_file = st.file_uploader(
    "Upload media for analysis",
    type=["mp4", "mov", "wav", "mp3", "jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded_file is not None:
    st.write(f"Selected file: {uploaded_file.name}")
    st.write(f"File size (bytes): {uploaded_file.size}")

    ext = Path(uploaded_file.name).suffix.lower()

    if ext in IMAGE_TYPES:
        st.image(uploaded_file, width="stretch")
        uploaded_file.seek(0)  # st.image read the file, so rewind it
        try:
            model_input, info = preprocess_image(uploaded_file)
        except ImagePreprocessingError as e:
            st.error(f"Error processing image: {e}")
            st.stop()
    elif ext in VIDEO_TYPES:
        st.video(uploaded_file)
        st.info("Video analysis is not implemented yet.")
        st.stop()
    elif ext in AUDIO_TYPES:
        st.audio(uploaded_file)
        st.info("Audio analysis is not implemented yet.")
        st.stop()

    with st.spinner("Analyzing the media, please wait..."):
        result = predict(model_input)
    st.session_state["result"] = result
    show_result(result)