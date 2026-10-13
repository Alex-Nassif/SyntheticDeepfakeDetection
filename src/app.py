import streamlit as st


st.title("Deepfake Detection System")


uploaded_file = st.file_uploader(
    "Upload media for analysis",
    type=["mp4", "mov", "wav", "mp3", "jpg", "png"]
)


if uploaded_file is not None:
    st.write("Selected file:", uploaded_file.name)
