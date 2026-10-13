import streamlit as st
import random as rd
import time


st.title("Deepfake Detection System")

st.subheader("Upload media for analysis")
uploaded_file = st.file_uploader(
    "Upload media for analysis",
    type=["mp4", "mov", "wav", "mp3", "jpg", "png"],
    label_visibility="collapsed"
)


if uploaded_file is not None:
    st.write("Selected file:", uploaded_file.name)
    st.write("File size:", uploaded_file.size)

    #simulates the analysis of the media file
    st.write("Analyzing the media, please wait...")
    wait_time = rd.randint(1, 5)
    time.sleep(wait_time)
    
    output = rd.randint(0,2)

    #simulates the different ouputs of the model
    if output == 0:
        st.write("### :green[The media is likely to be real.]")
    elif output == 1:
        st.write("### :red[The media is likely to be fake.]")
    else:
        st.write("### :orange[The media is inconclusive.]")

