import streamlit as st
import random as rd
import time

#hard coded values that will later be given by the model
confidence_interval = rd.randint(60, 100)
model_decision = rd.randint(0, 1)
TARGET_CONFIDENCE_INTERVAL = 75


st.title("Deepfake Detection System")

st.subheader("Upload media for analysis")
uploaded_file = st.file_uploader(
    "Upload media for analysis",
    type=["mp4", "mov", "wav", "mp3", "jpg", "png"],
    label_visibility="collapsed"
)

#takes in the model_decision and confidence_interval variables and displays the result of the analysis based on their values
def detection_classification():
    if  confidence_interval < TARGET_CONFIDENCE_INTERVAL:
        st.write("### :orange[The media is inconclusive.]")
    elif model_decision == 1:
        st.write("### :green[The media is likely to be real.]")
    else:
        st.write("### :red[The media is likely to be fake.]")


if uploaded_file is not None:
    st.write("Selected file:", uploaded_file.name)
    st.write("File size(bytes):", uploaded_file.size)

    
    #shows the model has started analyzing
    st.write("Analyzing the media, please wait...")

    #TODO: Remove when actual system is implemented
    #simulates wait time
    wait_time = rd.randint(1, 5)
    time.sleep(wait_time)

    #calls the detection_classification function to display the result of the analysis
    detection_classification()

    st.write("Confidence interval:", confidence_interval, "%")
    #TODO: Maybe add an explanation portion



        