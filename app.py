import streamlit as st
import requests
import threading
from speech_recognition import Recognizer, Microphone
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import groq
from groq import Groq
import os
from dotenv import load_dotenv
import threading
import io

load_dotenv()
key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=key)

recognizer = Recognizer()
microphone = Microphone()

st.title('Real-Time Speech Analysis and Transcription with Groq')   
if st.button("Start Listening"):
    with microphone as source:
        st.info("Listening...")
        audio = recognizer.listen(source)
        audio_data = audio.get_wav_data()

        if audio_data:
            st.title('Audio Transcript')
            try:
                audio_file = io.BytesIO(audio_data)
                audio_file.name = "audio.wav"
                transcription = client.audio.transcriptions.create(
                    model="distil-whisper-large-v3-en", 
                    file=audio_file,
                    prompt="provide an accurate transcription of the audio file using punctuations and capitalization as well."
                )
                st.write(transcription.text)
            except groq.NotFoundError as e:
                st.error(f"Error: {e.error['message']}")