import streamlit as st
import requests
import threading
from speech_recognition import Recognizer, Microphone
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from groq import Groq
import os
from dotenv import load_dotenv
import threading

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# def setup_google_sheet(sheet_name):
#     scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
#     credentials = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
#     client = gspread.authorize(credentials)
#     sheet = client.open(sheet_name).sheet1
#     return sheet

# SHEET_NAME = "Chat Analysis Results"
# sheet = setup_google_sheet(SHEET_NAME)


recognizer = Recognizer()
microphone = Microphone()

temp_audio_path = "temp_audio.wav"

if 'stop_listening' not in st.session_state:
    st.session_state.stop_listening = False
if 'transcriptions' not in st.session_state:
    st.session_state.transcriptions = []

def transcribe_audio(audio_data):
    temp_audio_path = "temp_audio.wav"
    with open(temp_audio_path, "wb") as temp_file:
        temp_file.write(audio_data)

    with open(temp_audio_path, "rb") as file:
        response = client.audio.transcriptions.create(file=file)

    os.remove(temp_audio_path)

    if response.status_code == 200:
        return response.text
    else:
        st.error(f"Error: {response.status_code}, {response.json()}")
        return None

# def analyze_emotion(audio_data):
#     url = HUME_BASE_URL
#     files = {"audio": audio_data}
#     response = requests.post(url, headers=hume_headers, files=files)
#     response.raise_for_status()
#     return response.json()

# def summarize_text(text):
#     url = f"{GROQ_BASE_URL}/summarize"
#     payload = {"text": text}
#     response = requests.post(url, headers=groq_headers, json=payload)
#     response.raise_for_status()
#     return response.json()["summary"]

# def analyze_sentiment(text):
#     url = f"{GROQ_BASE_URL}/sentiment"
#     payload = {"text": text}
#     response = requests.post(url, headers=groq_headers, json=payload)
#     response.raise_for_status()
#     return response.json()

# def store_in_google_sheet(sheet, transcription, emotion_results, sentiment, summary):
#     emotions = ", ".join([f"{e['name']} ({e['score']:.2f})" for e in emotion_results["results"]["emotions"]])
#     sentiment_label = sentiment["label"]
#     sentiment_score = sentiment["score"]
#     sheet.append_row([transcription, emotions, sentiment_label, sentiment_score, summary])

def listen_and_process():
    with microphone as source:
        st.info("Listening...")
        try:
            while not st.session_state.stop_listening:
                audio = recognizer.listen(source)
                st.write("Captured audio.")
                audio_data = audio.get_wav_data()

                transcription = transcribe_audio(audio_data)
                if transcription:
                    st.session_state.transcriptions.append(transcription)

                # emotion_results = analyze_emotion(audio_data)
                # emotions = ", ".join([f"{e['name']} ({e['score']:.2f})" for e in emotion_results["results"]["emotions"]])
                # st.write(f"**Emotions**: {emotions}")

                # sentiment = analyze_sentiment(transcription)
                # st.write(f"**Sentiment**: {sentiment['label']} (Confidence: {sentiment['score']:.2f})")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

st.title("Real-Time Audio Analysis Assistant")
st.write("Speak into the microphone.")

transcription_placeholder = st.empty()

if st.button("Start Listening"):
    st.session_state.stop_listening = False
    threading.Thread(target=listen_and_process).start()

if st.button("Stop Listening"):
    st.session_state.stop_listening = True

if st.session_state.transcriptions:
    with transcription_placeholder.container():
        st.write("### Transcriptions:")
        for i, transcription in enumerate(st.session_state.transcriptions):
            st.write(f"{i + 1}. {transcription}")