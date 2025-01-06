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
import io
import json


load_dotenv()
key1 = os.getenv("GROQ_API_KEY")

groqclient = Groq(api_key=key1)

recognizer = Recognizer()
microphone = Microphone()

st.title('Real-Time Speech Analysis and Transcription with Groq')

if 'stop_listening' not in st.session_state:
    st.session_state.stop_listening = False
if 'transcription_result' not in st.session_state:
    st.session_state.transcription_result = ""


scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
client = gspread.authorize(creds)
sheet = client.open("Chat Analysis Results").sheet1

def append_to_google_sheet(text, sentiment_result):
    try:
        sentiment = sentiment_result.get("Sentiment", "Unknown")
        scores = sentiment_result.get("Scores", {})
        sheet.append_row([
            text,
            sentiment,
            scores.get("Positive", 0),
            scores.get("Negative", 0),
            scores.get("Neutral", 0)
        ])
        print("Data has been successfully saved!")
    except Exception as e:
        st.error(f"Error saving the results: {str(e)}")


def analyze_sentiment(text):
    try:
        sentiment_completion = groqclient.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """You are a sentiment analysis assistant. For the given text, provide the most confident sentiment (Positive, Negative, Neutral) along with confidence scores for each sentiment.
                                   Respond in the following JSON format:
                                   {
                                       "Sentiment": "Positive/Negative/Neutral",
                                       "Scores": {
                                           "Positive": 0.85,
                                           "Negative": 0.10,
                                           "Neutral": 0.05
                                       }
                                   }"""
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.5
        )
        sentiment_result = sentiment_completion.choices[0].message.content
        return json.loads(sentiment_result)
    except Exception as e:
        st.error(f"Error analyzing sentiment: {str(e)}")
        return None
    

def transcribe_audio():
    transcription_placeholder = st.empty()
    sentiment_placeholder = st.empty()

    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        st.info("Listening...")
        while not st.session_state.stop_listening:
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                audio_data = audio.get_wav_data()

                if audio_data:
                    print("Audio data received.")
                    audio_file = io.BytesIO(audio_data)
                    audio_file.name = "audio.wav"
                    print("Sending audio data for transcription...")
                    transcription = groqclient.audio.transcriptions.create(
                        model="whisper-large-v3", 
                        file=audio_file,
                        prompt="provide an accurate transcription of the audio file using punctuations and capitalization as well."
                    )
                    st.session_state.transcription_result += transcription.text + " "
                    transcription_placeholder.text(st.session_state.transcription_result)

                    sentiment_result = analyze_sentiment(transcription.text)
                    if sentiment_result:
                        sentiment_text = sentiment_result.get("Sentiment", "Unknown")
                        scores = sentiment_result.get("Scores", {})
                        sentiment_display = (
                            f"### Sentiment: {sentiment_text}\n\n"
                            f"**Confidence Scores:**\n"
                            f"- Positive: {scores.get('Positive', 0):.2f}\n"
                            f"- Negative: {scores.get('Negative', 0):.2f}\n"
                            f"- Neutral: {scores.get('Neutral', 0):.2f}"
                        )
                        sentiment_placeholder.markdown(sentiment_display)

                        append_to_google_sheet(transcription.text, sentiment_result)
            except Exception as e:
                st.error(f"Error: {str(e)}")
                break

if not st.session_state.stop_listening:
    if st.button("Start Listening"):
        st.session_state.stop_listening = False
        transcription_placeholder = st.empty()
        sentiment_placeholder = st.empty()
        transcribe_audio()
else:
    if st.button("Stop Listening"):
        st.session_state.stop_listening = True
        st.write("Listening stopped.")

if st.session_state.transcription_result:
    print("Transcription received.")
    st.write(st.session_state.transcription_result)