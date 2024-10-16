import streamlit as st
import librosa
import numpy as np
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import google.generativeai as genai
from dotenv import load_dotenv
import os
import tempfile
import time
import assemblyai as aai

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")


# Function to perform speech-to-text using AssemblyAI with speaker labeling
def assemblyai_speech_to_text(audio_file_path):
    transcriber = aai.Transcriber()

    # Configure the transcription to enable speaker labels
    config = aai.TranscriptionConfig(speaker_labels=True)

    # Start transcription with the provided config
    transcript = transcriber.transcribe(audio_file_path, config=config)

    # Wait for the transcription to complete
    while transcript.status not in ["completed", "failed"]:
        time.sleep(5)  # Wait for 5 seconds before checking again
        transcript = transcriber.get_transcript(transcript.id)

    # Check if the transcription was completed successfully
    if transcript.status == "completed":
        transcribed_text = ""

        # Format speaker-specific utterances with bullet points and spacing
        if transcript.utterances:
            transcribed_text = "\n\n".join([f"• Speaker {utt.speaker}: {utt.text}" for utt in transcript.utterances])
        else:
            transcribed_text = transcript.text

        return transcribed_text
    else:
        return "Transcription failed."


# Use Gemini to summarize text
def generate_gemini_content(text, prompt):
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    # Advanced prompt for summarization and analysis
    full_prompt = (
        f"{prompt}\n"
        "Summarize the text in bullet points, highlighting the key points. "
        "Also, determine if the tone of the text is positive or negative."
    )
    response = model.generate_content(full_prompt + text)
    return response.text


# Custom audio processor class for microphone input
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.sr = 16000
        self.audio_data = np.zeros((0,), dtype=np.float32)

    def recv(self, frame):
        audio_data = np.frombuffer(frame.to_ndarray().flatten(), dtype=np.float32)
        self.audio_data = np.concatenate((self.audio_data, audio_data))
        return frame

    def get_audio_data(self):
        return self.audio_data


# Streamlit UI
st.title("English Speech-to-Text and Summarization App")

# Choose input method: file upload or microphone
option = st.selectbox("Choose input method:", ("Upload audio file", "Use microphone"))

# Upload audio file option
if option == "Upload audio file":
    audio_file = st.file_uploader("Upload your audio file", type=["wav", "mp3", "m4a"])
    if audio_file is not None:
        st.audio(audio_file, format="audio/wav")
        st.write("Converting speech to text...")

        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_file.read())
            temp_audio_path = temp_audio.name

        # Perform speech-to-text conversion using AssemblyAI
        transcribed_text = assemblyai_speech_to_text(temp_audio_path)
        st.write("Transcribed Text:", transcribed_text)

        # Perform summarization with Gemini
        st.write("Summarizing text...")
        prompt = "Please analyze the following text and summarize it:"
        summary = generate_gemini_content(transcribed_text, prompt)
        st.write("Summary:", summary)

        # Remove the temporary file
        os.remove(temp_audio_path)

# Microphone input option
elif option == "Use microphone":
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDRECV,
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
    )

    if webrtc_ctx.audio_processor:
        audio_data = webrtc_ctx.audio_processor.get_audio_data()
        if len(audio_data) > 0:
            st.write("Converting speech to text...")

            # Save audio data to a temporary WAV file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                librosa.output.write_wav(temp_audio.name, audio_data, sr=16000)
                temp_audio_path = temp_audio.name

            # Perform speech-to-text conversion using AssemblyAI
            transcribed_text = assemblyai_speech_to_text(temp_audio_path)
            st.write("Transcribed Text:", transcribed_text)

            # Perform summarization with Gemini
            st.write("Summarizing text...")
            prompt = "Please analyze the following text and summarize it:"
            summary = generate_gemini_content(transcribed_text, prompt)
            st.write("Summary:", summary)

            # Remove the temporary file
            os.remove(temp_audio_path)

# Option to clear cache if needed
if st.button("Clear Cache"):
    st.cache_resource.clear()
