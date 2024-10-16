import numpy as np
from streamlit_webrtc import AudioProcessorBase
import google.generativeai as genai
from dotenv import load_dotenv
import os
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
