import assemblyai as aai
import google.generativeai as genai
import os
import time

# Initialize API keys if set in environment variables
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def assemblyai_speech_to_text(audio_file_path):
    transcriber = aai.Transcriber()
    config = aai.TranscriptionConfig(speaker_labels=True)
    transcript = transcriber.transcribe(audio_file_path, config=config)

    while transcript.status not in ["completed", "failed"]:
        time.sleep(5)  # Wait for 5 seconds before checking again
        transcript = transcriber.get_transcript(transcript.id)

    if transcript.status == "completed":
        if transcript.utterances:
            transcribed_text = "\n\n".join([f"• Speaker {utt.speaker}: {utt.text}" for utt in transcript.utterances])
        else:
            transcribed_text = transcript.text
        return transcribed_text
    else:
        return "Transcription failed."


def generate_gemini_content(text, prompt):
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content(prompt + text)
    return response.text
