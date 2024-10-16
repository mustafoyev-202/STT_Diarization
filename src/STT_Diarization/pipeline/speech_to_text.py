import time
import assemblyai as aai


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
