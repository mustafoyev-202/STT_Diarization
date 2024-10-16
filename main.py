from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import os
import shutil

# Create a FastAPI instance
app = FastAPI()

# Setup templates and static file directories
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Define API key environment variable (optional if defined in utils.py)
os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"
os.environ["ASSEMBLYAI_API_KEY"] = "YOUR_ASSEMBLYAI_API_KEY"


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Render the home page with a form to upload audio files or select input method.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/uploadfile/")
async def upload_file(request: Request, file: UploadFile = File(...)):
    """
    Endpoint to handle file uploads and process with AssemblyAI.
    """
    # Save the uploaded file temporarily
    temp_file_path = f"temp_files/{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Perform speech-to-text using the utility function
    transcribed_text = assemblyai_speech_to_text(temp_file_path)

    # Perform summarization with Gemini (using your existing utility function)
    summary_prompt = "Please analyze the following text and summarize it:"
    summary = generate_gemini_content(transcribed_text, summary_prompt)

    # Remove the temporary file after processing
    os.remove(temp_file_path)

    # Render the result page with transcription and summary
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "transcribed_text": transcribed_text,
            "summary": summary,
        },
    )

# If running locally, start the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
