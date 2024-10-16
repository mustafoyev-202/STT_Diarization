from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import shutil
from main_utils import assemblyai_speech_to_text, generate_gemini_content

app = FastAPI()

# Setup static and template directories
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/uploadfile/")
async def upload_file(request: Request, file: UploadFile = File(...)):
    # Check file type
    if file.content_type not in ["audio/wav", "audio/mp3", "audio/m4a"]:
        return templates.TemplateResponse("index.html", {"request": request, "error": "Unsupported file type."})

    # Save uploaded file
    temp_file_path = f"temp_files/{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Perform speech-to-text
        transcribed_text = assemblyai_speech_to_text(temp_file_path)

        # Summarization
        summary_prompt = "Please analyze the following text and summarize it:"
        summary = generate_gemini_content(transcribed_text, summary_prompt)

        # Clean up temp file
        os.remove(temp_file_path)

        return templates.TemplateResponse("index.html", {
            "request": request,
            "transcribed_text": transcribed_text,
            "summary": summary
        })

    except Exception as e:
        # If something goes wrong
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Error occurred: {str(e)}"
        })

# If running locally, start the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

