import google.generativeai as genai


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
