import google.generativeai as genai

def setup_gemini(api_key):
    genai.configure(api_key=api_key)
    model=genai.GenerativeModel('gemini-2.0-flash')
    return model

def correct_ocr_with_llm(model, raw_text):
    prompt = f"""You are an expert at reading text from documents.
The following text was extracted using OCR and may contain errors.
Please:
1. Fix obvious OCR errors (e.g., 'rn' misread as 'm', '1' misread as 'l')
2. Fix broken words that should be joined
3. Fix spacing issues
4. If unsure about a word, keep the original

RAW OCR TEXT:
{raw_text}


Return ONLY the corrected text, nothing else."""
    response = model.generate_content(prompt)
    return response.text

if __name__ == "__main__":
    model = setup_gemini("YOUR_API_KEY_HERE")
    test = correct_ocr_with_llm(model, "Hel1o Wor1d th1s 1s a test")
    print(test)


