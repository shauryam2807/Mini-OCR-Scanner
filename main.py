from step1_pdf_to_image import pdf_to_images
from step2_preprocess import preprocess_image
from step3_detect_text import detect_text_regions, crop_text_regions
from step4_recognize_text import TrOCRRecognizer
from step5_llm_correction import setup_gemini, correct_ocr_with_llm
import os

def run_ocr_pipeline(pdf_path, gemini_api_key=None):
    print("=" * 60)
    print("MINI OCR SCANNER")
    print("=" * 60)

    # Step 1: PDF to images
    print("\nStep 1: Converting PDF to images...")
    page_images = pdf_to_images(pdf_path)

    # Load TrOCR once
    print("\nLoading TrOCR model...")
    recognizer = TrOCRRecognizer()

    all_page_texts = []

    for page_num, page_img in enumerate(page_images, 1):
        print(f"\nProcessing Page {page_num}/{len(page_images)}")

        # Step 2: Clean the image
        clean_path, _ = preprocess_image(page_img)

        # Step 3: Find text regions
        boxes = detect_text_regions(clean_path,
            output_dir=f"output/detections/page_{page_num}")
        if len(boxes) == 0:
            continue
        crops = crop_text_regions(clean_path, boxes,
            output_dir=f"output/crops/page_{page_num}")

        # Step 4: Read the text
        texts = recognizer.recognize_batch(crops)
        page_text = " ".join(texts)
        all_page_texts.append(page_text)

    full_text = "\n\n".join(all_page_texts)

    os.makedirs("output", exist_ok=True)
    with open("output/raw_ocr.txt", "w", encoding="utf-8") as f:
        f.write(full_text)

    # Step 5: LLM correction (optional)
    if gemini_api_key:
        print("\nStep 5: LLM correction...")
        model = setup_gemini(gemini_api_key)
        corrected = correct_ocr_with_llm(model, full_text)
        with open("output/corrected_text.txt", "w", encoding="utf-8") as f:
            f.write(corrected)

    print("\nDONE! Check the output/ folder.")
    return full_text

if __name__ == "__main__":
    GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
    run_ocr_pipeline("sample_document.pdf", gemini_api_key=GEMINI_KEY)
