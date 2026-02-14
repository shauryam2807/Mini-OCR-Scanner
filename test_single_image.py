"""Quick test: run OCR pipeline on a single image (skip PDF conversion)"""
from step2_preprocess import preprocess_image
from step3_detect_text import detect_text_regions, crop_text_regions
from step4_recognize_text import TrOCRRecognizer
from step5_llm_correction import setup_gemini, correct_ocr_with_llm
import os

IMAGE_PATH = "book_header.png"
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")

print("="*50)
print("SINGLE IMAGE OCR TEST")
print("="*50)

# Step 2: Preprocess
print("\nStep 2: Preprocessing image...")
clean_path, _ = preprocess_image(IMAGE_PATH)
print(f"Cleaned image saved to: {clean_path}")

# Step 3: Detect text regions
print("\nStep 3: Detecting text regions...")
boxes = detect_text_regions(clean_path, output_dir="output/test_detections")
print(f"Detected {len(boxes)} text regions")

if len(boxes) == 0:
    print("No text detected!")
    exit()

# Step 3b: Crop text regions
crops = crop_text_regions(clean_path, boxes, output_dir="output/test_crops")
print(f"Cropped {len(crops)} text regions")

# Step 4: Recognize text
print("\nStep 4: Recognizing text with TrOCR...")
recognizer = TrOCRRecognizer()
texts = recognizer.recognize_batch(crops)
raw_text = " ".join(texts)

print(f"\n{'='*50}")
print("RAW OCR OUTPUT:")
print("="*50)
print(raw_text)

os.makedirs("output", exist_ok=True)
with open("output/test_raw_ocr.txt", "w", encoding="utf-8") as f:
    f.write(raw_text)
print(f"\nRaw text saved to: output/test_raw_ocr.txt")

# Step 5: LLM correction
print("\nStep 5: Correcting with Gemini...")
model = setup_gemini(GEMINI_KEY)
corrected = correct_ocr_with_llm(model, raw_text)
print(f"\n{'='*50}")
print("CORRECTED TEXT:")
print("="*50)
print(corrected)
with open("output/test_corrected.txt", "w", encoding="utf-8") as f:
    f.write(corrected)

print("\n✅ Done!")
