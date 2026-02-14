from transformers import TrOCRProcessor,VisionEncoderDecoderModel
from PIL import Image
import torch
import os


class TrOCRRecognizer:
    def __init__(self, model_name="microsoft/trocr-base-printed"):
        print(f"Loading model: {model_name}")
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")
    def recognize_single(self, image_path):
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text
    def recognize_batch(self, image_paths, batch_size=4):
        all_texts = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            pixel_values = self.processor(
                images=images, return_tensors="pt"
            ).pixel_values.to(self.device)
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values)
            texts = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            all_texts.extend(texts)
            print(f"Recognized {min(i+batch_size, len(image_paths))}/{len(image_paths)}")
        return all_texts
