from pdf2image import convert_from_path
import os

def pdf_to_images(pdf_path, output_dir="output", dpi=300):
    os.makedirs(output_dir, exist_ok=True)
    images = convert_from_path(pdf_path, dpi=dpi)
    saved_paths = []
    for i, img in enumerate(images):
        path = os.path.join(output_dir, f"page_{i+1}.png")
        img.save(path, "PNG")
        saved_paths.append(path)
        print(f"Saved: {path} ({img.size[0]}x{img.size[1]})")
    return saved_paths

if __name__ == "__main__":
    pages = pdf_to_images("sample_document.pdf")
    print(f"\nConverted {len(pages)} pages")
