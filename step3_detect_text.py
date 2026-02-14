from craft_text_detector import Craft
import cv2
import os

def detect_text_regions(image_path, output_dir="output/detections"):
    os.makedirs(output_dir, exist_ok=True)
    craft = Craft(output_dir=output_dir, crop_type="box", cuda=False, text_threshold=0.7, link_threshold=0.4, low_text=0.4)
    prediction = craft.detect_text(image_path)
    print(f"Found {len(prediction['boxes'])} text regions")
    img = cv2.imread(image_path)
    for box in prediction["boxes"]:
        box = box.astype(int)
        cv2.polylines(img, [box], True, (0, 255, 0), 2)
    viz_path = os.path.join(output_dir, "detected.png")
    cv2.imwrite(viz_path, img)
    craft.unload_craftnet_model()
    craft.unload_refinenet_model()
    return prediction["boxes"]

def crop_text_regions(image_path, boxes, output_dir="output/crops"):
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(image_path)
    cropped_paths = []
    for i, box in enumerate(boxes):
        box = box.astype(int)
        x_min = max(0, min(box[:, 0]))
        y_min = max(0, min(box[:, 1]))
        x_max = min(img.shape[1], max(box[:, 0]))
        y_max = min(img.shape[0], max(box[:, 1]))
        pad = 5
        crop = img[max(0, y_min-pad):y_max+pad, max(0, x_min-pad):x_max+pad]
        if crop.size > 0:
            crop_path = os.path.join(output_dir, f"word_{i:04d}.png")
            cv2.imwrite(crop_path, crop)
            cropped_paths.append(crop_path)
    cropped_paths.sort()
    return cropped_paths
