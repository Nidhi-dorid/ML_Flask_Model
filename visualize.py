import torch
from ultralytics import YOLO
import os
import sys

# Patch for PyTorch 2.6+ if needed
original_load = torch.load
def patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

def main():
    model_path = "model/best.pt"
    image_path = "not_pothole_image.jpg"

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)

    print(f"Running detection on {image_path}...")
    # save=True will save the result to runs/detect/predictX/
    results = model(image_path, save=True)

    # Find where it was saved
    save_dir = results[0].save_dir
    saved_image = os.path.join(save_dir, os.path.basename(image_path))

    print("-" * 40)
    print("DETECTION COMPLETE")
    print(f"Result saved to: {saved_image}")
    print("-" * 40)
    
    # Optional: try to open the image if on desktop
    # os.startfile(saved_image) 

if __name__ == "__main__":
    main()
