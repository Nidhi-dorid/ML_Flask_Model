import torch
from ultralytics import YOLO

# Fix for PyTorch 2.6+ compatibility
# Monkey-patch torch.load to use weights_only=False as ultralytics calls it internally
import torch
original_load = torch.load
def patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load


model = YOLO("model/best.pt")

# Run detection and show result
model(r"C:\Users\nidhi\OneDrive\Documents\Major_project\files\not_pothole_image.jpg", save=True)