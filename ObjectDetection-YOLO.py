import cv2
import torch
import numpy as np

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load image
img = cv2.imread('download.jpg')

# Run object detection
results = model(img)

# Extract detected objects and their confidences
objects = results.xyxyn[0][:, :-1].numpy()
confidences = results.xyxyn[0][:, -1].numpy()

# Define a threshold for object confidence
threshold = 0.5

# Filter objects by confidence
objects = objects[confidences > threshold]

# Count objects
object_count = len(objects)

# Display result
print(f"Detected {object_count} objects.")