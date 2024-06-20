import cv2
import numpy as np
from ultralytics import YOLO

# buat deteksi satu gambar aja

# Load the YOLO model
model = YOLO("deteksi-motor-n.pt")


# Read the image
image_path = "D:\\deteksi-motor-yolo\\images\\motor0013_png.rf.c316e15437de72d4dd11b53ca8079991.jpg"



image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print(f"Error: Unable to load image {image_path}")
    exit()

# Use the model to detect objects in the image
results = model(image)

# Create a named window with the ability to resize
cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)

# Set the desired window size (e.g., 800x600)
cv2.resizeWindow("YOLO Detection", 640, 640)

# Iterate over detected objects
for result in results:
    for box in result.boxes:
        # Extract coordinates, confidence score, and class ID
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        score = box.conf[0]  # Confidence score
        class_id = int(box.cls[0])  # Class ID
        
        # Get the label for the class ID
        label = f'{model.names[class_id]} {score:.2f}'
        
        # Draw the bounding box and label on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the image with detections
cv2.imshow("YOLO Detection", image)

# Wait until a key is pressed and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
