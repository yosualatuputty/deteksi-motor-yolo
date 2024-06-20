import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("deteksi-motor-n.pt")

# Initialize the video
cap = cv2.VideoCapture("C:\\Users\\ASUS\\Videos\\CCTV Jl. MT. Haryono Arah Barat_Kota Malang_21 31_'9.mkv")

# Create a named window with the ability to resize
cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)

# Set the desired window size (e.g., 800x600)
cv2.resizeWindow("YOLO Detection", 1080, 720)

# Define colors for different classes
colors = {
    'motorcycle': (0, 255, 255),  # Yellow
    'person': (0, 0, 255),         # Red
    'spion': (255, 0, 0),        # Blue
    'helm': (0, 255, 0)       # Orange
}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Use the model to detect objects in the frame
    results = model(frame)
    
    # Iterate over detected objects
    for result in results:
        for box in result.boxes:
            # Extract coordinates, confidence score, and class ID
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            score = box.conf[0]  # Confidence score
            class_id = int(box.cls[0])  # Class ID
            
            # Get the label for the class ID
            label = f'{model.names[class_id]} {score:.2f}'
            
            # Set color based on class
            color = colors.get(model.names[class_id], (0, 255, 0))  # Default to green if class not found
            
            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Display the frame with detections
    cv2.imshow("YOLO Detection", frame)
    
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()

