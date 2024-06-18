import os
import cv2
from ultralytics import YOLO

# algoritma alternatif ga perlu export foto motor individual (belom selesai)
def export_individual_motorcycles(image_path, model, output_dir):
    # Load the image
    image = cv2.imread(image_path)

    # Perform detection
    results = model(image)

    # Prepare output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each detected motorcycle
    for idx, result in enumerate(results):
        motorcycle_attrs = []

        # Draw bounding boxes and labels
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().numpy()
            class_id = box.cls[0].int().numpy()
            confidence = box.conf[0].item()

            # Define your class ids
            MOTORCYCLE_CLASS_ID = 1
            HELMET_CLASS_ID = 0
            PERSON_CLASS_ID = 3
            MIRROR_CLASS_ID = 2

            # Check class IDs and draw bounding boxes and labels
            if class_id == MOTORCYCLE_CLASS_ID:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Motorcycle {confidence:.2f}"
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Crop the motorcycle region
                motorcycle_img = image[y1:y2, x1:x2].copy()

                # Save the cropped motorcycle image with attributes
                cv2.imwrite(os.path.join(output_dir, f"motorcycle_{idx}.jpg"), motorcycle_img)

            elif class_id == HELMET_CLASS_ID:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"Helmet {confidence:.2f}"
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            elif class_id == PERSON_CLASS_ID:
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f"Person {confidence:.2f}"
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            elif class_id == MIRROR_CLASS_ID:
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)
                label = f"Mirror {confidence:.2f}"
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Save the image with bounding boxes and labels (optional)
    cv2.imwrite(os.path.join(output_dir, "result_with_boxes.jpg"), image)

def main():
    # Initialize the YOLO model
    model = YOLO("best.pt")

    # Path to the image
    image_path = "C:\\Users\\YOSUA N LATUPUTTY\\deteksi-motor\\images\\motor0006_png.rf.e510c12cc4196baa59bc98ebfccda1a1.jpg"
    output_dir = "C:\\Users\\YOSUA N LATUPUTTY\\deteksi-motor\\results"

    # Export individual motorcycles with attributes and bounding boxes
    export_individual_motorcycles(image_path, model, output_dir)

if __name__ == "__main__":
    main()
