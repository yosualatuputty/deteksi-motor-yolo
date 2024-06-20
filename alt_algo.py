import os
import cv2
import logging
import shutil
from datetime import datetime
from ultralytics import YOLO



def export_individual_motorcycles(image_path, model, output_dir):
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
         
    # Prepare output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate unique log filename based on timestamp
    log_filename = datetime.now().strftime('detection_log_%Y%m%d_%H%M%S.txt')
    log_path = os.path.join(output_dir, log_filename)

    # Set up logging
    logging.basicConfig(filename=log_path, level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

    # Load the image
    image = cv2.imread(image_path)

    # Perform detection
    results = model(image)

    # Define your class ids
    MOTORCYCLE_CLASS_ID = 1
    HELMET_CLASS_ID = 0
    PERSON_CLASS_ID = 2
    MIRROR_CLASS_ID = 3

    motorcycle_idx = 0  # Initialize motorcycle index

    # Process each detected object
    for result in results:
        num_helmets = 0
        num_persons = 0
        num_mirrors = 0

        for box in result.boxes:
            class_id = box.cls[0].int().numpy()

            if class_id == MOTORCYCLE_CLASS_ID:  # Motorcycle class ID
                x1, y1, x2, y2 = box.xyxy[0].int().numpy()
                confidence = box.conf[0].item()

                # Crop the motorcycle region
                motorcycle_img = image[y1:y2, x1:x2].copy()

                # Log motorcycle details
                logging.info(f"Detected Motorcycle {motorcycle_idx}:")
                logging.info(f"  Bounding Box: ({x1}, {y1}), ({x2}, {y2})")
                logging.info(f"  Confidence: {confidence:.2f}")

                # Draw bounding boxes and labels for attributes
                num_persons = 0
                num_mirrors = 0
                num_helmets = 0
                for attr_box in result.boxes:
                    attr_class_id = attr_box.cls[0].int().numpy()
                    attr_confidence = attr_box.conf[0].item()
                    attr_x1, attr_y1, attr_x2, attr_y2 = attr_box.xyxy[0].int().numpy()

                    # Check if the attribute is within the motorcycle bounding box
                    if attr_class_id in [HELMET_CLASS_ID, PERSON_CLASS_ID, MIRROR_CLASS_ID]:
                        if x1 <= attr_x1 <= x2 and y1 <= attr_y1 <= y2:
                            color = (0, 0, 0)
                            label = ""
                            attr_name = ""

                            if attr_class_id == HELMET_CLASS_ID:
                                color = (0, 0, 255)
                                label = f"Helmet {attr_confidence:.2f}"
                                attr_name = "Helmet"
                                num_helmets += 1
                            elif attr_class_id == PERSON_CLASS_ID:
                                color = (255, 0, 0)
                                label = f"Person {attr_confidence:.2f}"
                                attr_name = "Person"
                                num_persons += 1
                            elif attr_class_id == MIRROR_CLASS_ID:
                                color = (255, 255, 0)
                                label = f"Mirror {attr_confidence:.2f}"
                                attr_name = "Mirror"
                                num_mirrors += 1
                            
                            # Log attribute details
                            logging.info(f"  Detected {attr_name}:")
                            logging.info(f"    Bounding Box: ({attr_x1}, {attr_y1}), ({attr_x2}, {attr_y2})")
                            logging.info(f"    Confidence: {attr_confidence:.2f}")

                            # Draw attribute bounding box and label on cropped motorcycle image
                            cv2.rectangle(motorcycle_img, (attr_x1 - x1, attr_y1 - y1), (attr_x2 - x1, attr_y2 - y1), color, 2)
                            cv2.putText(motorcycle_img, label, (attr_x1 - x1, attr_y1 - y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Draw bounding box and label on the original image (optional)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Motorcycle {confidence:.2f}"
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Save the cropped motorcycle image with attributes
                cv2.imwrite(os.path.join(output_dir, f"motorcycle_{motorcycle_idx}.jpg"), motorcycle_img)
                motorcycle_idx += 1  # Increment motorcycle index
                
                # Log totals for each motorcycle
                logging.info(f"  Total Helmets: {num_helmets}")
                logging.info(f"  Total Persons: {num_persons}")
                logging.info(f"  Total Mirrors: {num_mirrors}")
                logging.info("")  # Add a line break after each motorcycle entry for better readability

    # Save the image with bounding boxes and labels (optional)
    cv2.imwrite(os.path.join(output_dir, "result_with_boxes.jpg"), image)

def main():
    # Initialize the YOLO model
    model = YOLO("deteksi-motor-l.pt")

    # Path to the image
    image_path = "C:\\Users\\YOSUA N LATUPUTTY\\deteksi-motor\\images\\motor0322_png.rf.66ed0519ca89110a494f16874faa78d6.jpg"
    output_dir = "C:\\Users\\YOSUA N LATUPUTTY\\deteksi-motor\\results\\" + datetime.now().strftime('result_%Y%m%d_%H%M%S')

    # Export individual motorcycles with attributes and bounding boxes
    export_individual_motorcycles(image_path, model, output_dir)

if __name__ == "__main__":
    main()
