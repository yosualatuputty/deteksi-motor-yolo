import os
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO

# Fungsi untuk memuat model YOLOv8
def load_model(model_path):
    return YOLO(model_path)

# Fungsi untuk mendeteksi objek dan menyimpan cropping hasil deteksi
def detect_and_crop(model, image_path, save_dir, class_id):
    image = cv2.imread(image_path)
    results = model(image)
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    cropped_images = []
    for result in results:
        boxes = result.boxes  # Access bounding boxes
        for box in boxes:
            if int(box.cls) == class_id:  # Class ID for motor
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to int
                score = box.conf[0]  # Confidence score
                class_id = int(box.cls[0])  # Class ID
                
                if score >= 0.9:
                    # Menghitung panjang dan lebar bounding box
                    panjang = x2 - x1
                    lebar = y2 - y1
                    
                    # Check if bounding box size is 64x171
                    if panjang >= 40 and lebar >= 130:
                        cropped_img = image[y1:y2, x1:x2]
                        cropped_path = os.path.join(save_dir, f"{Path(image_path).stem}_crop_{x1}_{y1}.jpg")
                        cv2.imwrite(cropped_path, cropped_img)
                        cropped_images.append(cropped_path)
    
    return cropped_images

# Fungsi untuk mendeteksi orang, spion, dan helm
def detect_objects(model, image_path, save_dir):
    image = cv2.imread(image_path)
    results = model(image)
    
    person_count = 0
    mirror_count = 0
    helmet_count = 0
    
    for result in results:
        boxes = result.boxes  # Access bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls)
            label = model.names[class_id]
            
            if label == 'person':
                person_count += 1
            elif label == 'spion':
                mirror_count += 1
            elif label == 'helm':
                helmet_count += 1
            
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    save_path = os.path.join(save_dir, f"{Path(image_path).name}")
    cv2.imwrite(save_path, image)
    
    return person_count, mirror_count, helmet_count

# Main program
if __name__ == "__main__":
    motor_model_path = "best.pt"
    object_model_path = "best.pt"
    
    images_dir = "C:/Users/YOSUA N LATUPUTTY/deteksi-motor/images"
    cropped_dir = "C:/Users/YOSUA N LATUPUTTY/deteksi-motor/cropped_images"
    results_dir = "C:/Users/YOSUA N LATUPUTTY/deteksi-motor/results"
    
    motor_model = load_model(motor_model_path)
    object_model = load_model(object_model_path)
    
    total_counts = {"person": 0, "mirror": 0, "helmet": 0}
    
    for image_file in os.listdir(images_dir):
        image_path = os.path.join(images_dir, image_file)
        
        cropped_images = detect_and_crop(motor_model, image_path, cropped_dir, class_id=1)  # Assume class_id=3 is for motor
        
        for cropped_image in cropped_images:
            person_count, mirror_count, helmet_count = detect_objects(object_model, cropped_image, results_dir)
            
            total_counts["person"] += person_count
            total_counts["mirror"] += mirror_count
            total_counts["helmet"] += helmet_count
            
            os.remove(cropped_image)
            print(f"Image: {cropped_image}, Persons: {person_count}, Mirrors: {mirror_count}, Helmets: {helmet_count}")
   
    
    print("Total counts:", total_counts)
