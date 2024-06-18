from ultralytics import YOLO

# buat deteksi dari video bawaan YOLO

# Load a model
model = YOLO("deteksi-motor-n.pt")  # build a new model from scratch

# Use the model
# model.train(data="config.yaml", epochs=100)  # train the model

model.track(source="C:\\Users\\YOSUA N LATUPUTTY\\deteksi-motor\\Video\\2024-06-07 11-34-16.mkv",  show=True, tracker="bytetrack.yaml")

