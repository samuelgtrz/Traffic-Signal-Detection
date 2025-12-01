from ultralytics import YOLO

# Ruta al modelo base (puedes cambiar a yolov8m.pt, yolov8l.pt, etc.)
model = YOLO("yolov8n.pt")

# Entrenamiento
model.train(
    data="data.yaml",        # ruta al archivo data.yaml
    epochs=300,               # número de épocas
    imgsz=640,               # tamaño de imagen
    batch=32,                # tamaño de batch (ajustar si hay poca memoria)
    device=0,                # usar GPU
    workers=4,               # número de dataloader workers
    patience=30,             # early stopping
    project="runs_yolo",     # carpeta donde se guardará el entrenamiento
    name="finetune_v1",      # nombre del experimento
)
