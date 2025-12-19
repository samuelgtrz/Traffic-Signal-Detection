#Script to fine tune YOLO models
from ultralytics import YOLO
import torch
import os
import sys

# === COMPROBAR ESTRUCTURA DEL DATASET ===
def check_dataset_structure(base_dir):
    expected_dirs = [
    "images/train",
    "labels/train",
    "images/val",
    "labels/val",
]


    for d in expected_dirs:
        path = os.path.join(base_dir, d)
        if not os.path.exists(path):
            print(f"No existe la carpeta: {path}")
            return False
        if not os.listdir(path):
            print(f"La carpeta {path} está vacía.")

    print("Estructura del dataset verificada correctamente.")

    return True


if __name__ == "__main__":

    BASE_DIR ="/mnt/netapp2/Store_uni/home/usc/cursos/curso1589/VISION/practica6competicion/deteccion-de-sinais-de-trafico/dataset"
    DATA_YAML = BASE_DIR + "/data.yaml"

    print("=== INICIO DEL ENTRENAMIENTO YOLOv8 ===")
    print(f"Ruta base del dataset: {BASE_DIR}")
    print(f"Archivo YAML: {DATA_YAML}")

    # === COMPROBAR GPU ===
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU detectada: {gpu_name}")
        device = 'cuda'
    else:
        print("No se detectó GPU. Se entrenará en CPU.")
        device = 'cpu'

    if not check_dataset_structure(BASE_DIR):
        print("ERROR: Estructura del dataset incompleta. Revisa las rutas.")
        sys.exit(1)

    # === CARGAR MODELO YOLOv8 ===
    try:
        model = YOLO("yolov8s.pt")
        print("Modelo YOLOv8 cargado correctamente.")
    except Exception as e:
        print("Error al cargar el modelo YOLOv8:", e)
        sys.exit(1)

    # === ENTRENAMIENTO (FINE-TUNING) ===
    print("Iniciando entrenamiento...")
    results = model.train(
        data=DATA_YAML,        		# Ruta al archivo YAML
        epochs=700,
	patience=80,			# Si el modelo deja de mejorar durante 30 épocas seguidas, el entrenamiento se detiene automáticamente
        imgsz=640,              	# Tamaño de entrada
        batch=32,               	# Tamaño del batch
        lr0=0.0005,            		# Learning rate inicial
        optimizer="AdamW",      	# Optimizador
        device=device,          	# GPU o CPU
        name="yolo_trafico_N_parametros_augment",	# Carpeta de resultados
        pretrained=True, 		# Utiliza pesos preentrenados
	augment=True,
	mosaic=1.0,
    	mixup=0.2,
    	degrees=5,
    	shear=2,
    	translate=0.2,
    	scale=0.9,           	# Utiliza data augmentation
        workers=4,             		# Número de hilos para carga de datos
    )

    # === VALIDACIÓN POST-ENTRENAMIENTO ===
    print("Evaluando el modelo...")
    metrics = model.val(data=DATA_YAML)
    print(metrics)

    print("Entrenamiento finalizado correctamente.")
