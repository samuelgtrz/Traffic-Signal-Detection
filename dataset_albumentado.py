#Creacion de un dataset con data augmentation de la libreria Albumentation
import albumentations as A
import cv2
import os
import glob
import shutil
from tqdm import tqdm

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = "\dataset"
OUTPUT_DIR = "\dataset_aug"

# Rutas de entrada (Train)
INPUT_IMAGES_TRAIN = os.path.join(BASE_DIR, 'images', 'train')
INPUT_LABELS_TRAIN = os.path.join(BASE_DIR, 'labels', 'train')

# Rutas de salida (Train)
OUTPUT_IMAGES_TRAIN = os.path.join(OUTPUT_DIR, 'images', 'train')
OUTPUT_LABELS_TRAIN = os.path.join(OUTPUT_DIR, 'labels', 'train')

# --- DEFINIR PIPELINE DE ALBUMENTATIONS ---
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    # Rotación ligera y escalado
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
    # Efectos de clima/luz (útil para tráfico)
    A.OneOf([
        A.RandomBrightnessContrast(p=1),
        A.CLAHE(clip_limit=2),
        A.HueSaturationValue(p=1),
    ], p=0.3),
    A.OneOf([
        A.Blur(blur_limit=3, p=1),
        A.MotionBlur(blur_limit=3, p=1),
    ], p=0.2),
    # Ruido (simula baja calidad de cámara)
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def setup_directories():
    """Crea la estructura de carpetas y copia los archivos necesarios."""
    if os.path.exists(OUTPUT_DIR):
        print(f"Nota: La carpeta de salida ya existe: {OUTPUT_DIR}")
    
    os.makedirs(OUTPUT_IMAGES_TRAIN, exist_ok=True)
    os.makedirs(OUTPUT_LABELS_TRAIN, exist_ok=True)

    # 1. Copiar carpeta VAL tal cual (no se aumenta validación)
    print("Copiando carpeta 'val' original (sin cambios)...")
    for folder in ['images', 'labels']:
        src = os.path.join(BASE_DIR, folder, 'val')
        dst = os.path.join(OUTPUT_DIR, folder, 'val')
        if os.path.exists(src):
            if os.path.exists(dst): shutil.rmtree(dst) # Limpiar si existe
            shutil.copytree(src, dst)
    
    # 2. Copiar data.yaml
    yaml_src = os.path.join(BASE_DIR, 'data.yaml')
    if os.path.exists(yaml_src):
        shutil.copy(yaml_src, os.path.join(OUTPUT_DIR, 'data.yaml'))
        print("Copiado data.yaml")

def augment_train_data():
    image_paths = glob.glob(os.path.join(INPUT_IMAGES_TRAIN, '*.jpg')) # Cambia a *.png si usas png
    # También busca .png por si acaso tienes mezcla
    image_paths += glob.glob(os.path.join(INPUT_IMAGES_TRAIN, '*.png'))
    
    print(f"Procesando {len(image_paths)} imágenes de TRAIN para aumentar...")

    for img_path in tqdm(image_paths):
        # Leer imagen
        image = cv2.imread(img_path)
        if image is None: continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Leer label
        txt_name = os.path.basename(img_path).rsplit('.', 1)[0] + '.txt'
        label_path = os.path.join(INPUT_LABELS_TRAIN, txt_name)
        
        bboxes = []
        class_labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    class_labels.append(int(parts[0]))
                    bboxes.append([float(x) for x in parts[1:]])
        
        # --- GENERACIÓN ---
        # 1. Guardar la ORIGINAL en la nueva carpeta
        # (Es importante mantener tus 1100 originales en el nuevo dataset)
        try:
            filename_orig = os.path.basename(img_path)
            cv2.imwrite(os.path.join(OUTPUT_IMAGES_TRAIN, filename_orig), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if os.path.exists(label_path):
                shutil.copy(label_path, os.path.join(OUTPUT_LABELS_TRAIN, txt_name))
        except Exception as e:
            print(f"Error copiando original {filename_orig}: {e}")

        # 2. Generar COPIAS AUMENTADAS (Ejemplo: crear 2 versiones extra por cada imagen)
        for i in range(2): 
            try:
                transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                
                # Si perdimos las bboxes en la transformación (raro), no guardamos
                if len(bboxes) > 0 and len(transformed['bboxes']) == 0:
                    continue
                
                # Guardar imagen aug
                aug_filename = f"{os.path.splitext(filename_orig)[0]}_aug_{i}.jpg"
                save_img_path = os.path.join(OUTPUT_IMAGES_TRAIN, aug_filename)
                cv2.imwrite(save_img_path, cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR))
                
                # Guardar label aug
                save_lbl_path = os.path.join(OUTPUT_LABELS_TRAIN, os.path.splitext(aug_filename)[0] + '.txt')
                with open(save_lbl_path, 'w') as f:
                    for cls, bbox in zip(transformed['class_labels'], transformed['bboxes']):
                        f.write(f"{cls} {' '.join(map(str, bbox))}\n")
                        
            except Exception as e:
                print(f"Error aumentando {img_path}: {e}")

if __name__ == "__main__":
    setup_directories()
    augment_train_data()
    print("\n¡Proceso terminado!")
    print(f"Tu nuevo dataset está en: {OUTPUT_DIR}")
    print("Recuerda actualizar la ruta 'path' dentro del nuevo data.yaml si usas rutas absolutas.")
