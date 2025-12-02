import os
import cv2

# Ruta raíz de tu dataset YOLO
ROOT = r"C:\\Users\\sam20\\OneDrive\\Documentos\\IA\\CuartoIA\\Vision_por_computador\\practica6competicion\\deteccion-de-sinais-de-trafico\\dataset"

IMG_DIR = os.path.join(ROOT, "images")
LBL_DIR = os.path.join(ROOT, "labels")

CLASSES = [c.strip() for c in open(os.path.join(ROOT, "classes.txt"))]

# Salida KITTI
OUT = os.path.join(ROOT, "kitti_dataset")
os.makedirs(OUT, exist_ok=True)
for sp in ["train", "val"]:
    os.makedirs(os.path.join(OUT, sp, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUT, sp, "labels"), exist_ok=True)

def convert_split(split):
    print(f"[INFO] Convirtiendo {split}...")
    with open(os.path.join(ROOT, f"{split}.txt")) as f:
        img_list = f.read().strip().split("\n")

    for img_path in img_list:
        img_path = img_path.replace("\\", "/")
        img_name = os.path.basename(img_path)
        base = os.path.splitext(img_name)[0]

        # Ruta a imagen real
        img = cv2.imread(img_path)
        if img is None:
            print("No se pudo leer la imagen:", img_path)
            continue

        h, w, _ = img.shape

        # Ruta label YOLO
        yolo_label = os.path.join(LBL_DIR, split, base + ".txt")
        if not os.path.exists(yolo_label):
            continue

        kitti_lines = []

        # Convertir anotaciones YOLO → KITTI
        with open(yolo_label) as f:
            for line in f:
                c, xc, yc, bw, bh = map(float, line.split())

                xmin = int((xc - bw / 2) * w)
                ymin = int((yc - bh / 2) * h)
                xmax = int((xc + bw / 2) * w)
                ymax = int((yc + bh / 2) * h)

                cls = CLASSES[int(c)]

                kitti_line = f"{cls} 0.0 0.0 0.0 {xmin} {ymin} {xmax} {ymax} 0 0 0 0 0 0 0\n"
                kitti_lines.append(kitti_line)

        # Guardar imagen y label KITTI
        cv2.imwrite(os.path.join(OUT, split, "images", img_name), img)
        with open(os.path.join(OUT, split, "labels", base + ".txt"), "w") as f:
            f.writelines(kitti_lines)

    print(f"[OK] Conversión de {split} terminada.")

# Convertir TRAIN y VAL
convert_split("train")
convert_split("val")
