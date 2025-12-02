import pandas as pd
import os
from PIL import Image

csv_path = "C:\\Users\\sam20\\OneDrive\\Documentos\\IA\\CuartoIA\\Vision_por_computador\\practica6competicion\\deteccion-de-sinais-de-trafico\\train.csv"
images_dir = "C:\\Users\\sam20\\OneDrive\\Documentos\\IA\\CuartoIA\\Vision_por_computador\\practica6competicion\\deteccion-de-sinais-de-trafico\\train"
labels_dir = "C:\\Users\\sam20\\OneDrive\\Documentos\\IA\\CuartoIA\\Vision_por_computador\\practica6competicion\\deteccion-de-sinais-de-trafico\\labels"

os.makedirs(labels_dir, exist_ok=True)

df = pd.read_csv(csv_path)
df["image_id"] = df["image_id"].astype(str)
df["image_id"] = df["image_id"].apply(lambda x: x if x.endswith(".jpg") else x + ".jpg")

for image_id, group in df.groupby("image_id"):
    # Ruta de imagen
    img_path = os.path.join(images_dir, image_id)
    if not os.path.exists(img_path):
        print("No existe la imagen:", img_path)
        continue
    
    # Cargar imagen para obtener ancho y alto
    img = Image.open(img_path)
    W, H = img.size #deberia ser siempre 640x640

    lines = []
    for _, row in group.iterrows():
        x_min = row["bbox_x"]
        y_min = row["bbox_y"]
        w = row["bbox_width"]
        h = row["bbox_height"]
        cls = row["category_id"]

        # Convertir a YOLO
        x_center = (x_min + w/2) / W
        y_center = (y_min + h/2) / H
        w_norm = w / W
        h_norm = h / H

        lines.append(f"{cls} {x_center} {y_center} {w_norm} {h_norm}")

    # Guardar archivo YOLO
    label_path = os.path.join(labels_dir, image_id.replace(".jpg",".txt").replace(".png",".txt"))
    with open(label_path, "w") as f:
        f.write("\n".join(lines))
