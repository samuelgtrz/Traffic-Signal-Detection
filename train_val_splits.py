import os
import shutil
import random


images_src = "C:\\Users\\sam20\\OneDrive\\Documentos\\IA\\CuartoIA\\Vision_por_computador\\practica6competicion\\deteccion-de-sinais-de-trafico\\train"
labels_src = "C:\\Users\\sam20\\OneDrive\\Documentos\\IA\\CuartoIA\\Vision_por_computador\\practica6competicion\\deteccion-de-sinais-de-trafico\\labels"
output_dir = "C:\\Users\\sam20\\OneDrive\\Documentos\\IA\\CuartoIA\\Vision_por_computador\\practica6competicion\\deteccion-de-sinais-de-trafico\\dataset"

train_ratio = 0.8   # 80% train, 20% val

# Crear estructura YOLO
paths = [
    f"{output_dir}/images/train",
    f"{output_dir}/images/val",
    f"{output_dir}/labels/train",
    f"{output_dir}/labels/val",
]

for p in paths:
    os.makedirs(p, exist_ok=True)

# Listar todas las imágenes del train
images = [f for f in os.listdir(images_src) if f.lower().endswith((".jpg", ".png"))]

print(f"Total imágenes encontradas: {len(images)}")

# Shuffle para dividir aleatoriamente
random.shuffle(images)

split_idx = int(len(images) * train_ratio)
train_imgs = images[:split_idx]
val_imgs = images[split_idx:]

print(f"Train: {len(train_imgs)} imágenes")
print(f"Val:   {len(val_imgs)} imágenes")

def copy_pair(img_list, subset):
    """
    Copia imagen y su .txt correspondiente a la carpeta subset (train/val)
    """
    for img_name in img_list:
        img_src_path = os.path.join(images_src, img_name)

        txt_name = img_name.rsplit(".", 1)[0] + ".txt"
        txt_src_path = os.path.join(labels_src, txt_name)

        # Destinos
        img_dst = os.path.join(output_dir, "images", subset, img_name)
        txt_dst = os.path.join(output_dir, "labels", subset, txt_name)

        # Copiar imagen
        shutil.copy(img_src_path, img_dst)

        # Copiar label si existe
        if os.path.exists(txt_src_path):
            shutil.copy(txt_src_path, txt_dst)
        else:
            print(f"WARNING: no existe label para {img_name}")

# Copiar train
copy_pair(train_imgs, "train")

# Copiar val
copy_pair(val_imgs, "val")

print("\n Dataset preparado correctamente en la carpeta:", output_dir)
