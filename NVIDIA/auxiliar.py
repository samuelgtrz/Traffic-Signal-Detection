import os
import yaml

# Ruta a tu data.yaml
DATA_YAML = r"C:\\Users\\sam20\\OneDrive\\Documentos\\IA\\CuartoIA\\Vision_por_computador\\practica6competicion\\deteccion-de-sinais-de-trafico\\dataset\\data.yaml"

# Cargar yaml
with open(DATA_YAML, "r") as f:
    cfg = yaml.safe_load(f)

# Extraer rutas del yaml
base_path = cfg["path"]
train_rel = cfg["train"]
val_rel   = cfg["val"]

# Construir rutas absolutas correctas
train_dir = os.path.join(base_path, train_rel).replace("\\", "/")
val_dir   = os.path.join(base_path, val_rel).replace("\\", "/")

print("Train dir:", train_dir)
print("Val dir:", val_dir)

def collect_images(folder):
    exts = {".jpg", ".jpeg", ".png"}
    imgs = []
    for root, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                imgs.append(os.path.join(root, f).replace("\\", "/"))
    return imgs

train_imgs = collect_images(train_dir)
val_imgs   = collect_images(val_dir)

# Guardar train.txt y val.txt en el mismo directorio que data.yaml
output_dir = os.path.dirname(DATA_YAML)

with open(os.path.join(output_dir, "train.txt"), "w") as f:
    f.write("\n".join(train_imgs))

with open(os.path.join(output_dir, "val.txt"), "w") as f:
    f.write("\n".join(val_imgs))

print(f"✔ train.txt creado ({len(train_imgs)} imágenes)")
print(f"✔ val.txt creado ({len(val_imgs)} imágenes)")
