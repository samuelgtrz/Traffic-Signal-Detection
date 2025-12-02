import os
import pandas as pd
from ultralytics import YOLO

# Cargar modelo entrenado
model = YOLO("C:\\Users\\sam20\\OneDrive\\Documentos\\IA\\CuartoIA\\Vision_por_computador\\practica6competicion\\deteccion-de-sinais-de-trafico\\resultados_yolo\\yolo_trafico\\weights\\best.pt")

test_path = "C:\\Users\\sam20\\OneDrive\\Documentos\\IA\\CuartoIA\\Vision_por_computador\\practica6competicion\\deteccion-de-sinais-de-trafico\\test"
preds = model.predict(test_path, save=False, conf=0.25)

rows = []
row_id = 0

for result in preds:
    img_name = os.path.basename(result.path)
    image_id = img_name

    for box in result.boxes:
        cls = int(box.cls[0].item())
        conf = float(box.conf[0].item())

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        width = x2 - x1
        height = y2 - y1

        # Volver al ID original (1-21)
        category_id = cls + 1

        rows.append([
            row_id,
            image_id,
            x1, y1, width, height,
            category_id,
            conf
        ])
        row_id += 1

df = pd.DataFrame(rows, columns=[
    "row_id",
    "image_id",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
    "category_id",
    "score"
])

df.to_csv("C:\\Users\\sam20\\OneDrive\\Documentos\\IA\\CuartoIA\\Vision_por_computador\\practica6competicion\\deteccion-de-sinais-de-trafico\\resultados_yolo\\submission.csv", index=False)
print("CSV generado como testsubmission.csv")
