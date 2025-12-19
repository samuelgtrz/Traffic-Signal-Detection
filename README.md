# 🛑 Traffic Signal Detection

Proyecto de detección de señales de tráfico mediante modelos de detección de objetos, utilizando principalmente YOLO.

El repositorio incluye el dataset con sus etiquetas, scripts de entrenamiento e inferencia, y los resultados obtenidos con distintos modelos.

## Ejemplo de validación para el modelo de YOLOv8 L con las métricas obtenidas
<p align="center">
  <img src="https://github.com/samuelgtrz/Traffic-Signal-Detection/blob/main/resultados_yolo/yolo_trafico_L/val_batch1_pred.jpg?raw=true" width="600">
</p>

<p align="center">
  <img src="https://github.com/samuelgtrz/Traffic-Signal-Detection/blob/main/resultados_yolo/yolo_trafico_L/BoxPR_curve.png?raw=true" width="600">
</p>

---

## 🧠 Descripción del proyecto

El objetivo del proyecto es entrenar un modelo capaz de detectar señales de tráfico en imágenes, a partir de anotaciones con *bounding boxes* y etiquetas de clase.

El flujo general es:
1. Leer las anotaciones desde un CSV.
2. Preparar el dataset en el formato requerido por el modelo.
3. Entrenar el modelo de detección.
4. Realizar inferencias sobre imágenes nuevas.
5. Generar las predicciones en el formato indicado.

---

## 📄 Archivos CSV

### `train.csv`

Contiene las anotaciones del conjunto de entrenamiento.  
Incluye, para cada imagen:
- Las **bounding boxes** de las señales.
- Las **etiquetas de clase** asociadas a cada bounding box.

Este archivo se utiliza para generar las anotaciones necesarias durante el entrenamiento del modelo.

---

### `sample_submission.csv`

Archivo de ejemplo que muestra **el formato correcto en el que deben enviarse las inferencias**.

Sirve como referencia para:
- La estructura del CSV final de predicciones.
- El formato de las bounding boxes y clases en la inferencia.
- La forma en la que se deben identificar las imágenes.

Las predicciones generadas por el modelo deben seguir exactamente este formato.

---

## ⚙️ Fine-tuning del modelo

Para realizar el *fine-tuning* del modelo, ejecuta el script `train.py` ajustando los parámetros según tus necesidades (arquitectura, épocas, tamaño de batch, etc.):

---


## 🧪 Inferencia

Para realizar inferencia sobre nuevas imágenes:

1. Carga en la variable `model` el modelo entrenado que desees utilizar.
2. Ejecuta el script de inferencia correspondiente.

El resultado de la inferencia se generará automáticamente en un archivo **CSV**, que se guardará en la **misma carpeta**, siguiendo el formato especificado en `sample_submission.csv`.


