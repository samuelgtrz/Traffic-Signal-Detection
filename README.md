Instrucciones:
Primero se descargan los datos de kaggle
Luego ejecutamos transform_to_coco.py para pasar la informacion del csv a la estructura requerida para usar YOLO (necesitamos primero las etiquetas en formato .txt)
Ejecutamos train_val_split.py para reorganizar la estructura de archivos a la que YOLO necesita
Despues ejecutamos fix_classes porque YOLO necesita que las clases empiecen en 0 y eso no sucede en nuestro caso
