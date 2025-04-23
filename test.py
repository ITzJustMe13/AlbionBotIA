from IPython import display
display.clear_output()

import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import roboflow

import ultralytics
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

# prever em novas imagens
confidence_level = 0.1
input_path = 'captured_images'
output_path = 'detections'
class_names = model.names

for file in os.listdir(input_path):
    if file.lower().endswith((".png")):
        image = cv2.imread(os.path.join(input_path, file))
        results = model.predict(source=image,
                                conf=confidence_level)  # gerar previsões acima de determinada confiança, e guardar imagens

        output_filename = f"prediction_{file}"
        output_filepath = os.path.join(output_path, output_filename)

        for result in results:
            result.save(filename=output_filepath)
            print("==== Resultados Previsão ====")
            print("Imagem: " + os.path.join(input_path, file))
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes (x_min, y_min, x_max, y_max)
            scores = result.boxes.conf.cpu().numpy()  # Score de confiança
            labels = result.boxes.cls.cpu().numpy()  # Índice da classe

            for i in range(len(boxes)):
                class_id = labels[i]
                class_label = class_names[class_id] if class_id in class_names else "Desconhecido"

                print(f"--- Objeto {i + 1} ---")
                print(f"Class: {class_label} (ID: {class_id})")
                print(f"Coordenadas Bounding Box: {boxes[i]}")
                print(f"Confiança: {scores[i]:.4f}")
                print("-------------------")

            print("\n")