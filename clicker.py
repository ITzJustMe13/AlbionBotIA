import cv2
import numpy as np
import pyautogui
import time
import mss
import keyboard
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}

sct = mss.mss()
clicker_ativo = False

print("Pressiona 's' para iniciar, 'e' para parar, 'q' para sair.")

while True:
    # Teclas para controlo
    if keyboard.is_pressed("s"):
        clicker_ativo = True
        print("[Clicker Ativado]")

    if keyboard.is_pressed("e"):
        clicker_ativo = False
        print("[Clicker Desativado]")

    if keyboard.is_pressed("q"):
        print("A sair...")
        break

    # Captura o ecrã
    screenshot = sct.grab(monitor)
    img = np.array(screenshot)
    frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    results = model(frame, verbose=False)

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Só atua se o modo estiver ativo
            if clicker_ativo and conf > 0.7 and cls == 1:
                print(f"Classe: {cls} | Conf: {conf:.2f} | Click: ({cx}, {cy})")
                pyautogui.moveTo(cx, cy)
                pyautogui.click()
                time.sleep(3.0)

    # Mostrar janela com deteções
    annotated = results[0].plot()
    cv2.imshow("Deteção em tempo real", annotated)

    if cv2.waitKey(1) == ord("q"):  # redundante, mas mantém-se
        break

cv2.destroyAllWindows()
