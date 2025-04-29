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

def recurso_presente(img):
    results = model.predict(img)
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if conf > 0.8 and cls == 1:
                return True
    return False

while True:

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

    results = model.predict(frame)

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if clicker_ativo and conf > 0.8 and cls == 1:
                print(f"Classe: {cls} | Conf: {conf:.2f} | Click: ({cx}, {cy})")
                pyautogui.moveTo(cx, cy)
                pyautogui.click()

                start_time = time.time()

                # Verificação do recurso antes de entrar no loop de espera
                screenshot_check = sct.grab(monitor)
                img_check = np.array(screenshot_check)
                frame_check = cv2.cvtColor(img_check, cv2.COLOR_BGRA2BGR)

                # Se o recurso não está presente, sai imediatamente
                if not recurso_presente(frame_check):
                    print("Resource collected.")
                    break

                # Se o recurso está presente, começa a esperar no loop com tempo máximo
                while True:
                    time.sleep(0.5)
                    screenshot_check = sct.grab(monitor)
                    img_check = np.array(screenshot_check)
                    frame_check = cv2.cvtColor(img_check, cv2.COLOR_BGRA2BGR)

                    if not recurso_presente(frame_check):
                        print("Resource collected.")
                        break

                    if time.time() - start_time > 8:
                        print("Max time reached.")
                        break

        # annotated = results[0].plot()
        # cv2.imshow("Deteção em tempo real", annotated)

cv2.destroyAllWindows()
