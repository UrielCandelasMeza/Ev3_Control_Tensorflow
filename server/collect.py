import cv2
from cvzone.HandTrackingModule import HandDetector
import mediapipe as mp
import numpy as np
import math
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Tamaño de la imagen
IMAGE_SIZE = 300


# Folders destino para la recoleccion de datos
folder = "Datasets/Forward"
#folder = "Datasets/Left"
#folder = "Datasets/Right"
#folder = "Datasets/Back"
#folder = "Datasets/Stop"
counter = 0


#detector = HandDetector(maxHands=1)

def calculateBbox(landmarks_px, x, y, image, h, w):
  if landmarks_px:
    x_coords = [x for x, y in landmarks_px]
    y_coords = [y for x, y in landmarks_px]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Margen al bbox
    margin = 30
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(w, x_max + margin)
    y_max = min(h, y_max + margin)

    # Recortar mano
    hand_roi = image[y_min:y_max, x_min:x_max]

    #hand_roi_shape = hand_roi.shape

    # Pantalla blanca para obtener datos
    imageWhite = np.ones((IMAGE_SIZE, IMAGE_SIZE,3), np.uint8) * 255


    # Calcula la relacion entre la altura y el ancho
    height = y_max - y_min
    width = x_max - x_min
    aspectRatio = height / width 

    # Esto busca normalizar la imagen para que sea siempre igual
    # dejando un fondo blanco y la imagen en medio con altura constante
    if aspectRatio > 1:
      # Si es mayor a uno entonces realiza un calculo para que el alto 
      # siempre toque el borde de la pantalla

      K = IMAGE_SIZE / height
      wCalc = math.ceil(K*width)

      # Realiza la nueva asignacion de tamaño
      imageResize = cv2.resize(hand_roi,(wCalc,IMAGE_SIZE))
      #imageResizeShape = imageResize.shape

      # Realiza el calculo para dejar la imagen en medio
      wGap = math.ceil((IMAGE_SIZE - wCalc)/2)

      # Asigna la posicion del ancho y el alto a la imagen con fondo
      imageWhite[:,wGap:wCalc+wGap] = imageResize
    else :
      # Si no entonces realiza un calculo  para que el ancho siempre
      # toque el borde de la pantalla
      K = IMAGE_SIZE / width
      hCalc = math.ceil(K*height)

      # Realiza la nueva asignacion de tamaño
      imageResize = cv2.resize(hand_roi,(IMAGE_SIZE,hCalc))

      # Realiza el calculo para dejar la imagen en medio
      hGap = math.ceil((IMAGE_SIZE - hCalc)/2)

      # Asigna la posicion del ancho y el alto a la imagen con fondo
      imageWhite[hGap:hCalc + hGap,:] = imageResize

    
    return hand_roi, imageWhite
  return 0


cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()

    # Si no se obtuvo la imagen entonces ignora el frame
    if not success:
      print("Ignoring empty camera frame.")
      continue

    # Pasa de Azul, Verde y Rojo a RGB
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #Se obtienen los datos de los landmarks de la mano
    results = hands.process(image)

    # Pasa de RGB a Azul, Verde y Rojo
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)    

    if results.multi_hand_landmarks:
    # Si se obtienen las landmarks se recorre cada una de ellas
      for hand_landmarks in results.multi_hand_landmarks:

        # Se dibujan las landmarks
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        
        # Obtner Lanmarks de las manos
        h,w,_ = image.shape
        landmarks_px = []
        for landmark in hand_landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            landmarks_px.append((x, y))

        # Calcular la bounding box de las manos
        hand_roi, imageWhite = calculateBbox(landmarks_px, x, y, image, h, w)


        # Dibujar la pantalla con la mano recortada del resto de la imagen
        if hand_roi.size != 0 and imageWhite.size != 0:
            cv2.imshow("Mano recortada", hand_roi)

            # Muestra la imagen con todo y bordes para el entrenamiento
            cv2.imshow("ImageWhite", imageWhite)
        
    # Voltea la imagen principal para que se vea chido
    cv2.imshow('MediaPipe Hands', cv2.flip(image,1))

    capture = cv2.waitKey(1)

    if capture == ord("s"):
      counter += 1
      cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imageWhite)
      print(counter)

    # Letra 'q' para salir
    if cv2.waitKey(5) == ord("q"):
      break

cap.release()


