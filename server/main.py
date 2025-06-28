import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import math
import socket
import time
import threading

# Inicialización de MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Configuración del servidor
HOST = '0.0.0.0'  # Escucha en todas las interfaces
PORT = 4041

# Variables globales
conn = None
addr = None
server_socket = None
connection_active = False
last_command_time = 0
last_command = ""
command_delay = 0.7  # medio segundo entre comandos

# Cargar el modelo
try:
    model = tf.keras.models.load_model("Models/optimized_model.keras")
    print("Modelo cargado exitosamente")
except Exception as e:
    print(f"Error cargando el modelo: {e}")
    exit()

class_labels = ["Back", "Forward", "Left", "Right", "Stop"]
IMAGE_SIZE = 300

def start_server():
    global conn, addr, server_socket, connection_active
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        print(f"Servidor escuchando en {HOST}:{PORT}")
        
        while True:
            try:
                conn, addr = server_socket.accept()
                connection_active = True
                print(f"Conexión establecida desde {addr}")
                
                threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()
                
            except Exception as e:
                print(f"Error aceptando conexión: {e}")
                if conn:
                    conn.close()
                break
                
    except Exception as e:
        print(f"Error iniciando servidor: {e}")
    finally:
        if server_socket:
            server_socket.close()

def handle_client(connection, client_address):
    global conn, addr, connection_active
    try:
        while connection_active:
            # Verificar si la conexión sigue activa
            try:
                # Intento de recibir un pequeño paquete (podría ser un ping/heartbeat)
                connection.settimeout(1.0)
                data = connection.recv(1, socket.MSG_PEEK)
                if data == b'':
                    raise ConnectionError("Cliente desconectado")
            except socket.timeout:
                # Timeout es normal, significa que no hay datos pero conexión activa
                pass
            except Exception as e:
                print(f"Error verificando conexión: {e}")
                break
                
    except Exception as e:
        print(f"Error en handle_client: {e}")
    finally:
        connection.close()
        conn = None
        addr = None
        connection_active = False
        print(f"Conexión con {client_address} cerrada")

def send_command(command):
    global conn, last_command_time, last_command
    
    if not conn:
        print("No hay conexión activa para enviar comando")
        return False
    
    current_time = time.time()
    if (current_time - last_command_time) <= command_delay:
        return False
    
    try:
        if command != last_command:
            conn.sendall(command.encode('utf-8'))
            last_command_time = current_time
            #time.sleep(1)
            print(f"Comando enviado: {command}")
            last_command = command
        return True
    except Exception as e:
        print(f"Error enviando comando: {e}")
        if conn:
            conn.close()
        return False

def predict_class(image_white, x_min, y_min, x_max, y_max, image):
    if image_white is None:
        return None, 0
    
    try:
        img_for_pred = cv2.resize(image_white, (IMAGE_SIZE, IMAGE_SIZE))
        img_for_pred = np.expand_dims(img_for_pred, axis=0)
        img_for_pred = img_for_pred / 255.0
        
        prediction = model.predict(img_for_pred)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # Dibujar resultados
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(
            image, 
            f"{class_labels[predicted_class]} ({confidence:.2f})", 
            (x_min, y_min - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.9, 
            (0, 255, 0), 
            2
        )
        
        return predicted_class, confidence
        
    except Exception as e:
        print(f"Error en predicción: {e}")
        return None, 0

def generate_normalized_image(landmarks_px, image, h, w):
    if not landmarks_px:
        return None, None, None, 0
    
    try:
        x_coords = [x for x, y in landmarks_px]
        y_coords = [y for x, y in landmarks_px]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        margin = 30
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(w, x_max + margin)
        y_max = min(h, y_max + margin)

        hand_roi = image[y_min:y_max, x_min:x_max]
        if hand_roi.size == 0:
            return None, None, None, 0

        image_white = np.ones((IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8) * 255
        height, width = y_max - y_min, x_max - x_min
        aspect_ratio = height / width

        if aspect_ratio > 1:
            k = IMAGE_SIZE / height
            w_calc = math.ceil(k * width)
            img_resized = cv2.resize(hand_roi, (w_calc, IMAGE_SIZE))
            w_gap = math.ceil((IMAGE_SIZE - w_calc) / 2)
            image_white[:, w_gap:w_calc + w_gap] = img_resized
        else:
            k = IMAGE_SIZE / width
            h_calc = math.ceil(k * height)
            img_resized = cv2.resize(hand_roi, (IMAGE_SIZE, h_calc))
            h_gap = math.ceil((IMAGE_SIZE - h_calc) / 2)
            image_white[h_gap:h_calc + h_gap, :] = img_resized

        predicted_class, confidence = predict_class(image_white, x_min, y_min, x_max, y_max, image)
        return hand_roi, image_white, predicted_class, confidence
        
    except Exception as e:
        print(f"Error generando imagen normalizada: {e}")
        return None, None, None, 0

threading.Thread(target=start_server, daemon=True).start()

# Captura de video
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error abriendo la cámara")
    exit()

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    static_image_mode=False
    ) as hands:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignorando frame vacío")
            continue

        # Procesamiento de imagen
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                h, w, _ = image.shape
                landmarks_px = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

                hand_roi, image_white, predicted_class, confidence = generate_normalized_image(landmarks_px, image, h, w)

                if predicted_class is not None and confidence > 0.9:
                    command = class_labels[predicted_class]
                    send_command(command)

        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Limpieza
cap.release()
cv2.destroyAllWindows()
if conn:
    conn.close()
if server_socket:
    server_socket.close()