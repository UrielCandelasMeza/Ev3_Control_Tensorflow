import socket
from movement import Movement
import time

host = '192.168.0.2'
port = 4041
last_command = ""

robot = Movement()

def start_client():
    
    max_retries = 5
    retries = 0

    while retries < max_retries:

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            s.connect((host, port))
            print("Conectado al servidor: " +  str(host) + ":" + str(port))
            retries = 0  # Reiniciar contador de reintentos al conectar exitosamente
            
            while True:
                data = s.recv(1024)
                if not data:
                    robot.stop()
                    break
                data = data.decode('utf-8').strip().lower()

                if data == "forward":
                    robot.forward()
                elif data == "back":
                    robot.back()
                elif data == "left":
                    robot.turn_left()
                elif data == "right":  
                    robot.turn_right()
                elif data == "stop":
                    robot.steering.off()
                elif data == "salir":
                    print("Saliendo del cliente...")
                    break
                
        except socket.timeout:
            retries += 1
            print("Timeout de conexion (intento " + str(retries) + "/" + str(max_retries) + ")")
            
        except ConnectionRefusedError:
            print("Servidor no disponible. Reintentando...")
            time.sleep(2)
            retries += 1
            
        except Exception as e:
            print("Error inesperado: " + str(e))
            break
            
        finally:
            s.close()
            if retries >= max_retries:
                print("Numero maximo de reintentos alcanzado")

if __name__ == "__main__":
    start_client()