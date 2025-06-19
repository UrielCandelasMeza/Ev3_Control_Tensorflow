# server.py
import socket

def start_server():
    host = '0.0.0.0'  # Escucha en todas las interfaces
    port = 4041   # Puerto no privilegiado (> 1023)
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        print(f"Servidor escuchando en {host}:{port}")
        
        conn, addr = s.accept()
        with conn:
            print(f"Conexión establecida desde {addr}")
            while True:
                #data = conn.recv(1024)
                #if not data:
                #    break
                #print(f"Mensaje recibido: {data.decode('utf-8')}")
                respuesta = input("Respuesta para el cliente: ")
                if respuesta == "salir":
                    break
                conn.sendall(respuesta.encode('utf-8'))

if __name__ == "__main__":
    start_server()