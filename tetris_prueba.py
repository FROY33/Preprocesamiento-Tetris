import socket
import json
import time

def enviar_movimiento(rotacion, columna):
    try:
        # Nos conectamos al servidor (el juego de Tetris)
        cliente = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        cliente.connect(('192.168.1.214', 9999))
        print("Conectado al juego de Tetris!")

        movimiento = {"rotacion": rotacion, "columna": columna}
        mensaje = json.dumps(movimiento) + "\n"
        
        # Enviamos el movimiento
        cliente.sendall(mensaje.encode('utf-8'))
        print(f"Movimiento enviado: {movimiento}")

        # Esperamos un poco para dejar que la pieza se mueva
        time.sleep(1)

        cliente.close()
        print("Desconectado.\n")
    except ConnectionRefusedError:
        print("No se pudo conectar. Asegúrate de que tetris.py esté corriendo primero.")

if __name__ == "__main__":
    print("Prueba de control remoto de Tetris")
    print("----------------------------------")
    
    # Prueba 1: Mover a la columna 1, rotación 1
    enviar_movimiento(rotacion=1, columna=7)