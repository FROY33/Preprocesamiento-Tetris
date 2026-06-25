import cv2
import json
import socket
import numpy as np
from agente_tetris import AgenteTetris

HOST = "192.168.1.214"
PORT = 9999

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((HOST, PORT))

""" CONSTANTES GLOBALES """
PIEZAS = {
    "O": {(0,0),(0,1),(1,0),(1,1)},
    "I": {(0,0),(0,1),(0,2),(0,3)},
    "T": {(0,1),(1,0),(1,2),(1,1)},
    "L": {(1,0),(0,2),(1,2),(1,1)},
    "J": {(1,0),(1,1),(1,2),(0,0)},
    "S": {(0,1),(0,2),(1,0),(1,1)},
    "Z": {(0,0),(0,1),(1,1),(1,2)}
}

matriz_anterior = np.zeros((20, 10), dtype=np.uint8)
matriz_estado = np.zeros((20, 10), dtype=np.uint8)

""" FUNCION PARA DETECTAR EL TABLERO Y NORMALIZARLO A 200x400 """
def detectar_tablero(frame, frame_umbral):
    # Buscar contornos rectangulares
    contornos, _ = cv2.findContours(frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    candidatos = []

    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if area < 20000 or area > 60000: # Ignorar contornos muy pequeños o muy grandes
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # Solo seleccionar rectangulos
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspecto = h / w

            # El tablero 10×20 tiene aspect ratio 2.0
            if 1.8 < aspecto < 2.2:
                candidatos.append((area, approx))
                
    if not candidatos:
        return None, None

    # Tomar el candidato de mayor área
    _, mejor = max(candidatos, key=lambda x: x[0])

    # Ordenar esquinas: top-left, top-right, bottom-right, bottom-left
    pts = mejor.reshape(4, 2).astype(np.float32)
    pts = ordenar_esquinas(pts)

    # Rectificar a imagen canónica 200×400 (20px por celda)
    W, H = 200, 400
    dst = np.float32([[0,0],[W,0],[W,H],[0,H]])
    M = cv2.getPerspectiveTransform(pts, dst)
    tablero_rectificado = cv2.warpPerspective(frame, M, (W, H))
    tablero_umbral = cv2.warpPerspective(frame_umbral, M, (W, H))

    return tablero_rectificado, tablero_umbral

""" DEVUELVE LAS ESQUINAS QUE DEFINEN EL CONTORNO DEL TABLERO ORDENADAS """
def ordenar_esquinas(pts):
    tl = min(pts, key=lambda p: p[0]+p[1])
    br = max(pts, key=lambda p: p[0]+p[1])
    tr = max(pts, key=lambda p: p[0]-p[1])
    bl = min(pts, key=lambda p: p[0]-p[1])
    
    return np.float32([tl, tr, br, bl])

""" DEVUELVE LA MATRIZ CON EL ESTADO DEL TABLERO MEDIANTE UN PROMEDIO """
def matriz_umbral(tablero_umbral):
    alto, ancho = tablero_umbral.shape

    filas = alto // 20
    columnas = ancho // 20

    resultado = np.zeros((filas, columnas), dtype=np.uint8)

    for i in range(filas):
        for j in range(columnas):

            y1 = i * 20
            y2 = (i + 1) * 20

            x1 = j * 20
            x2 = (j + 1) * 20

            celda = tablero_umbral[y1:y2, x1:x2]

            promedio = np.mean(celda)

            if promedio > 200:
                resultado[i, j] = 1

    return resultado

""" DEVUELVE EL TIPO DE PIEZA QUE ESTÁ CAYENDO """
def determinar_tipo_pieza(pieza):
    ys, xs = np.where(pieza == 1)

    if len(xs) != 4:
        return None
    
    coords = set(zip(
        ys - np.min(ys),
        xs - np.min(xs)
    ))

    for nombre, patron in PIEZAS.items():
        if coords == patron:
            return nombre

    return None

""" DEVUELVE TRUE SI LA PIEZA EN MOVIMIENTO CAYO """
def pieza_cayo(pieza_activa, tablero_fijo):
    ys, xs = np.where(pieza_activa == 1)

    if len(xs) != 4:
        return False

    for y, x in zip(ys, xs):

        # tocó el fondo
        if y == 19:
            return True

        # bloque fijo debajo
        if tablero_fijo[y + 1, x] == 1:
            return True

    return False

""" CICLO PRINCIPAL """
cam = cv2.VideoCapture(1)
tablero_fijo = np.zeros((20, 10), dtype=np.uint8)

# Crear clase agente
agente = AgenteTetris()
movimiento_encontrado = False

if not cam.isOpened():
    print("No se pude abrir la camara")
    exit()
while True:
    ret, frame = cam.read()
    if not ret:
        print("No se puede recibir la imagen")
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    
    # Umbralizar
    ret, frame_umbral = cv2.threshold(frame, 20, 255, cv2.THRESH_BINARY)
    
    # Bordes Canny
    frame1 = cv2.Canny(blur, 50, 150)
    
    # Dilate para que los bordes de pieza se vean como uno solo
    kernel = np.ones((3, 3), np.uint8)
    frame1 = cv2.dilate(frame1, kernel, iterations=1)
    
    tablero, tablero_umbral = detectar_tablero(frame1, frame_umbral)
    
    if tablero_umbral is not None:
        matriz_estado = matriz_umbral(tablero_umbral)

    # Detectar reinicio del juego (si la cámara ve muchos menos bloques de los que recordamos)
    if int(np.sum(tablero_fijo)) > int(np.sum(matriz_estado)) + 10:
        tablero_fijo = np.zeros((20, 10), dtype=np.uint8)
    pieza_activa = np.logical_and(matriz_estado, np.logical_not(tablero_fijo)).astype(np.uint8)
    print(tablero_fijo)
   
    if pieza_cayo(pieza_activa, tablero_fijo):
        # Usamos logical_or para solo agregar los 4 bloques exactos de la pieza que cayó,
        # evitando copiar "glitches" (celdas faltantes) de la cámara en ese instante.
        tablero_fijo = np.logical_or(tablero_fijo, pieza_activa).astype(np.uint8)
         # Limpiar matemáticamente las líneas completas en tablero_fijo para adelantarnos a la animación
        filas_buenas = [i for i in range(20) if not np.all(tablero_fijo[i, :] == 1)]
        lineas_borradas = 20 - len(filas_buenas)
        if lineas_borradas > 0:
            tablero_fijo = np.vstack((np.zeros((lineas_borradas, 10), dtype=np.uint8), tablero_fijo[filas_buenas]))
        agente.pieza_fijada()
        movimiento_encontrado = False
    else:        
        tipo_pieza = determinar_tipo_pieza(pieza_activa)

        if tipo_pieza is not None and not movimiento_encontrado:
            print(tipo_pieza)

            movimiento = agente.decidir_movimiento(
                tablero_fijo,
                tipo_pieza
            )

            if movimiento is not None:
                movimiento_encontrado = True
                mensaje = json.dumps(movimiento) + "\n"
                client.sendall(mensaje.encode())

                print("Enviado:", movimiento)
    
    matriz_anterior = matriz_estado.copy()

    # Mostrar imagen
    cv2.imshow('WebCam Kanny', frame1)
    
    # Mostrar tablero (si hay)
    if tablero is not None:
        cv2.imshow('Tablero', tablero)
        cv2.imshow('Tablero umbral', tablero_umbral)
        
    if cv2. waitKey(1) == ord('q'):
        break
    
cam.release()
cv2.destroyAllWindows()
