import cv2
import time
import numpy as np

""" CONSTANTES GLOBALES """
PIEZAS = {
    "O": {(0,0),(0,1),(1,0),(1,1)},
    "I": {(0,0),(0,1),(0,2),(0,3)},
    "T": {(0,0),(0,1),(0,2),(1,1)},
    "L": {(0,0),(1,0),(2,0),(2,1)},
    "J": {(0,1),(1,1),(2,1),(2,0)},
    "S": {(0,1),(0,2),(1,0),(1,1)},
    "Z": {(0,0),(0,1),(1,1),(1,2)}
}

ultimo_movimiento = time.time()

matriz_anterior = np.zeros((20, 10), dtype=np.uint8)

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

"""

DEVUELVE EL TAMAÑO REAL DE CADA CELDA * revisar si se va a utilizar
def inferir_tamano_celda(img_rectificada):
    gris = img_rectificada

    # Proyección horizontal → detectar filas
    proj_h = gris.mean(axis=1)
    # Proyección vertical → detectar columnas
    proj_v = gris.mean(axis=0)

    h_celda = detectar_periodo(proj_h)
    w_celda = detectar_periodo(proj_v)

    # Fallback si no se detecta período
    if h_celda is None:
        h_celda = img_rectificada.shape[0] // 20
    if w_celda is None:
        w_celda = img_rectificada.shape[1] // 10

    return int(h_celda), int(w_celda)

ENCUENTRA EL PERIODO DE LA GRILLA * revisar si se va a utilizar
def detectar_periodo(senal):
    senal = senal - senal.mean()
    autocorr = np.correlate(senal, senal, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # Mitad positiva

    # Buscar el primer pico después de lag = 3
    from scipy.signal import find_peaks
    picos, _ = find_peaks(autocorr, distance=3)

    if len(picos) == 0:
        return None

    # Período en píxeles
    return picos[0]

"""

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
def pieza_cayo():
    if time.time() - ultimo_movimiento > 0.5:
        return True
    
    return False

""" CICLO PRINCIPAL """
cam = cv2.VideoCapture(1)

tablero_fijo = np.zeros((20, 10), dtype=np.uint8)

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
    matriz_estado = matriz_umbral(tablero_umbral)

    pieza_activa = np.logical_and(matriz_estado, np.logical_not(tablero_fijo)).astype(np.uint8)

    if not np.array_equal(matriz_estado, matriz_anterior):
        ultimo_movimiento = time.time()

    if pieza_cayo():
        tablero_fijo = matriz_estado.copy()
    else:
        print(determinar_tipo_pieza(pieza_activa))
    
    matriz_anterior = matriz_estado.copy()

    # Mostrar imagen
    cv2.imshow('WebCam Kanny', frame1)
    
    # Mostrar tablero (si hay)
    if tablero is not None:
        cv2.imshow('Tablero', tablero)
        cv2.imshow('Tablero umbral', tablero_umbral)
        
        print(matriz_estado, "\n")
        
    if cv2. waitKey(1) == ord('q'):
        break
    
cam.release()
cv2.destroyAllWindows()