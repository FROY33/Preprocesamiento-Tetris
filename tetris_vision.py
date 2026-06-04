import cv2
import numpy as np

FILAS = 20
COLS = 10

PX_CELDA = 20
ASPECT_MIN = 1.6
ASPECT_MAX = 2.6

IMG_W = COLS  * PX_CELDA   # 200
IMG_H = FILAS * PX_CELDA   # 400

def detectar_tablero(frame, original):
    """
    Encuentra el bounding box del tablero Tetris en el frame.
    Retorna la región recortada y la transformación homográfica.
    Asume que el tablero tiene un borde visible o contraste claro.
    """
    
    # Buscar contornos grandes y rectangulares
    contornos, _ = cv2.findContours(frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    candidatos = []

    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if area < 20000 or area > 60000:  # ignorar contornos pequeños
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:  # es un cuadrilátero
            x, y, w, h = cv2.boundingRect(approx)
            aspect = h / w
            # print(aspect)
            # El tablero 10×20 tiene aspect ratio ~2.0
            if 1.8 < aspect < 2.2:
                candidatos.append((area, approx))
                

    if not candidatos:
        return None, None, None

    # Tomar el candidato de mayor área
    _, mejor = max(candidatos, key=lambda x: x[0])

    # Ordenar esquinas: top-left, top-right, bottom-right, bottom-left
    pts = mejor.reshape(4, 2).astype(np.float32)
    pts = ordenar_esquinas(pts)

    # Rectificar a imagen canónica 200×400 (20px por celda)
    W, H = 200, 400
    dst = np.float32([[0,0],[W,0],[W,H],[0,H]])
    M = cv2.getPerspectiveTransform(pts, dst)
    rectificado = cv2.warpPerspective(frame, M, (W, H))
    tablero_umbral = cv2.warpPerspective(original, M, (W, H))

    return rectificado, M, tablero_umbral

def ordenar_esquinas(pts):
    """Ordena 4 puntos: TL, TR, BR, BL."""
    centro = pts.mean(axis=0)
    def angulo(p):
        return np.arctan2(p[1]-centro[1], p[0]-centro[0])
    # pts_ord = sorted(pts, key=angulo)
    
    # arctan2 da orden: BR, TR, TL, BL → reordenar
    tl = min(pts, key=lambda p: p[0]+p[1])
    br = max(pts, key=lambda p: p[0]+p[1])
    tr = max(pts, key=lambda p: p[0]-p[1])
    bl = min(pts, key=lambda p: p[0]-p[1])
    return np.float32([tl, tr, br, bl])


def inferir_tamano_celda(img_rectificada):
    """
    Detecta las líneas de la grilla para medir el tamaño real de cada celda.
    Retorna (h_celda, w_celda).
    """
    gris = img_rectificada

    # Proyección horizontal → detectar filas
    proj_h = gris.mean(axis=1)  # promedio por fila de píxeles
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


def detectar_periodo(senal):
    """
    Usa autocorrelación para encontrar el período de la grilla.
    """
    senal = senal - senal.mean()
    autocorr = np.correlate(senal, senal, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # mitad positiva

    # Buscar el primer pico después de lag=3
    from scipy.signal import find_peaks
    picos, _ = find_peaks(autocorr, distance=3)

    if len(picos) == 0:
        return None

    return picos[0]  # período en píxeles
def matriz_umbral(tablero_umbral, h, w):
    """
    Recibe:
        hsv_img  : imagen en formato HSV (numpy array)
        h        : altura de cada celda
        w        : ancho de cada celda
        umbral_s : umbral para la saturación promedio

    Devuelve:
        matriz binaria (0 y 1) donde:
            1 -> saturación promedio > umbral_s
            0 -> saturación promedio <= umbral_s
    """

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

cam = cv2.VideoCapture(1)

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
    
    # Dilate para ver mejor
    kernel = np.ones((3, 3), np.uint8)
    frame1 = cv2.dilate(frame1, kernel, iterations = 1)
    
    tablero, M, tablero_umbral = detectar_tablero(frame1, frame_umbral)
    hCelda,wCelda = inferir_tamano_celda(tablero)
    matrizSat = matriz_umbral(tablero_umbral, hCelda, wCelda)
    
    # Mostrar imagen
    cv2.imshow('WebCam Kanny', frame1)
    cv2.imshow("umbral", tablero_umbral)    
    
    if tablero is not None:
        cv2.imshow('Tablero', tablero)
        cv2.imshow("umbral", tablero_umbral)
        
        print(matrizSat,"\n")
        
    if cv2. waitKey(1) == ord('q'):
        break
    
cam.release()
cv2.destroyAllWindows()