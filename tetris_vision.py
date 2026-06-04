"""
tetris_vision.py
================
Sistema de visión por computadora para extracción de estado de Tetris.

Pipeline:
  Frame de cámara
    → detectar_tablero()       — localiza y rectifica el tablero
    → inferir_tamano_celda()   — mide el tamaño real de cada celda
    → SeparadorPieza.actualizar() — separa pieza activa del tablero fijo
    → clasificar_pieza()       — identifica tipo y orientación
    → construir_estado()       — empaqueta el estado para el agente

Uso:
  python tetris_vision.py [--camara 0] [--debug]

Dependencias:
  pip install opencv-python numpy scipy
"""

import cv2
import numpy as np
import argparse
import json
import time
from scipy.signal import find_peaks

# ──────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────

FILAS = 20
COLS  = 10

# Tamaño canónico de la imagen rectificada (píxeles)
# 20px por celda → 200×400
PX_CELDA = 20
IMG_W = COLS  * PX_CELDA   # 200
IMG_H = FILAS * PX_CELDA   # 400

# Saturación mínima (HSV) para considerar una celda ocupada
MIN_SATURACION = 50

# Porcentaje mínimo de píxeles saturados dentro de una celda
MIN_OCUPACION = 0.30

# Umbral de diferencia entre frames para detectar movimiento
UMBRAL_DIFF = 20

# Filas superiores donde puede aparecer la pieza activa (fallback)
FILAS_ZONA_ACTIVA = 6

# Aspect ratio esperado del tablero (alto/ancho) con tolerancia
ASPECT_MIN = 1.6
ASPECT_MAX = 2.6

# Margen interior al samplear cada celda (evita bordes)
MARGEN_CELDA = 2

# ──────────────────────────────────────────────
# DEFINICIÓN DE PIEZAS Y ROTACIONES
# ──────────────────────────────────────────────

PIEZAS_ROTACIONES = {
    'I': [
        ((0,0),(0,1),(0,2),(0,3)),
        ((0,0),(1,0),(2,0),(3,0)),
    ],
    'O': [
        ((0,0),(0,1),(1,0),(1,1)),
    ],
    'T': [
        ((0,1),(1,0),(1,1),(1,2)),
        ((0,0),(1,0),(1,1),(2,0)),
        ((0,0),(0,1),(0,2),(1,1)),
        ((0,1),(1,0),(1,1),(2,1)),
    ],
    'S': [
        ((0,1),(0,2),(1,0),(1,1)),
        ((0,0),(1,0),(1,1),(2,1)),
    ],
    'Z': [
        ((0,0),(0,1),(1,1),(1,2)),
        ((0,1),(1,0),(1,1),(2,0)),
    ],
    'J': [
        ((0,0),(1,0),(1,1),(1,2)),
        ((0,0),(0,1),(1,0),(2,0)),
        ((0,0),(0,1),(0,2),(1,2)),
        ((0,1),(1,1),(2,0),(2,1)),
    ],
    'L': [
        ((0,2),(1,0),(1,1),(1,2)),
        ((0,0),(1,0),(2,0),(2,1)),
        ((0,0),(0,1),(0,2),(1,0)),
        ((0,0),(0,1),(1,1),(2,1)),
    ],
}

# Índice invertido: forma_normalizada → (tipo, índice_rotación)
def _construir_indice():
    idx = {}
    for tipo, rotaciones in PIEZAS_ROTACIONES.items():
        for i, rot in enumerate(rotaciones):
            idx[_normalizar(rot)] = (tipo, i)
    return idx

def _normalizar(celdas):
    """Traslada celdas para que empiecen en (0,0) y las ordena."""
    celdas = list(celdas)
    min_f = min(c[0] for c in celdas)
    min_c = min(c[1] for c in celdas)
    return tuple(sorted((f - min_f, c - min_c) for f, c in celdas))

INDICE_PIEZAS = _construir_indice()

# ──────────────────────────────────────────────
# ETAPA 1 — DETECTAR Y RECTIFICAR EL TABLERO
# ──────────────────────────────────────────────

def ordenar_esquinas(pts):
    """
    Ordena 4 puntos en el orden: TL, TR, BR, BL.
    """
    tl = min(pts, key=lambda p: p[0] + p[1])
    br = max(pts, key=lambda p: p[0] + p[1])
    tr = max(pts, key=lambda p: p[0] - p[1])
    bl = min(pts, key=lambda p: p[0] - p[1])
    return np.float32([tl, tr, br, bl])


def detectar_tablero(frame, M_cache=None):
    """
    Localiza el tablero Tetris en el frame y lo rectifica a IMG_W × IMG_H.

    Parámetros
    ----------
    frame    : imagen BGR capturada por la cámara
    M_cache  : última homografía válida (fallback si falla la detección)

    Retorna
    -------
    img_rect : imagen rectificada (IMG_W × IMG_H) o None si falla todo
    M        : matriz de homografía 3×3
    """
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Dilatar bordes para cerrar posibles gaps
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidatos = []
    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if area < 5000:
            continue

        peri  = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            if w == 0:
                continue
            aspect = h / w
            if ASPECT_MIN < aspect < ASPECT_MAX:
                candidatos.append((area, approx))

    if not candidatos:
        # Fallback: reusar la última homografía válida
        if M_cache is not None:
            img_rect = cv2.warpPerspective(frame, M_cache, (IMG_W, IMG_H))
            return img_rect, M_cache
        return None, None

    _, mejor = max(candidatos, key=lambda x: x[0])
    pts = mejor.reshape(4, 2).astype(np.float32)
    pts = ordenar_esquinas(pts)

    dst = np.float32([[0, 0], [IMG_W, 0], [IMG_W, IMG_H], [0, IMG_H]])
    M   = cv2.getPerspectiveTransform(pts, dst)
    img_rect = cv2.warpPerspective(frame, M, (IMG_W, IMG_H))

    return img_rect, M

# ──────────────────────────────────────────────
# ETAPA 2 — INFERIR TAMAÑO DE CELDA
# ──────────────────────────────────────────────

def _detectar_periodo(senal):
    """
    Usa autocorrelación para encontrar el período (en píxeles)
    de una señal periódica (proyección de la grilla).
    Retorna None si no se encuentra período confiable.
    """
    senal = senal.astype(float) - senal.mean()
    autocorr = np.correlate(senal, senal, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]   # mitad positiva

    picos, props = find_peaks(autocorr, distance=3, prominence=10)
    if len(picos) == 0:
        return None

    # El primer pico significativo es el período
    return int(picos[0])


def inferir_tamano_celda(img_rect):
    """
    Mide el tamaño real de cada celda a partir de la periodicidad
    de la grilla en la imagen rectificada.

    Retorna (h_celda, w_celda) en píxeles.
    Si la detección falla, usa el valor nominal (IMG_H/FILAS, IMG_W/COLS).
    """
    gris = cv2.cvtColor(img_rect, cv2.COLOR_BGR2GRAY)

    proj_h = gris.mean(axis=1)   # proyección horizontal → período vertical
    proj_v = gris.mean(axis=0)   # proyección vertical   → período horizontal

    h_celda = _detectar_periodo(proj_h) or (IMG_H // FILAS)
    w_celda = _detectar_periodo(proj_v) or (IMG_W // COLS)

    # Sanidad: los valores deben estar cerca del nominal
    nominal_h = IMG_H // FILAS
    nominal_w = IMG_W // COLS
    if not (nominal_h * 0.5 < h_celda < nominal_h * 1.5):
        h_celda = nominal_h
    if not (nominal_w * 0.5 < w_celda < nominal_w * 1.5):
        w_celda = nominal_w

    return int(h_celda), int(w_celda)

# ──────────────────────────────────────────────
# ETAPA 3 — EXTRACCIÓN DE CELDAS OCUPADAS
# ──────────────────────────────────────────────

def extraer_celdas_ocupadas(img_rect, h_celda, w_celda):
    """
    Retorna un set de (fila, col) de celdas que tienen contenido visible
    (saturación alta en HSV).
    """
    hsv = cv2.cvtColor(img_rect, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(float)

    celdas = set()
    for f in range(FILAS):
        for c in range(COLS):
            y0 = f * h_celda + MARGEN_CELDA
            y1 = (f + 1) * h_celda - MARGEN_CELDA
            x0 = c * w_celda + MARGEN_CELDA
            x1 = (c + 1) * w_celda - MARGEN_CELDA

            if y1 <= y0 or x1 <= x0:
                continue

            region = sat[y0:y1, x0:x1]
            ocupacion = (region > MIN_SATURACION).mean()
            if ocupacion > MIN_OCUPACION:
                celdas.add((f, c))

    return celdas

# ──────────────────────────────────────────────
# ETAPA 4 — SEPARAR PIEZA ACTIVA DEL TABLERO FIJO
# ──────────────────────────────────────────────

class SeparadorPieza:
    """
    Mantiene el estado entre frames para separar la pieza activa
    del tablero fijo mediante diferencia temporal.
    """

    def __init__(self):
        self._tablero_acumulado = np.zeros((FILAS, COLS), dtype=int)
        self._frames_sin_cambio = 0
        self._ultima_pieza      = set()

    def actualizar(self, img_actual, img_anterior, h_celda, w_celda):
        """
        Compara dos frames rectificados consecutivos.

        Retorna
        -------
        celdas_pieza   : set de (f,c) que corresponden a la pieza activa
        celdas_tablero : set de (f,c) del tablero fijo
        """
        # Diferencia entre frames
        diff      = cv2.absdiff(img_actual, img_anterior)
        diff_gris = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, diff_bin = cv2.threshold(diff_gris, UMBRAL_DIFF, 255, cv2.THRESH_BINARY)

        # Qué celdas cambiaron
        celdas_cambiaron = set()
        for f in range(FILAS):
            for c in range(COLS):
                y0 = f * h_celda + MARGEN_CELDA
                y1 = (f + 1) * h_celda - MARGEN_CELDA
                x0 = c * w_celda + MARGEN_CELDA
                x1 = (c + 1) * w_celda - MARGEN_CELDA
                if y1 <= y0 or x1 <= x0:
                    continue
                region = diff_bin[y0:y1, x0:x1]
                if region.mean() > 10:
                    celdas_cambiaron.add((f, c))

        # Celdas ocupadas en el frame actual
        celdas_ocupadas = extraer_celdas_ocupadas(img_actual, h_celda, w_celda)

        # Pieza = ocupadas que cambiaron
        celdas_pieza   = celdas_ocupadas & celdas_cambiaron

        # Tablero fijo = ocupadas que NO cambiaron
        celdas_tablero = celdas_ocupadas - celdas_cambiaron

        # ── Fallback A: pieza no se movió este frame ──
        if len(celdas_pieza) != 4 and len(self._ultima_pieza) == 4:
            # Verificar si la última pieza sigue estando en el frame
            if self._ultima_pieza.issubset(celdas_ocupadas):
                celdas_pieza   = self._ultima_pieza
                celdas_tablero = celdas_ocupadas - celdas_pieza
                self._frames_sin_cambio += 1
            else:
                self._frames_sin_cambio = 0
        else:
            self._frames_sin_cambio = 0

        # ── Fallback B: buscar grupo de 4 en zona superior ──
        if len(celdas_pieza) != 4:
            celdas_pieza = self._fallback_zona_superior(
                celdas_ocupadas, celdas_tablero
            )
            celdas_tablero = celdas_ocupadas - celdas_pieza

        # Actualizar acumulado del tablero fijo
        for f, c in celdas_tablero:
            self._tablero_acumulado[f][c] = 1

        # Si se detectó una línea completa (fila llena), limpiarla del acumulado
        self._limpiar_lineas_completas()

        self._ultima_pieza = celdas_pieza
        return celdas_pieza, celdas_tablero

    def _fallback_zona_superior(self, celdas_ocupadas, celdas_tablero):
        """
        Busca en las filas superiores un grupo de 4 celdas conectadas
        que no pertenezcan al tablero fijo acumulado.
        """
        candidatas = {
            (f, c) for f, c in celdas_ocupadas
            if f < FILAS_ZONA_ACTIVA and self._tablero_acumulado[f][c] == 0
        }

        if len(candidatas) == 0:
            return set()

        # BFS para encontrar componentes conectadas
        visitadas = set()
        componentes = []

        def bfs(inicio):
            cola = [inicio]
            comp = set()
            while cola:
                nodo = cola.pop()
                if nodo in visitadas:
                    continue
                visitadas.add(nodo)
                comp.add(nodo)
                f, c = nodo
                for df, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    vecino = (f+df, c+dc)
                    if vecino in candidatas and vecino not in visitadas:
                        cola.append(vecino)
            return comp

        for celda in candidatas:
            if celda not in visitadas:
                comp = bfs(celda)
                componentes.append(comp)

        if not componentes:
            return set()

        # La componente más grande (idealmente 4 celdas) es la pieza
        mejor = max(componentes, key=len)
        return mejor

    def _limpiar_lineas_completas(self):
        """
        Detecta filas llenas en el acumulado (evento de line clear)
        y las elimina, bajando todo lo de arriba.
        """
        filas_llenas = [
            f for f in range(FILAS)
            if self._tablero_acumulado[f].sum() == COLS
        ]
        for f in filas_llenas:
            self._tablero_acumulado[1:f+1] = self._tablero_acumulado[0:f]
            self._tablero_acumulado[0] = 0

    def reset(self):
        self._tablero_acumulado  = np.zeros((FILAS, COLS), dtype=int)
        self._frames_sin_cambio  = 0
        self._ultima_pieza       = set()

# ──────────────────────────────────────────────
# ETAPA 5 — CLASIFICAR PIEZA Y ORIENTACIÓN
# ──────────────────────────────────────────────

def clasificar_pieza(celdas_pieza):
    """
    Identifica el tipo de pieza y su orientación comparando la forma
    normalizada contra el índice de piezas conocidas.

    Retorna
    -------
    tipo        : 'I','O','T','S','Z','J','L'  o  None
    orientacion : 0=norte, 1=este, 2=sur, 3=oeste  o  None
    """
    if len(celdas_pieza) != 4:
        return None, None

    forma = _normalizar(celdas_pieza)
    return INDICE_PIEZAS.get(forma, (None, None))

# ──────────────────────────────────────────────
# ETAPA 6 — CONSTRUIR ESTADO PARA EL AGENTE
# ──────────────────────────────────────────────

def construir_estado(celdas_tablero, celdas_pieza, tipo, orientacion):
    """
    Empaqueta toda la información extraída en un dict listo para el agente.

    Retorna
    -------
    {
      "tablero"         : lista 20×10 de 0/1  (0=vacío, 1=ocupado),
      "pieza"           : str o None,
      "orientacion"     : int o None,
      "posicion_pieza"  : lista de [fila, col] con las 4 celdas absolutas
    }
    """
    tablero_matriz = np.zeros((FILAS, COLS), dtype=int)
    for f, c in celdas_tablero:
        if 0 <= f < FILAS and 0 <= c < COLS:
            tablero_matriz[f][c] = 1

    return {
        "tablero"        : tablero_matriz.tolist(),
        "pieza"          : tipo,
        "orientacion"    : orientacion,
        "posicion_pieza" : sorted([list(p) for p in celdas_pieza]),
    }

# ──────────────────────────────────────────────
# AGENTE PLACEHOLDER
# ──────────────────────────────────────────────

def agente_decide(estado):
    """
    Placeholder del agente decisor.
    Recibe el estado y retorna una acción.

    Acciones posibles:
      'rotar'         — girar la pieza 90° en sentido horario
      'mover_der'     — mover la pieza una celda a la derecha
      'mover_izq'     — mover la pieza una celda a la izquierda
      'bajar'         — bajar la pieza una fila
      'bajar_rapido'  — bajar la pieza al fondo (hard drop)
      'nada'          — no hacer nada este frame

    Reemplaza este método con tu agente real (reglas, RL, etc.)
    """
    # Ejemplo trivial: siempre moverse a la derecha
    return 'nada'

# ──────────────────────────────────────────────
# VISUALIZACIÓN DEBUG
# ──────────────────────────────────────────────

def dibujar_debug(img_rect, celdas_pieza, celdas_tablero, h_celda, w_celda,
                  tipo, orientacion, accion):
    """
    Dibuja la imagen rectificada con las celdas coloreadas para debug.
    """
    vis = img_rect.copy()

    # Tablero fijo → azul semitransparente
    overlay = vis.copy()
    for f, c in celdas_tablero:
        x0, y0 = c * w_celda, f * h_celda
        cv2.rectangle(overlay, (x0, y0), (x0+w_celda, y0+h_celda),
                      (180, 60, 60), -1)
    cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)

    # Pieza activa → verde
    for f, c in celdas_pieza:
        x0, y0 = c * w_celda, f * h_celda
        cv2.rectangle(vis, (x0+2, y0+2), (x0+w_celda-2, y0+h_celda-2),
                      (0, 230, 80), 2)

    # Grilla
    for f in range(FILAS + 1):
        cv2.line(vis, (0, f*h_celda), (IMG_W, f*h_celda), (60,60,60), 1)
    for c in range(COLS + 1):
        cv2.line(vis, (c*w_celda, 0), (c*w_celda, IMG_H), (60,60,60), 1)

    # Texto de estado
    texto = f"Pieza: {tipo or '?'}  Rot: {orientacion}  Accion: {accion}"
    cv2.putText(vis, texto, (4, IMG_H - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1, cv2.LINE_AA)

    return vis

# ──────────────────────────────────────────────
# LOOP PRINCIPAL
# ──────────────────────────────────────────────

def imprimir_tablero_ascii(estado):
    """Imprime el estado en la terminal de forma legible."""
    import os
    os.system('cls' if os.name == 'nt' else 'clear')

    pieza       = estado["pieza"] or "?"
    orientacion = estado["orientacion"]
    posicion    = {tuple(p) for p in estado["posicion_pieza"]}
    tablero     = estado["tablero"]

    print(f"  Pieza: {pieza}   Orientación: {orientacion}")
    print("  ┌" + "──" * COLS + "┐")
    for f in range(FILAS):
        fila = "  │"
        for c in range(COLS):
            if (f, c) in posicion:
                fila += "██"      # pieza activa
            elif tablero[f][c]:
                fila += "░░"      # tablero fijo
            else:
                fila += "  "      # vacío
        fila += "│"
        print(fila)
    print("  └" + "──" * COLS + "┘")
    print(f"  Celdas pieza: {sorted(posicion)}")

def main(camara=1, debug=False, guardar_estado=False):
    """
    Loop principal del sistema de visión.

    Parámetros
    ----------
    camara        : índice de la cámara (default 0)
    debug         : mostrar ventana con visualización
    guardar_estado: imprimir el estado JSON en stdout cada frame
    """
    cap = cv2.VideoCapture(camara)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara {camara}")

    separador  = SeparadorPieza()
    img_ant    = None
    M_cache    = None
    h_celda    = IMG_H // FILAS
    w_celda    = IMG_W // COLS

    print("[INFO] Sistema iniciado. Presiona 'q' para salir, 'r' para resetear.")

    fps_tiempo  = time.time()
    fps_frames  = 0
    FPS_OBJETIVO = 3
    INTERVALO    = 1.0 / FPS_OBJETIVO  # 0.333 seg entre frames
    ultimo_frame = 0.0

    while True:
        ahora = time.time()
        if ahora - ultimo_frame < INTERVALO:
            cv2.waitKey(1)
            continue
        ultimo_frame = ahora
        
        ret, frame = cap.read()
        if not ret:
            print("[WARN] No se pudo leer frame de la cámara.")
            continue

        # ── Etapa 1: detectar y rectificar tablero ──
        img_rect, M = detectar_tablero(frame, M_cache)
        if img_rect is None:
            if debug:
                cv2.imshow("Tetris Vision", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        M_cache = M

        # ── Etapa 2: inferir tamaño de celda ──
        h_celda, w_celda = inferir_tamano_celda(img_rect)

        # ── Etapas 3-4: separar pieza y tablero ──
        if img_ant is not None:
            celdas_pieza, celdas_tablero = separador.actualizar(
                img_rect, img_ant, h_celda, w_celda
            )
        else:
            # Primer frame: solo tablero, sin pieza
            celdas_pieza   = set()
            celdas_tablero = extraer_celdas_ocupadas(img_rect, h_celda, w_celda)

        # ── Etapa 5: clasificar pieza ──
        tipo, orientacion = clasificar_pieza(celdas_pieza)

        # ── Etapa 6: construir estado ──
        estado = construir_estado(celdas_tablero, celdas_pieza, tipo, orientacion)
        
        if args.inspect:
            imprimir_tablero_ascii(estado)

        # ── Agente ──
        accion = agente_decide(estado)

        # ── Salida ──
        if guardar_estado:
            print(json.dumps({
                "pieza"          : estado["pieza"],
                "orientacion"    : estado["orientacion"],
                "posicion_pieza" : estado["posicion_pieza"],
                "tablero"        : estado["tablero"],
                "accion"         : accion,
            }))

        # ── Debug visual ──
        if debug:
            vis = dibujar_debug(img_rect, celdas_pieza, celdas_tablero,
                                h_celda, w_celda, tipo, orientacion, accion)

            # FPS
            fps_frames += 1
            if time.time() - fps_tiempo >= 1.0:
                fps = fps_frames / (time.time() - fps_tiempo)
                fps_tiempo  = time.time()
                fps_frames  = 0
                cv2.setWindowTitle("Tetris Vision", f"Tetris Vision — {fps:.1f} FPS")

            cv2.imshow("Tetris Vision", vis)

        img_ant = img_rect.copy()

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            separador.reset()
            img_ant = None
            print("[INFO] Estado reseteado.")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Sistema detenido.")


# ──────────────────────────────────────────────
# ENTRADA
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tetris Vision System")
    parser.add_argument("--camara", type=int, default=0,
                        help="Índice de la cámara (default: 0)")
    parser.add_argument("--debug", action="store_true",
                        help="Mostrar ventana de visualización")
    parser.add_argument("--json", action="store_true",
                        help="Imprimir estado JSON en stdout cada frame")
    parser.add_argument("--inspect", action="store_true",
                        help="Mostrar tablero ASCII en terminal")
    args = parser.parse_args()

    main(camara=args.camara, debug=args.debug, guardar_estado=args.json)