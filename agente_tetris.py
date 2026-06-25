import numpy as np

PIEZAS = {
    "I": [
        np.array([[1, 1, 1, 1]]), # I horizontal
        np.array([[1], [1], [1], [1]]) # I vertical
    ],
    "O": [
        np.array([[1,1], [1,1]]) # Cuadrado (no rota)
    ],
    "T": [
        np.array([[1, 1, 1], [0, 1, 0]]), # Punta abajo
        np.array([[1, 0], [1, 1], [1, 0]]),  # Punta izquierda
        np.array([[0, 1, 0], [1, 1, 1]]), # Punta arriba
        np.array([[0, 1], [1, 1], [0, 1]]) # Punta derecha
    ],
    "L": [
        np.array([[1, 0], [1, 0], [1, 1]]), # L normal
        np.array([[0, 0, 1], [1, 1, 1]]), # L acostada
        np.array([[1, 1], [0, 1], [0, 1]]), # L invertida vertical
        np.array([[1, 1, 1], [1, 0, 0]])  # L acostada invertida
    ],
    "J": [
        np.array([[0, 1], [0, 1], [1, 1]]), # J normal
        np.array([[1, 1, 1], [0, 0, 1]]), # J acostada
        np.array([[1, 1], [1, 0], [1, 0]]), # J invertida vertical
        np.array([[1, 0, 0], [1, 1, 1]])  # J acostada invertida
    ],
    "S": [
        np.array([[0, 1, 1], [1, 1, 0]]), # S horizontal
        np.array([[1, 0], [1, 1], [0, 1]])  # S vertical
    ],
    "Z": [
        np.array([[1, 1, 0], [0, 1, 1]]), # Z horizontal
        np.array([[0, 1], [1, 1], [1, 0]])  # Z vertical
    ]
}

class AgenteTetris:

    """ CONSTRUCTOR """
    def __init__(self, pesos):
        if pesos is None:
            self.pesos = {
                "lineas": 10,
                "altura": 0.5,
                "huecos": 5,
                "rugosidad": 0.3
            }
        else:
            self.pesos = pesos

        self.esperando_pieza = True
        self.pieza_anterior = None
        self.contador = 0

    """ MODO ESPERA """
    def pieza_fijada(self):
        self.esperando_pieza = True

    """ ENCONTRAR MOVIMIENTO """
    def decidir_movimiento(self, tablero_fijo, tipo_pieza):

        # Si recibe el mismo tipo de pieza 5 veces la selecciona
        if (self.esperando_pieza):
            if (tipo_pieza == self.pieza_anterior):
                self.pieza_anterior = tipo_pieza
                self.contador += 1

                if (self.contador == 5):
                    self.esperando_pieza = False
                    self.contador = 0
            else:
                self.pieza_anterior = tipo_pieza

            return None
        else:
            mejor_score = -float('inf')
            mejor_movimiento = None
            
            # Probar todos los posibles movimientos y tomar el que maximice el score 
            for id_rotacion, rotacion in enumerate(PIEZAS[tipo_pieza]): 
                ancho_pieza = rotacion.shape[1]
                for columna in range(10 - ancho_pieza + 1):

                    # Simular la caída de la pieza en la columna seleccionada
                    tablero_resultante, lineas_eliminadas = self.simular_caida(tablero_fijo, rotacion, columna)
                    
                    # Evaluar el estado del tablero resultante
                    score = self.evaluar(lineas_eliminadas, tablero_resultante)
                    
                    if score > mejor_score:
                        mejor_score = score
                        mejor_movimiento = {
                            "rotacion": id_rotacion,
                            "columna": columna
                        }
                        
            return mejor_movimiento 
        
    """ SIMULAR CAIDA DE LA PIEZA Y ELIMINAR LINEAS """
    def simular_caida(self, tablero_fijo, rotacion, column):
        tablero_simulado = tablero_fijo.copy()
        fila = 0

        while not self.hay_colision(
            tablero_simulado,
            rotacion,
            fila + 1,
            column
        ):
            fila += 1

        alto = rotacion.shape[0]
        ancho = rotacion.shape[1]

        for r in range(alto):
            for c in range(ancho):

                if rotacion[r][c]:

                    tablero_simulado[fila + r][column + c] = 1

        filas_restantes = []
        lineas = 0

        for fila_tablero in tablero_simulado:

            if all(fila_tablero):
                lineas += 1
            else:
                filas_restantes.append(fila_tablero)

        while len(filas_restantes) < 20:
            filas_restantes.insert(0, [0]*10)

        return filas_restantes, lineas
    
    """" DETECTAR COLISIONES """
    def hay_colision(self, tablero, pieza, fila, columna):
        alto = pieza.shape[0]
        ancho = pieza.shape[1]

        for r in range(alto):
            for c in range(ancho):

                if pieza[r][c] == 0:
                    continue

                tablero_fila = fila + r
                tablero_col = columna + c

                # Sale del tablero
                if tablero_fila >= 20:
                    return True

                # Choca con bloque fijo
                if tablero[tablero_fila][tablero_col]:
                    return True

        return False
    
    """ FUNCIONES PARA CALCULAR EL SCORE """

    # Altura del tablero fijo (w2)
    def altura_agregada(self, tablero):
        total = 0

        for col in range(10):
            for fila in range(20):
                if tablero[fila][col]:
                    total += 20 - fila
                    break

        return total
    
    # Alturas de las columnas (para obtener rugosidad)
    def alturas_columnas(self, tablero):
        alturas = []

        for col in range(10):
            altura = 0
            for fila in range(20):
                if tablero[fila][col]:
                    altura = 20 - fila
                    break

            alturas.append(altura)

        return alturas

    # Contar huecos (w3)
    def contar_huecos(self, tablero):
        huecos = 0

        for col in range(10):
            bloque_encontrado = False

            for fila in range(20):
                if tablero[fila][col]:
                    bloque_encontrado = True
                elif bloque_encontrado:
                    huecos += 1

        return huecos
    
    # Calcular rugosidad (w4)
    def rugosidad(self, tablero):
        alturas = self.alturas_columnas(tablero)
        total = 0

        for i in range(9):
            total += abs(alturas[i] - alturas[i+1])

        return total
    
    """ CALCULAR SCORE """
    def evaluar(self, lineas_eliminadas, tablero):
        altura = self.altura_agregada(tablero)
        huecos = self.contar_huecos(tablero)
        rugosidad = self.rugosidad(tablero)

        p = self.pesos

        score = (p["lineas"] * lineas_eliminadas - p["altura"] * altura - p["huecos"] * huecos - p["rugosidad"] * rugosidad)

        return score