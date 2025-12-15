import numpy as np
from scipy.signal import convolve2d


#%% CALCULA FSIM

"""
Esta función calcula el índice de similitud de características (FSIM) entre dos imágenes. A partir de una 
imagen de referencia y una imagen distorsionada, ya sea en escala de grises (0–255) o en formato RGB
(0–255), se estiman características perceptuales relevantes para medir
qué tan similares son ambas imágenes.
En particular, se calcula el mapa de fase de congruencia y el mapa de gradiente, y luego se combinan 
estas dos características para producir un valor escalar de similitud. Mientras más alto sea el valor de FSIM,
más parecidas se consideran las imágenes.
"""


def calcularSimilitudFSIM(imagenReferencia, imagenDistorsion):
    # Detectar si la imagen es en escala de grises o RGB
    if imagenReferencia.ndim == 2:
        canalLuminanciaRef = imagenReferencia.astype(float)
        canalLuminanciaDis = imagenDistorsion.astype(float)
        esEscalaGrises = True
    else:
        # RGB
        imagenRefFloat = imagenReferencia.astype(float)
        imagenDisFloat = imagenDistorsion.astype(float)

        # Conversión aproximada a espacio YIQ (luminancia y crominancias)
        canalLuminanciaRef = (
            0.299 * imagenRefFloat[:, :, 0] +
            0.587 * imagenRefFloat[:, :, 1] +
            0.114 * imagenRefFloat[:, :, 2]
        )
        canalLuminanciaDis = (
            0.299 * imagenDisFloat[:, :, 0] +
            0.587 * imagenDisFloat[:, :, 1] +
            0.114 * imagenDisFloat[:, :, 2]
        )

        canalIRef = (
            0.596 * imagenRefFloat[:, :, 0] -
            0.274 * imagenRefFloat[:, :, 1] -
            0.322 * imagenRefFloat[:, :, 2]
        )
        canalIDis = (
            0.596 * imagenDisFloat[:, :, 0] -
            0.274 * imagenDisFloat[:, :, 1] -
            0.322 * imagenDisFloat[:, :, 2]
        )
        canalQRef = (
            0.211 * imagenRefFloat[:, :, 0] -
            0.523 * imagenRefFloat[:, :, 1] +
            0.312 * imagenRefFloat[:, :, 2]
        )
        canalQDis = (
            0.211 * imagenDisFloat[:, :, 0] -
            0.523 * imagenDisFloat[:, :, 1] +
            0.312 * imagenDisFloat[:, :, 2]
        )
        esEscalaGrises = False

    filas, columnas = canalLuminanciaRef.shape
    factorSubmuestreo = max(1, round(min(filas, columnas) / 256))

    def submuestreoPromedio(matriz):#Reduce la resolución de la matriz promediando bloques de tamaño F x F.
        kernelPromedio = np.ones((factorSubmuestreo, factorSubmuestreo)) / (factorSubmuestreo ** 2)
        matrizSuavizada = convolve2d(matriz, kernelPromedio, mode='same', boundary='symm')
        return matrizSuavizada[0:filas:factorSubmuestreo, 0:columnas:factorSubmuestreo]

    canalLuminanciaRef = submuestreoPromedio(canalLuminanciaRef)
    canalLuminanciaDis = submuestreoPromedio(canalLuminanciaDis)

    # Fase de congruencia (todavía falta traducir phasecong2)
    faseCongruenciaRef = phasecong2(canalLuminanciaRef)
    faseCongruenciaDis = phasecong2(canalLuminanciaDis)

    # Gradientes (filtros tipo Sobel modificados)
    kernelDx = np.array([[3, 0, -3],
                         [10, 0, -10],
                         [3, 0, -3]]) / 16.0
    kernelDy = np.array([[3, 10, 3],
                         [0, 0, 0],
                         [-3, -10, -3]]) / 16.0

    gradienteXRef = convolve2d(canalLuminanciaRef, kernelDx, mode='same', boundary='symm')
    gradienteYRef = convolve2d(canalLuminanciaRef, kernelDy, mode='same', boundary='symm')
    magnitudGradienteRef = np.sqrt(gradienteXRef ** 2 + gradienteYRef ** 2)

    gradienteXDis = convolve2d(canalLuminanciaDis, kernelDx, mode='same', boundary='symm')
    gradienteYDis = convolve2d(canalLuminanciaDis, kernelDy, mode='same', boundary='symm')
    magnitudGradienteDis = np.sqrt(gradienteXDis ** 2 + gradienteYDis ** 2)

    # Constantes de estabilización
    constantePC = 0.85
    constanteGradiente = 160

    # Similitud de fase de congruencia
    similitudPC = (
        2 * faseCongruenciaRef * faseCongruenciaDis + constantePC
    ) / (faseCongruenciaRef ** 2 + faseCongruenciaDis ** 2 + constantePC)

    # Similitud de gradiente
    similitudGradiente = (
        2 * magnitudGradienteRef * magnitudGradienteDis + constanteGradiente
    ) / (magnitudGradienteRef ** 2 + magnitudGradienteDis ** 2 + constanteGradiente)

    faseCongruenciaMax = np.maximum(faseCongruenciaRef, faseCongruenciaDis)

    matrizSimilitud = similitudGradiente * similitudPC * faseCongruenciaMax
    indiceFsim = np.sum(matrizSimilitud) / np.sum(faseCongruenciaMax)

    if esEscalaGrises:
        return indiceFsim, indiceFsim

    return indiceFsim, indiceFsim


def phasecong2(imagen: np.ndarray) -> np.ndarray:
    raise NotImplementedError(
        "Pendiente."
    )
