import numpy as np
import math

"""
Esta función genera un vector de pasos aleatorios siguiendo la distribución de Lévy. 
Este tipo de pasos se utiliza en algoritmos metaheurísticos para introducir saltos 
largos en el espacio de búsqueda,lo que ayuda a mejorar la exploración y a escapar de óptimos locales.
El vector resultante tiene una dimensión definida por el parámetro de entrada y contiene los valores 
correspondientes a los pasos Lévy, los cuales se emplean para actualizar soluciones dentro del proceso de
optimización.
"""

def generarPasoLevy(dimension):

    beta = 1.5

    numerador = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
    denominador = math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma = (numerador / denominador) ** (1 / beta)

    vectorU = np.random.randn(dimension) * sigma
    vectorV = np.random.randn(dimension)

    pasosLevy = vectorU / (np.abs(vectorV) ** (1 / beta))
    return pasosLevy
