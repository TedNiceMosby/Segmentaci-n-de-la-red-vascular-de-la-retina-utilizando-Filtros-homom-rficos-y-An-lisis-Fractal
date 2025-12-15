import numpy as np


#%% CALCULAR LA MEDIA 
"""
Esta función calcula la media ponderada de una porción específica del histograma, recorriendo los niveles de 
intensidad desde indiceInicio hasta indiceFin - 1. En este cálculo, cada índice se utiliza como peso y
el valor correspondiente del histograma como la probabilidad asociada.
El resultado es una media ponderada que representa el valor promedio de intensidad dentro del intervalo definido. 
Si la suma de las probabilidades en dicho intervalo es cero, la función devuelve 0.0 para evitar divisiones indefinidas.
"""

def calcularMediaPonderada(indiceInicio, indiceFin, histograma):

    sumaPesos = 0.0
    sumaFrecuencias = 0.0

    for nivel in range(indiceInicio, indiceFin):
        sumaPesos += nivel * histograma[nivel]
        sumaFrecuencias += histograma[nivel]

    if sumaPesos == 0 and sumaFrecuencias == 0:
        return 0.0

    return sumaPesos / sumaFrecuencias
