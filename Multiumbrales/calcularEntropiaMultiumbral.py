import numpy as np



#%% ENTROPIA CRUZADA
"""
Esta función calcula la medida MCEM (criterio de entropía) para un conjunto de umbrales propuestos. A partir de una imagen en escala de
grises y su histograma, el histograma se divide en distintas clases definidas por los umbrales, y sobre cada una de ellas se calcula una
medida de entropía (Dt)
La medida obtenida se utiliza como función objetivo a minimizar por el algoritmo HHO, ya que valores más pequeños de 
Dt indican una mejor calidad en la umbralización y permiten encontrar los umbrales óptimos para la segmentación de la imagen.
"""

def calcularEntropiaMultiumbral(imagen, umbrales, histograma):

    alto, ancho = imagen.shape[:2]
    histogramaNormalizado = histograma.astype(np.float64) / (alto * ancho)

    listaUmbrales = list(sorted(int(t) for t in umbrales))
    numeroUmbrales = len(listaUmbrales)

    listaEntropias = []
    listaMedias = []

    for indiceClase in range(numeroUmbrales + 1):# Recorremos cada clase definida por los umbrales
        if indiceClase == 0:
            valorInferior = 0
            valorSuperior = listaUmbrales[indiceClase] - 1
        elif indiceClase == numeroUmbrales:
            valorInferior = listaUmbrales[indiceClase - 1]
            valorSuperior = 255
        else:
            valorInferior = listaUmbrales[indiceClase - 1]
            valorSuperior = listaUmbrales[indiceClase] - 1

        sumaPonderada = 0.0
        sumaProbabilidades = 0.0
        for nivel in range(valorInferior, valorSuperior + 1):
            sumaPonderada += nivel * histogramaNormalizado[nivel]
            sumaProbabilidades += histogramaNormalizado[nivel]

        if sumaProbabilidades == 0:
            mediaClase = sumaPonderada / (sumaProbabilidades + np.finfo(float).eps)
        else:
            mediaClase = sumaPonderada / sumaProbabilidades

        listaMedias.append(mediaClase)

        entropiaClase = 0.0
        for nivel in range(valorInferior, valorSuperior + 1):
            if mediaClase == 0:
                entropiaClase += (
                    nivel * histogramaNormalizado[nivel] *
                    np.log(mediaClase + np.finfo(float).eps)
                )
            else:
                entropiaClase += nivel * histogramaNormalizado[nivel] * np.log(mediaClase)

        listaEntropias.append(entropiaClase)

    entropiaImagen = 0.0 # Entropía de la imagen completa
    for nivel in range(256):
        if nivel == 0:
            continue
        entropiaImagen += nivel * histogramaNormalizado[nivel] * np.log(nivel)

    valorDt = entropiaImagen - sum(listaEntropias)
    return float(valorDt)
