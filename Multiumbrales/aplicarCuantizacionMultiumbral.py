import numpy as np


"""
Esta función aplica un proceso de cuantización por rangos utilizando múltiples umbrales. La imagen en 
escala de grises se divide en varias clases de intensidad definidas por la lista de umbrales proporcionada.
Cada píxel de la imagen es asignado al rango correspondiente según su valor de intensidad, y se reemplaza por 
el límite inferior de dicho rango. El resultado es una imagen cuantizada que conserva el tamaño de la imagen original 
y resalta regiones según los intervalos definidos.
"""

def aplicarCuantizacionMultiumbral(imagen: np.ndarray, umbrales):
    
    umbralesOrdenados = sorted(int(t) for t in umbrales)# Asegurar orden de los umbrales
    limites = [0] + umbralesOrdenados + [255]

    imagenCuantizada = np.zeros_like(imagen, dtype=np.uint8)

    for indiceRango in range(len(limites) - 1):
        limiteInferior = limites[indiceRango]
        limiteSuperior = limites[indiceRango + 1]

        mascaraRango = (imagen >= limiteInferior) & (imagen <= limiteSuperior)
        imagenCuantizada[mascaraRango] = limiteInferior

    return imagenCuantizada
