import cv2
import numpy as np
from skimage.filters import threshold_multiotsu
import matplotlib.pyplot as plt

#%% OBTENER LA ZONA ROI

'''
Obtenemos la zona ROI (Region of Interest)
A partir de una imagen en escala de grises, crea una máscara que segmenta las regiones de 
interés que son normalmente las más brillantes, las limpia morfológicamente y rellena los agujeros 
internos, devolviendo la máscara en formato booleano y en 0–255.
'''

def obtenerMascara(imagenGris, invertir=False, tamanoKernel=7):

    
    if imagenGris.dtype == np.float64:#Nos tenemos que asegurar que sea uint8
        imagenGris = np.uint8(np.clip(imagenGris * 255, 0, 255))
    elif imagenGris.dtype != np.uint8:
        imagenGris = np.uint8(imagenGris)

    umbrales = threshold_multiotsu(imagenGris, classes=3) #Umbralización múltiple en 3 clases
    regionesIntensidad = np.digitize(imagenGris, bins=umbrales)

    mascaraBinaria = np.uint8((regionesIntensidad == 1) | (regionesIntensidad == 2)) * 255 #Nos quedamos con las regiones 1 y 2 


    elementoEstructurante = np.ones((tamanoKernel, tamanoKernel), np.uint8)# Cierre morfológico para unir regiones y cerrar huecos pequeños
    mascaraCerrada = cv2.morphologyEx(mascaraBinaria, cv2.MORPH_CLOSE, elementoEstructurante)

    alto, ancho = mascaraCerrada.shape[:2] #Rellenar agujeros internos con flood fill
    imagenFloodFill = mascaraCerrada.copy()
    mascaraFloodFill = np.zeros((alto + 2, ancho + 2), np.uint8)

    cv2.floodFill(imagenFloodFill, mascaraFloodFill, (0, 0), 255)# Flood fill desde la esquinaa

    floodInvertida = cv2.bitwise_not(imagenFloodFill)
    mascaraRellena = mascaraCerrada | floodInvertida

    if invertir:
        mascaraRellena = cv2.bitwise_not(mascaraRellena)

    mascaraBooleana = mascaraRellena == 255 #Generar máscara booleana y versión 0–255
    mascara255 = np.uint8(mascaraBooleana) * 255

    return mascaraBooleana, mascara255

