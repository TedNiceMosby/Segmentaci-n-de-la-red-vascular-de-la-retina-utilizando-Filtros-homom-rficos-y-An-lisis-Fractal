import cv2
import numpy as np
import matplotlib.pyplot as plt

def sobreponerMascaraVerde(imagenRGB,
                           mascaraBinaria,
                           alpha: float = 0.3):
    if len(mascaraBinaria.shape) == 3:
        mascaraBinaria = cv2.cvtColor(mascaraBinaria, cv2.COLOR_BGR2GRAY)

    mascaraBinaria = mascaraBinaria.astype(np.uint8)

    _, mascaraBinaria = cv2.threshold(mascaraBinaria, 127, 255, cv2.THRESH_BINARY)

    # Comprobar tamaños
    if imagenRGB.shape[:2] != mascaraBinaria.shape[:2]:
        raise ValueError("La imagen RGB y la máscara deben tener el mismo tamaño (H, W).")


    mascara_bool = mascaraBinaria == 255

    resultado = imagenRGB.copy()


    verde = np.array([0, 0, 255], dtype=np.uint8)

    resultado[mascara_bool] = (
        (1 - alpha) * imagenRGB[mascara_bool].astype(np.float32) +
        alpha * verde.astype(np.float32)
    ).astype(np.uint8)
    
    

    return resultado