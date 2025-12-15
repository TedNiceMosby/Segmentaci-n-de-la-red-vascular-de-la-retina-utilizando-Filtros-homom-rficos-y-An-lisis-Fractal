import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float
from scipy.stats import entropy
from pyswarm import pso


#%% PREPROCESAMIENTO 

"""
Cargar imagen verde
Carga una imagen y devuelve su canal verde y la imagen RGB.
"""
def cargarImagenVerde(rutaImagen):
    img = io.imread(rutaImagen)
    img = img_as_float(img)
    if img.ndim == 3:  # RGB
        verde = img[:, :, 1]
        color = img
    else:  # Escala de grises
        verde = img
        color = np.stack([img, img, img], axis=-1)
    return verde.astype(np.float32), color.astype(np.float32)


#%% ENTROPÍA DE SHANNON
'''
Entropía de Shannon
Se calcula el grado de aleatoriedad o incertidumbre que existe en la señal

INPUT-> imagen normalizada entre 0 a 1

'''

def calcularEntropia(imagen):
    imagen = np.clip(imagen, 0, 1)
    hist, _ = np.histogram(imagen, bins=256, range=(0, 1), density=True)
    hist = hist[hist > 0]
    return entropy(hist, base=2)


#%% FUNCION OBJETIVO FILTRO HOMOMÓRFICO 

def funcionObjetivo(Duv, sigma):
    denom = 2.0 * (sigma ** 2) + 1e-12
    H = 1.0 - np.exp(- Duv / denom)
    return H


def crearFiltroGaussiano(imagenRef, sigma):
    filas, columnas = imagenRef.shape
    cx, cy = columnas // 2, filas // 2
    U, V = np.meshgrid(np.arange(-cx, columnas - cx),
                       np.arange(-cy, filas - cy))
    Duv = U**2 + V**2
    H = funcionObjetivo(Duv, sigma)
    return H.astype(np.float32)


def aplicarFiltroHomomorfico(imagen, sigma, mascara=None):
    imagen = np.clip(imagen, 0, 1)
    imagenLog = np.log1p(imagen)
    F = np.fft.fft2(imagenLog)
    Fshift = np.fft.fftshift(F)

    H = crearFiltroGaussiano(imagen, sigma)
    Gshift = H * Fshift
    G = np.fft.ifftshift(Gshift)
    g = np.fft.ifft2(G)
    gReal = np.real(g)
    salida = np.expm1(gReal)

    salidaNorm = (salida - salida.min()) / (salida.max() - salida.min() + 1e-12)
    if mascara is not None:
        salidaNorm = mascara * salidaNorm + (1 - mascara) * imagen
    return salidaNorm.astype(np.float32)


def objetivoSigma(sigmaArray, imagen):
    sigma = sigmaArray[0]
    imgFiltrada = aplicarFiltroHomomorfico(imagen, sigma)
    entropiaVal = calcularEntropia(imgFiltrada)
    return -entropiaVal


#%% APLICAR PSO 

"""
Encontrar sigma 

Optimiza el valor de sigma usando PSO para maximizar la entropía

"""
def encontrarSigma(imagen, sigmaMin, sigmaMax, swarmSize=10, maxIter=5):
    print("-> Optimizando SIGMA (PSO) :)...")
    (sigmaOpt,), costo = pso(objetivoSigma, [sigmaMin], [sigmaMax],
                             args=(imagen,),
                             swarmsize=swarmSize,
                             maxiter=maxIter,
                             minfunc=1e-4,
                             minstep=1e-4)
    return sigmaOpt, -costo



#%% GRAFICAS 

"""
Gráfica 1
Muestra la imagen anterior a su filtrado y la filtrada.
"""
def graficarResultado(original, homomorfica, sigmaOpt):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(original, cmap="gray")
    axs[0].set_title("Filtro gaussiano",
                     fontfamily='serif', fontstyle='italic', fontsize=15)
    axs[0].axis("off")

    axs[1].imshow(homomorfica, cmap="gray")
    axs[1].set_title(f"Filtro Homomórfico (D0 = {sigmaOpt:.2f})",
                     fontfamily='serif', fontstyle='italic', fontsize=15)
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()
   
    
   
"""
Grafica 2 
Compara histogramas antes y después del filtrado homomórfico.
"""
def graficarHistComparativa(imagenAntes, imagenDespues, nombreImagen=""):

    plt.figure(figsize=(12, 6))
    plt.hist(imagenAntes.ravel(), bins=256, range=(0, 1), alpha=0.6, label='Antes (Suavizada)')
    plt.hist(imagenDespues.ravel(), bins=256, range=(0, 1), alpha=0.7, label='Después (Homomórfico)')
    titulo = f'Comparación de Histogramas "{nombreImagen}"' if nombreImagen else 'Comparación de Histogramas'
    plt.title(titulo, fontfamily='serif', fontstyle='italic', fontsize=15)
    plt.xlabel('Intensidad', fontsize=12, fontfamily='serif'); plt.ylabel('Frecuencia', fontsize=12, fontfamily='serif')
    plt.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(prop={'family':'serif','size':10})
    plt.show()


