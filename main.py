from FiltroGaussiano.filtroGaussiano import cargarImagen, elegirSigmaGauss, filtroGaussiano, graficarCurvasSigma
from FiltroHomomorfico.filtroHomomorfico import encontrarSigma, aplicarFiltroHomomorfico, graficarResultado,  graficarHistComparativa
from AnalisisFractal.analisisDimensionFractalAuxiliar import visualizarEvolucionSegmentacion, clasificadorVessel, unirSegmentos, clasificadorVesselPot

# FILTRO FRANGI
from modelFrangiLayers import crearRedNeuronal

from Complementos.mascaraROI import obtenerMascara
from Multiumbrales.obtenerUmbrales import obtenerUmbrales

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from pathlib import Path
import cv2
plt.close('all')



#%% MAIN 

if __name__ == "__main__":

    
    rangoSigma = np.arange(0.2, 4.1, 0.2)
    
    #--------------ELEGIR IMAGEN-------------------- 
    rutaEjemplo = r"C:\Users\USUARIO\Desktop\..."

    imagenVerde, imagenColor = cargarImagen(rutaEjemplo)
    if imagenVerde is None:
        raise SystemExit("No se pudo cargar la imagen de ejemplo.")
                
    #------------COMPLEMENTOS-----------------------
    mask_bool, mask_255 = obtenerMascara(imagenVerde)

    #--------------BUSCAR SIGMA ÓPTIMO-------------------- 
    resultado = elegirSigmaGauss(imagenVerde, rangoSigma)
    sigmaOptimo = resultado["sigmaOptima"]
    print(f"Sigma de consenso: {sigmaOptimo:.3f}")
    
    #--------------IMAGEN CON EL FILTRO GAUSSIANO-------------------- 
    imagenGauss = filtroGaussiano(imagenVerde, sigmaOptimo)
    
    #--------------GRAFICAS DEL FILTRO GAUSSIANO-------------------- 
    nombreImagen = os.path.basename(rutaEjemplo)
    graficarCurvasSigma(resultado, nombreImagen)
    
    #--------------FILTRO HOMOMÓRFICO + PSO--------------------
    sigmaMin = 0.2
    sigmaMax = 150.0
    swarmSize = 20
    maxIter = 30
    guardar = False
    
    sigmaOpt, entropiaOpt = encontrarSigma(imagenGauss, sigmaMin, sigmaMax, swarmSize, maxIter)
    print(f"Sigma óptima = {sigmaOpt:.4f} | Entropía = {entropiaOpt:.4f}")
    
    imagenHomomorfica = aplicarFiltroHomomorfico(imagenGauss, sigmaOpt)
    
    #--------------GRAFICAS DEL FILTRO HOMOMORFICO--------------------
    graficarResultado(imagenGauss, imagenHomomorfica, sigmaOpt)
    graficarHistComparativa(imagenGauss, imagenHomomorfica, nombreImagen="")
    
    #--------------FILTRO FRANGI--------------------
    
    def rutaRelativa(archivoRelativo):
        rutaScript = Path(__file__).resolve().parent
        return rutaScript / archivoRelativo
    pesosPath = rutaRelativa("AnalisisFractal/modelRetina.h5")   

    # Opciones de procesamiento
    invertirEntrada = True
    normalizarEntrada = False
    
    # Crea el modelo 
    x, model = crearRedNeuronal()
    
    # 1. Nos aseguramos de que está en el rango correcto para Frangi (uint8)
    imagenUint8 = (imagenHomomorfica * 255).astype(np.uint8) if imagenHomomorfica.max() <= 1 else imagenHomomorfica
    
    # 2. Cargar los pesos del modelo
    if pesosPath.exists():
        model.load_weights(str(pesosPath))
        print(f"[OK] Pesos cargados desde: {pesosPath}")
    else:
        raise FileNotFoundError(f"No existe el archivo de pesos: {pesosPath}")
    
    # 3. Invertir la imagen si es necesario
    entrada = imagenUint8.copy()
    if invertirEntrada:
        entrada = cv2.bitwise_not(entrada)

    # 4. Formatear entrada para la red
    entrada = np.expand_dims(np.expand_dims(entrada, axis=2), axis=0).astype(np.float32)
    if normalizarEntrada:
        entrada /= 255.0
    
    # 5. Ejecutar predicción
    pred = model.predict(entrada)
    pred2d = pred[0, :, :, 0] if pred.ndim == 4 else pred[:, :, 0]
    
    # 6. Normalizar salida a 0–255 y convertir a uint8
    imagenFrangi = (pred2d * 255).astype(np.uint8)
    

    
    #--------------OBTENER UMBRAL--------------------
    umbral =  obtenerUmbrales(imagenFrangi, 2)
    umbral = min(umbral)
    umbrales =[umbral, 255]

    
    plt.figure(figsize=(12,6))
    
    # ---------------------IMAGEN FRANGI ----------------
    plt.subplot(1, 2, 1)
    plt.imshow(imagenFrangi, cmap='gray')
    plt.title("Imagen Frangi",  fontfamily='serif', fontstyle='italic', fontsize=20)
    plt.axis('off')
    
    # ---------------- HISTOGRAMA ----------------
    plt.subplot(1, 2, 2)
    hist_vals, bins, _ = plt.hist(imagenFrangi.ravel(), bins=256, range=[0,255], alpha=0.7)
    plt.title("Histograma de intensidades", fontfamily='serif', fontstyle='italic', fontsize=19)
    plt.xlabel("Intensidad", fontsize=12, fontfamily='serif')
    plt.ylabel("Frecuencia", fontsize=12, fontfamily='serif')
    
    #Líneas rojas
    for u in umbrales:
        plt.axvline(x=u, color='red', linestyle='--', linewidth=5)#2
    
    u = umbrales[0]                
    mid = (u + 255) / 2 #Punto medio entre umbral y 255
    
        #línea azul
    plt.axvline(x=mid, color='blue', linestyle='-', linewidth=3)#
    
    # Altura para flechas
    ymax = hist_vals.max()
    y_text_1 = ymax * 0.72
    y_arrow_1 = ymax * 0.60
    y_text_2 = ymax * 0.52
    y_arrow_2 = ymax * 0.40
    
    # Intervalo 1
    plt.annotate(
        'Intervalo 2',
        xy=((u + mid)/2, y_arrow_1),    # punto intermedio
        xytext=((u + mid)/2, y_text_1), # texto arriba
        ha='center', va='bottom',
        fontsize=12, fontfamily='serif'
    )
    plt.annotate(
        '',
        xy=(u, y_arrow_1), xytext=(mid, y_arrow_1),
        arrowprops=dict(arrowstyle='<->', color='blue', lw=2)
    )
    
    # Intervalo 2
    plt.annotate(
        'Intervalo 1',
        xy=((mid + 255)/2, y_arrow_2),
        xytext=((mid + 255)/2, y_text_2),
        ha='center', va='bottom',
        fontsize=12, fontfamily='serif'
    )
    plt.annotate(
        '',
        xy=(mid, y_arrow_2), xytext=(255, y_arrow_2),
        arrowprops=dict(arrowstyle='<->', color='blue', lw=2)
    )
    plt.tight_layout()
    plt.show()
    
        
    # #--------------ANÁLISIS FRACTAL CON BOX COUNTING--------------------
    Iteraciones, DF_iteraciones = visualizarEvolucionSegmentacion(
        imagenFrangi,
        clasificadorVessel,
        unirSegmentos,
        clasificadorVesselPot,
        umbrales,
        mask_bool
    )
    
    # ##--------------IMAGEN SEGMENTADA--------------------
    imagen_segmentada = Iteraciones[-1]
    fig = plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(Iteraciones[0], cmap='gray')
    plt.title('Intervalo 1',  fontsize=12, fontfamily='serif')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(Iteraciones[1], cmap='gray')
    plt.title('Intervalo 2',  fontsize=12, fontfamily='serif')
    plt.axis('off')
    
    fig.suptitle(
        'Segmentación de la red vascular',
        fontfamily='serif', fontstyle='italic', fontsize=15
    )

    
    