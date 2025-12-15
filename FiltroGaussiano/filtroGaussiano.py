import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, img_as_float



#%% EXTRACCION DEL CANAL G

'''
Primer paso consiste en obtener el canal verde la imagen,
INPUT-> la ruta de la imagen 
OUTPUT-> imagen en el canaL G, imagen RGB

'''
def cargarImagen(ruta):
    try:
        imagenColor = img_as_float(io.imread(ruta))
        if imagenColor.ndim == 2:  #Ya esta en escala de grises
            imagenVerde = imagenColor
        else:
            imagenVerde = imagenColor[:, :, 1] #Extraer el canal verde
        return imagenVerde, imagenColor
    
    except Exception as e:
        
        print(f"[ERROR] No se pudo cargar '{ruta}': {e}")
        return None, None


#%% VARIABLES DE ANALISIS 

'''
Varianza del laplaciano (VoL)
Es una mascara de segundo orden que permite ver los cambios de intensidad como los
bordes del objeto en la imagen.

INPUT -> canal verde + Filtro gaussiano (es de tipo float), 
OUTPUT -> valor del VoL
'''
def varianzaLaplacianoImagen(imagenGauss):
    VoL = float(filters.laplace(imagenGauss).var())
    return VoL 

'''
La energía de alta frecuencia residual (HREF)
Representa la energia total de la imagen original que se ha suprimido durante 
el proceso de suavizado y actua como señal de grado de perdida de datos de alta
frecuencia

HFER = energia(residual) / energia(original) en 2D.

INPUT -> canal verde + Filtro gaussiano, 
OUTPUT -> valor del VoL
'''
def energiaAltaFrecuencia(imagenVerde, imagenGauss):
    residual = imagenVerde - imagenGauss
    e_res = float(np.sum(residual ** 2))
    e_org = float(np.sum(imagenVerde ** 2))
    if e_org == 0: # Evitamos divisiones sobre 0 
        return 0.0
    HREF = e_res / e_org 
    return HREF


#%% ENCONTRAR MEJOR PUNTO
'''
Indice Punto Codo

INPUT -> vector de los valores VoL cuando se aplica el filtro gaussiano, 
OUTPUT -> indice del punto codo
OBJETIVO El índice del punto más alejado de la recta que une el primer y último valor de la curva. 
         Este índice es interpretado como el punto de codo en la gráfica
'''

def indicePuntoCodo(y):
    y = np.asarray(y)
    n = y.size
    if n == 0:
        return 0
    puntos = np.column_stack((np.arange(n), y))
    p0 = puntos[0]
    p1 = puntos[-1]
    v = p1 - p0
    nrm = np.linalg.norm(v)
    if nrm == 0:
        return 0
    u = v / nrm
    d0 = puntos - p0
    proy = np.dot(d0, u)
    par = np.outer(proy, u)
    perp = d0 - par
    dist = np.linalg.norm(perp, axis=1)
    puntoCodo = int(dist.argmax())
    return puntoCodo


'''
Indice de máxima aceletacion 

INPUT -> vector de los valores HREF de la imagen con filtro gaussiano, 
OUTPUT -> el indice del punto de aceleracion 
OBJETIVO -> El indice donde la segunda derivada alcanza su máximo. 
            Ese índice identifica el punto de máxima aceleración de la curva

'''

def indiceMaximaAceleracion(y):
    y = np.asarray(y)
    if y.size < 3:
        return 0
    d1 = np.diff(y)
    d2 = np.diff(d1)
    puntoMaxA = int(d2.argmax()) + 1
    return puntoMaxA

#%% FILTRO GAUSSIANO 


def filtroGaussiano(imagenVerde, sigma):
    imagenGauss = filters.gaussian(imagenVerde, sigma)
    return imagenGauss


#%% ELEGIR SIGMA 


"""
Elegir sigma del filtro gaussiano
Recorre un rango de sigma para filtro gaussiano y calcula:
    
- VoL (varianza del laplaciano de la imagen suavizada)
- HFER (energía de alta frecuencia perdida)

Elige utilizando el criterio de:    
  * sigma_vol por punto de codo en VoL
  * sigma_hfer por máxima aceleración en HFER
Hace un promedio de ambas variables = (sigma_vol + sigma_hfer)/2.

"""



def elegirSigmaGauss(imagenVerde, rangoSigma):
    
    print("-> Buscando sigma óptimo (filtro Gauss) ...")
    valoresVoL, valoresHFER = [], []

    for sigma in rangoSigma:
        imagenGauss = filtroGaussiano(imagenVerde, sigma)
        vol = varianzaLaplacianoImagen(imagenGauss)
        hfer = energiaAltaFrecuencia(imagenVerde, imagenGauss)
        valoresVoL.append(vol)
        valoresHFER.append(hfer)

    valoresVoL = np.array(valoresVoL)
    valoresHFER = np.array(valoresHFER)

    idxVol = indicePuntoCodo(valoresVoL)
    sigmaVol = float(rangoSigma[idxVol])

    idxHfer = indiceMaximaAceleracion(valoresHFER)
    sigmaHfer = float(rangoSigma[idxHfer])

    sigmaConsenso = (sigmaVol + sigmaHfer) / 2.0

    return {
        "rangoSigma": rangoSigma,
        "valoresVoL": valoresVoL,
        "valoresHFER": valoresHFER,
        "idxVol": idxVol,
        "sigmaVol": sigmaVol,
        "idxHfer": idxHfer,
        "sigmaHfer": sigmaHfer,
        "sigmaOptima": sigmaConsenso,
    }


#%%  GRAFICA DE SIGMA 

"""
Esta funcion se encrga de Grafica VoL y HFER contra sigma y 
marca los puntos elegidos.
"""


def graficarCurvasSigma(resultado: dict, nombreImagen: str = ""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    titulo = f'Análisis de sigma del filtro gaussiano' if nombreImagen else 'Análisis de sigma del Filtro gaussiano'
    fig.suptitle(titulo, fontfamily='serif', fontstyle='italic', fontsize=18)

    rangoSigma = resultado["rangoSigma"]

    # VoL
    ax1.plot(rangoSigma, resultado["valoresVoL"], 'o-', label='VoL (nitidez)', ms=4)
    ax1.scatter(resultado["sigmaVol"],
                resultado["valoresVoL"][resultado["idxVol"]],
                s=150, zorder=5, label=f'Punto de codo σ={resultado["sigmaVol"]:.2f}')
    ax1.set_title('VoL vs. Sigma',  fontsize=12, fontfamily='serif'); 
    ax1.set_xlabel('Sigma (σ)',  fontsize=12, fontfamily='serif'); 
    ax1.set_ylabel('Varianza del Laplaciano',  fontsize=12, fontfamily='serif')
    ax1.legend(); 
    ax1.grid(True, ls='--', linewidth=0.5, alpha=0.7)

    # HFER
    ax2.plot(rangoSigma, resultado["valoresHFER"], 'o-', label='HFER (detalle perdido)', ms=4)
    ax2.scatter(resultado["sigmaHfer"],
                resultado["valoresHFER"][resultado["idxHfer"]],
                s=150, zorder=5, label=f'Acel. máx σ={resultado["sigmaHfer"]:.2f}')
    ax2.set_title('HFER vs. Sigma',  fontsize=12, fontfamily='serif'); 
    ax2.set_xlabel('Sigma (σ)',  fontsize=12, fontfamily='serif'); 
    ax2.set_ylabel('Energía de Alta Frecuencia Residual ',  fontsize=12, fontfamily='serif')
    ax2.legend(); 
    ax2.grid(True, ls='--', linewidth=0.5, alpha=0.7)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

