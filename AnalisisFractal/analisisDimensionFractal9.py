from skimage import morphology, measure
import numpy as np
import matplotlib.pyplot as plt
import cv2

from skimage.util import view_as_blocks
from concurrent.futures import ThreadPoolExecutor



plt.close('all')

#%% OBTENER LOS VALORES DE UMBRAL 
'''
Obtenemos los pixeles que son validos que se encuentran en la zona de interes y eso se realiza a traves
de la máscara ROI y los umbrales obteniendo tanto los pixeles que se encuentran en el rango de interes como
los que se encuentran en el umbral, se hacen los siguientes pasos:
Los guardamos en mayor (los pixeles blancos asociados a la red vascular) a menor. 
Estas las dividimos en diferentes mascaras dependiendo del numero de iteraciones. 
num_iter: normalmente 3 (33.3%, 33.3%, 33.3%)
Regresa un array con las diferentes pixeles de interes que cumplen con estas condiciones
    
    
Importante**
FH_magnitude debe estar en uint8    
'''

def dividirUmbralesBins(FH_magnitude, umbrales, ROI_bool, num_iter=3):
    low, high = umbrales  
    FH_norm = FH_magnitude.astype(np.float32)
    FH_norm = (FH_norm - FH_norm.min()) / (FH_norm.max() - FH_norm.min() + 1e-8)
    FH_255 = (FH_norm * 255).astype(np.uint8)

    mask_rango = (FH_255 >= low) & (FH_255 <= high) & ROI_bool

    coords = np.argwhere(mask_rango)  # shape (N, 2)
    
    if coords.size == 0:
        return []  # Si no hay píxeles, devolvemos listas vacías

    # Ordenar esos píxeles de mayor a menor intensidad
    intensidades = FH_255[mask_rango]
    order = np.argsort(intensidades)[::-1]  # descendente
    coords_ord = coords[order]

    total = len(coords_ord)
    
    chunk = int(np.ceil(total / num_iter))  # tamaño de cada tercio

    mascaras_iter = []
    for i in range(num_iter):
        inicio = i * chunk
        fin = min((i + 1) * chunk, total)
        if inicio >= fin:  # ya no hay más píxeles
            mascaras_iter.append(np.zeros_like(ROI_bool, dtype=bool))
            continue

        m = np.zeros_like(ROI_bool, dtype=bool)
        subcoords = coords_ord[inicio:fin]
        m[subcoords[:, 0], subcoords[:, 1]] = True
        mascaras_iter.append(m)

    return mascaras_iter

#%% ETIQUETAR IMAGEN POR SEGMENTOS 

'''
    Etiquetar segmentos
Etiquetamos los segmentos que se encuentran unidos por una conectividad de dos con la finalidad de analizarlos 
independientemente (posteriormente se obtendra el DF)
*etiquetaXConectividad
binarizada- imagen binaria que se obtuvo mediante el umbral dinamico
conectividad- un criterio que se utiliza para ver que tan juntos estan los segmentos
-> imagen_label, num_etiquetas
imagen_label- es la imagen etiquetada donde cada pixel que se considero un conjunto estara unida mediante 1111,222,333 etc
num_etiquetas- me ayuda a saber cuantos fragmentos identifico


IMPORTANTE*
Aqui podemos descartar los fragmentos con deteminado tamaño de pixeles

'''

def etiquetaXConectividad(binarizada, conectividad=2):

    # Eliminar objetos muy pequeños
    binarizada = morphology.remove_small_objects(binarizada, min_size=3) #15
    
    # Etiquetar regiones conectadas
    imagen_label = measure.label(binarizada, connectivity=conectividad)
    
    num_etiquetas = np.max(imagen_label)
    print('Número de etiquetas:', num_etiquetas)

    return imagen_label, num_etiquetas



#%% FUNCIONES OPTIMIZADAS

def contarCajasImagen(imagen_binaria, tam):
    filas, columnas = imagen_binaria.shape
    filas_corte = (filas // tam) * tam
    columnas_corte = (columnas // tam) * tam
    imagen_recortada = imagen_binaria[:filas_corte, :columnas_corte]
    bloques = view_as_blocks(imagen_recortada, block_shape=(tam, tam))
    bloques_contenidos = np.any(bloques, axis=(2, 3))
    return np.sum(bloques_contenidos)

def calcularDF(imagen_binaria):
    if imagen_binaria is None:
        return None
    tamanos = []
    conteos = []
    max_tam = min(imagen_binaria.shape) // 2
    K = [2**i for i in range(1, int(np.log2(max_tam)))]

    for tam in K:
        cajas = contarCajasImagen(imagen_binaria, tam)
        if cajas > 0:
            tamanos.append(1 / tam)
            conteos.append(cajas)

    if len(tamanos) < 2:
        return None

    log_tamanos = np.log(tamanos)
    log_conteos = np.log(conteos)
    pendiente, _ = np.polyfit(log_tamanos, log_conteos, 1)
    return pendiente


#%% DETECTAR LAS VENAS 

'''
    Detectar fragmentos
Paso 1- Clasificar los fragmentos de acuerdo con DF (Ruido o posible vena) mediante la DF del articulo 1.35
Paso 2- Si es posible venas debe compararse con Ivessel
            Si se conecta con Ivessel -> es vena
            Otro no lo es 


*clasificadorVessel*
imagen_label- mediante un for van introduciendo imagenes por segmento, e 'i' me dice con que numero esta etiquetado el segmento de 
interes y regreso 
    Etiqueta 1: no es vena
    Etiqueta 2: posible vena
    
*Clasificador potencial* 
mediante los label se percata si es parte del fragmento o no se calculan las numeros de etiquetas iniciales
se suma la imagen con las I vessel reales con el fragmento que estoy analizando, aplico erosion y si el numero de etiquetas aumenta
es parte del segmento y se vuelve I vessel

'''    
def clasificadorVessel(imagen_label, i, DF):
    I_vessel_pot = np.zeros_like(imagen_label)
    I_vessel_non = np.zeros_like(imagen_label)

    if DF is None:
        etiqueta = 0
        return I_vessel_non, etiqueta  # etiqueta 0 = fragmento descartado

    if DF > 1.9: #1.35
        I_vessel_non = imagen_label == i
        etiqueta = 1
        return I_vessel_non, etiqueta

    elif DF <= 1.9: #1.35
        I_vessel_pot = imagen_label == i
        etiqueta = 2
        return I_vessel_pot, etiqueta


"""
Esta función determina si un fragmento candidato presenta unión con una red vascular ya existente. 
A partir de una máscara binaria del fragmento potencial y de la máscara binaria de la red aceptada, 
se evalúa si existeconectividad suficiente entre ambas estructuras.

El criterio de evaluación depende del modo seleccionado. En el modo "adjacency", el fragmento candidato 
se acepta si presenta contacto directo con la red vascular existente. De manera opcional, el modo
"bridging" permite verificar si el fragmento actúa como un puente que conecta al menos dos componentes 
diferentes de la red. La función devuelve un valor booleano que indica si el fragmento cumple con el
criterio de unión establecido.
"""

def clasificadorVesselPot(I_vessel_pot, I_vessel, modo="bridging"):    
    I_vessel_pot = I_vessel_pot.astype(bool)# Convertimos a booleanos por seguridad
    I_vessel = I_vessel.astype(bool)

    

    if not np.any(I_vessel):# Si no hay venas previas, aceptamos el primer segmento
        return True

    # Kernel 3x3 para 8-conectividad
    kernel = np.ones((3, 3), np.uint8)

    if modo == "adjacency":
        I_vessel_dilatada = cv2.dilate(I_vessel.astype(np.uint8), kernel, iterations=1).astype(bool)
        conectado = np.any(I_vessel_pot & I_vessel_dilatada)
        return bool(conectado)


    # En este modo buscamos si el candidato une (en una misma componente conectada)
    # al menos dos componentes distintos de la red original.
    if modo == "bridging":
        # Etiquetas de la red original
        labels_base = measure.label(I_vessel, connectivity=2)

        # Unión candidato + red
        union = (I_vessel | I_vessel_pot).astype(np.uint8)
        union_dilatada = cv2.dilate(union, kernel, iterations=1).astype(bool)

        labels_union = measure.label(union_dilatada, connectivity=2)

        # Etiquetas de componentes de "union" donde aparece el candidato
        etiquetas_union_candidato = np.unique(labels_union[I_vessel_pot])
        etiquetas_union_candidato = etiquetas_union_candidato[etiquetas_union_candidato > 0]

        for et in etiquetas_union_candidato:
            # Pixels de esta componente en la imagen unida
            mask_comp = (labels_union == et)

            # ¿Cuántas componentes distintas de la red original están metidas aquí?
            etiquetas_base_en_comp = np.unique(labels_base[mask_comp])
            etiquetas_base_en_comp = etiquetas_base_en_comp[etiquetas_base_en_comp > 0]

            # Si hay al menos 2 componentes originales diferentes, el candidato hace de puente
            if len(etiquetas_base_en_comp) >= 2:
                return True

        return False
'''
Une lógicamente una lista de máscaras binarias sin crear un arreglo 3D,
evitando problemas de memoria.
'''
def unirSegmentos(segmentos):
   if not segmentos:
       # Si está vacío, retornar una imagen vacía mínima de 1x1
       return np.zeros((1, 1), dtype=np.uint8)

   # Comenzamos con una máscara del tamaño del primer segmento
   unidos = np.zeros_like(segmentos[0], dtype=bool)

   for seg in segmentos:
       unidos |= seg.astype(bool)  # Unión incremental

   return unidos.astype(np.uint8)

#%% CONDICION DE PARO


def calculandoIncrementoVasos(Iteraciones, cont):
    
    if cont > 0 and cont < len(Iteraciones):
        
        imagen_actual = Iteraciones[cont]
        imagen_anterior = Iteraciones[cont - 1]
        
        # Diferencia píxel a píxel
        diferencia = imagen_actual - imagen_anterior
        
        # Contar cuántos pixeles nuevos de vasos aparecieron
        incremento = np.sum(diferencia == 1)  #
        
        # Tamaño total de la imagen
        fil, col = imagen_actual.shape
        total = fil * col
        
        # C1 normalizado
        C1 = incremento / total
        
        return C1
    
    else:
        # No se puede calcular si es la primera iteración
        return 0

    
def calculandoVelocidad(C1_list, cont):
    if cont > 0 and cont < len(C1_list):
        C2 = C1_list[cont] - C1_list[cont - 1]
        return C2
    else:
        return 0

def calculandoAceleracion(C2_list, cont):
    if cont > 0 and cont < len(C2_list):
        C3 = C2_list[cont] - C2_list[cont - 1]
        return C3
    else:
        return 0
    

#%% ANALISIS WAVELET

"""
Esta función dibuja una figura con tres subgráficos para la iteración indicada. 
El primero muestra la máscara binaria de los objetos detectados, el segundo la unión de 
los segmentos que cumplen el criterio DF y el tercero la unión de los segmentos que no lo cumplen.
"""
def mostrarResultadosIteracion(cont,
                               binaria_iter,
                               segmentos_DF_ok,
                               segmentos_DF_no):

    def unirSegmentosLocal(lista_segmentos):
        if lista_segmentos:
            return np.logical_or.reduce(lista_segmentos)
        else:
            return np.zeros_like(binaria_iter, dtype=bool)

    img_ok  = unirSegmentosLocal(segmentos_DF_ok)
    img_no  = unirSegmentosLocal(segmentos_DF_no)

    plt.figure(figsize=(10, 3))

    plt.subplot(1, 3, 1) #objetos detectados (binaria)
    plt.imshow(binaria_iter, cmap='gray')
    plt.title(f'Iter {cont+1} - Detectados')
    plt.axis('off')

    plt.subplot(1, 3, 2) #cumplen DF
    plt.imshow(img_ok, cmap='gray')
    plt.title(f'Iter {cont+1} - Cumplen DF')
    plt.axis('off')

    plt.subplot(1, 3, 3) # NO cumplen DF
    plt.imshow(img_no, cmap='gray')
    plt.title(f'Iter {cont+1} - NO cumplen DF')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


"""
Esta función genera un kernel tipo Mexican Hat (segunda derivada de una Gaussiana). 
El parámetro M define el tamaño del kernel (forzándolo a ser impar) y sigma controla 
el ancho de la campana.
"""
def kernelSombreroMexicano(M=31, sigma=4.0):
    if M % 2 == 0:
        M += 1
    x = np.linspace(-3*sigma, 3*sigma, M)
    kernel = (1 - (x**2)/(sigma**2)) * np.exp(-(x**2)/(2*sigma**2))
    kernel -= kernel.mean() # Opcional: normalizar para que la media sea ~0
    return kernel


"""
Esta función calcula un umbral para los valores de DF usando un histograma y 
un filtro Mexican Hat. El umbral se obtiene a partir de un valle en la señal filtrada 
y solo se acepta si su valor es mayor que min_df.

Devuelve el umbral calculado (o None si no es válido), el índice del bin
correspondiente, los centros de los bins y la señal filtrada.
"""
def calcularUmbralSombreroMexicano(df_values, n_bins=40, min_df=0.15, sigma=4.0):

    df_vals = np.array([v for v in df_values if v is not None], dtype=float)# Limpiar DF: quitar None y NaN
    df_vals = df_vals[np.isfinite(df_vals)]

    if df_vals.size < 2:
        return None, None, None, None

    # Histograma
    hist, bin_edges = np.histogram(df_vals, bins=n_bins)

    if np.all(hist == 0):
        return None, None, None, None

    x_centros = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Mexican Hat
    M = len(hist)# 40
    kernel = kernelSombreroMexicano(M=M, sigma=sigma)  # dentro se vuelve 41
    señal_filtrada = np.convolve(hist, kernel, mode='same')  # longitud 41


    if len(señal_filtrada) != len(x_centros):
        diff = len(señal_filtrada) - len(x_centros)

        start = diff // 2
        señal_filtrada = señal_filtrada[start:start + len(x_centros)]

    if len(señal_filtrada) != len(x_centros):
        min_len = min(len(señal_filtrada), len(x_centros))
        señal_filtrada = señal_filtrada[:min_len]
        x_centros = x_centros[:min_len]


    mascara_validos = x_centros > min_df

    if not np.any(mascara_validos):
        return None, None, x_centros, señal_filtrada

    idx_validos = np.where(mascara_validos)[0]
    idx_min_local = np.argmin(señal_filtrada[mascara_validos])
    idx_min_global = idx_validos[idx_min_local]

    umbral = (bin_edges[idx_min_global] + bin_edges[idx_min_global + 1]) / 2.0

    return umbral, idx_min_global, x_centros, señal_filtrada




#Visualiza el análisis de umbral con sombrero mexicano para una iteración.

def visualizarSombreroMexicano(indice_iteracion,
                               df_values,
                               x_centros_df,
                               señal_df,
                               umbral_iter,
                               min_df=0.2,
                               n_bins_hist=40):
    
    # Limpiar lista de DF para el histograma
    df_vals = np.array([v for v in df_values if v is not None], dtype=float)
    df_vals = df_vals[np.isfinite(df_vals)]

    if df_vals.size == 0:
        print(f"[Iteración {indice_iteracion+1}] No hay valores DF válidos para visualizar.")
        return

    if x_centros_df is None or señal_df is None:
        print(f"[Iteración {indice_iteracion+1}] No hay información de sombrero mexicano (centros o señal nulos).")
        plt.figure(figsize=(8, 4))
        plt.hist(df_vals, bins=n_bins_hist, alpha=0.7, edgecolor='black')
        plt.xlabel('Dimensión fractal (DF)')
        plt.ylabel('Frecuencia')
        plt.title(f'Histograma DF - Iteración {indice_iteracion+1}')
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.show()
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histograma izquierda
    ax_hist = axes[0]
    ax_hist.hist(df_vals, bins=n_bins_hist, alpha=0.7, edgecolor='black')
    ax_hist.set_xlabel('Dimensión fractal (DF)')
    ax_hist.set_ylabel('Frecuencia')
    ax_hist.set_title(f'Histograma DF - Iteración {indice_iteracion+1}')
    ax_hist.grid(alpha=0.3, linestyle='--')

    ax_hist.axvline(min_df, color='orange', linestyle='--', label=f'DF mínima ({min_df:.2f})')
    if umbral_iter is not None:
        ax_hist.axvline(umbral_iter, color='red', linestyle='--', label=f'Umbral MexHat ({umbral_iter:.3f})')
    ax_hist.legend(loc='best')

    # Señal filtrada derecha
    ax_filter = axes[1]
    ax_filter.plot(x_centros_df, señal_df, marker='o')
    ax_filter.set_xlabel('Dimensión fractal (DF)')
    ax_filter.set_ylabel('Respuesta filtrada')
    ax_filter.set_title('Señal Mexican Hat sobre el histograma DF')
    ax_filter.grid(alpha=0.3, linestyle='--')

    if umbral_iter is not None:
        ax_filter.axvline(umbral_iter, color='red', linestyle='--', label='Umbral MexHat')
        ax_filter.legend(loc='best')

    fig.suptitle(f'Análisis de umbral (sombrero mexicano) - Iteración {indice_iteracion+1}')
    plt.tight_layout()
    plt.show()
    
    
#%% MAIN calculo de DF

"""
Esta función evalúa la evolución de la segmentación mediante un esquema de dos iteraciones basado 
en la medida DF, el análisis de histogramas y la conectividad entre segmentos.

En la primera iteración, se calculan los valores DF de los segmentos, se construye un histograma y se 
aplica un filtro Mexican Hat para obtener un umbral mayor a 0.2. Los segmentos con DF superior al umbral se
aceptan directamente, mientras que aquellos con DF menor o igual se aceptan solo si están conectados a 
la red vascular ya aceptada.

En la segunda iteración, se repite el cálculo de DF y del umbral, y solo se aceptan los segmentos que 
superan el umbral y además presentan unión con la red total de venas. Los segmentos que no cumplen estas
condiciones se rechazan definitivamente. Si en alguna iteración no se puede calcular un umbral válido, no se 
acepta ningún segmento por DF.
"""

def visualizarEvolucionSegmentacion(FH_magnitude,
                                    clasificadorVessel, 
                                    unirSegmentos,       
                                    clasificadorVesselPot,
                                    umbrales,
                                    ROI_bool):


    # Máscaras por rango de intensidades dentro del ROI 
    mascaras_iter = dividirUmbralesBins(FH_magnitude, umbrales, ROI_bool, num_iter=2)

    cont = 0
    I_vessel = []# fragmentos aceptados (lista de máscaras)
    I_vessel_non = [] # fragmentos rechazados (lista de máscaras)
    Iteraciones = [] # imagen de vasos global por iteración
    C1_list, C2_list = [], []

    DF_iteraciones = []  # lista de listas de DF por iteración

    # Máscara acumulada de todos los vasos aceptados (venas totales)
    I_vessel_unido = None

    for binaria in mascaras_iter:
        print(f"\n===== Iteración (bin de intensidad) {cont+1} =====")

        # Inicializar máscara acumulada la primera vez
        if I_vessel_unido is None:
            I_vessel_unido = np.zeros_like(binaria, dtype=bool)

        # Si en este bin no hay píxeles dentro del ROI, solo actualizamos métricas y seguimos
        if not np.any(binaria):
            DF_iteraciones.append([])
            Iteraciones.append(I_vessel_unido.astype(np.uint8))

            C1 = calculandoIncrementoVasos(Iteraciones, cont)
            C1_list.append(C1)
            C2 = calculandoVelocidad(C1_list, cont)
            C2_list.append(C2)
            C3 = calculandoAceleracion(C2_list, cont)
            print(f'C1:{C1}, C2:{C2}, C3:{C3}')

            mostrarResultadosIteracion(cont,
                                       binaria_iter=binaria,
                                       segmentos_DF_ok=[],
                                       segmentos_DF_no=[])
            cont += 1
            continue

        # Etiquetamos segmentos en la máscara binaria de esta iteración
        imagen_label, num_etiquetas = etiquetaXConectividad(binaria, conectividad=1)

        # Listas solo para visualización en esta iteración
        segmentos_DF_ok_iter = []   # segmentos aceptados en esta iteración
        segmentos_DF_no_iter = []   # segmentos rechazados en esta iteración

        # Paso 1: obtenemos DF de cada segmento
        def calcularSegmento(i):
            seg = (imagen_label == i)
            DF = calcularDF(seg)
            return i, seg, DF

        with ThreadPoolExecutor() as executor:
            resultados = list(executor.map(calcularSegmento,
                                           range(1, num_etiquetas + 1)))

        DF_actual = [DF for (_, _, DF) in resultados]

        #Paso 2: calcular umbral por Mexican Hat sobre DF de esta iteración 
        umbral_iter, idx_min_dfbin, x_centros_df, señal_df = calcularUmbralSombreroMexicano(
            DF_actual,
            n_bins=40,
            min_df=0.2,
            sigma=4.0
        )

        if umbral_iter is not None:
            print(f"Umbral MexHat (iter {cont+1}): {umbral_iter:.6f}")
        else:
            print(f"Umbral MexHat (iter {cont+1}): None (no se encontró valle > 0.2)")


        # Paso 3: aplicar reglas de aceptación según la iteración 
        for i, imagen_binaria_i, DF in resultados:
            aceptado = False

            
            if (DF is None) or (umbral_iter is None):# Si no hay DF o no hay umbral válido, no puede ser aceptado por DF
                aceptado = False
            else:
                # ITERACIÓN 1
                if cont == 0:
                    #DF > umbral -> aceptación directa
                    if DF > umbral_iter:
                        aceptado = True
                    else:
                        # DF <= umbral -> se acepta si se une con las ya aceptadas
                        if np.any(I_vessel_unido):
                            conectado = clasificadorVesselPot(
                                imagen_binaria_i.astype(np.uint8),
                                I_vessel_unido.astype(np.uint8),
                                modo="bridging"
                            )
                            if conectado:
                                aceptado = True

                # ITERACIÓN 2
                elif cont == 1:
                    # En la segunda iteración:
                    # Solo aceptamos si DF > umbral y además se une con las venas totales
                    if DF > umbral_iter and np.any(I_vessel_unido):
                        conectado = clasificadorVesselPot(
                            imagen_binaria_i.astype(np.uint8),
                            I_vessel_unido.astype(np.uint8),
                            modo="adjacency"
                        )
                        if conectado:
                            aceptado = True

                else:
                    aceptado = False

            # Actualizamos máscaras globales
            if aceptado:
                I_vessel.append(imagen_binaria_i)
                I_vessel_unido |= imagen_binaria_i.astype(bool)
                segmentos_DF_ok_iter.append(imagen_binaria_i)
            else:
                I_vessel_non.append(imagen_binaria_i)
                segmentos_DF_no_iter.append(imagen_binaria_i)

        # Guardamos los DF de esta iteración
        DF_iteraciones.append(DF_actual)
        print(f"DF_iteración {cont+1}: {DF_actual}")

        # Imagen de vasos de esta iteración = máscara acumulada hasta ahora
        Iteraciones.append(I_vessel_unido.astype(np.uint8))

        C1 = calculandoIncrementoVasos(Iteraciones, cont)
        C1_list.append(C1)
        C2 = calculandoVelocidad(C1_list, cont)
        C2_list.append(C2)
        C3 = calculandoAceleracion(C2_list, cont)
        print(f'C1:{C1}, C2:{C2}, C3:{C3}')
        cont += 1

    return Iteraciones, DF_iteraciones