from skimage import morphology, measure
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.util import view_as_blocks
from concurrent.futures import ThreadPoolExecutor

'''
ESTE CÓDIGO PERMITE VISUALIZAR LAS GRÁFICAS EN CADA ETAPA DEL PROCESO 
'''

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

    # Normalizar FH_magnitude a 0-255 para que sea comparable con los umbrales
    FH_norm = FH_magnitude.astype(np.float32)
    FH_norm = (FH_norm - FH_norm.min()) / (FH_norm.max() - FH_norm.min() + 1e-8)
    FH_255 = (FH_norm * 255).astype(np.uint8)

    # Pixels válidos: dentro del ROI y dentro del rango de umbrales
    mask_rango = (FH_255 >= low) & (FH_255 <= high) & ROI_bool

    coords = np.argwhere(mask_rango)  # shape (N, 2)
    
    if coords.size == 0:
        return []# Si no hay píxeles, devolvemos listas vacías

    # Ordenar esos píxeles de mayor a menor intensidad
    intensidades = FH_255[mask_rango]
    order = np.argsort(intensidades)[::-1]# descendente
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


def clasificadorVesselPot(I_vessel_pot, I_vessel, modo="bridging"):

    # Convertimos a booleanos por seguridad
    I_vessel_pot = I_vessel_pot.astype(bool)
    I_vessel = I_vessel.astype(bool)

    # Si no hay venas previas, aceptamos el primer segmento
    if not np.any(I_vessel):
        return True

    kernel = np.ones((3, 3), np.uint8)

    if modo == "adjacency":
        # Dilatamos la red existente para permitir unión por proximidad (1 pixel)
        I_vessel_dilatada = cv2.dilate(I_vessel.astype(np.uint8), kernel, iterations=1).astype(bool)

        # El candidato se considera conectado si toca la red dilatada
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

def unirSegmentos(segmentos):
   if not segmentos:
       # Si está vacío, retornar una imagen vacía mínima de 1x1
       return np.zeros((1, 1), dtype=np.uint8)

   # Comenzamos con una máscara del tamaño del primer segmento
   unidos = np.zeros_like(segmentos[0], dtype=bool)

   for seg in segmentos:
       unidos |= seg.astype(bool)#Unión incremental

   return unidos.astype(np.uint8)


#%% CONDICION DE PARO
def calculandoIncrementoVasos(Iteraciones, cont):
    
    #if cont > 0 and len(Iteraciones) > 1:
    if cont > 0 and cont < len(Iteraciones):
        
        imagen_actual = Iteraciones[cont]
        imagen_anterior = Iteraciones[cont - 1]
        
        #diferencia píxel a píxel
        diferencia = imagen_actual - imagen_anterior
        
        #cntar cuántos pixeles nuevos de vasos aparecieron
        incremento = np.sum(diferencia == 1)#Solo cuenta donde el pixel pasó de 0 a 1
        
        #tamaño total de la imagen
        fil, col = imagen_actual.shape
        total = fil * col
        
        # C1 normalizado
        C1 = incremento / total
        
        return C1
    
    else:

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
    

#%% MAIN calculo de DF

def mostrarResultadosIteracion(cont,
                               binaria_iter,
                               segmentos_DF_ok,
                               segmentos_DF_no):

   #Función interna para unir segmentos con misma forma que binaria_iter
    def unirSegmentosLocal(lista_segmentos):
        if lista_segmentos:
            return np.logical_or.reduce(lista_segmentos)
        else:
            return np.zeros_like(binaria_iter, dtype=bool)

    img_ok  = unirSegmentosLocal(segmentos_DF_ok)
    img_no  = unirSegmentosLocal(segmentos_DF_no)

    plt.figure(figsize=(9, 5))

    #objetos detectados (binaria)
    plt.subplot(1, 2, 1)
    plt.imshow(binaria_iter, cmap='gray')
    plt.title(f'Binarización\n en el intervalo {cont+1}',  fontfamily='serif', fontstyle='italic', fontsize=15)
    plt.axis('off')

    #cumplen DF
    plt.subplot(1, 2, 2)
    plt.imshow(img_ok, cmap='gray')
    plt.title('Vasos validados', fontfamily='serif', fontstyle='italic', fontsize=15)
    plt.axis('off')


"""
Genera un kernel tipo Mexican Hat que correponde a la segubda derivada de Gauss.
"""
def kernelSombreroMexicano(M=31, sigma=4.0):
    if M % 2 == 0:
        M += 1
    x = np.linspace(-3*sigma, 3*sigma, M)
    kernel = (1 - (x**2)/(sigma**2)) * np.exp(-(x**2)/(2*sigma**2))
    kernel -= kernel.mean()
    return kernel



def calcularUmbralSombreroMexicano(df_values, n_bins=40, min_df=0.15, sigma=4.0,
                                   p_max=99, margin_bins=2, modo_valle="simple"):

    df_vals = np.array([v for v in df_values if v is not None], dtype=float)
    df_vals = df_vals[np.isfinite(df_vals)]

    if df_vals.size < 2:
        return None, None, None, None

    # Histograma
    hist, bin_edges = np.histogram(df_vals, bins=n_bins)

    if np.all(hist == 0):
        return None, None, None, None

    x_centros = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    #Mexican Hat
    M = len(hist)
    kernel = kernelSombreroMexicano(M=M, sigma=sigma)
    señal_filtrada = np.convolve(hist, kernel, mode='same')

    #Ajuste de longitudes
    if len(señal_filtrada) != len(x_centros):
        diff = len(señal_filtrada) - len(x_centros)
        start = diff // 2
        señal_filtrada = señal_filtrada[start:start + len(x_centros)]

    if len(señal_filtrada) != len(x_centros):
        min_len = min(len(señal_filtrada), len(x_centros))
        señal_filtrada = señal_filtrada[:min_len]
        x_centros = x_centros[:min_len]

    p_lim = np.percentile(df_vals, p_max)
    mascara_validos = (x_centros > min_df) & (x_centros < p_lim)

    if not np.any(mascara_validos):
        return None, None, x_centros, señal_filtrada

    idx_validos = np.where(mascara_validos)[0]


    idx_validos = idx_validos[(idx_validos >= margin_bins) &
                              (idx_validos < len(señal_filtrada) - margin_bins)]

    if idx_validos.size == 0:
        return None, None, x_centros, señal_filtrada

    def es_valle_simple(y, i):
        # Valle básico: y[i] < y[i-1] y y[i] < y[i+1]
        return (y[i] < y[i-1]) and (y[i] < y[i+1])

    def es_valle_strict(y, i, m):
        # Valle más estricto: baja hacia i y luego sube
        izq = y[i-m:i+1]
        der = y[i:i+m+1]
        baja = np.all(np.diff(izq) <= 0)
        sube = np.all(np.diff(der) >= 0)
        return baja and sube

    # Ordenar candidatos por valor (más bajo primero)
    candidatos = idx_validos[np.argsort(señal_filtrada[idx_validos])]

    idx_min_global = None
    for idx in candidatos:
        if modo_valle == "strict":
            ok = es_valle_strict(señal_filtrada, idx, margin_bins)
        else:
            ok = es_valle_simple(señal_filtrada, idx)

        if ok:
            idx_min_global = int(idx)
            break


    if idx_min_global is None:
        idx_min_local = np.argmin(señal_filtrada[mascara_validos])
        idx_min_global = int(np.where(mascara_validos)[0][idx_min_local])

    umbral = (bin_edges[idx_min_global] + bin_edges[idx_min_global + 1]) / 2.0
    return umbral, idx_min_global, x_centros, señal_filtrada




def visualizarSombreroMexicano(indice_iteracion,
                               df_values,
                               x_centros_df,
                               señal_df,
                               umbral_iter,
                               min_df=0.2,
                               n_bins_hist=40):
    
    df_vals = np.array([v for v in df_values if v is not None], dtype=float)# Limpiar lista de DF para el histograma
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
    
    plt.figure(figsize=(8,5))
    plt.hist(df_vals, bins=n_bins_hist, alpha=0.7, edgecolor='black')    
    plt.xlabel('DF', fontsize=12, fontfamily='serif')
    plt.ylabel('Frecuencia', fontsize=12, fontfamily='serif')
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(
        f"Histograma {indice_iteracion+1}.png",
        dpi=600,
        bbox_inches='tight',
        pad_inches=0
    )
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # Histograma izquierda
    ax_hist = axes[0]
    ax_hist.hist(df_vals, bins=n_bins_hist, alpha=0.7, edgecolor='black')
    ax_hist.set_xlabel('Dimensión fractal (DF)', fontsize=12, fontfamily='serif')
    ax_hist.set_ylabel('Frecuencia', fontsize=12, fontfamily='serif')
    ax_hist.set_title('Histograma DF',fontsize=12, fontfamily='serif')
    ax_hist.grid(alpha=0.3, linestyle='--')

    if umbral_iter is not None:
        ax_hist.axvline(umbral_iter, color='red', linestyle='--', label=f'Umbral {indice_iteracion+1} ({umbral_iter:.3f})')
    ax_hist.legend(loc='best')

    # Señal filtrada derecha, es decir, punto de cambio en el histograma 
    ax_filter = axes[1]
    ax_filter.plot(x_centros_df, señal_df, marker='o')
    ax_filter.set_xlabel('Dimensión fractal (DF)', fontsize=12, fontfamily='serif')
    ax_filter.set_ylabel('Respuesta filtrada', fontsize=12, fontfamily='serif')
    ax_filter.set_title('Trasformada wavelet', fontsize=12, fontfamily='serif')
    ax_filter.grid(alpha=0.3, linestyle='--')

    if umbral_iter is not None:
        ax_filter.axvline(umbral_iter, color='red', linestyle='--', label=f'Umbral {indice_iteracion+1}')
        ax_filter.legend(loc='best')

    fig.suptitle(f'Análisis del umbral - Intervalo {indice_iteracion+1}',  fontfamily='serif', fontstyle='italic', fontsize=15)
    plt.tight_layout()
    plt.show()



def visualizarEvolucionSegmentacion(FH_magnitude,
                                    clasificadorVessel,   
                                    unirSegmentos,
                                    clasificadorVesselPot,
                                    umbrales,
                                    ROI_bool):

    #Máscaras (bins) por rango de intensidades dentro del ROI, por numero de iteraciones
    #en este caso son dos, como se indica en el escrito
    mascaras_iter = dividirUmbralesBins(FH_magnitude, umbrales, ROI_bool, num_iter=2)

    cont = 0
    Iteraciones = []# Imagen de vasos global por iteración (acumulada)
    DF_iteraciones = []# Lista de DF por iteración
    C1_list, C2_list = [], []

    
    I_vessel_unido = np.zeros_like(ROI_bool, dtype=bool)# Máscara acumulada de venas aceptadas en todas las iteraciones

    for binaria in mascaras_iter:
        #Si no hay píxeles en este bin, se salta
        if binaria is None or not np.any(binaria):
            DF_iteraciones.append([])
            Iteraciones.append(I_vessel_unido.astype(np.uint8))
            cont += 1
            continue

        print(f"\n===== ITERACIÓN {cont+1} =====")

        # Etiquetado por conectividad (4-conectividad)
        imagen_label, num_etiquetas = etiquetaXConectividad(binaria, conectividad=1)
        print(f"Total de segmentos etiquetados en iteración {cont+1}: {num_etiquetas}")

        # Listas solo para visualización en esta iteración
        segmentos_DF_ok_iter = []#segmentos aceptados en esta iteración
        segmentos_DF_no_iter = []#segmentos rechazados en esta iteración

        #1- Se obtiene DF de cada segmento
        def calcularSegmento(i):
            seg = (imagen_label == i)
            DF = calcularDF(seg)
            return i, seg, DF

        DF_actual = [None] * num_etiquetas

        with ThreadPoolExecutor() as executor:
            resultados = list(executor.map(calcularSegmento,
                                           range(1, num_etiquetas + 1)))

        for i, seg, DF in resultados:
            DF_actual[i - 1] = DF

        #2-Calcular umbral por Mexican Hat sobre DF de esta iteración
        umbral_iter, idx_min_dfbin, x_centros_df, señal_df = calcularUmbralSombreroMexicano(
            DF_actual,
            n_bins=40,
            min_df=0.2,
            sigma=4.0
        )

        #Visualización del sombrero mexicano por iteración
        
        visualizarSombreroMexicano(
            indice_iteracion=cont,
            df_values=DF_actual,
            x_centros_df=x_centros_df,
            señal_df=señal_df,
            umbral_iter=umbral_iter,
            min_df=0.05,
            n_bins_hist=40
        )

        #Visualización de la RED VASCULAR según el subumbral de DF 
        
        if umbral_iter is not None:
            mascara_sup = np.zeros_like(binaria, dtype=bool)  # DF > subumbral
            mascara_inf = np.zeros_like(binaria, dtype=bool)  # DF <= subumbral

            for idx_etiqueta in range(1, num_etiquetas + 1):
                DF = DF_actual[idx_etiqueta - 1]
                if DF is None or not np.isfinite(DF):
                    continue

                if DF > umbral_iter:
                    mascara_sup[imagen_label == idx_etiqueta] = True
                else:
                    mascara_inf[imagen_label == idx_etiqueta] = True

            # Figura: red vascular separada por subumbral
            plt.figure(figsize=(8, 4))

            plt.subplot(1, 2, 1)
            plt.imshow(mascara_sup, cmap="gray")
            plt.title(f"Iter {cont+1}  DF > subumbral", fontsize=12, fontfamily='serif')
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(mascara_inf, cmap="gray")
            plt.title(f"Iter {cont+1}  DF ≤ subumbral", fontsize=12, fontfamily='serif')
            plt.axis("off")

            plt.suptitle(f"Red vascular separada por subumbral DF Iteración {cont+1}", fontfamily='serif', fontstyle='italic', fontsize=15)
            plt.tight_layout()
            plt.show()
            

            mascara_sup_mod = mascara_sup.copy()
            mascara_inf_mod = mascara_inf.copy()
            
            
            mascara_sup_mod[ROI_bool == 0] = 255
            mascara_inf_mod[ROI_bool == 0] = 255

            #Subp 1
            plt.figure()
            plt.imshow(mascara_sup_mod, cmap="gray")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(
                f"Intervalo{cont+1}_umbral1.png",
                dpi=600, 
                bbox_inches='tight', 
                pad_inches=0 
            )
            plt.close()
            
            #Subp 2
            plt.figure()
            plt.imshow(mascara_inf_mod, cmap="gray")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(
                f"Intervalo{cont+1}_umbral2.png",
                dpi=600,           
                bbox_inches='tight',  
                pad_inches=0          
            )
            plt.close()

        for idx_etiqueta in range(1, num_etiquetas + 1):
            DF = DF_actual[idx_etiqueta - 1]
            if DF is None or not np.isfinite(DF) or umbral_iter is None:
                #Si el DF es inválido o no hay umbral, se rechaza el segmento
                segmentos_DF_no_iter.append((imagen_label == idx_etiqueta).astype(np.uint8))
                continue

            seg_mask = (imagen_label == idx_etiqueta)
            aceptado = False

            if cont == 0:
                ###ITERACIÓN 1 DF es la red vascular gruesa
                if DF > umbral_iter:
                    aceptado = True
                else:
                    #Si ya existe red acumulada, verificamos si el segmento sirve de puente
                    if np.any(I_vessel_unido):
                        aceptado = clasificadorVesselPot(seg_mask,
                                                         I_vessel_unido,
                                                         modo="bridging")
            else:
                #I#TERACIÓN 2 DF > umbral y adyacente a la red global acumulada
                if DF > umbral_iter and np.any(I_vessel_unido):
                    aceptado = clasificadorVesselPot(seg_mask,
                                                     I_vessel_unido,
                                                     modo="adjacency")

            if aceptado:
                I_vessel_unido |= seg_mask
                segmentos_DF_ok_iter.append(seg_mask.astype(np.uint8))
            else:
                segmentos_DF_no_iter.append(seg_mask.astype(np.uint8))

        #Guardamos los DF de esta iteración
        DF_iteraciones.append(DF_actual)
        print(f"DF_iteración {cont+1}: {DF_actual}")

        #Imagen de vasos de esta iteración = máscara acumulada hasta ahora
        Iteraciones.append(I_vessel_unido.astype(np.uint8))

        #Métricas C1, C2, C3 que sib la velocidad y aceleracion de como se incorporan los segmentos 
        try:
            C1 = calculandoC1(Iteraciones, cont)
            C1_list.append(C1)
            C2 = calculandoVelocidad(C1_list, cont)
            C2_list.append(C2)
            C3 = calculandoAceleracion(C2_list, cont)
        except NameError:

            pass

    #Visualización de máscaras aceptadas/rechazadas
        try:
            mostrarResultadosIteracion(cont,
                                       binaria_iter=binaria,
                                       segmentos_DF_ok=segmentos_DF_ok_iter,
                                       segmentos_DF_no=segmentos_DF_no_iter)
        except NameError:
            pass

        cont += 1

    return Iteraciones, DF_iteraciones
