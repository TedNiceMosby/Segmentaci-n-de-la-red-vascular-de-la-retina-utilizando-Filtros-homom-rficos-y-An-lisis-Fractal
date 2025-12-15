import numpy as np
import plotly.graph_objs as go
from pathlib import Path


"""
Esta función genera una proyección 3D del globo ocular a partir de una imagen de fondo de ojo y su 
máscara vascular, y guarda el resultado en un archivo HTML interactivo. La máscara se utiliza para resaltar la red
de vasos añadiendo relieve sobre la superficie reconstruida.

Permite reducir la resolución mediante un factor de submuestreo y controlar la altura del relieve aplicado 
en la zona vascular. Devuelve la figura 3D generada (Plotly) para poder visualizarla o manipularla
posteriormente.
"""


def generarProyeccion3D(Imagen,
                        mascara_vasos,
                        escala_reduccion=4,
                        altura_relieve=15,
                        nombre_html="proyeccion_desde_centro_esfera.html"):

    Imagen = np.asarray(Imagen)
    
    if Imagen.ndim == 2:# Si viene en escala de grises, replicamos a 3 canales
        Imagen = np.stack([Imagen]*3, axis=-1)

    
    if Imagen.dtype != np.float32 and Imagen.dtype != np.float64:# Normalizar entre 0 y 1
        imagen_normalizada = Imagen.astype(np.float32) / 255.0
    else:
        imagen_normalizada = np.clip(Imagen, 0, 1)

    
    mascara_vasos = np.asarray(mascara_vasos)# Asegurar máscara booleana
    if mascara_vasos.dtype != bool:
        mascara_vasos = mascara_vasos > 0

    # *IMPORTANTE -REDUCCIÓN DE RESOLUCIÓn
    imagen_reducida = imagen_normalizada[::escala_reduccion, ::escala_reduccion, :]
    mascara_venas_reducida = mascara_vasos[::escala_reduccion, ::escala_reduccion]

    alto_img, ancho_img = mascara_venas_reducida.shape


    centro_x = ancho_img // 2 #DEFINICIÓN DEL CENTRO DE LA ESFERA 
    centro_y = alto_img // 2

    grid_y, grid_x = np.meshgrid(np.arange(alto_img),
                                 np.arange(ancho_img),
                                 indexing='ij')

    coord_relativas_x = grid_x - centro_x
    coord_relativas_y = grid_y - centro_y
    distancia_cuadrada = coord_relativas_x**2 + coord_relativas_y**2

    radio_esfera = min(centro_x, centro_y) - 1
    profundidad_base = np.zeros_like(distancia_cuadrada, dtype=float)

    region_valida = distancia_cuadrada < radio_esfera**2
    profundidad_base[region_valida] = np.sqrt(radio_esfera**2 - distancia_cuadrada[region_valida])
    profundidad_con_relieve = profundidad_base + (mascara_venas_reducida * altura_relieve)#APLICACIÓN DEL RELIEVE 
    
    pixeles_rgb = imagen_reducida.reshape(-1, 3) #MAPEO DE COLOR ORIGINA

    def convertir_a_hex(rgb):
        return '#%02x%02x%02x' % tuple((np.clip(rgb, 0, 1) * 255).astype(int))

    colores_hex = [convertir_a_hex(pixel) for pixel in pixeles_rgb]
    
    x_total = coord_relativas_x.flatten()
    y_total = coord_relativas_y.flatten()
    z_total = profundidad_con_relieve.flatten()
    mascara_valida_flat = region_valida.flatten()

    x_muestra = x_total[mascara_valida_flat]
    y_muestra = y_total[mascara_valida_flat]
    z_muestra = z_total[mascara_valida_flat]
    colores_muestra = [colores_hex[i] for i in range(len(colores_hex)) if mascara_valida_flat[i]]

    figura_3d = go.Figure(data=[go.Scatter3d(
        x=x_muestra,
        y=y_muestra,
        z=z_muestra,
        mode='markers',
        marker=dict(
            size=1.5,
            color=colores_muestra,
            opacity=1
        )
    )])

    figura_3d.update_layout(
        title='Proyección 3D',
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, t=30, b=0)
    )

    carpeta_modulo = Path(__file__).resolve().parent
    ruta_html = carpeta_modulo / nombre_html
    figura_3d.write_html(str(ruta_html))

    return figura_3d, ruta_html


def mostrarProyeccion3D(Imagen, mascara_vasos):
    import webview

    #Generar la proyección 3D y obtener la ruta del HTML
    _, ruta_html = generarProyeccion3D(
        Imagen,
        mascara_vasos,
        escala_reduccion=2,
        altura_relieve=15,
        nombre_html="proyeccion_desde_centro_esfera.html"
    )

    #Abrir el HTML con webview usando el file
    webview.create_window("Proyección 3D", ruta_html.as_uri())
    webview.start()












