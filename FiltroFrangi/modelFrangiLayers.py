from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.models import Sequential, Model
from FiltroFrangi.frangi_module import FrangiLayer, ScaleLayer, Scaling01


#%% Crea la red Frangi, el numero de neuronas depende de sigma 

"""
Esta red combina múltiples respuestas del filtro de Frangi calculadas para distintos valores de sigma y 
aprende un factor de escala para cada una, con el objetivo de producir una respuesta final normalizada que
resalte estructuras tipo vasos.
Esta función construye la arquitectura principal de la red basada en Frangi a partir de una imagen de entrada 
en escala de grises con forma (alto, ancho, 1). Para cada sigma, se aplica una capa Frangi seguida de una capa de 
escala entrenable (ScaleLayer) que ajusta la contribución de esa respuesta. Luego, se utiliza una operación
de máximo (Maximum) para seleccionar, píxel por píxel, la respuesta más alta entre todos los sigmas. 
Finalmente, una capa de normalización (Scaling01) lleva la salida al rango [0, 1].
"""

def crearRedNeuronal():
    inp = Input(shape=(None, None, 1))
    abc_list = []
    
    for sigma in [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 5, 6, 7, 8, 9, 10]:     
    # Esta parte se encarga de:
    #Aplicar FrangiLayer para cada sigma.
    # Aplicar ScaleLayer para que la red aprenda un peso distinto para cada respuesta.
    #Guardar cada salida en una lista.
    
        ml = FrangiLayer(sigma=sigma)(inp) # from frani_module
        ml = ScaleLayer()(ml) #from frangi_module
        abc_list.append(ml)
    
    x = keras.layers.Maximum()(abc_list)
    x = Scaling01()(x) #from frangi_module
    model = Model(inp, x)
    model.summary()
    
    return x, model