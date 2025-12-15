import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.models import Sequential, Model


#%% FUNCIONES AUXILIARES DEL FILTRO GAUSSIANO

#Genera un kernel Gaussiano 2D para ser usado en convolución.

def _gaussian_kernel_2d(sigma, filter_shape, n_channels, dtype=tf.float32):
    # El centro del kernel
    center = filter_shape // 2

    
    x = tf.cast(tf.range(-center, center + 1), dtype=dtype)# Crea una cuadrícula de coordenadas 2D (x, y)
    y = tf.cast(tf.range(-center, center + 1), dtype=dtype)
    xx, yy = tf.meshgrid(x, y)

    kernel = tf.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))# Aplica la fórmula Gaussiana

    sum_kernel = tf.reduce_sum(kernel)# Normaliza el kernel para que la suma sea 1
    kernel = kernel / sum_kernel

    # Prepara el kernel para tf.nn.depthwise_conv2d:
    # [height, width, in_channels, channel_multiplier=1]
    kernel = tf.expand_dims(kernel, axis=-1)  # [height, width, 1]
    kernel = tf.expand_dims(kernel, axis=-1)  # [height, width, 1, 1]
    kernel = tf.tile(kernel, [1, 1, n_channels, 1]) # [height, width, n_channels, 1]

    return kernel

def _gaussian_filter2d_custom(inputs, sigma, filter_shape): #Realiza un filtrado Gaussiano 2D en los inputs usando tf.nn.depthwise_conv2d.
    n_channels = tf.shape(inputs)[-1]
    kernel = _gaussian_kernel_2d(sigma, filter_shape, n_channels)
    kernel = tf.cast(kernel, inputs.dtype)

    # Calcula el padding necesario para la convolución 'VALID' para simular 'REFLECT'
    padding_size = (filter_shape - 1) // 2
    paddings = [[0, 0], [padding_size, padding_size], [padding_size, padding_size], [0, 0]]
    padded_inputs = tf.pad(inputs, paddings, "REFLECT")

    return tf.nn.depthwise_conv2d(# Aplica la convolución. depthwise_conv2d aplica el mismo filtro canal a canal.
        padded_inputs,
        kernel,
        strides=[1, 1, 1, 1], # stride de 1
        padding='VALID'       # Usamos 'VALID' porque ya aplicamos el padding
    )

#%% ESCALADO 0 A 1
"""
Esta capa tiene como función normalizar un tensor al rango [0, 1]. Para ello, primero se calcula el valor mínimo 
del tensor y se resta a todos sus elementos, desplazando los datos para que comiencen en cero. Posteriormente, el 
resultado se divide entre el valor máximo obtenido, logrando así una escala normalizada.
Este proceso es fundamental porque la combinación final de múltiples capas Frangi puede producir salidas con distintos 
rangos numéricos, por lo que se requiere una salida estandarizada para su correcta
integración con otros módulos del sistema.
"""

class Scaling01(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs):
        min_value = tf.math.reduce_min(inputs)
        max_value = tf.math.reduce_max(inputs)
        # Maneja la división por cero si max_value == min_value
        denominator = tf.maximum(max_value - min_value, tf.keras.backend.epsilon())
        return_tensor = tf.math.divide(tf.math.subtract(inputs, min_value), denominator)
        return return_tensor

    def get_config(self):
        config = super(Scaling01, self).get_config()
        return config
    
#%% CAPA ENTRENABLE
"""
Esta capa se encarga de aplicar un factor de escala entrenable sobre la imagen filtrada por Frangi. 
Cada valor de sigma cuenta con su propia ScaleLayer, lo que permite al modelo aprender de forma independiente
cuánto peso asignar a cada respuesta del filtro. El parámetro 'factorEscala' es un valor entrenable que se 
ajusta automáticamente durante el proceso de entrenamiento.
"""
class ScaleLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs): #Capa con un parámetro de escala entrenable, restringido a [0, 1].
        super().__init__(**kwargs)

    def build(self, input_shape):
        # Restricción MinMaxNorm es una capa Keras estándar, no requiere addons
        kernel_constraint_scale = tf.keras.constraints.MinMaxNorm(min_value=0.0, max_value=1.0)
        initializer = tf.keras.initializers.Ones()

        self.scale = self.add_weight(name="scale",
                                     shape=(1, 1), initializer=initializer, trainable=True,
                                     constraint=kernel_constraint_scale
                                    )

    def call(self, inputs):
        return inputs * self.scale[0,0]

    def get_config(self):
        config = super(ScaleLayer, self).get_config()
        return config

#%% CAPA MAIN FRANGI EN 2D

"""
Esta capa se encarga de aplicar el filtro Frangi a una imagen en escala de grises mediante una secuencia 
de operaciones bien definidas. Primero, se realiza un suavizado Gaussiano en función del valor de sigma, seguido
del cálculo de derivadas segundas para obtener la matriz Hessiana. A partir de esta, se construyen los tensores 
Hxx, Hxy y Hyy, y se calculan sus autovalores.
Posteriormente, se obtienen distintas medidas, incluyendo la raíz cuadrada de la relación entre autovalores y 
una métrica diseñada para detectar estructuras tubulares tipo vasos. Finalmente, estas medidas se combinan para generar 
la respuesta final del filtro Frangi.
"""

class FrangiLayer(keras.layers.Layer):
    def __init__(self, sigma=1.0, **kwargs):
        self.sigma = tf.constant(sigma, dtype=tf.float32) # Convertir a tf.constant
        super().__init__(**kwargs)

    def build(self, input_shape):

        # Las restricciones y los inicializadores son estándar de Keras
        kernel_constraint_beta = tf.keras.constraints.MinMaxNorm(
            min_value=0.05, max_value=10.0, rate=1.0, axis=0
        )
        kernel_constraint_gamma = tf.keras.constraints.MinMaxNorm(
            min_value=0.05, max_value=50.0, rate=1.0, axis=0
        )

        my_initializer_beta = tf.keras.initializers.RandomUniform(minval=0.05, maxval=10.0, seed=None)
        my_initializer_gamma = tf.keras.initializers.RandomUniform(minval=0.05, maxval=50.0, seed=None)

        self.beta = self.add_weight(name="beta",
            shape=(1,1), initializer=my_initializer_beta, trainable=True, constraint=kernel_constraint_beta)
        self.gamma = self.add_weight(name="gamma",
                                     shape=(1, 1), initializer=my_initializer_gamma, trainable=True,
                                     constraint=kernel_constraint_gamma)

    def get_config(self):
        config = super(FrangiLayer, self).get_config()
        config.update({
            'sigma': self.sigma.numpy().item() # Guardar el valor float en la config
        })
        return config

    def hessian_matrix(self, inputs, sigma):# Función actualizada para usar el filtro Gaussiano implementado con TensorFlow Core
        # size of the gaussian kernel (see scipy implementation)
        truncate = 4.0
        # Calcular filter_shape como un entero impar
        filter_shape = 2 * tf.cast(truncate * sigma + 0.5, tf.int32) + 1

        # Uso de la función _gaussian_filter2d_custom que solo utiliza TensorFlow Core
        ggg = _gaussian_filter2d_custom(inputs, sigma, filter_shape)

        # first partial derivative
        # tf.image.image_gradients es una función de TensorFlow Core
        ggg = tf.image.image_gradients(ggg)
        # second partial derivative
        gggx = tf.image.image_gradients(ggg[0]) # dxx, dxy
        gggy = tf.image.image_gradients(ggg[1]) # dyx, dyy

        # La matriz Hessiana es simétrica: dxy = dyx (cercana)
        return [gggx[0], gggx[1], gggy[1], gggy[0]] # dxx, dxy, dyy, dyx (para M00, M01, M11, M10)


    def calculate_frangi_sigma(self, inputs, sigma, beta, gamma): #Calcula el filtro de Frangi para un solo sigma.
        gamma_2 = gamma * gamma
        beta_2 = beta * beta
        hm = self.hessian_matrix(inputs, sigma)

        # rescalling by sigma^2
        sigma_sq = sigma * sigma
        M00 = tf.math.scalar_mul(sigma_sq, hm[0]) # dxx
        M01 = tf.math.scalar_mul(sigma_sq, hm[1]) # dxy (== dyx)
        M11 = tf.math.scalar_mul(sigma_sq, hm[2]) # dyy
        # M10 = tf.math.scalar_mul(sigma_sq, hm[3]) # dyx

        # calculate eigenvalues of Hessian matrix in every point
        l_left = (M00 + M11) / 2.0001
        Mdiff = M00 - M11
        # El radicando del cuadrado es: (M00 - M11)^2 + 4 * M01^2
        l_right = 4.0 * tf.math.multiply(M01, M01) + tf.math.multiply(Mdiff, Mdiff)
        l_right = tf.math.sqrt(l_right) / 2.0001
        l1 = l_left + l_right
        l2 = l_left - l_right

        # order eigenvalues by ascending module: |l1| < |l2|
        l1_abs = tf.math.abs(l1)
        l2_abs = tf.math.abs(l2)
        l1g = tf.greater(l1_abs, l2_abs) # True si |l1| > |l2|

        # l1_sorted = min(|l1|, |l2|), l2_sorted = max(|l1|, |l2|)
        l1_sorted = tf.where(l1g, l2, l1) # El de menor módulo
        l2_sorted = tf.where(l1g, l1, l2) # El de mayor módulo

        l1 = l1_sorted
        l2 = l2_sorted

        # replace zeros by small values
        l2_divider = tf.where(tf.equal(l2, 0), tf.keras.backend.epsilon(), l2)

        # calculate RB = |l1| / |l2| (aspect ratio)
        RB = tf.math.divide(tf.math.abs(l1), tf.math.abs(l2_divider))
        RB_2 = tf.math.multiply(RB, RB)

        # calculate S = sqrt(l1^2 + l2^2) (frobenius norm of hessian)
        S_2 = tf.math.multiply(l1, l1) + tf.math.multiply(l2, l2)

        # calculate Frangi filter response
        exp_line = tf.math.exp(-tf.math.divide(RB_2, 2.00001 * beta_2))
        exp_back = 1.0 - (tf.math.exp(-tf.math.divide(S_2, 2.0001 * gamma_2)))
        VS = tf.math.multiply(exp_line, exp_back)

        # Filter response is zero if l2 (el mayor autovalor en módulo) > 0
        l2g0 = tf.greater(l2, 0.0)
        l2g0 = tf.cast(l2g0, tf.bool)
        VS = tf.where(l2g0, 0.0, VS)

        return VS

    def call(self, inputs):
        sigmas = [self.sigma]
        # Acceder a beta y gamma como escalares
        beta = self.beta[0,0]
        gamma = self.gamma[0,0]
        VS = None

        for sigma in sigmas:
            # Convierte sigma a float si es necesario
            sigma = tf.cast(sigma, tf.float32)

            if VS is None:
                VS = self.calculate_frangi_sigma(inputs, sigma, beta, gamma)
            else:
                VS1 = self.calculate_frangi_sigma(inputs, sigma, beta, gamma)
                # get maximum of filters
                VS = tf.math.maximum(VS, VS1)
        return VS
