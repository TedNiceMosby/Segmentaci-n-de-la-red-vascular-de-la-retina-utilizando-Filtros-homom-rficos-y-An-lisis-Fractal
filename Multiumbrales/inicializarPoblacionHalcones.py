import numpy as np

#%% INICIALIZA POBLACIÓN
"""
Esta función inicializa la población de halcones para el algoritmo de optimización HHO aplicado al problema de 
umbralización. Cada halcón representa una posible solución, definida por un conjunto de umbrales que serán optimizados 
durante el proceso de búsqueda.
La población se genera respetando los límites inferior y superior del espacio de búsqueda, los cuales pueden 
definirse como valores escalares o como arreglos independientes para cada variable de decisión. El resultado es un 
arreglo bidimensional con los valores iniciales de los umbrales, redondeados a enteros, listo para iniciar el proceso de
optimización.
"""


def inicializarPoblacionHalcones(numeroHalcones, numeroVariables, limiteSuperior, limiteInferior):

    limiteSuperior = np.array(limiteSuperior, dtype=float)
    limiteInferior = np.array(limiteInferior, dtype=float)

    if limiteSuperior.size == 1:
        poblacionUmbrales = (
            np.random.rand(numeroHalcones, numeroVariables) *
            (limiteSuperior - limiteInferior) + limiteInferior
        )
    else:
        poblacionUmbrales = np.zeros((numeroHalcones, numeroVariables))
        for indiceVariable in range(numeroVariables):
            maximo = limiteSuperior[indiceVariable]
            minimo = limiteInferior[indiceVariable]
            poblacionUmbrales[:, indiceVariable] = (
                np.random.rand(numeroHalcones) * (maximo - minimo) + minimo
            )

    return np.rint(poblacionUmbrales).astype(int)
