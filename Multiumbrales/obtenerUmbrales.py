import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from Multiumbrales.inicializarPoblacionHalcones import inicializarPoblacionHalcones
from Multiumbrales.calcularEntropiaMultiumbral import calcularEntropiaMultiumbral
from Multiumbrales.generarPasoLevy import generarPasoLevy
from Multiumbrales.aplicarCuantizacionMultiumbral import aplicarCuantizacionMultiumbral
import matplotlib.pyplot as plt

plt.close('all')


"""
Obtiene los umbrales óptimos de la imagen usando HHO + MCEM.
A partir de una imagen en escala de grises (por ejemplo la salida de un filtro de Frangi), aplica el algoritmo 
Harris Hawks Optimization (HHO) utilizando como función objetivo la medida de entropía multiumbral (calcularEntropiaMultiumbral). 
Al final, calcula la imagen multiumbralizada, imprime métricas de calidad y regresa la lista de umbrales óptimos encontrados.
Patra ello se coloca la imagen en escala de grises y regresa un arreglo con los valores optimos y ordenados
*IMPORTANTE entrada en uint8 (0-255)

"""
#%% MAIN de los umbrales 

def obtenerUmbrales(imagenFrangi, numeroUmbrales):

    imagenGris = imagenFrangi.astype(np.uint8)

    histograma, _ = np.histogram(imagenGris, bins=256, range=(0, 255)) # Histograma de 256 niveles

    #Parámetros HHO
    #numeroUmbrales = 2      # número de umbrales a buscar
    umbralMinimo = 1
    umbralMaximo = 255
    numeroHalcones = 30     # tamaño de la población
    maxIteraciones = 100    # número de iteraciones de HHO

    mejorUmbral = np.zeros(numeroUmbrales, dtype=int)
    mejorEnergia = np.inf

    poblacionUmbrales = inicializarPoblacionHalcones(
        numeroHalcones, numeroUmbrales, umbralMaximo, umbralMinimo
    )

    historialEnergia = []

    for iteracion in range(maxIteraciones):
        # Evaluar cada halcón
        for indiceHalcon in range(poblacionUmbrales.shape[0]):
            # Asegurar límites
            umbralesActuales = poblacionUmbrales[indiceHalcon, :].copy()
            umbralesActuales[umbralesActuales > umbralMaximo] = umbralMaximo
            umbralesActuales[umbralesActuales < umbralMinimo] = umbralMinimo
            umbralesActuales = np.sort(np.rint(umbralesActuales).astype(int))

            valorFitness = calcularEntropiaMultiumbral(
                imagenGris, umbralesActuales, histograma
            )

            if valorFitness < mejorEnergia:
                mejorEnergia = valorFitness
                mejorUmbral = umbralesActuales

            poblacionUmbrales[indiceHalcon, :] = umbralesActuales

        # Actualización HHO 
        energiaInicial = 2 * (1 - (iteracion / maxIteraciones))

        for indiceHalcon in range(poblacionUmbrales.shape[0]):
            energiaAleatoria = 2 * np.random.rand() - 1
            energiaEscape = energiaInicial * energiaAleatoria

            if abs(energiaEscape) >= 1:
                # Fase de exploración
                probExploracion = np.random.rand()
                indiceHalconAleatorio = np.random.randint(0, numeroHalcones)
                halconAleatorio = poblacionUmbrales[indiceHalconAleatorio, :]

                if probExploracion < 0.5:
                    poblacionUmbrales[indiceHalcon, :] = (
                        halconAleatorio -
                        np.random.rand() *
                        np.abs(halconAleatorio - 2 * np.random.rand() * poblacionUmbrales[indiceHalcon, :])
                    )
                else:
                    poblacionUmbrales[indiceHalcon, :] = (
                        (mejorUmbral - np.mean(poblacionUmbrales, axis=0)) -
                        np.random.rand() *
                        ((umbralMaximo - umbralMinimo) * np.random.rand() + umbralMinimo)
                    )
            else:
                # Fase de explotación
                probExplotacion = np.random.rand()

                if probExplotacion >= 0.5 and abs(energiaEscape) < 0.5:
                    poblacionUmbrales[indiceHalcon, :] = np.abs(
                        mejorUmbral -
                        energiaEscape *
                        np.abs(mejorUmbral - poblacionUmbrales[indiceHalcon, :])
                    )

                if probExplotacion >= 0.5 and abs(energiaEscape) >= 0.5:
                    fuerzaSalto = 2 * (1 - np.random.rand())
                    poblacionUmbrales[indiceHalcon, :] = np.abs(
                        (mejorUmbral - poblacionUmbrales[indiceHalcon, :]) -
                        energiaEscape *
                        np.abs(fuerzaSalto * mejorUmbral - poblacionUmbrales[indiceHalcon, :])
                    )

                if probExplotacion < 0.5 and abs(energiaEscape) >= 0.5:
                    fuerzaSalto = 1 - np.random.rand()
                    umbralesPropuestos1 = np.abs(
                        mejorUmbral -
                        energiaEscape *
                        np.abs(fuerzaSalto * mejorUmbral - poblacionUmbrales[indiceHalcon, :])
                    ).astype(int)
                    umbralesPropuestos1 = np.clip(
                        umbralesPropuestos1, umbralMinimo, umbralMaximo
                    )

                    if calcularEntropiaMultiumbral(
                        imagenGris, umbralesPropuestos1, histograma
                    ) < calcularEntropiaMultiumbral(
                        imagenGris, poblacionUmbrales[indiceHalcon, :], histograma
                    ):
                        poblacionUmbrales[indiceHalcon, :] = umbralesPropuestos1
                    else:
                        umbralesPropuestos2 = np.abs(
                            mejorUmbral -
                            energiaEscape *
                            np.abs(fuerzaSalto * mejorUmbral - poblacionUmbrales[indiceHalcon, :]) +
                            np.random.rand(numeroUmbrales) * generarPasoLevy(numeroUmbrales)
                        ).astype(int)
                        umbralesPropuestos2 = np.clip(
                            umbralesPropuestos2, umbralMinimo, umbralMaximo
                        )

                        if calcularEntropiaMultiumbral(
                            imagenGris, umbralesPropuestos2, histograma
                        ) < calcularEntropiaMultiumbral(
                            imagenGris, poblacionUmbrales[indiceHalcon, :], histograma
                        ):
                            poblacionUmbrales[indiceHalcon, :] = umbralesPropuestos2

                if probExplotacion < 0.5 and abs(energiaEscape) < 0.5:
                    fuerzaSalto = 2 * (1 - np.random.rand())
                    umbralesPropuestos1 = np.abs(
                        mejorUmbral -
                        energiaEscape *
                        np.abs(fuerzaSalto * mejorUmbral - np.mean(poblacionUmbrales, axis=0))
                    ).astype(int)
                    umbralesPropuestos1 = np.clip(
                        umbralesPropuestos1, umbralMinimo, umbralMaximo
                    )

                    if calcularEntropiaMultiumbral(
                        imagenGris, umbralesPropuestos1, histograma
                    ) < calcularEntropiaMultiumbral(
                        imagenGris, poblacionUmbrales[indiceHalcon, :], histograma
                    ):
                        poblacionUmbrales[indiceHalcon, :] = umbralesPropuestos1
                    else:
                        umbralesPropuestos2 = np.abs(
                            mejorUmbral -
                            energiaEscape *
                            np.abs(fuerzaSalto * mejorUmbral - np.mean(poblacionUmbrales, axis=0)) +
                            np.random.rand(numeroUmbrales) * generarPasoLevy(numeroUmbrales)
                        ).astype(int)
                        umbralesPropuestos2 = np.clip(
                            umbralesPropuestos2, umbralMinimo, umbralMaximo
                        )

                        if calcularEntropiaMultiumbral(
                            imagenGris, umbralesPropuestos2, histograma
                        ) < calcularEntropiaMultiumbral(
                            imagenGris, poblacionUmbrales[indiceHalcon, :], histograma
                        ):
                            poblacionUmbrales[indiceHalcon, :] = umbralesPropuestos2

           
            poblacionUmbrales[indiceHalcon, :] = np.clip(  # Nos seguramos enteros y orden de los umbrales
                np.rint(poblacionUmbrales[indiceHalcon, :]).astype(int),
                umbralMinimo,
                umbralMaximo
            )
            poblacionUmbrales[indiceHalcon, :] = np.sort(poblacionUmbrales[indiceHalcon, :])

        historialEnergia.append(mejorEnergia)

    
    imagenUmbralizada = aplicarCuantizacionMultiumbral(imagenGris, mejorUmbral) # Segmentación final con los mejores umbrales

    # Métricas de calidad
    valorPsnr = peak_signal_noise_ratio(imagenGris, imagenUmbralizada, data_range=255)
    valorSsim = structural_similarity(imagenGris, imagenUmbralizada, data_range=255)

    print("Fitness (MCEM):", mejorEnergia)
    print("Umbrales óptimos:", mejorUmbral)
    print("PSNR:", valorPsnr)
    print("SSIM:", valorSsim)

    return mejorUmbral


