import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import scrolledtext
import customtkinter as ctk
from PIL import Image, ImageTk
import numpy as np
import threading
import cv2
import os
from pathlib import Path


from FiltroGaussiano.filtroGaussiano import filtroGaussiano, elegirSigmaGauss
from FiltroHomomorfico.filtroHomomorfico import (
    encontrarSigma,
    aplicarFiltroHomomorfico,
)

from modelFrangiLayers import crearRedNeuronal

from AnalisisFractal.analisisDimensionFractal9 import (
    visualizarEvolucionSegmentacion, 
    clasificadorVessel, 
    unirSegmentos, 
    clasificadorVesselPot
)


from Complementos.proyeccion3D1 import generarProyeccion3D, mostrarProyeccion3D
from Complementos.mascaraROI import obtenerMascara
from Multiumbrales.obtenerUmbrales import obtenerUmbrales
from Complementos.imagenConMascara import sobreponerMascaraVerde

from skimage import io, img_as_float
from skimage.color import rgb2gray


def normalizarAUint8(img):

    arr = np.array(img, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    mn = arr.min()
    mx = arr.max()
    if mx - mn < 1e-6:
        return np.zeros_like(arr, dtype=np.uint8)
    esc = (arr - mn) / (mx - mn)
    return (esc * 255.0).clip(0, 255).astype(np.uint8)



def cargarImagen(ruta):
    try:
        imagenColor = img_as_float(io.imread(ruta))

        if imagenColor.ndim == 2:
            imagenProcesada = imagenColor

        elif imagenColor.ndim == 3 and imagenColor.shape[2] == 3:
            imagenProcesada = imagenColor[:, :, 1]  # Canal verde

        elif imagenColor.ndim == 3 and imagenColor.shape[2] == 4:
            imagenProcesada = imagenColor[:, :, 1]  # Canal verde

        else:
            print("[AVISO] Número de canales inesperado. Convirtiendo a escala de grises.")
            imagenProcesada = rgb2gray(imagenColor)

        return imagenProcesada, imagenColor

    except Exception as e:
        print(f"[ERROR] No se pudo cargar '{ruta}': {e}")
        return None, None


class PaginaBase(tk.Frame):
    def __init__(self, app, contenedor, *args, **kwargs):
        super().__init__(contenedor, *args, **kwargs)
        self.app = app

class BarraLateral(tk.Frame):
    def __init__(self, app, contenedor, *args, **kwargs):
        super().__init__(contenedor, *args, **kwargs)
        self.app = app

        self.config(bg="#c3c3c3")

        ANCHO_BOTON = 18

        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=1)
        self.grid_rowconfigure(3, weight=0)
        self.grid_columnconfigure(0, weight=1)

        self.indicadorActivo = None

        self.marcoSuperior = tk.Frame(self, bg="#c3c3c3")
        self.marcoSuperior.grid(row=0, column=0, sticky="nsew", pady=(15, 10))

        self.filaCargar = tk.Frame(self.marcoSuperior, bg="#c3c3c3")#CARGAR IMAGEN
        self.filaCargar.grid(row=0, column=0, sticky="ew", pady=10)
        self.filaCargar.grid_columnconfigure(1, weight=1)

        self.indicadorCargar = tk.Frame(self.filaCargar, bg="#c3c3c3", width=8, height=30)
        self.indicadorCargar.grid(row=0, column=0, sticky="nsw")
        self.indicadorCargar.grid_propagate(False)

        self.botonCargar = tk.Button(
            self.filaCargar,
            text="Cargar imagen",
            font=('Arial', 15, 'bold'),
            fg='#158aff',
            bg='#c3c3c3',
            activebackground='#c3c3c3',
            activeforeground='#158aff',
            bd=0,
            width=ANCHO_BOTON,
            anchor="w",
            command=lambda: self.cambiarPagina("cargar")
        )
        self.botonCargar.grid(row=0, column=1, sticky="w", padx=10)

        self.filaSegmentada = tk.Frame(self.marcoSuperior, bg="#c3c3c3")# SEGMENTADA
        self.filaSegmentada.grid(row=1, column=0, sticky="ew", pady=10)
        self.filaSegmentada.grid_columnconfigure(1, weight=1)

        self.indicadorSegmentada = tk.Frame(self.filaSegmentada, bg="#c3c3c3", width=8, height=30)
        self.indicadorSegmentada.grid(row=0, column=0, sticky="nsw")
        self.indicadorSegmentada.grid_propagate(False)

        self.botonSegmentada = tk.Button(
            self.filaSegmentada,
            text="Imagen segmentada",
            font=('Arial', 15, 'bold'),
            fg='#158aff',
            bg='#c3c3c3',
            activebackground='#c3c3c3',
            activeforeground='#158aff',
            bd=0,
            width=ANCHO_BOTON,
            anchor="w",
            command=lambda: self.cambiarPagina("segmentada")
        )
        self.botonSegmentada.grid(row=0, column=1, sticky="w", padx=10)

        self.filaAbout = tk.Frame(self, bg="#c3c3c3") #ABOUT
        self.filaAbout.grid(row=1, column=0, sticky="ew", pady=10)
        self.filaAbout.grid_columnconfigure(1, weight=1)

        self.indicadorAbout = tk.Frame(self.filaAbout, bg="#c3c3c3", width=8, height=30)
        self.indicadorAbout.grid(row=0, column=0, sticky="nsw")
        self.indicadorAbout.grid_propagate(False)

        self.botonAbout = tk.Button(
            self.filaAbout,
            text="About",
            font=('Arial', 15, 'bold'),
            fg='#158aff',
            bg='#c3c3c3',
            activebackground='#c3c3c3',
            activeforeground='#158aff',
            bd=0,
            width=ANCHO_BOTON,
            anchor="w",
            command=lambda: self.cambiarPagina("about")
        )
        self.botonAbout.grid(row=0, column=1, sticky="w", padx=10)

        self.filaSalir = tk.Frame(self, bg="#c3c3c3") #SALIR
        self.filaSalir.grid(row=3, column=0, sticky="ew", pady=(10, 20))
        self.filaSalir.grid_columnconfigure(1, weight=1)

        self.indicadorSalir = tk.Frame(self.filaSalir, bg="#c3c3c3", width=8, height=30)
        self.indicadorSalir.grid(row=0, column=0, sticky="nsw")
        self.indicadorSalir.grid_propagate(False)

        self.botonSalir = tk.Button(
            self.filaSalir,
            text="Salir",
            font=('Arial', 15, 'bold'),
            fg='red',
            bg='#c3c3c3',
            activebackground='#c3c3c3',
            activeforeground='red',
            bd=0,
            width=ANCHO_BOTON,
            anchor="w",
            command=self.app.cerrarAplicacion
        )
        self.botonSalir.grid(row=0, column=1, sticky="w", padx=10)

        self._hoverConfig = {
            self.botonCargar:     {"base_bg": "#c3c3c3", "base_fg": "#158aff", "hover_bg": "#158aff", "hover_fg": "white"},
            self.botonSegmentada: {"base_bg": "#c3c3c3", "base_fg": "#158aff", "hover_bg": "#158aff", "hover_fg": "white"},
            self.botonAbout:      {"base_bg": "#c3c3c3", "base_fg": "#158aff", "hover_bg": "#158aff", "hover_fg": "white"},
            self.botonSalir:      {"base_bg": "#c3c3c3", "base_fg": "red",     "hover_bg": "#b30000", "hover_fg": "white"},
        }

        self._configurarHover(self.botonCargar, self.indicadorCargar)
        self._configurarHover(self.botonSegmentada, self.indicadorSegmentada)
        self._configurarHover(self.botonAbout, self.indicadorAbout)
        self._configurarHover(self.botonSalir, self.indicadorSalir)

    def _configurarHover(self, boton, indicador):
        boton.bind("<Enter>", lambda e: self._encenderHover(boton, indicador))
        boton.bind("<Leave>", lambda e: self._apagarHover(boton, indicador))

    def _encenderHover(self, boton, indicador):
        if indicador is not self.indicadorActivo:
            if boton is self.botonSalir:
                indicador.config(bg="#b30000")
            else:
                indicador.config(bg="#158aff")

        cfg = self._hoverConfig.get(boton)
        if cfg:
            boton.config(bg=cfg["hover_bg"], fg=cfg["hover_fg"])

    def _apagarHover(self, boton, indicador):
        if indicador is not self.indicadorActivo:
            indicador.config(bg="#c3c3c3")

        cfg = self._hoverConfig.get(boton)
        if cfg:
            boton.config(bg=cfg["base_bg"], fg=cfg["base_fg"])

    def _limpiarIndicadores(self):
        self.indicadorCargar.config(bg="#c3c3c3")
        self.indicadorSegmentada.config(bg="#c3c3c3")
        self.indicadorAbout.config(bg="#c3c3c3")
        self.indicadorSalir.config(bg="#c3c3c3")

    def _marcarActivo(self, nombrePagina):
        self._limpiarIndicadores()
        if nombrePagina == "cargar":
            self.indicadorCargar.config(bg="#158aff")
            self.indicadorActivo = self.indicadorCargar
        elif nombrePagina == "segmentada":
            self.indicadorSegmentada.config(bg="#158aff")
            self.indicadorActivo = self.indicadorSegmentada
        elif nombrePagina == "about":
            self.indicadorAbout.config(bg="#158aff")
            self.indicadorActivo = self.indicadorAbout
        else:
            self.indicadorActivo = None

    def cambiarPagina(self, nombrePagina):
        if nombrePagina == "salir":
            self.indicadorActivo = None
            self.app.cerrarAplicacion()
            return

        self._marcarActivo(nombrePagina)
        self.app.mostrarPagina(nombrePagina)


class PaginaCargarImagen(PaginaBase):
    def __init__(self, app, contenedor, *args, **kwargs):
        super().__init__(app, contenedor, *args, **kwargs)

        self.grid_rowconfigure(0, weight=4)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Zona superior (rojo / luego rojo + OTRO PANEL
        
        self.zonaSuperiorRoja = tk.Frame(self, bg="#4a90e2")
        self.zonaSuperiorRoja.grid(row=0, column=0, sticky="nsew")

        # Zona inferior (azul con log) (SIN CAMBIOS)
        self.zonaInferiorAzul = tk.Frame(self, bg="#4a90e2")
        self.zonaInferiorAzul.grid(row=1, column=0, sticky="nsew")

        self._contenedorLog = tk.Frame(self.zonaInferiorAzul, bg="#4a90e2")
        self._contenedorLog.pack(fill="both", expand=True, padx=10, pady=10)

        from tkinter import ttk  

        self._progresoVar = tk.DoubleVar(value=0.0)

        self._estiloProgreso = ttk.Style()
        self._estiloProgreso.configure(
            "Green.Horizontal.TProgressbar",
            background="green"  # barra verde
        )

        self.barraProgreso = ttk.Progressbar(
            self._contenedorLog,
            orient="horizontal",
            mode="determinate",
            maximum=100.0,
            variable=self._progresoVar,
            style="Green.Horizontal.TProgressbar",
            takefocus=0
        )
        self.barraProgreso.pack(fill="x", expand=False)

        self.areaLog = scrolledtext.ScrolledText(
            self._contenedorLog,
            wrap="word",
            font=("Consolas", 11),
            bg="#1f3c88",
            fg="white",
            state="disabled",
            height=6
        )
        self.areaLog.pack(fill="both", expand=True)

        self.botonSeleccionarImagen = self.crearBotonAzul(
            contenedor=self.zonaSuperiorRoja,
            texto="Selecciona tu imagen",
            accion=self.accionSeleccionarImagen,
            anchoPixeles=210,
            altoPixeles=50,
            tamanoLetra=14
        )
        self.botonSeleccionarImagen.place(relx=0.5, rely=0.5, anchor="center")
        self.panelRojo = None
        self.panelNaranja = None
        self.etiquetaImagen = None
        self.rutaImagenActual = None
        self.modeloFrangi = None

        def rutaRelativa(archivoRelativo):
            rutaScript = Path(__file__).resolve().parent
            return rutaScript / archivoRelativo

        self.rutaPesosFrangi = rutaRelativa("FiltroFrangi/modelRetina.h5")

    def crearBotonAzul(self, contenedor, texto, accion,
                       anchoPixeles=150, altoPixeles=40, tamanoLetra=12):
        return ctk.CTkButton(
            master=contenedor,
            text=texto,
            fg_color="#0B3D91",
            hover_color="#102F5E",
            text_color="white",
            corner_radius=25,
            width=anchoPixeles,
            height=altoPixeles,
            font=("Arial", tamanoLetra, "bold"),
            command=accion
        )

    def escribirLog(self, mensaje):
        self.areaLog.configure(state="normal")
        self.areaLog.insert("end", mensaje + "\n")
        self.areaLog.see("end")
        self.areaLog.configure(state="disabled")
        self.areaLog.update_idletasks()

    def _ui(self, func, *args, **kwargs):

        self.after(0, lambda: func(*args, **kwargs))

    def _log_ts(self, mensaje):

        self._ui(self.escribirLog, mensaje)
        
    def _inicializarProgreso(self):

        self._setProgreso(0.0)

    def _setProgreso(self, valor):

        try:
            v = float(valor)
        except Exception:
            v = 0.0
        v = max(0.0, min(100.0, v))

        def _aplicar():
            self._progresoVar.set(v)
            self.barraProgreso.update_idletasks()

        self.after(0, _aplicar)

    def construirZonaSuperiorDividida(self):
        for hijo in self.zonaSuperiorRoja.winfo_children():
            hijo.destroy()

        self.zonaSuperiorRoja.grid_rowconfigure(0, weight=1)
        self.zonaSuperiorRoja.grid_columnconfigure(0, weight=3)
        self.zonaSuperiorRoja.grid_columnconfigure(1, weight=1)

        self.panelRojo = tk.Frame(
            self.zonaSuperiorRoja,
            bg = 'black',
            highlightbackground="black",
            highlightthickness=1,
            width=600,
            height=400
        )
        self.panelRojo.grid(row=0, column=0, sticky="nsew")
        self.panelRojo.grid_propagate(False)

        self.panelNaranja = tk.Frame(
            self.zonaSuperiorRoja,
            bg="#CDB891",
            highlightbackground="black",
            highlightthickness=1,
            width=220
        )
        self.panelNaranja.grid(row=0, column=1, sticky="nsew")
        self.panelNaranja.grid_propagate(False)

        self.winfo_toplevel().update_idletasks()

    def mostrarImagenInicial(self, rutaImagen):
        self.rutaImagenActual = rutaImagen

        for hijo in self.panelRojo.winfo_children():
            hijo.destroy()

        self.panelRojo.update_idletasks()
        anchoPanel = self.panelRojo.winfo_width()
        altoPanel = self.panelRojo.winfo_height()

        if anchoPanel < 50 or altoPanel < 50:
            anchoPanel = max(anchoPanel, 400)
            altoPanel = max(altoPanel, 300)

        try:
            img = Image.open(rutaImagen)
            iw, ih = img.size

            escala = min(anchoPanel / iw, altoPanel / ih)
            nuevoAncho = int(iw * escala)
            nuevoAlto = int(ih * escala)

            imgRed = img.resize((nuevoAncho, nuevoAlto), Image.LANCZOS)
            imgTk = ImageTk.PhotoImage(imgRed)

            self.panelRojo.img_ref = imgTk

            #self.etiquetaImagen = tk.Label(self.panelRojo, image=imgTk, bg="#d9534f")
            self.etiquetaImagen = tk.Label(self.panelRojo, image=imgTk, bg="black")
            self.etiquetaImagen.place(relx=0.5, rely=0.5, anchor="center")

        except Exception as e:
            print(f"Error mostrando imagen: {e}")
            etiquetaError = tk.Label(
                self.panelRojo,
                text=f"Error: {str(e)}",
                bg="#d9534f",
                fg="white"
            )
            etiquetaError.place(relx=0.5, rely=0.5, anchor="center")

    def poblarPanelNaranja(self, rutaImagen):

        for hijo in self.panelNaranja.winfo_children():
            hijo.destroy()
    
        tituloInicio = tk.Label(
            self.panelNaranja,
            text="Inicio",
            bg="#CDB891",  # mismo fondo del panel naranja (NO cambia colores)
            fg="black",
            font=("Arial", 14, "bold") # estilo similar a la referencia
        )

        tituloInicio.pack(pady=(20, 25))
    
        botonEjecutar = self.crearBotonAzul(
            contenedor=self.panelNaranja,
            texto="Ejecutar proceso",
            accion=self.accionEjecutarProceso,
            anchoPixeles=200,
            altoPixeles=40,
            tamanoLetra=12
        )
        botonEjecutar.pack(pady=(0, 25))  # separación vertical como en la referencia

        botonCargarNueva = self.crearBotonAzul(
            contenedor=self.panelNaranja,
            texto="Cargar nueva imagen",
            accion=self.accionCargarNuevaImagen,
            anchoPixeles=200,
            altoPixeles=40,
            tamanoLetra=12
        )
        botonCargarNueva.pack(pady=(0, 0))
    
        # 5) Mostrar imagen (SIN CAMBIOS)
        self.mostrarImagenInicial(rutaImagen)

    def accionSeleccionarImagen(self):
        rutaSeleccionada = filedialog.askopenfilename(
            title="Selecciona una imagen de retina",
            filetypes=[
                ("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("Todos los archivos", "*.*")
            ]
        )

        if not rutaSeleccionada:
            self.escribirLog("[INFO] No se seleccionó imagen.")
            return

        self.app.rutaImagenActual = rutaSeleccionada

        if hasattr(self, "botonSeleccionarImagen"):
            self.botonSeleccionarImagen.place_forget()

        self.construirZonaSuperiorDividida()

        def _continuar():
            self.poblarPanelNaranja(rutaSeleccionada)
            self.escribirLog(f"[OK] Imagen cargada: {rutaSeleccionada}")

        self.after(0, _continuar)

    def accionCargarNuevaImagen(self):
        rutaSeleccionada = filedialog.askopenfilename(
            title="Selecciona una nueva imagen de retina",
            filetypes=[
                ("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("Todos los archivos", "*.*")
            ]
        )

        if not rutaSeleccionada:
            self.escribirLog("[INFO] No se seleccionó nueva imagen.")
            return

        self.app.rutaImagenActual = rutaSeleccionada
        self.construirZonaSuperiorDividida()

        def _continuar():
            self.poblarPanelNaranja(rutaSeleccionada)
            self.escribirLog(f"[OK] Nueva imagen cargada: {rutaSeleccionada}")

        self.after(0, _continuar)


    def accionEjecutarProceso(self):

        self._inicializarProgreso()

        hilo = threading.Thread(target=self._pipelineSeguro)
        hilo.daemon = True
        hilo.start()

    def _pipelineSeguro(self):
        try:
            self._ejecutarPipelineAvanzado()
            self._ui(self.app.mostrarPagina, "segmentada")
        except Exception as e:

            self._log_ts(f"[ERROR] {e}")
            self._ui(messagebox.showerror, "Error en pipeline", str(e))

    def _cargarModeloFrangiSiNecesario(self):
        if self.modeloFrangi is not None:
            return
        self._log_ts("[4] Ejecutando red tipo Frangi multiescala...")
        x, modelo = crearRedNeuronal()

        if not self.rutaPesosFrangi or not Path(self.rutaPesosFrangi).exists():
            raise RuntimeError(f"No existe el archivo de pesos: {self.rutaPesosFrangi}")

        modelo.load_weights(str(self.rutaPesosFrangi))
        self._log_ts(f"[OK] Pesos cargados desde: {self.rutaPesosFrangi}")

        self.modeloFrangi = modelo
        self._log_ts("[OK] Modelo Frangi listo para predecir.")

    def _ejecutarPipelineAvanzado(self):
        if not hasattr(self.app, "rutaImagenActual") or self.app.rutaImagenActual is None:
            raise RuntimeError("No hay imagen cargada.")

        ruta = self.app.rutaImagenActual

        # Lectura de la imagen
        self._log_ts("[1] Leyendo imagen en escala de grises...")
        imagenVerde, imagenColor = cargarImagen(ruta)
        imagenVerde_uint8 = imagenVerde.astype(np.uint8)
        if imagenVerde_uint8 is None:
            raise RuntimeError("No pude leer la imagen.")
        self.app.imagenOriginal_uint8 = imagenVerde_uint8.copy()
        self._setProgreso(100.0 * 1 / 6) 

        mask_bool, mask_255 = obtenerMascara(imagenVerde)

        # Filtro gaussiano
        self._log_ts("[2] Buscando sigma óptima para el Filtro gaussiano.")
        rangoSigma = np.arange(0.2, 4.1, 0.5)
        resultadoSigma = elegirSigmaGauss(imagenVerde, rangoSigma)
        sigmaOptima = float(resultadoSigma.get("sigmaOptima", 1.0))
        self._log_ts(f"[2] Sigma óptima = {sigmaOptima:.3f}")

        self._log_ts("[2] Aplicando filtro gaussiano.")
        imagenGauss = filtroGaussiano(imagenVerde, sigmaOptima)
        self._setProgreso(100.0 * 2 / 6)  

        #Filtro homomórfico
        self._log_ts("[3] Buscando parámetros homomórficos en conjunto con el PSO.")
        sigmaMin = 0.2
        sigmaMax = 100.0
        swarmSize = 30
        maxIter = 10

        sigmaOpt, entropiaOpt = encontrarSigma(
            imagenGauss,
            sigmaMin,
            sigmaMax,
            swarmSize,
            maxIter
        )
        self._log_ts(
            f"[3] Sigma óptima homomórfica = {sigmaOpt:.4f} | Entropía = {entropiaOpt:.4f}"
        )

        self._log_ts("[3] Aplicando filtro homomórfico óptimo.")
        imagenHomomorfica = aplicarFiltroHomomorfico(imagenGauss, sigmaOpt)

        imagenHomomorfica_uint8 = (imagenHomomorfica * 255.0).clip(0, 255).astype(np.uint8)
        self._setProgreso(100.0 * 3 / 6) 

        # Filtro Frangi
        self._cargarModeloFrangiSiNecesario()

        imgParaRed = cv2.bitwise_not(imagenHomomorfica_uint8)
        tin = np.expand_dims(np.expand_dims(imgParaRed, axis=2), axis=0).astype(np.float32)


        self._log_ts("[4] Ejecutando predicción de la red Frangi.")
        pred = self.modeloFrangi.predict(tin)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]

        if pred.ndim == 4:
            pred2d = pred[0, :, :, 0]
        elif pred.ndim == 3:
            pred2d = pred[:, :, 0]
        else:
            pred2d = np.squeeze(pred)

        pred2d = (np.squeeze(pred) * 255).clip(0, 255).astype(np.uint8)
        self._setProgreso(100.0 * 4 / 6)  

        # Multiumbralización
        self._log_ts("[5] Obteniendo el umbral...")
        umbral = obtenerUmbrales(pred2d, 2)
        umbral = min(umbral)
        umbrales = [umbral, 255]

        self._log_ts(f"[5] El mejor umbral es {umbral} ...")
        self._setProgreso(100.0 * 5 / 6) 

        # Análisis fractal
        iteraciones, DF_iteraciones = visualizarEvolucionSegmentacion(
            pred2d,
            clasificadorVessel,
            unirSegmentos,
            clasificadorVesselPot,
            umbrales,
            mask_bool
        )

        mascaraFinal = iteraciones[-1]
        mascaraFinal_uint8 = (mascaraFinal.astype(bool).astype(np.uint8) * 255)

        self.app.imagenSegmentada_uint8 = mascaraFinal_uint8.copy()
        self._log_ts("[5] Segmentación final lista.")
        self._log_ts("[6] Puedes visualizar la proyección 3D desde la pestaña 'Imagen segmentada'.")
        self._log_ts("[OK] Pipeline completado sin errores.")

        self._setProgreso(100.0)


#%% PAGINA SEGMENTADA

class PaginaSegmentada(PaginaBase):
    def __init__(self, app, contenedor, *args, **kwargs):
        super().__init__(app, contenedor, *args, **kwargs)

        self.config(bg="#ffffff")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.contenedorGeneral = tk.Frame(self, bg="#ffffff")
        self.contenedorGeneral.grid(row=0, column=0, sticky="nsew")

        # Panel izquierdo verde
        self.panelIzquierdoVerde = tk.Frame(self.contenedorGeneral, bg="black")
        self.panelIzquierdoVerde.place(relx=0.0, rely=0.0, relwidth=0.66, relheight=1.0)

        # Control interno de actualización
        self._idActualizacionResultado = None
        self.etiquetaResultado = None

        self._imagenBasePil = None
        self._modoImagen = None   

        self.panelIzquierdoVerde.bind("<Configure>", self._eventoCambioTamanoResultado)

        # Panel derecho morado
        self.panelDerechoMorado = tk.Frame(self.contenedorGeneral, bg="#CDB891")
        self.panelDerechoMorado.place(relx=0.66, rely=0.0, relwidth=0.34, relheight=1.0)
        self.marcoResultado = tk.Label(
            self.panelIzquierdoVerde,
            bg="#1A1A1A",
            text="(Aún no hay resultado)"
        )
        self.marcoResultado.pack(expand=True, fill="both", padx=20, pady=20)

        tituloOpciones = tk.Label(
            self.panelDerechoMorado,
            text="Opciones de salida",
            bg="#CDB891",
            fg="black",
            font=('Arial', 15, 'bold')
        )
        tituloOpciones.pack(pady=(20, 10))

        # Variables para hacerlas exclusivas
        self.varGrises = ctk.IntVar(value=1)    # por defecto: grises activo
        self.varOriginal = ctk.IntVar(value=0)

        self.checkGrises = ctk.CTkCheckBox(
            self.panelDerechoMorado,
            text="Imagen binaria",
            font=("Arial", 12),
            variable=self.varGrises,
            command=self._onCheckGrises
        )
        self.checkGrises.pack(pady=10, padx=20, anchor="w")

        self.checkOriginal = ctk.CTkCheckBox(
            self.panelDerechoMorado,
            text="Imagen original",
            font=("Arial", 12),
            variable=self.varOriginal,
            command=self._onCheckOriginal
        )
        self.checkOriginal.pack(pady=10, padx=20, anchor="w")

        self.botonVer3D = ctk.CTkButton(
            self.panelDerechoMorado,
            text="Visualizar 3D",
            fg_color="#0B3D91",
            hover_color="#102F5E",
            text_color="white",
            corner_radius=25,
            width=200,
            height=40,
            font=("Arial", 12, "bold"),
            command=self.accionVisualizar3D
        )
        self.botonVer3D.pack(pady=20, padx=20, fill="x")

        self.botonGuardar = ctk.CTkButton(
            self.panelDerechoMorado,
            text="Guardar imagen",
            fg_color="#0B3D91",
            hover_color="#102F5E",
            text_color="white",
            corner_radius=25,
            width=200,
            height=40,
            font=("Arial", 12, "bold"),
            command=self.accionGuardarImagen
        )
        self.botonGuardar.pack(pady=10, padx=20, fill="x")

    def _onCheckGrises(self):#Se ejecuta cuando el usuario cambia la casilla 'Imagen a escala de grises'
        if self.varGrises.get() == 1:
            self.varOriginal.set(0)
        else:
            # No permitir que ambas estén apagadas
            self.varGrises.set(1)

        self._modoImagen = "grises"
        self._imagenBasePil = None
        self._actualizarSegunTamPanel()

    def _onCheckOriginal(self):#Se ejecuta cuando el usuario cambia la casilla 'Imagen original
        if self.varOriginal.get() == 1:
            self.varGrises.set(0)
        else:
            # No permitir que ambas estén apagadas
            self.varOriginal.set(1)

        self._modoImagen = "original"
        self._imagenBasePil = None
        self._actualizarSegunTamPanel()

    def _eventoCambioTamanoResultado(self, event):
        if self._idActualizacionResultado:
            self.after_cancel(self._idActualizacionResultado)

        if hasattr(self.app, "imagenSegmentada_uint8") and self.app.imagenSegmentada_uint8 is not None:
            if event.width > 1 and event.height > 1:
                self._idActualizacionResultado = self.after(
                    100,
                    lambda: self._actualizarResultado(event.width, event.height)
                )

    def _actualizarSegunTamPanel(self):
        w = self.panelIzquierdoVerde.winfo_width()
        h = self.panelIzquierdoVerde.winfo_height()
        if w > 1 and h > 1:
            self._actualizarResultado(w, h)


    def _cargarImagenBase(self):


        # Determinar modo actual
        if self.varGrises.get() == 1:
            modo = "grises"
        elif self.varOriginal.get() == 1:
            modo = "original"
        else:
            modo = "grises"
            self.varGrises.set(1)
            self.varOriginal.set(0)

        if self._imagenBasePil is not None and self._modoImagen == modo:
            return self._imagenBasePil

        self._modoImagen = modo

        # MODO GRIS
        if modo == "grises":
            if not hasattr(self.app, "imagenSegmentada_uint8") or self.app.imagenSegmentada_uint8 is None:
                return None
            img = self.app.imagenSegmentada_uint8.astype(np.uint8)
            pil = Image.fromarray(img, mode="L")  # escala de grises

        #MODO ORIGINAL (OVERLAY VERDE)
        else:

            img_bgr = None
            ruta = getattr(self.app, "rutaImagenActual", None)

            if ruta:
                try:

                    img = cv2.imread(ruta, cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        if img.ndim == 2:
                            # Gris -> BGR
                            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                        elif img.ndim == 3 and img.shape[2] == 3:
                            # Ya BGR
                            img_bgr = img
                        elif img.ndim == 3 and img.shape[2] == 4:
                            # BGRA -> BGR
                            img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                except Exception as e:
                    print(f"Error leyendo imagen original con cv2: {e}")

            # Fallback si no se pudo leer con cv2
            if img_bgr is None and hasattr(self.app, "imagenOriginal_uint8") and self.app.imagenOriginal_uint8 is not None:
                try:
                    base = self.app.imagenOriginal_uint8
                    base = base.astype(np.uint8)
                    if base.ndim == 2:
                        img_bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
                    elif base.ndim == 3 and base.shape[2] == 3:

                        img_bgr = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
                except Exception as e:
                    print(f"Error creando imagen BGR desde imagenOriginal_uint8: {e}")

            if img_bgr is None:
                return None

            # Máscara binaria 
            if not hasattr(self.app, "imagenSegmentada_uint8") or self.app.imagenSegmentada_uint8 is None:
                return None

            mask = self.app.imagenSegmentada_uint8
            mask = np.asarray(mask)

            if mask.ndim == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = mask.astype(np.uint8)

            h_img, w_img = img_bgr.shape[:2]
            if mask.shape[:2] != (h_img, w_img):
                mask = cv2.resize(mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
            try:
                overlay_bgr = sobreponerMascaraVerde(img_bgr, mask, alpha=0.3)
            except Exception as e:
                print(f"Error en sobreponerMascaraVerde: {e}")
                return None

            overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(overlay_rgb)

        self._imagenBasePil = pil
        return self._imagenBasePil

    def _crearPhotoImageRedimensionada(self, imagenBase, anchoPanel, altoPanel):
        w, h = imagenBase.size
        escala = min((anchoPanel - 40) / w, (altoPanel - 40) / h)
        escala = max(escala, 1e-6)

        nuevo_w = max(int(w * escala), 1)
        nuevo_h = max(int(h * escala), 1)

        imagenRedimensionada = imagenBase.resize((nuevo_w, nuevo_h), Image.LANCZOS)
        return ImageTk.PhotoImage(imagenRedimensionada)

    def _actualizarResultado(self, ancho, alto):
        if not hasattr(self.app, "imagenSegmentada_uint8") or self.app.imagenSegmentada_uint8 is None:
            return

        if not self.etiquetaResultado:
            return

        try:
            imagenBase = self._cargarImagenBase()
            if imagenBase is None:
                return

            imagenTk = self._crearPhotoImageRedimensionada(imagenBase, ancho, alto)
            self.etiquetaResultado.config(image=imagenTk)
            self.etiquetaResultado.image = imagenTk
        except Exception as e:
            print(f"Error actualizando resultado: {e}")


    def mostrarResultado(self):
        if not hasattr(self.app, "imagenSegmentada_uint8") or self.app.imagenSegmentada_uint8 is None:
            self.marcoResultado.config(text="No hay imagen procesada todavía.", font=('Arial', 15, 'bold'), fg="white")
            return

    
        if self.varGrises.get() == 0 and self.varOriginal.get() == 0:
            self.varGrises.set(1)
            self.varOriginal.set(0)

        self._modoImagen = None
        self._imagenBasePil = None

        self.marcoResultado.pack_forget()

        if not self.etiquetaResultado:
            self.etiquetaResultado = tk.Label(self.panelIzquierdoVerde, bg="black")
            self.etiquetaResultado.place(relx=0.5, rely=0.5, anchor="center")

        self._actualizarSegunTamPanel()

    def accionVisualizar3D(self):
        # Verificar que exista segmentación
        if not hasattr(self.app, "imagenSegmentada_uint8") or self.app.imagenSegmentada_uint8 is None:
            messagebox.showinfo("3D", "Primero ejecuta el pipeline en 'Cargar imagen'.")
            return

        ruta = getattr(self.app, "rutaImagenActual", None)
        if not ruta:
            messagebox.showinfo("3D", "No se encontró la imagen original cargada.")
            return

        try:

            img = cv2.imread(ruta, cv2.IMREAD_UNCHANGED)
            img_para_3d = None

            if img is not None:
                if img.ndim == 2:
  
                    img_para_3d = img.astype(np.uint8)
                elif img.ndim == 3 and img.shape[2] == 3:

                    img_para_3d = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elif img.ndim == 3 and img.shape[2] == 4:

                    img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    img_para_3d = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            if img_para_3d is None and hasattr(self.app, "imagenOriginal_uint8") and self.app.imagenOriginal_uint8 is not None:
                base = self.app.imagenOriginal_uint8.astype(np.uint8)
                if base.ndim == 2:
                    img_para_3d = base 
                elif base.ndim == 3 and base.shape[2] == 3:
                    img_para_3d = base

            if img_para_3d is None:
                messagebox.showerror("Error 3D", "No se pudo reconstruir la imagen original para la proyección 3D.")
                return

            mascara = self.app.imagenSegmentada_uint8 #Preparar máscara binaria
            mascara = np.asarray(mascara)

            if mascara.ndim == 3:
                mascara = cv2.cvtColor(mascara, cv2.COLOR_BGR2GRAY)
            mascara = mascara.astype(np.uint8)
            h_img, w_img = img_para_3d.shape[:2] #Ajustar tamaños si no coinciden
            if mascara.shape[:2] != (h_img, w_img):
                mascara = cv2.resize(mascara, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
            mostrarProyeccion3D(img_para_3d, mascara) #Lanzar la proyección 3D

        except Exception as e:
            messagebox.showerror("Error 3D", str(e))

    def accionGuardarImagen(self):
        if not hasattr(self.app, "imagenSegmentada_uint8") or self.app.imagenSegmentada_uint8 is None:
            messagebox.showinfo("Guardar", "No hay segmentación para guardar.")
            return

        rutaGuardar = filedialog.asksaveasfilename(
            title="Guardar imagen segmentada",
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPG", "*.jpg *.jpeg"),
                ("BMP", "*.bmp"),
                ("TIFF", "*.tif *.tiff")
            ]
        )
        if not rutaGuardar:
            return

        imagenBinaria = self.app.imagenSegmentada_uint8.astype(np.uint8)
        Image.fromarray(imagenBinaria, mode="L").save(rutaGuardar)
        messagebox.showinfo("Guardar", f"Imagen guardada en:\n{rutaGuardar}")

#%% PAGINA ABOUT
class PaginaAbout(PaginaBase):
    def __init__(self, app, contenedor, *args, **kwargs):
        super().__init__(app, contenedor, *args, **kwargs)

        self.config(bg="white")

        marcoInfo = tk.Frame(self, bg="white")
        marcoInfo.pack(expand=True, fill="both")

        titulo = tk.Label(
            marcoInfo,
            text="Acerca de la herramienta",
            font=("Arial", 20, "bold"),
            bg="white",
            fg="#158aff"
        )
        titulo.pack(pady=20)

        texto = tk.Label(
            marcoInfo,
            text=(
                "Esta interfaz permite:\n"
                "1. Cargar imagen de retina\n"
                "2. Suavizar mediante el Filtro gaussiano\n"
                "3. Mejorar contraste con el Filtro homomórfico optimizado con PSO\n"
                "4. Segmentar red vascular con red tipo Frangi + refinamiento DF\n"
                "5. Proyectar la red vascular en 3D\n"
                "6. Guardar resultados y visualizarlos"
            ),
            font=("Arial", 14),
            bg="white",
            fg="black",
            justify="center"
        )
        texto.pack(pady=10)


class Aplicacion(tk.Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        
        self.rutaImagenActual = None

        self.imagenOriginal_uint8 = None

        self.imagenSegmentada_uint8 = None


        self.X_points_3D = None
        self.Y_points_3D = None
        self.Z_points_3D = None
        self.datosOjo3D = None


        self.title("Acerca de")
        self.minsize(1100, 600)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)

        self.barraLateral = BarraLateral(self, self)
        self.barraLateral.grid(row=0, column=0, sticky="nsw")

        self.marcoPaginas = tk.Frame(self, bg="#ffffff")
        self.marcoPaginas.grid(row=0, column=1, sticky="nsew")

        self.paginas = {
            "cargar": PaginaCargarImagen(self, self.marcoPaginas, bg="#ffffff"),
            "segmentada": PaginaSegmentada(self, self.marcoPaginas, bg="#ffffff"),
            "about": PaginaAbout(self, self.marcoPaginas, bg="#ffffff")
        }

        
        self.barraLateral._marcarActivo("about")
        self.mostrarPagina("about")

    def mostrarPagina(self, nombrePagina):
        # limpiar lo que estuviera antes
        for hijo in self.marcoPaginas.winfo_children():
            hijo.pack_forget()

        pagina = self.paginas[nombrePagina]
        pagina.pack(fill="both", expand=True)

        # actualizar barra lateral visualmente
        self.barraLateral._marcarActivo(nombrePagina)
        if nombrePagina == "cargar":
            self.title("Cargar imagen")

        elif nombrePagina == "segmentada":
            self.title("Imagen segmentada")
            try:
                pagina.mostrarResultado()
            except AttributeError:
                pass

        elif nombrePagina == "about":
            self.title("Acerca de")
        else:
            self.title("Interfaz Retina")

    def cerrarAplicacion(self):
        self.destroy()

#%% MAIN

if __name__ == "__main__":
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")

    app = Aplicacion()
    app.mainloop()
