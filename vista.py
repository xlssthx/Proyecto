"""
Vista del sistema de seguridad - Interfaz gráfica
"""
import tkinter as tk
from tkinter import messagebox, ttk
import datetime
from componentes.slider_filtro import SliderFiltro
from componentes.panel_alertas import PanelAlertas
from componentes.panel_camaras import PanelCamaras
from utils.constantes import TITULO_APP, COLOR_FONDO, COLOR_BARRA_ESTADO


class SistemaSeguridadVista:
    """Interfaz gráfica del sistema de seguridad"""
    
    def __init__(self, root):
        self.root = root
        self._configurar_ventana()
        self._crear_variables()
        self._crear_widgets()
        self._iniciar_reloj()

    def _configurar_ventana(self):
        """Configura la ventana principal"""
        self.root.title(TITULO_APP)
        self.root.state('zoomed')
        self.root.configure(bg=COLOR_FONDO)

    def _crear_variables(self):
        """Crea las variables de control"""
        self.filtro_var = tk.StringVar(value="Todas")
        self.callbacks = {}

    def _crear_widgets(self):
        """Crea todos los widgets de la interfaz"""
        self._crear_header()
        self._crear_controles()
        self._crear_pestanas()
        self._crear_barra_estado()

    def _crear_header(self):
        """Crea el header con título y estado"""
        # Título
        tk.Label(
            self.root,
            text=TITULO_APP,
            font=("Arial", 16, "bold"),
            bg=COLOR_FONDO
        ).pack(pady=10, anchor="w", padx=20)

        # Estado
        frame_estado = tk.Frame(self.root, bg=COLOR_FONDO)
        frame_estado.place(relx=0.99, rely=0.06, anchor="e")
        
        self.label_estado = tk.Label(
            frame_estado,
            text="Estado: En mantenimiento",
            font=("Arial", 12, "bold"),
            fg="red",
            bg=COLOR_FONDO
        )
        self.label_estado.pack()

    def _crear_controles(self):
        """Crea los controles principales"""
        frame = tk.Frame(self.root, bg=COLOR_FONDO)
        frame.pack(fill="x", padx=20, pady=5)

        # Botones de control
        self.btn_iniciar = tk.Button(
            frame,
            text="Iniciar Monitoreo",
            width=15
        )
        self.btn_iniciar.pack(side=tk.LEFT, padx=5)

        self.btn_detener = tk.Button(
            frame,
            text="Detener",
            width=10
        )
        self.btn_detener.pack(side=tk.LEFT, padx=5)

        # Separador
        tk.Frame(
            frame,
            width=2,
            bg="#cccccc"
        ).pack(side=tk.LEFT, fill="y", padx=15, pady=5)

        # Slider de filtros
        self.slider_filtro = SliderFiltro(
            frame,
            self.filtro_var,
            self._on_filtro_cambiado
        )

    def _crear_pestanas(self):
        """Crea el sistema de pestañas"""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Estilo
        style = ttk.Style()
        style.configure(
            'TNotebook.Tab',
            font=('Arial', 11, 'bold'),
            padding=[20, 10]
        )

        # Pestaña 1: Vista completa
        self._crear_tab_completa()

        # Pestaña 2: Solo cámaras
        self._crear_tab_camaras()

        # Pestaña 3: Solo alertas
        self._crear_tab_alertas()

    def _crear_tab_completa(self):
        """Crea la pestaña de vista completa"""
        tab = tk.Frame(self.notebook, bg=COLOR_FONDO)
        self.notebook.add(tab, text="Vista completa")

        frame_principal = tk.Frame(tab, bg=COLOR_FONDO)
        frame_principal.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Panel de cámaras
        frame_camaras = tk.LabelFrame(
            frame_principal,
            text="Monitoreo",
            font=("Arial", 12, "bold"),
            bg=COLOR_FONDO,
            width=1000
        )
        frame_camaras.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        frame_camaras.pack_propagate(False)

        self.panel_camaras_completa = PanelCamaras(
            frame_camaras,
            "completa",
            self._on_eliminar_camara
        )

        self.btn_agregar_completa = tk.Button(
            frame_camaras,
            text="Agregar cámara",
            command=self._on_agregar_camara
        )
        self.btn_agregar_completa.pack(side=tk.BOTTOM, pady=7)

        # Panel de alertas
        frame_alertas = tk.LabelFrame(
            frame_principal,
            text="Alertas y eventos",
            font=("Arial", 12, "bold"),
            bg=COLOR_FONDO
        )
        frame_alertas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        self.panel_alertas_completa = PanelAlertas(frame_alertas, "completa")

    def _crear_tab_camaras(self):
        """Crea la pestaña solo cámaras"""
        tab = tk.Frame(self.notebook, bg=COLOR_FONDO)
        self.notebook.add(tab, text="Cámaras")

        frame = tk.LabelFrame(
            tab,
            text="Monitoreo de Cámaras",
            font=("Arial", 12, "bold"),
            bg=COLOR_FONDO
        )
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.panel_camaras_solo = PanelCamaras(
            frame,
            "solo",
            self._on_eliminar_camara
        )

        self.btn_agregar_solo = tk.Button(
            frame,
            text="Agregar cámara",
            command=self._on_agregar_camara
        )
        self.btn_agregar_solo.pack(side=tk.BOTTOM, pady=7)

    def _crear_tab_alertas(self):
        """Crea la pestaña solo alertas"""
        tab = tk.Frame(self.notebook, bg=COLOR_FONDO)
        self.notebook.add(tab, text="Alertas / Historial")

        frame = tk.LabelFrame(
            tab,
            text="Gestión de Alertas y Eventos",
            font=("Arial", 12, "bold"),
            bg=COLOR_FONDO
        )
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.panel_alertas_solo = PanelAlertas(frame, "solo")

    def _crear_barra_estado(self):
        """Crea la barra de estado inferior"""
        frame = tk.Frame(self.root, bg=COLOR_BARRA_ESTADO, height=25)
        frame.pack(fill="x", side=tk.BOTTOM)

        self.label_fecha_hora = tk.Label(
            frame,
            text="",
            bg=COLOR_BARRA_ESTADO
        )
        self.label_fecha_hora.pack(side=tk.LEFT, padx=10)

    def _iniciar_reloj(self):
        """Inicia el reloj de la barra de estado"""
        self._actualizar_reloj()

    def _actualizar_reloj(self):
        """Actualiza la fecha y hora"""
        ahora = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        self.label_fecha_hora.config(text=f"Fecha y hora: {ahora}")
        self.root.after(1000, self._actualizar_reloj)

    # ========== MÉTODOS PÚBLICOS ==========

    def actualizar_estado(self, estado):
        """Actualiza el estado del sistema"""
        color = "red" if estado == "Alerta" else "green"
        self.label_estado.config(text=f"Estado: {estado}", fg=color)

    def mostrar_mensaje(self, titulo, mensaje):
        """Muestra un mensaje al usuario"""
        messagebox.showinfo(titulo, mensaje)

    def agregar_camara(self, id_camara, nombre):
        """Agrega una cámara a ambas pestañas"""
        self.panel_camaras_completa.agregar_camara(id_camara, nombre)
        self.panel_camaras_solo.agregar_camara(id_camara, nombre)

    def eliminar_camara(self, id_camara):
        """Elimina una cámara de ambas pestañas"""
        self.panel_camaras_completa.eliminar_camara(id_camara)
        self.panel_camaras_solo.eliminar_camara(id_camara)

    def actualizar_video_camara(self, id_camara, imagen_tk):
        """Actualiza el video de una cámara en ambas pestañas"""
        self.panel_camaras_completa.actualizar_video(id_camara, imagen_tk)
        self.panel_camaras_solo.actualizar_video(id_camara, imagen_tk)

    def agregar_alerta(self, alerta):
        """Agrega una alerta a ambas pestañas"""
        callback = lambda id_a, w: self._on_marcar_revisada(id_a, w)
        self.panel_alertas_completa.agregar_alerta(alerta, callback)
        self.panel_alertas_solo.agregar_alerta(alerta, callback)

    def aplicar_filtro_alertas(self, filtro):
        """Aplica el filtro a ambas pestañas de alertas"""
        self.panel_alertas_completa.aplicar_filtro(filtro)
        self.panel_alertas_solo.aplicar_filtro(filtro)

    def eliminar_alerta_visual(self, id_alerta):
        """Elimina visualmente una alerta"""
        self.panel_alertas_completa.eliminar_alerta(id_alerta)
        self.panel_alertas_solo.eliminar_alerta(id_alerta)

    # ========== CALLBACKS ==========

    def set_callback(self, nombre, funcion):
        """Establece un callback"""
        self.callbacks[nombre] = funcion

    def _on_agregar_camara(self):
        """Maneja el evento de agregar cámara"""
        if 'agregar_camara' in self.callbacks:
            self.callbacks['agregar_camara']()

    def _on_eliminar_camara(self, id_camara):
        """Maneja el evento de eliminar cámara"""
        if 'eliminar_camara' in self.callbacks:
            self.callbacks['eliminar_camara'](id_camara)

    def _on_filtro_cambiado(self, filtro):
        """Maneja el cambio de filtro"""
        self.aplicar_filtro_alertas(filtro)
        if 'filtro_cambiado' in self.callbacks:
            self.callbacks['filtro_cambiado'](filtro)

    def _on_marcar_revisada(self, id_alerta, widget):
        """Maneja marcar alerta como revisada"""
        if 'marcar_revisada' in self.callbacks:
            self.callbacks['marcar_revisada'](id_alerta, widget)
