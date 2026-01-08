"""
Componente del slider de filtros personalizado
"""
import tkinter as tk
from utils.constantes import COLORES_FILTRO, TIPOS_ALERTAS


class SliderFiltro:
    """Slider personalizado para filtrar alertas"""
    
    def __init__(self, parent, filtro_var, callback=None):
        """
        Inicializa el slider de filtros
        
        Args:
            parent: Widget padre
            filtro_var: Variable tk.StringVar para el filtro actual
            callback: Función a llamar cuando cambia el filtro
        """
        self.parent = parent
        self.filtro_var = filtro_var
        self.callback = callback
        self.filtro_index = 0
        self.slider_arrastrando = False
        
        self._crear_widgets()
        self._dibujar_slider()
        self._configurar_eventos()

    def _crear_widgets(self):
        """Crea los widgets del slider"""
        # Icono de búsqueda
        tk.Label(
            self.parent, 
            text="🔍", 
            bg="#f0f0f0",
            font=("Arial", 12)
        ).pack(side=tk.LEFT, padx=(0, 5))

        # Canvas para el slider
        self.canvas = tk.Canvas(
            self.parent,
            width=500,
            height=35,
            bg="#f0f0f0",
            highlightthickness=0,
            bd=0
        )
        self.canvas.pack(side=tk.LEFT, padx=5)

        # Badge del filtro actual
        self.label_filtro = tk.Label(
            self.parent,
            text="Todas",
            bg="#26E439",
            fg="white",
            font=("Arial", 8, "bold"),
            relief="flat",
            bd=0,
            padx=8,
            pady=4
        )
        self.label_filtro.pack(side=tk.LEFT, padx=8)

    def _configurar_eventos(self):
        """Configura los eventos del mouse"""
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

    def _dibujar_slider(self):
        """Dibuja el slider completo"""
        self.canvas.delete("all")
        
        # Configuración
        colores = list(COLORES_FILTRO.values())
        labels = ["Todas", "Tipo 1", "Tipo 2", "Tipo 3", "Tipo 4"]
        ancho_total = 480
        inicio_x = 20
        ancho_seccion = ancho_total / 5
        
        # Línea de fondo
        self.canvas.create_rectangle(
            inicio_x, 18, inicio_x + ancho_total, 22,
            fill="#e0e0e0",
            outline="#cccccc",
            width=1
        )
        
        # Marcadores de posición
        for i in range(5):
            x = inicio_x + (i * ancho_seccion) + (ancho_seccion / 2)
            
            # Círculo marcador
            radio = 4 if i == self.filtro_index else 3
            color = colores[i] if i == self.filtro_index else "#bdbdbd"
            
            self.canvas.create_oval(
                x - radio, 20 - radio,
                x + radio, 20 + radio,
                fill=color,
                outline="#999999",
                width=1
            )
            
            # Etiqueta
            font_weight = "bold" if i == self.filtro_index else "normal"
            color_texto = colores[i] if i == self.filtro_index else "#757575"
            
            self.canvas.create_text(
                x, 8,
                text=labels[i],
                font=("Arial", 7, font_weight),
                fill=color_texto
            )
        
        # Handle (bolita deslizable)
        x_handle = inicio_x + (self.filtro_index * ancho_seccion) + (ancho_seccion / 2)
        
        # Sombra
        self.canvas.create_oval(
            x_handle - 11, 21 - 11,
            x_handle + 11, 21 + 11,
            fill="#d0d0d0",
            outline=""
        )
        
        # Handle principal
        self.canvas.create_oval(
            x_handle - 10, 20 - 10,
            x_handle + 10, 20 + 10,
            fill=colores[self.filtro_index],
            outline="white",
            width=3,
            tags="handle"
        )
        
        # Punto interior
        self.canvas.create_oval(
            x_handle - 3, 20 - 3,
            x_handle + 3, 20 + 3,
            fill="white",
            outline=""
        )
        
        self.canvas.config(cursor="hand2")

    def _calcular_indice(self, x):
        """Calcula el índice del filtro basado en la posición X"""
        inicio_x = 20
        ancho_seccion = 480 / 5
        
        if inicio_x <= x <= inicio_x + 480:
            return max(0, min(4, int((x - inicio_x) / ancho_seccion)))
        return self.filtro_index

    def _on_click(self, event):
        """Maneja el clic en el slider"""
        nuevo_index = self._calcular_indice(event.x)
        
        if nuevo_index != self.filtro_index:
            self.filtro_index = nuevo_index
            self._dibujar_slider()
            self._aplicar_filtro()
        
        self.slider_arrastrando = True

    def _on_drag(self, event):
        """Maneja el arrastre del slider"""
        if not self.slider_arrastrando:
            return
        
        nuevo_index = self._calcular_indice(event.x)
        
        if nuevo_index != self.filtro_index:
            self.filtro_index = nuevo_index
            self._dibujar_slider()
            self._aplicar_filtro()

    def _on_release(self, event):
        """Maneja cuando se suelta el mouse"""
        self.slider_arrastrando = False

    def _aplicar_filtro(self):
        """Aplica el filtro seleccionado"""
        filtro = TIPOS_ALERTAS[self.filtro_index]
        self.filtro_var.set(filtro)
        
        # Actualizar badge
        self.label_filtro.config(
            text=filtro,
            bg=COLORES_FILTRO.get(filtro, "#2196F3")
        )
        
        # Llamar al callback si existe
        if self.callback:
            self.callback(filtro)

    def set_filtro(self, filtro):
        """Establece el filtro programáticamente"""
        try:
            self.filtro_index = TIPOS_ALERTAS.index(filtro)
            self._dibujar_slider()
            self._aplicar_filtro()
        except ValueError:
            pass
