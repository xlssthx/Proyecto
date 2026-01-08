"""
Componente del panel de cámaras
"""
import tkinter as tk
from utils.constantes import TAMANO_FEED_VIDEO


class PanelCamaras:
    """Panel para mostrar feeds de video de cámaras"""
    
    def __init__(self, parent, nombre_id, callback_eliminar=None):
        """
        Inicializa el panel de cámaras
        
        Args:
            parent: Widget padre
            nombre_id: Identificador único ('completa' o 'solo')
            callback_eliminar: Función a llamar al eliminar cámara
        """
        self.parent = parent
        self.nombre_id = nombre_id
        self.callback_eliminar = callback_eliminar
        self.camaras = {}
        
        self._crear_widgets()

    def _crear_widgets(self):
        """Crea los widgets del panel"""
        # Grid para las cámaras
        self.grid_frame = tk.Frame(self.parent, bg="#f0f0f0")
        self.grid_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Configurar grid 2x2
        self.grid_frame.grid_rowconfigure(0, weight=1)
        self.grid_frame.grid_rowconfigure(1, weight=1)
        self.grid_frame.grid_columnconfigure(0, weight=1)
        self.grid_frame.grid_columnconfigure(1, weight=1)

    def agregar_camara(self, id_camara, nombre):
        """Agrega una cámara al panel"""
        num_camaras = len(self.camaras)
        row = num_camaras // 2
        col = num_camaras % 2
        
        # Frame de la cámara
        frame = tk.Frame(
            self.grid_frame,
            borderwidth=2,
            relief="groove",
            bg="white",
            width=480,
            height=400
        )
        frame.grid(row=row, column=col, padx=10, pady=10)
        frame.grid_propagate(False)
        
        # Header con nombre y botón eliminar
        header = tk.Frame(frame, bg="white", height=35)
        header.pack(fill="x", padx=5, pady=5)
        header.pack_propagate(False)
        
        tk.Label(
            header,
            text=nombre,
            bg="white",
            font=("Arial", 10, "bold")
        ).pack(side=tk.LEFT)
        
        btn_eliminar = tk.Button(
            header,
            text="✕",
            bg="red",
            fg="white",
            font=("Arial", 8, "bold"),
            width=2,
            height=1,
            command=lambda: self._on_eliminar(id_camara)
        )
        btn_eliminar.pack(side=tk.RIGHT)
        
        # Panel de video
        ancho, alto = TAMANO_FEED_VIDEO
        video_panel = tk.Label(
            frame,
            bg="gray15",
            width=ancho,
            height=alto
        )
        video_panel.pack(padx=5, pady=(0, 5))
        
        # Guardar referencia
        self.camaras[id_camara] = {
            'frame': frame,
            'panel': video_panel,
            'btn_eliminar': btn_eliminar,
            'row': row,
            'col': col
        }
        
        return video_panel

    def eliminar_camara(self, id_camara):
        """Elimina una cámara del panel"""
        if id_camara in self.camaras:
            self.camaras[id_camara]['frame'].destroy()
            del self.camaras[id_camara]
            self.reorganizar_grid()

    def reorganizar_grid(self):
        """Reorganiza las cámaras en el grid 2x2"""
        camaras_lista = list(self.camaras.items())
        
        for idx, (cam_id, cam_data) in enumerate(camaras_lista):
            row = idx // 2
            col = idx % 2
            cam_data['frame'].grid(row=row, column=col, padx=10, pady=10)
            cam_data['row'] = row
            cam_data['col'] = col

    def actualizar_video(self, id_camara, imagen_tk):
        """Actualiza el frame de video de una cámara"""
        if id_camara in self.camaras:
            panel = self.camaras[id_camara]['panel']
            panel.imgtk = imagen_tk
            panel.configure(image=imagen_tk)

    def obtener_panel(self, id_camara):
        """Obtiene el panel de una cámara específica"""
        if id_camara in self.camaras:
            return self.camaras[id_camara]['panel']
        return None

    def _on_eliminar(self, id_camara):
        """Maneja el evento de eliminar cámara"""
        if self.callback_eliminar:
            self.callback_eliminar(id_camara)

    def limpiar(self):
        """Elimina todas las cámaras"""
        for id_camara in list(self.camaras.keys()):
            self.eliminar_camara(id_camara)
