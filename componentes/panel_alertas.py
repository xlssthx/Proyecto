"""
Componente del panel de alertas
"""
import tkinter as tk
from tkinter import ttk
from utils.constantes import COLORES_FILTRO


class PanelAlertas:
    """Panel para mostrar alertas activas e historial"""
    
    def __init__(self, parent, nombre_id):
        """
        Inicializa el panel de alertas
        
        Args:
            parent: Widget padre
            nombre_id: Identificador único ('completa' o 'solo')
        """
        self.parent = parent
        self.nombre_id = nombre_id
        self.alertas_widgets = {}
        self.historial_items = {}
        
        self._crear_widgets()

    def _crear_widgets(self):
        """Crea todos los widgets del panel"""
        # Título alertas activas
        tk.Label(
            self.parent,
            text="Alertas activas",
            bg="#f0f0f0",
            anchor="w",
            font=("Arial", 10, "bold")
        ).pack(fill="x", pady=(5, 0), padx=5)
        
        # Contenedor de alertas activas con scroll
        alertas_container = tk.Frame(self.parent)
        alertas_container.pack(fill="x", pady=5, padx=5)
        
        # Canvas y scrollbar
        self.canvas = tk.Canvas(
            alertas_container,
            height=150,
            bg="#fff5f5",
            highlightthickness=1,
            highlightbackground="#ffcccc"
        )
        scrollbar = ttk.Scrollbar(
            alertas_container,
            orient="vertical",
            command=self.canvas.yview
        )
        
        # Frame interior
        self.frame_alertas = tk.Frame(self.canvas, bg="#fff5f5")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        # Empaquetar
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill="both", expand=True)
        
        # Crear ventana en canvas
        self.canvas_frame_id = self.canvas.create_window(
            (0, 0),
            window=self.frame_alertas,
            anchor="nw"
        )
        
        # Configurar eventos de scroll
        self.frame_alertas.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )
        self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.itemconfig(
                self.canvas_frame_id,
                width=e.width
            )
        )

        # Mensaje inicial
        self.label_vacio = tk.Label(
            self.frame_alertas,
            text="No hay alertas activas",
            bg="#fff5f5",
            fg="#999999",
            font=("Arial", 10, "italic")
        )
        self.label_vacio.pack(pady=20)

        # Separador
        tk.Frame(
            self.parent,
            height=2,
            bg="#cccccc"
        ).pack(fill="x", pady=10, padx=5)

        # Título historial
        tk.Label(
            self.parent,
            text="Historial de Eventos",
            bg="#f0f0f0",
            anchor="w",
            font=("Arial", 10, "bold")
        ).pack(fill="x", pady=(5, 0), padx=5)
        
        # Crear Treeview para historial
        self._crear_historial()

    def _crear_historial(self):
        """Crea el treeview del historial"""
        tree_frame = tk.Frame(self.parent)
        tree_frame.pack(fill="both", expand=True, pady=5, padx=5)
        
        # Scrollbars
        scroll_y = ttk.Scrollbar(tree_frame)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        scroll_x = ttk.Scrollbar(tree_frame, orient="horizontal")
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Treeview
        columns = ("ID", "Fecha/Hora", "Cámara", "Tipo", "Confianza")
        self.tree = ttk.Treeview(
            tree_frame,
            columns=columns,
            show="headings",
            height=10,
            yscrollcommand=scroll_y.set,
            xscrollcommand=scroll_x.set
        )
        
        # Configurar scrollbars
        scroll_y.config(command=self.tree.yview)
        scroll_x.config(command=self.tree.xview)
        
        # Configurar columnas
        self.tree.heading("ID", text="ID")
        self.tree.heading("Fecha/Hora", text="Fecha/Hora")
        self.tree.heading("Cámara", text="Cámara")
        self.tree.heading("Tipo", text="Tipo")
        self.tree.heading("Confianza", text="Confianza %")
        
        self.tree.column("ID", width=50)
        self.tree.column("Fecha/Hora", width=150)
        self.tree.column("Cámara", width=100)
        self.tree.column("Tipo", width=200)
        self.tree.column("Confianza", width=100)
        
        self.tree.pack(side=tk.LEFT, fill="both", expand=True)

    def agregar_alerta(self, alerta, callback_revisar):
        """Agrega una alerta visual al panel"""
        # Ocultar mensaje vacío
        self.label_vacio.pack_forget()
        
        # Crear widget de alerta
        widget = self._crear_widget_alerta(alerta, callback_revisar)
        
        # Guardar referencia
        self.alertas_widgets[alerta["id_alerta"]] = {
            'widget': widget,
            'tipo': alerta['tipo']
        }
        
        # Agregar al historial
        self.agregar_al_historial(alerta)

    def _crear_widget_alerta(self, alerta, callback_revisar):
        """Crea un widget de alerta individual"""
        # Frame principal
        frame = tk.Frame(
            self.frame_alertas,
            bg="white",
            relief="solid",
            borderwidth=1
        )
        frame.pack(fill="x", padx=5, pady=3)
        
        # Color según tipo
        color = COLORES_FILTRO.get(alerta["tipo"], "#FF0000")
        frame.config(highlightbackground=color, highlightthickness=2)
        
        # Contenido
        contenido = tk.Frame(frame, bg="white")
        contenido.pack(fill="both", expand=True, padx=8, pady=6)
        
        # Header
        header = tk.Frame(contenido, bg="white")
        header.pack(fill="x")
        
        icono = "🚨" if "robo" in alerta["tipo"].lower() else "⚠️"
        tk.Label(
            header,
            text=icono,
            bg="white",
            font=("Arial", 14)
        ).pack(side=tk.LEFT)
        
        tk.Label(
            header,
            text=alerta["tipo"],
            bg="white",
            font=("Arial", 10, "bold"),
            fg=color
        ).pack(side=tk.LEFT, padx=5)
        
        # Info
        tk.Label(
            contenido,
            text=f"Cámara {int(alerta['id_camara'])+1} • {alerta['fecha_hora']} • Confianza: {alerta['confianza']:.1f}%",
            bg="white",
            font=("Arial", 8),
            fg="#666666"
        ).pack(anchor="w")
        
        # Botón revisar
        btn = tk.Button(
            contenido,
            text="Revisar",
            bg=color,
            fg="white",
            font=("Arial", 8, "bold"),
            relief="flat",
            cursor="hand2",
            command=lambda: callback_revisar(alerta["id_alerta"], frame)
        )
        btn.pack(anchor="e", pady=(5, 0))
        
        return frame

    def agregar_al_historial(self, alerta):
        """Agrega una alerta al historial"""
        valores = (
            alerta["id_alerta"],
            alerta["fecha_hora"],
            f"Cámara {int(alerta['id_camara'])+1}",
            alerta["tipo"],
            f"{alerta['confianza']:.1f}%"
        )
        
        item = self.tree.insert("", 0, values=valores)
        
        self.historial_items[alerta["id_alerta"]] = {
            'item': item,
            'tipo': alerta['tipo']
        }

    def aplicar_filtro(self, filtro):
        """Aplica un filtro a las alertas"""
        hay_visibles = False
        
        # Filtrar alertas activas
        for id_alerta, data in self.alertas_widgets.items():
            widget = data['widget']
            tipo = data['tipo']
            
            if filtro == "Todas" or tipo == filtro:
                widget.pack(fill="x", padx=5, pady=3)
                hay_visibles = True
            else:
                widget.pack_forget()
        
        # Mostrar mensaje si no hay alertas
        if not hay_visibles:
            self.label_vacio.pack(pady=20)
        
        # Filtrar historial
        for id_alerta, data in self.historial_items.items():
            item = data['item']
            tipo = data['tipo']
            
            if filtro == "Todas" or tipo == filtro:
                try:
                    self.tree.reattach(item, '', 0)
                except:
                    pass
            else:
                try:
                    self.tree.detach(item)
                except:
                    pass

    def eliminar_alerta(self, id_alerta):
        """Elimina una alerta del panel"""
        if id_alerta in self.alertas_widgets:
            del self.alertas_widgets[id_alerta]
