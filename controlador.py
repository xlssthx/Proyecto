"""
Controlador del sistema de seguridad - Coordinador
"""
import threading
import time
import cv2
from PIL import Image, ImageTk
from modelo import SistemaSeguridad
from vista import SistemaSeguridadVista
from utils.constantes import MAX_CAMARAS, RESOLUCION_CAMARA, COOLDOWN_ALERTAS_SEGUNDOS


class SistemaSeguridadControlador:
    """Controlador que coordina el modelo y la vista"""
    
    def __init__(self, root):
        self.root = root
        self.modelo = SistemaSeguridad()
        self.vista = SistemaSeguridadVista(root)
        
        self.monitoreo_activo = False
        self.hilos_camara = {}
        self.capturas_video = {}
        self.filtro_actual = "Todas"
        
        # Control de cooldown para alertas
        self.ultima_alerta_por_tipo = {}
        
        self._conectar_callbacks()
        self._iniciar_camaras_automatico()

    def _conectar_callbacks(self):
        """Conecta los callbacks entre vista y controlador"""
        # Botones principales
        self.vista.btn_iniciar.config(command=self.iniciar_monitoreo)
        self.vista.btn_detener.config(command=self.detener_monitoreo)
        
        # Callbacks de la vista
        self.vista.set_callback('agregar_camara', self.agregar_camara)
        self.vista.set_callback('eliminar_camara', self.eliminar_camara)
        self.vista.set_callback('filtro_cambiado', self._on_filtro_cambiado)
        self.vista.set_callback('marcar_revisada', self.marcar_alerta_revisada)

    def _iniciar_camaras_automatico(self):
        """Detecta y agrega cámaras automáticamente"""
        for i in range(MAX_CAMARAS):
            if len(self.modelo.camaras_activas) >= MAX_CAMARAS:
                break
            self._agregar_camara_real(str(i), f"Cámara {i+1}", f"Ubicación {i+1}")

    def _agregar_camara_real(self, id_camara, nombre, ubicacion):
        """Agrega una cámara real al sistema"""
        if len(self.modelo.camaras_activas) >= MAX_CAMARAS:
            return False
        
        if any(cam["id"] == id_camara for cam in self.modelo.camaras_activas):
            return False
        
        try:
            # Intentar abrir la cámara
            cap = cv2.VideoCapture(int(id_camara))
            if not cap.isOpened():
                return False
            
            # Configurar resolución
            ancho, alto = RESOLUCION_CAMARA
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
            
            # Guardar captura
            self.capturas_video[id_camara] = cap
            
            # Agregar al modelo
            self.modelo.agregar_camara(id_camara, nombre, ubicacion)
            
            # Agregar a la vista
            self.vista.agregar_camara(id_camara, f"{nombre} ({ubicacion})")
            
            # Si está en monitoreo, iniciar hilo
            if self.monitoreo_activo:
                self._iniciar_hilo_camara(id_camara)
            
            return True
            
        except Exception as e:
            print(f"Error al agregar cámara {id_camara}: {e}")
            return False

    def agregar_camara(self):
        """Agrega una nueva cámara (callback público)"""
        if len(self.modelo.camaras_activas) >= MAX_CAMARAS:
            self.vista.mostrar_mensaje(
                "Límite alcanzado",
                f"Solo se permiten {MAX_CAMARAS} cámaras máximo"
            )
            return
        
        # Buscar siguiente cámara disponible
        for i in range(10):
            id_camara = str(i)
            if not any(cam["id"] == id_camara for cam in self.modelo.camaras_activas):
                if self._agregar_camara_real(id_camara, f"Cámara {i+1}", f"Ubicación {i+1}"):
                    return
        
        self.vista.mostrar_mensaje(
            "Error",
            "No se encontraron más cámaras disponibles"
        )

    def eliminar_camara(self, id_camara):
        """Elimina una cámara del sistema"""
        # Liberar captura
        if id_camara in self.capturas_video:
            if self.capturas_video[id_camara].isOpened():
                self.capturas_video[id_camara].release()
            del self.capturas_video[id_camara]
        
        # Eliminar del modelo
        self.modelo.eliminar_camara(id_camara)
        
        # Eliminar de la vista
        self.vista.eliminar_camara(id_camara)
        
        # Limpiar hilo
        if id_camara in self.hilos_camara:
            del self.hilos_camara[id_camara]

    def iniciar_monitoreo(self):
        """Inicia el monitoreo de todas las cámaras"""
        if self.monitoreo_activo:
            return
        
        self.monitoreo_activo = True
        self.modelo.estado_sistema = "Monitoreo"
        self.vista.actualizar_estado("Monitoreo")
        
        # Iniciar hilos para cada cámara
        for camara in self.modelo.camaras_activas:
            id_camara = camara["id"]
            
            # Reabrir cámara si es necesario
            if id_camara.isdigit():
                if id_camara not in self.capturas_video or \
                   not self.capturas_video[id_camara].isOpened():
                    cap = cv2.VideoCapture(int(id_camara))
                    if cap.isOpened():
                        ancho, alto = RESOLUCION_CAMARA
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
                        self.capturas_video[id_camara] = cap
                    else:
                        continue
            
            self._iniciar_hilo_camara(id_camara)
        
        self.vista.mostrar_mensaje("Monitoreo", "El monitoreo se ha iniciado")

    def detener_monitoreo(self):
        """Detiene el monitoreo"""
        if not self.monitoreo_activo:
            return
        
        self.monitoreo_activo = False
        self.modelo.estado_sistema = "Detenido"
        self.vista.actualizar_estado("Detenido")
        
        time.sleep(0.2)
        self.vista.mostrar_mensaje("Monitoreo", "El monitoreo se ha detenido")

    def _iniciar_hilo_camara(self, id_camara):
        """Inicia el hilo de procesamiento de una cámara"""
        hilo = threading.Thread(
            target=self._procesar_video,
            args=(id_camara,)
        )
        hilo.daemon = True
        hilo.start()
        self.hilos_camara[id_camara] = hilo

    def _procesar_video(self, id_camara):
        """Procesa el video de una cámara (ejecuta en hilo)"""
        # Intentar importar el módulo de detección
        try:
            from proto1 import detectar_personas
            usar_deteccion = True
        except ImportError:
            usar_deteccion = False
            print(f"Advertencia: proto1.py no disponible, sin detección IA")
        
        while self.monitoreo_activo and id_camara in self.capturas_video:
            cap = self.capturas_video[id_camara]
            
            if not cap.isOpened():
                break
            
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            # Aplicar detección si está disponible
            detecciones = None
            if usar_deteccion:
                try:
                    frame, detecciones = detectar_personas(frame)
                    if detecciones:
                        self._procesar_detecciones(detecciones, id_camara)
                except Exception as e:
                    print(f"Error en detección: {e}")
                    usar_deteccion = False
            
            # Actualizar frame en la vista
            self._actualizar_frame_video(id_camara, frame)
            
            time.sleep(0.03)

    def _actualizar_frame_video(self, id_camara, frame):
        """Actualiza el frame de video en la vista"""
        try:
            # Convertir a RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            # Redimensionar
            img_resized = img.resize((470, 350), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(image=img_resized)
            
            # Actualizar en el hilo principal
            self.root.after_idle(
                self.vista.actualizar_video_camara,
                id_camara,
                img_tk
            )
        except Exception as e:
            print(f"Error actualizando video: {e}")

    def _procesar_detecciones(self, detecciones, id_camara):
        """Procesa las detecciones y genera alertas"""
        if not detecciones:
            return
        
        for deteccion in detecciones:
            tipo = deteccion.get('tipo', 'Desconocido')
            confianza = deteccion.get('confianza', 0) * 100
            
            # Ignorar detecciones normales de persona
            if tipo.lower() in ['persona', 'person', 'people']:
                continue
            
            # Mapear a tipo de alerta
            tipo_alerta = self._mapear_tipo_alerta(tipo)
            
            # Verificar filtro actual
            if self.filtro_actual != "Todas" and tipo_alerta != self.filtro_actual:
                continue
            
            # Verificar cooldown
            if not self._puede_generar_alerta(id_camara, tipo_alerta):
                continue
            
            # Crear alerta
            alerta = self.modelo.agregar_alerta(id_camara, tipo_alerta, confianza)
            
            # Mostrar en vista
            self.root.after(0, self.vista.agregar_alerta, alerta)
            
            print(f"🚨 ALERTA: {tipo_alerta} en Cámara {int(id_camara)+1} ({confianza:.1f}%)")

    def _mapear_tipo_alerta(self, tipo_deteccion):
        """Mapea el tipo de detección a tipo de alerta"""
        tipo_lower = tipo_deteccion.lower()
        
        if 'sosteniendo' in tipo_lower or 'holding' in tipo_lower:
            return "Sosteniendo mercancía"
        elif 'sospechoso' in tipo_lower or 'suspicious' in tipo_lower:
            return "Comportamiento sospechoso"
        elif 'escondiendo' in tipo_lower or 'hiding' in tipo_lower:
            return "Escondiendo mercancía"
        elif 'robo' in tipo_lower or 'theft' in tipo_lower or 'stealing' in tipo_lower:
            return "Posible robo"
        else:
            return "Comportamiento sospechoso"

    def _puede_generar_alerta(self, id_camara, tipo_alerta):
        """Verifica si puede generar una alerta (sistema de cooldown)"""
        clave = f"{id_camara}_{tipo_alerta}"
        ahora = time.time()
        
        if clave in self.ultima_alerta_por_tipo:
            tiempo_transcurrido = ahora - self.ultima_alerta_por_tipo[clave]
            if tiempo_transcurrido < COOLDOWN_ALERTAS_SEGUNDOS:
                return False
        
        self.ultima_alerta_por_tipo[clave] = ahora
        return True

    def _on_filtro_cambiado(self, filtro):
        """Maneja el cambio de filtro"""
        self.filtro_actual = filtro

    def marcar_alerta_revisada(self, id_alerta, widget):
        """Marca una alerta como revisada"""
        if self.modelo.marcar_alerta_revisada(id_alerta):
            # Animar widget
            widget.config(bg="#e8f5e9")
            
            # Eliminar de la vista
            self.vista.eliminar_alerta_visual(id_alerta)
            
            # Destruir widget después de animación
            self.root.after(300, widget.destroy)

    def __del__(self):
        """Limpieza al cerrar el programa"""
        try:
            self.monitoreo_activo = False
            time.sleep(0.3)
            
            for cap in self.capturas_video.values():
                if cap.isOpened():
                    cap.release()
        except:
            pass
