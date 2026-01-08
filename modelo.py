"""
Modelo del sistema de seguridad - Lógica de negocio
"""
import datetime


class SistemaSeguridad:
    """Maneja la lógica de negocio del sistema de seguridad"""
    
    def __init__(self):
        """Inicializa el sistema de seguridad con valores predeterminados"""
        self.historial_alertas = []
        self.estado_sistema = "Monitoreo"
        self.camaras_activas = []
        self.contador_alertas = 0

    def agregar_camara(self, id_camara, nombre, ubicacion):
        """Agrega una cámara al sistema"""
        self.camaras_activas.append({
            "id": id_camara,
            "nombre": nombre,
            "ubicacion": ubicacion
        })

    def eliminar_camara(self, id_camara):
        """Elimina una cámara del sistema"""
        self.camaras_activas = [
            cam for cam in self.camaras_activas 
            if cam["id"] != id_camara
        ]

    def agregar_alerta(self, id_camara, tipo_alerta, confianza=0):
        """Agrega una nueva alerta al sistema"""
        self.contador_alertas += 1
        alerta = {
            "id_alerta": self.contador_alertas,
            "fecha_hora": datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
            "id_camara": id_camara,
            "tipo": tipo_alerta,
            "confianza": confianza,
            "estado": "Activa"
        }
        self.historial_alertas.append(alerta)
        return alerta

    def marcar_alerta_revisada(self, id_alerta):
        """Marca una alerta como revisada"""
        for alerta in self.historial_alertas:
            if alerta["id_alerta"] == id_alerta:
                alerta["estado"] = "Revisada"
                return True
        return False

    def obtener_alertas_filtradas(self, filtro="Todas"):
        """Obtiene alertas según el filtro seleccionado"""
        if filtro == "Todas":
            return self.historial_alertas
        else:
            return [
                a for a in self.historial_alertas 
                if a["tipo"] == filtro
            ]

    def obtener_camara(self, id_camara):
        """Obtiene información de una cámara específica"""
        for cam in self.camaras_activas:
            if cam["id"] == id_camara:
                return cam
        return None

    def obtener_total_camaras(self):
        """Retorna el número total de cámaras activas"""
        return len(self.camaras_activas)
