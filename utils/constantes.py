"""
Constantes y configuraciones del sistema
"""

# Colores por tipo de alerta
COLORES_FILTRO = {
    "Todas": "#26E439",
    "Permanencia sospechosa": "#ECDA38",
    "Interacción con el vehiculo": "#D777A1",
    "Vehiculo sin actividad luminosa": "#FF9800",
    "Posible robo de autopartes": "#CB1616",
    "Persona portando arma": "#050C55"
}

# Tipos de alertas disponibles
TIPOS_ALERTAS = [
    "Todas",
    "Permanencia sospechosa",
    "Interacción con el vehiculo",
    "Vehiculo sin actividad luminosa",
    "Posible robo de autopartes",
    "Persona portando arma"
]

# Configuración de cámaras
MAX_CAMARAS = 4
RESOLUCION_CAMARA = (640, 480)
TAMANO_FEED_VIDEO = (470, 350)

# Configuración de alertas
COOLDOWN_ALERTAS_SEGUNDOS = 5

# Configuración de UI
TITULO_APP = "Sistema de seguridad"
COLOR_FONDO = "#f0f0f0"
COLOR_BARRA_ESTADO = "#d9d9d9"

# Dimensiones del slider
SLIDER_ANCHO = 500
SLIDER_ALTO = 35
SLIDER_INICIO_X = 20
