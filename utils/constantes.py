"""
Constantes y configuraciones del sistema SIVISEC
"""

# Colores por tipo de alerta
COLORES_FILTRO = {
    "Todas": "#26E439",
    "Sosteniendo mercancía": "#ECDA38",
    "Comportamiento sospechoso": "#D777A1",
    "Escondiendo mercancía": "#FF9800",
    "Posible robo": "#CB1616"
}

# Tipos de alertas disponibles
TIPOS_ALERTAS = [
    "Todas",
    "Sosteniendo mercancía",
    "Comportamiento sospechoso",
    "Escondiendo mercancía",
    "Posible robo"
]

# Configuración de cámaras
MAX_CAMARAS = 4
RESOLUCION_CAMARA = (640, 480)
TAMANO_FEED_VIDEO = (470, 350)

# Configuración de alertas
COOLDOWN_ALERTAS_SEGUNDOS = 5

# Configuración de UI
TITULO_APP = "Sistema de Seguridad SIVISEC"
COLOR_FONDO = "#f0f0f0"
COLOR_BARRA_ESTADO = "#d9d9d9"

# Dimensiones del slider
SLIDER_ANCHO = 500
SLIDER_ALTO = 35
SLIDER_INICIO_X = 20
