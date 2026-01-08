import cv2
import mediapipe as mp
from ultralytics import YOLO
import math
import time

# --- Inicialización global ---
model = YOLO("yolov8n.pt")  # Modelo YOLOv8 (ligero)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- Memorias y parámetros ---
personas_memoria = {}
DISTANCIA_MAX = 80
TIEMPO_PERMANENCIA = 3.5  # Tiempo para considerar permanencia sospechosa
FRAMES_MIN = 3
ALPHA_S = 0.25  # Suavizado más estable

# --- Funciones auxiliares ---
def colision(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    return xA < xB and yA < yB

def centro(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2)//2, (y1 + y2)//2)

def suavizar(prev, act, alpha=ALPHA_S):
    return int(prev*(1-alpha) + act*alpha)

# --- Detección principal ---
def detectar_personas(frame):
    global personas_memoria

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # --- Detección YOLO ---
    results = model(frame, verbose=False)
    personas, vehiculos, armas = [], [], []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if cls == 0:  # persona
                personas.append((x1, y1, x2, y2))
            elif cls in [2, 3, 5, 7]:  # vehículos (car, motorcycle, bus, truck)
                vehiculos.append((x1, y1, x2, y2))
            # Nota: YOLOv8n no detecta armas por defecto, necesitarías un modelo especializado
            # elif cls == ARMA_CLASS_ID:  # arma (requiere modelo custom)
            #     armas.append((x1, y1, x2, y2))

    # --- Detección manos ---
    mano_boxes = []
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
            ys = [int(lm.y * h) for lm in hand_landmarks.landmark]
            mano_boxes.append((min(xs), min(ys), max(xs), max(ys)))

    nuevas_memorias = {}
    detecciones_actuales = []

    # --- Seguimiento de personas y lógica de estado ---
    for p_box in personas:
        px, py = centro(p_box)
        persona_id = None

        # Asociación con memorias anteriores
        for pid, data in personas_memoria.items():
            ox, oy = data["centro"]
            if math.dist((px, py), (ox, oy)) < DISTANCIA_MAX:
                persona_id = pid
                px = suavizar(ox, px)
                py = suavizar(oy, py)
                break

        if persona_id is None:
            persona_id = len(personas_memoria) + 1

        # Recuperar estado previo
        data_ant = personas_memoria.get(persona_id, {
            "estado": "normal",
            "tiempo_inicio": time.time(),
            "frames_estado": 0,
            "interactuando_vehiculo": False
        })
        estado = data_ant["estado"]
        tiempo_inicio = data_ant["tiempo_inicio"]
        frames_estado = data_ant["frames_estado"]
        interactuando_vehiculo = data_ant["interactuando_vehiculo"]

        # --- Lógica de transición de estados ---
        cerca_vehiculo = any(
            colision(p_box, v) or math.dist(centro(p_box), centro(v)) < 150
            for v in vehiculos
        )
        
        tiene_arma = any(colision(p_box, a) for a in armas)
        
        nuevo_estado = estado
        tiempo_permanencia = time.time() - tiempo_inicio

        # PRIORIDAD 1: Detectar arma (máxima prioridad)
        if tiene_arma:
            nuevo_estado = "arma"
            frames_estado = 0
        
        # PRIORIDAD 2: Interacción con vehículo
        elif cerca_vehiculo and mano_boxes:
            mano_cerca_vehiculo = any(
                colision(mano, v) for mano in mano_boxes for v in vehiculos
            )
            if mano_cerca_vehiculo:
                nuevo_estado = "interaccion"
                interactuando_vehiculo = True
                frames_estado += 1
            elif estado == "interaccion":
                frames_estado += 1
                if frames_estado >= FRAMES_MIN:
                    nuevo_estado = "posible_robo"
                    frames_estado = 0
        
        # PRIORIDAD 3: Permanencia sospechosa
        elif cerca_vehiculo:
            if tiempo_permanencia > TIEMPO_PERMANENCIA:
                nuevo_estado = "permanencia"
            else:
                nuevo_estado = "normal"
                frames_estado = 0
        
        # PRIORIDAD 4: Vehículo sin actividad luminosa (requiere análisis adicional)
        # Esta detección normalmente se hace a nivel de vehículo, no persona
        # elif detectar_vehiculo_sin_luces(vehiculos):
        #     nuevo_estado = "luminosa"
        
        # Estado normal
        else:
            if estado != "normal":
                tiempo_inicio = time.time()  # Reset timer
            nuevo_estado = "normal"
            frames_estado = 0
            interactuando_vehiculo = False

        # Actualizar memoria
        nuevas_memorias[persona_id] = {
            "estado": nuevo_estado,
            "centro": (px, py),
            "tiempo_inicio": tiempo_inicio,
            "frames_estado": frames_estado,
            "interactuando_vehiculo": interactuando_vehiculo
        }

        # Agregar detección solo si NO es estado normal
        if nuevo_estado != "normal":
            # Calcular confianza basada en el estado
            if nuevo_estado == "arma":
                confianza = 0.98
                tipo_alerta = "Persona portando arma"
            elif nuevo_estado == "posible_robo":
                confianza = 0.95
                tipo_alerta = "Posible robo de autopartes"
            elif nuevo_estado == "interaccion":
                confianza = 0.85
                tipo_alerta = "Interacción con el vehiculo"
            elif nuevo_estado == "permanencia":
                confianza = 0.80
                tipo_alerta = "Permanencia sospechosa"
            elif nuevo_estado == "luminosa":
                confianza = 0.75
                tipo_alerta = "Vehiculo sin actividad luminosa"
            else:
                confianza = 0.70
                tipo_alerta = "Permanencia sospechosa"

            detecciones_actuales.append({
                'tipo': tipo_alerta,
                'confianza': confianza,
                'persona_id': persona_id,
                'box': p_box
            })

    personas_memoria = nuevas_memorias

    # --- Dibujar resultados ---
    for pid, data in personas_memoria.items():
        estado = data["estado"]
        p_box = None
        for box in personas:
            if math.dist(centro(box), data["centro"]) < DISTANCIA_MAX:
                p_box = box
                break
        if not p_box:
            continue

        # Colores y etiquetas según estado
        if estado == "normal":
            color, label = (0, 255, 0), "Persona"
        elif estado == "permanencia":
            color, label = (0, 255, 255), "Permanencia sospechosa"
        elif estado == "interaccion":
            color, label = (0, 165, 255), "Interacción con vehiculo"
        elif estado == "luminosa":
            color, label = (255, 165, 0), "Sin actividad luminosa"
        elif estado == "posible_robo":
            color, label = (0, 0, 255), "Posible robo"
        elif estado == "arma":
            color, label = (128, 0, 128), "¡ARMA DETECTADA!"
        else:
            color, label = (255, 255, 0), "Alerta"

        cv2.rectangle(frame, (p_box[0], p_box[1]), (p_box[2], p_box[3]), color, 2)
        cv2.putText(frame, f"{label} (ID {pid})",
                    (p_box[0], p_box[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return frame, detecciones_actuales