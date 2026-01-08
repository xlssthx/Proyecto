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
TIEMPO_SOSPECHOSO = 3.5
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
    personas, botellas = [], []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if cls == 0:  # persona
                personas.append((x1, y1, x2, y2))
            elif cls == 39:  # botella
                botellas.append((x1, y1, x2, y2))

    # --- Detección manos ---
    mano_boxes = []
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
            ys = [int(lm.y * h) for lm in hand_landmarks.landmark]
            mano_boxes.append((min(xs), min(ys), max(xs), max(ys)))

    nuevas_memorias = {}
    detecciones_actuales = []  # ← NUEVA: Lista para almacenar las detecciones

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
            "ha_escondido": False,
            "tiempo_sospechoso": 0,
            "frames_estado": 0
        })
        estado = data_ant["estado"]
        ha_escondido = data_ant["ha_escondido"]
        tiempo_sospechoso = data_ant["tiempo_sospechoso"]
        frames_estado = data_ant["frames_estado"]

        # --- Lógica de transición de estados ---
        sostenida = any(
            colision(mano, p_box) and any(colision(mano, b) for b in botellas)
            for mano in mano_boxes
        )
        nuevo_estado = estado

        if estado != "comportamiento_sospechoso":
            if estado == "posible_robo" and sostenida:
                nuevo_estado = "comportamiento_sospechoso"
                tiempo_sospechoso = time.time()
                frames_estado = 0
            elif sostenida:
                nuevo_estado = "sosteniendo"
            elif estado == "sosteniendo" and not botellas:
                nuevo_estado = "escondiendo"
                ha_escondido = True
            elif estado == "escondiendo":
                if sostenida:
                    nuevo_estado = "sosteniendo"
                    frames_estado = 0
                elif mano_boxes and not sostenida:
                    frames_estado += 1
                    if frames_estado >= FRAMES_MIN:
                        nuevo_estado = "posible_robo"
                        frames_estado = 0
            elif estado == "normal":
                frames_estado = 0

        # --- Reset sospechoso si pasa el tiempo ---
        if nuevo_estado == "comportamiento_sospechoso":
            if time.time() - tiempo_sospechoso > TIEMPO_SOSPECHOSO:
                nuevo_estado = "normal"
                tiempo_sospechoso = 0

        nuevas_memorias[persona_id] = {
            "estado": nuevo_estado,
            "centro": (px, py),
            "ha_escondido": ha_escondido,
            "tiempo_sospechoso": tiempo_sospechoso,
            "frames_estado": frames_estado
        }

        # ← NUEVO: Agregar detección solo si NO es estado normal
        if nuevo_estado != "normal":
            # Calcular confianza basada en el estado y otros factores
            if nuevo_estado == "posible_robo":
                confianza = 0.95
            elif nuevo_estado == "comportamiento_sospechoso":
                confianza = 0.90
            elif nuevo_estado == "escondiendo":
                confianza = 0.85
            elif nuevo_estado == "sosteniendo":
                confianza = 0.80
            else:
                confianza = 0.75

            # Mapear estado a tipo de alerta
            tipo_alerta = None
            if nuevo_estado == "sosteniendo":
                tipo_alerta = "Sosteniendo mercancía"
            elif nuevo_estado == "escondiendo":
                tipo_alerta = "Escondiendo mercancía"
            elif nuevo_estado == "posible_robo":
                tipo_alerta = "Posible robo"
            elif nuevo_estado == "comportamiento_sospechoso":
                tipo_alerta = "Comportamiento sospechoso"

            if tipo_alerta:
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

        if estado == "normal":
            color, label = (0, 255, 0), "Persona"
        elif estado == "sosteniendo":
            color, label = (0, 255, 255), "Sosteniendo mercancía"
        elif estado == "escondiendo":
            color, label = (0, 165, 255), "Escondiendo mercancía"
        elif estado == "posible_robo":
            color, label = (0, 0, 255), "Posible robo"
        else:
            color, label = (255, 128, 255), "Comportamiento sospechoso"

        cv2.rectangle(frame, (p_box[0], p_box[1]), (p_box[2], p_box[3]), color, 2)
        cv2.putText(frame, f"{label} (ID {pid})",
                    (p_box[0], p_box[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # ← CORREGIDO: Devolver las detecciones reales
    return frame, detecciones_actuales