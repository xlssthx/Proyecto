from ultralytics import YOLO
import torch
import os
from datetime import datetime

# Ruta al dataset combinado
DATASET_PATH = "combined_bottles_hands_people/data.yaml"

# Configuración del modelo
MODEL_SIZE = "yolov8n.pt"  # Opciones: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
                            # n = nano (más rápido, menos preciso) RECOMENDADO con limitaciones de gpu
                            # s = small
                            # m = medium
                            # l = large
                            # x = xlarge (más lento, más preciso)
                            # No todos los modelos sirven para detección en tiempo real

# Hiperparámetros de entrenamiento
EPOCHS = 50               # Número de épocas (ajusta según tu paciencia)
IMG_SIZE = 416            # Tamaño de imagen (640 es estándar, 416 es más rápido)
BATCH_SIZE = 4            # Tamaño del batch (ajusta según tu RAM)
WORKERS = 2               # Número de workers para cargar datos
PATIENCE = 10             # Early stopping: detiene si no mejora en X épocas

# Carpeta para guardar resultados
PROJECT_NAME = "runs/detect"
RUN_NAME = f"bottles_hands_people_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Verificacion inicial
print("=" * 70)
print(" ENTRENAMIENTO DE YOLO - DETECTOR MULTI-CLASE")
print("=" * 70)
print()

# Verificar que existe el dataset
if not os.path.exists(DATASET_PATH):
    print(f"x Error: No se encontró el dataset en: {DATASET_PATH}")
    print("   Asegúrate de haber ejecutado 'combine_datasets.py' primero")
    exit(1)

print(f"✓ Dataset encontrado: {DATASET_PATH}")

# Verificar hardware disponible
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n  Hardware detectado:")
print(f"   Dispositivo: {device.upper()}")

if device == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("   ¡Perfecto! El entrenamiento será rápido")
else:
    print("   CPU detectada - El entrenamiento será LENTO")
    print("   Tiempo estimado: 25-40 horas")
    print("   Recomendación: Usa Google Colab con GPU gratuita")
    
    respuesta = input("\n¿Deseas continuar de todos modos? (s/n): ")
    if respuesta.lower() != 's':
        print("Entrenamiento cancelado. Usa Google Colab para entrenar más rápido.")
        exit(0)

# Configuración de modelo
print("\nConfiguración del entrenamiento:")
print(f"   Modelo base: {MODEL_SIZE}")
print(f"   Épocas: {EPOCHS}")
print(f"   Tamaño de imagen: {IMG_SIZE}x{IMG_SIZE}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Workers: {WORKERS}")
print(f"   Patience (early stopping): {PATIENCE}")
print(f"   Resultados en: {PROJECT_NAME}/{RUN_NAME}")

print("\n" + "=" * 70)
input("Presiona ENTER para iniciar el entrenamiento... ")
print("=" * 70)

# Cargar modelo base
print("\n Cargando modelo base...")
model = YOLO(MODEL_SIZE)
print(f"✓ Modelo {MODEL_SIZE} cargado")

# Entrenamiento
print("\n Iniciando entrenamiento...\n")
print(" Puedes detener el entrenamiento en cualquier momento con Ctrl+C")
print("   El mejor modelo se guardará automáticamente\n")

try:
    results = model.train(
        # Datos
        data=DATASET_PATH,
        
        # Hiperparámetros básicos
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        
        # Hardware
        device=device,
        workers=WORKERS,
        
        # Optimizaciones
        patience=PATIENCE,      # Early stopping
        save=True,              # Guardar checkpoints
        save_period=10,         # Guardar cada 10 épocas
        cache=False,            # No cachear en RAM (para laptops)
        
        # Carpetas de salida
        project=PROJECT_NAME,
        name=RUN_NAME,
        exist_ok=True,
        
        # Métricas y visualización
        plots=True,             # Generar gráficas
        verbose=True,           # Mostrar detalles
        
        # Augmentación de datos (puedes ajustar)
        hsv_h=0.015,           # Variación de tono
        hsv_s=0.7,             # Variación de saturación
        hsv_v=0.4,             # Variación de brillo
        degrees=0.0,           # Rotación
        translate=0.1,         # Translación
        scale=0.5,             # Escala
        shear=0.0,             # Distorsión
        perspective=0.0,       # Perspectiva
        flipud=0.0,            # Voltear verticalmente
        fliplr=0.5,            # Voltear horizontalmente (50%)
        mosaic=1.0,            # Mosaic augmentation
        mixup=0.0,             # Mixup augmentation
    )
    
    print("\n" + "=" * 70)
    print("✓ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    
except KeyboardInterrupt:
    print("\n\n¡  Entrenamiento interrumpido por el usuario")
    print("   El mejor modelo hasta ahora se guardó automáticamente")
    
except Exception as e:
    print(f"\n\nx Error durante el entrenamiento: {e}")
    exit(1)

# Validación final
print("\n Validando el modelo entrenado...")

try:
    metrics = model.val()
    
    print("\n" + "=" * 70)
    print("MÉTRICAS FINALES")
    print("=" * 70)
    print(f"   mAP50 (IoU=0.5): {metrics.box.map50:.3f}")
    print(f"   mAP50-95: {metrics.box.map:.3f}")
    print(f"   Precisión: {metrics.box.mp:.3f}")
    print(f"   Recall: {metrics.box.mr:.3f}")
    print("=" * 70)
    
except Exception as e:
    print(f"¡  No se pudo validar el modelo: {e}")

# Ubicación de los modelos
best_model_path = f"{PROJECT_NAME}/{RUN_NAME}/weights/best.pt"
last_model_path = f"{PROJECT_NAME}/{RUN_NAME}/weights/last.pt"

print("\n Modelos guardados:")
print(f"   Mejor modelo: {best_model_path}")
print(f"   Último modelo: {last_model_path}")

# Prueba de modelo
print("\n ¿Quieres probar el modelo con una imagen de validación?")
respuesta = input("   (s/n): ")

if respuesta.lower() == 's':
    # Buscar imágenes de validación
    val_images_dir = "combined_bottles_hands_people/valid/images"
    
    if os.path.exists(val_images_dir):
        val_images = [f for f in os.listdir(val_images_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if val_images:
            test_img = f"{val_images_dir}/{val_images[0]}"
            
            print(f"\n  Probando con: {test_img}")
            
            # Cargar el mejor modelo
            trained_model = YOLO(best_model_path)
            
            # Hacer predicción
            results = trained_model(test_img)
            
            # Guardar resultado
            output_path = f"{PROJECT_NAME}/{RUN_NAME}/test_prediction.jpg"
            results[0].save(output_path)
            
            print(f"✓ Predicción guardada en: {output_path}")
            
            # Mostrar detecciones
            print("\n Detecciones:")
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = results[0].names[cls_id]
                print(f"   - {class_name}: {conf:.2%} confianza")
        else:
            print("   No se encontraron imágenes de validación")
    else:
        print("   No se encontró el directorio de validación")

# Instrucciones finales
print("\n" + "=" * 70)
print(" ¡TODO LISTO!")
print("=" * 70)
print("\n Para usar tu modelo entrenado:")
print(f"""
from ultralytics import YOLO

# Cargar modelo entrenado
model = YOLO('{best_model_path}')

# Hacer predicciones
results = model('tu_imagen.jpg')

# Mostrar resultados
results[0].show()

# O guardar
results[0].save('resultado.jpg')

# Clases que detecta:
# 0: bottle (botella)
# 1: hand (mano)
# 2: people/person (persona)
""")

print("=" * 70)
print("\n Métricas de entrenamiento:")
print(f"   Abre: {PROJECT_NAME}/{RUN_NAME}/")
print(f"   Archivos: results.png, confusion_matrix.png, etc.")
print("\n" + "=" * 70)
