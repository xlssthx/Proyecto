"""
====================================================================
DETECTOR DE SEGURIDAD COMPLETO - YOLOv8n
Version optimizada para PC local
Tiempo estimado: 2-3 horas en GPU (NVIDIA RTX 3060 o superior)
====================================================================
REQUISITOS:
  Python: 3.8 - 3.11 (recomendado 3.10)
  Instalacion: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && pip install ultralytics roboflow PyYAML pillow opencv-python
  GPU: NVIDIA con CUDA 11.8+ (opcional pero muy recomendado)
====================================================================
"""

import os
import shutil
import yaml
import torch
from datetime import datetime
from pathlib import Path

print("=" * 70)
print("INSTALANDO DEPENDENCIAS")
print("=" * 70)
print("Ejecuta primero en terminal:")
print("  pip install ultralytics roboflow PyYAML torch torchvision")
print("=" * 70)

from roboflow import Roboflow
from ultralytics import YOLO

# ====================================================================
# CONFIGURACION
# ====================================================================
print("\n" + "=" * 70)
print("CONFIGURACION DEL SISTEMA")
print("=" * 70)

# API Key de Roboflow
API_KEY = "QOL8UwOKMB97adVbFAU3"

# Deteccion de GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU detectada: {gpu_name}")
    print(f"Memoria GPU: {gpu_mem:.1f} GB")
    
    # Ajuste automatico de batch size segun GPU
    if gpu_mem >= 15:
        batch_size = 32
    elif gpu_mem >= 12:
        batch_size = 24
    elif gpu_mem >= 8:
        batch_size = 16
    else:
        batch_size = 8
else:
    print("ADVERTENCIA: No se detecto GPU. El entrenamiento sera MUY lento.")
    print("Recomendacion: Instala CUDA y PyTorch con soporte GPU")
    batch_size = 4

# Configuracion de entrenamiento
CONFIG = {
    'model_size': 'yolov8n.pt',  # Modelo nano (mas rapido)
    'epochs': 50,                 # 50 epocas completas
    'patience': 10,               # Early stopping
    'batch_size': batch_size,
    'img_size': 384,              # Resolucion optimizada
}

print(f"\nModelo: {CONFIG['model_size']}")
print(f"Epocas: {CONFIG['epochs']}")
print(f"Batch size: {CONFIG['batch_size']}")
print(f"Imagen: {CONFIG['img_size']}px")
print(f"Dispositivo: {device.upper()}")
print(f"Tiempo estimado: 2-3 horas (con GPU)")

# Directorios de trabajo
BASE_DIR = Path.cwd()
DATASETS_DIR = BASE_DIR / 'datasets_temp'
OUTPUT_DIR = BASE_DIR / 'combined_dataset'
MODELS_DIR = BASE_DIR / 'models'

# Crear directorios
DATASETS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ====================================================================
# FUNCIONES AUXILIARES
# ====================================================================

def download_dataset_safe(rf, workspace, project, version, name):
    """Descarga dataset de Roboflow con manejo de errores"""
    dataset_folder = DATASETS_DIR / f"{project}-{version}"
    
    if dataset_folder.exists() and (dataset_folder / 'data.yaml').exists():
        print(f"[OK] '{name}' ya existe en cache")
        return str(dataset_folder)
    
    print(f"Descargando '{name}'...")
    try:
        proj = rf.workspace(workspace).project(project)
        
        # Cambiar al directorio temporal
        os.chdir(DATASETS_DIR)
        dataset = proj.version(version).download("yolov8")
        os.chdir(BASE_DIR)
        
        print(f"[OK] Descargado: {name}")
        return dataset.location
    except Exception as e:
        print(f"[ERROR] {name}: {e}")
        os.chdir(BASE_DIR)
        return None


def process_split(dataset_path, output_path, split, prefix, offset):
    """Procesa un split (train/valid/test) y ajusta las clases"""
    img_src = Path(dataset_path) / split / 'images'
    lbl_src = Path(dataset_path) / split / 'labels'
    
    if not img_src.exists():
        return 0
    
    img_dst = Path(output_path) / split / 'images'
    lbl_dst = Path(output_path) / split / 'labels'
    
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for img_file in img_src.iterdir():
        if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
        
        # Nuevo nombre de archivo
        new_name = f'{prefix}_{count:06d}'
        ext = img_file.suffix
        
        # Copiar imagen
        shutil.copy(img_file, img_dst / f'{new_name}{ext}')
        
        # Procesar etiquetas
        label_file = lbl_src / f'{img_file.stem}.txt'
        if label_file.exists():
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            adjusted = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    old_class = int(parts[0])
                    new_class = old_class + offset
                    parts[0] = str(new_class)
                    adjusted.append(' '.join(parts) + '\n')
            
            with open(lbl_dst / f'{new_name}.txt', 'w') as f:
                f.writelines(adjusted)
        
        count += 1
    
    return count


def cleanup_datasets(datasets_dir):
    """Elimina los datasets individuales para liberar espacio"""
    print("\n" + "=" * 70)
    print("LIMPIEZA DE DATASETS TEMPORALES")
    print("=" * 70)
    
    if datasets_dir.exists():
        try:
            size_before = sum(f.stat().st_size for f in datasets_dir.rglob('*') if f.is_file()) / 1e9
            shutil.rmtree(datasets_dir)
            print(f"[OK] Eliminados {size_before:.2f} GB de datasets temporales")
        except Exception as e:
            print(f"[ADVERTENCIA] No se pudo eliminar datasets_temp: {e}")
    else:
        print("[INFO] No hay datasets temporales que eliminar")


# ====================================================================
# DESCARGA DE DATASETS
# ====================================================================
print("\n" + "=" * 70)
print("DESCARGANDO 8 DATASETS")
print("=" * 70)

rf = Roboflow(api_key=API_KEY)

# Lista de datasets a descargar
datasets_config = [
    ("test-n0rkj", "bottles-ca5xq", 1, "Botellas"),
    ("faces-e24ww", "hands-clfdp", 1, "Manos"),
    ("leo-ueno", "people-detection-o4rdr", 11, "Personas"),
    ("mscprojects", "shoplifting-cuzf8", 2, "Robo"),
    ("yolov7test-u13vc", "weapon-detection-m7qso", 1, "Armas"),
    ("mahad-ahmed", "gun-and-knife-detection", 1, "Gun-Knife"),
    ("personal-g5mzf", "soda-can-object-detection", 3, "Latas"),
    ("thdmd9", "snack-m981x", 2, "Papitas")
]

datasets = []
for i, (workspace, project, version, name) in enumerate(datasets_config, 1):
    print(f"\n[{i}/8] {name}")
    print("-" * 70)
    ds = download_dataset_safe(rf, workspace, project, version, name)
    datasets.append(ds)
    
    # Fallback para Robo si falla
    if name == "Robo" and not ds:
        print("Intentando alternativas para Robo...")
        ds = download_dataset_safe(rf, "shoplifting-dataset", "shoplifting-v2", 2, "Robo V2")
        if not ds:
            ds = download_dataset_safe(rf, "ktun", "shoplifting-ajeze", 1, "Robo V3")
        datasets[-1] = ds

# Verificar que todos se descargaron
if not all(datasets):
    print("\n[ERROR] No se descargaron todos los datasets:")
    for i, (ds, (_, _, _, name)) in enumerate(zip(datasets, datasets_config), 1):
        status = "[OK]" if ds else "[FALLO]"
        print(f"  {status} [{i}] {name}")
    raise Exception("Revisa la configuracion de Roboflow")

print("\n[OK] Todos los datasets descargados")

# ====================================================================
# COMBINAR DATASETS
# ====================================================================
print("\n" + "=" * 70)
print("COMBINANDO 8 DATASETS")
print("=" * 70)

# Crear estructura de directorios
for split in ['train', 'valid', 'test']:
    (OUTPUT_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)

# Recopilar todas las clases
all_classes = []
print("\nCLASES POR DATASET:")
for ds_path, (_, _, _, ds_name) in zip(datasets, datasets_config):
    with open(Path(ds_path) / 'data.yaml', 'r') as f:
        data = yaml.safe_load(f)
    classes = data['names']
    all_classes.extend(classes)
    print(f"  {ds_name:15} -> {classes}")

print(f"\nTOTAL DE CLASES: {len(all_classes)}")
print(f"Clases: {', '.join(all_classes)}")

# Procesar cada split
total_images = 0
for split in ['train', 'valid', 'test']:
    print(f"\n[{split.upper()}]")
    print("-" * 70)
    
    current_offset = 0
    split_total = 0
    
    for i, (ds_path, (_, _, _, ds_name)) in enumerate(zip(datasets, datasets_config), 1):
        with open(Path(ds_path) / 'data.yaml', 'r') as f:
            data = yaml.safe_load(f)
        num_classes = len(data['names'])
        
        count = process_split(
            ds_path, OUTPUT_DIR, split, 
            f'd{i}', current_offset
        )
        
        print(f"  [{i}/8] {ds_name:15} -> {count:6,} imagenes (offset: {current_offset})")
        
        current_offset += num_classes
        split_total += count
    
    total_images += split_total
    print(f"  [OK] Total {split}: {split_total:,} imagenes")

# Crear data.yaml combinado
combined_yaml = {
    'path': str(OUTPUT_DIR.absolute()),
    'train': 'train/images',
    'val': 'valid/images',
    'test': 'test/images',
    'nc': len(all_classes),
    'names': all_classes
}

with open(OUTPUT_DIR / 'data.yaml', 'w') as f:
    yaml.dump(combined_yaml, f, default_flow_style=False, sort_keys=False)

print("\n" + "=" * 70)
print("COMBINACION COMPLETADA")
print("=" * 70)
print(f"Total de imagenes: {total_images:,}")
print(f"Total de clases: {len(all_classes)}")
print(f"Dataset guardado en: {OUTPUT_DIR}")

# ====================================================================
# LIMPIAR DATASETS TEMPORALES
# ====================================================================
cleanup_datasets(DATASETS_DIR)

# ====================================================================
# ENTRENAMIENTO
# ====================================================================
print("\n" + "=" * 70)
print("INICIANDO ENTRENAMIENTO CON YOLOV8N")
print("=" * 70)
print(f"Primera epoca: ~3-4 min")
print(f"Siguientes epocas: ~2-3 min cada una")
print(f"Tiempo total estimado: 2-3 horas")
print("=" * 70)

# Cargar modelo base
model = YOLO(CONFIG['model_size'])

try:
    # Entrenar
    results = model.train(
        data=str(OUTPUT_DIR / 'data.yaml'),
        epochs=CONFIG['epochs'],
        imgsz=CONFIG['img_size'],
        batch=CONFIG['batch_size'],
        patience=CONFIG['patience'],
        device=device,
        workers=8,
        
        # Optimizaciones para velocidad
        cache=True,           # Cachear imagenes en RAM
        amp=True,            # Automatic Mixed Precision
        close_mosaic=10,     # Desactivar mosaic en ultimas epocas
        
        # Data augmentation reducida (mas rapido)
        hsv_h=0.01,
        hsv_s=0.4,
        hsv_v=0.3,
        degrees=0.0,
        translate=0.05,
        scale=0.2,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.5,
        
        # Configuracion de guardado
        save=True,
        save_period=5,
        
        # Directorios
        project=str(MODELS_DIR / 'runs' / 'detect'),
        name='security_nano',
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    
    print("\n" + "=" * 70)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 70)
    
except KeyboardInterrupt:
    print("\n[ADVERTENCIA] Entrenamiento interrumpido por usuario")
    print("El modelo parcial se guardo en:", MODELS_DIR / 'runs' / 'detect' / 'security_nano')
    
except Exception as e:
    print(f"\n[ERROR] Fallo en entrenamiento: {e}")
    raise

# ====================================================================
# VALIDACION
# ====================================================================
print("\n" + "=" * 70)
print("VALIDANDO MODELO")
print("=" * 70)

try:
    metrics = model.val()
    
    print("METRICAS FINALES:")
    print("-" * 70)
    print(f"mAP50:          {metrics.box.map50:.4f}")
    print(f"mAP50-95:       {metrics.box.map:.4f}")
    print(f"Precision:      {metrics.box.mp:.4f}")
    print(f"Recall:         {metrics.box.mr:.4f}")
    print("-" * 70)
    
except Exception as e:
    print(f"[ERROR] Fallo en validacion: {e}")

# ====================================================================
# GUARDAR MODELO FINAL
# ====================================================================
print("\n" + "=" * 70)
print("GUARDANDO MODELO FINAL")
print("=" * 70)

best_model_src = MODELS_DIR / 'runs' / 'detect' / 'security_nano' / 'weights' / 'best.pt'
best_model_dst = MODELS_DIR / 'sivisecb.pt'

if best_model_src.exists():
    # Hacer backup si ya existe
    if best_model_dst.exists():
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = MODELS_DIR / f'sivisecb_backup_{timestamp}.pt'
        shutil.copy(best_model_dst, backup_name)
        print(f"[OK] Backup anterior: {backup_name}")
    
    # Guardar el mejor modelo con nombre fijo
    shutil.copy(best_model_src, best_model_dst)
    model_size = best_model_src.stat().st_size / 1e6
    print(f"[OK] Mejor modelo guardado: {best_model_dst}")
    print(f"     Tamano: {model_size:.1f} MB")
    print(f"     (Siempre se sobrescribe con el mejor modelo)")
else:
    print(f"[ERROR] No se encontro el modelo entrenado en: {best_model_src}")

# ====================================================================
# PRUEBA RAPIDA
# ====================================================================
print("\n" + "=" * 70)
print("PRUEBA RAPIDA DEL MODELO")
print("=" * 70)

if best_model_src.exists():
    trained_model = YOLO(str(best_model_src))
    
    val_images_dir = OUTPUT_DIR / 'valid' / 'images'
    val_images = list(val_images_dir.glob('*.jpg')) + list(val_images_dir.glob('*.png'))
    
    if val_images:
        test_img = val_images[0]
        print(f"\nProbando con: {test_img.name}")
        
        results = trained_model(str(test_img))
        
        output_img = BASE_DIR / 'test_prediction.jpg'
        results[0].save(str(output_img))
        
        print(f"[OK] Resultado guardado en: {output_img}")
        
        print("\nDetecciones encontradas:")
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = results[0].names[cls_id]
                print(f"   - {class_name}: {conf:.1%}")
        else:
            print("   - No se detectaron objetos")

# ====================================================================
# RESUMEN FINAL
# ====================================================================
print("\n" + "=" * 70)
print("PROCESO COMPLETADO")
print("=" * 70)
print(f"""
ESTADISTICAS:
   - Total de imagenes: {total_images:,}
   - Total de clases: {len(all_classes)}
   - Epocas completadas: {CONFIG['epochs']}
   - Modelo final: {best_model_dst.name}
   - Tamano del modelo: ~6 MB

CLASES DETECTADAS:
   OBJETOS: {[c for c in all_classes if c in ['bottle', 'hand', 'person', 'can', 'snack']]}
   CONDUCTAS: {[c for c in all_classes if c in ['normal', 'shoplifting', 'suspect']]}
   ARMAS: {[c for c in all_classes if any(w in c.lower() for w in ['weapon', 'gun', 'knife', 'pistol', 'rifle'])]}

ARCHIVOS GENERADOS:
   - Modelo entrenado: {best_model_dst}
   - Dataset combinado: {OUTPUT_DIR}
   - Resultados: {MODELS_DIR / 'runs' / 'detect' / 'security_nano'}
""")

print("\n" + "=" * 70)
print("EJEMPLO DE USO:")
print("=" * 70)
print("""
from ultralytics import YOLO

# Cargar modelo
model = YOLO('models/security_nano_best_XXXXXXXX_XXXXXX.pt')

# Webcam en tiempo real
model.predict(source=0, show=True, conf=0.5)

# Video
model.predict(source='video.mp4', save=True, conf=0.5)

# Imagen
results = model('imagen.jpg')
results[0].show()

# Sistema de alertas
for result in results:
    for box in result.boxes:
        cls_name = result.names[int(box.cls[0])].lower()
        confidence = float(box.conf[0])
        
        if any(w in cls_name for w in ['weapon', 'gun', 'knife']):
            print(f"ALERTA: Arma detectada - {cls_name} ({confidence:.0%})")
        elif any(w in cls_name for w in ['shoplifting', 'suspect']):
            print(f"SOSPECHA: {cls_name} ({confidence:.0%})")
        elif cls_name in ['can', 'snack', 'bottle']:
            print(f"Producto: {cls_name}")
""")

print("=" * 70)
print("FIN DEL SCRIPT")
print("=" * 70)
