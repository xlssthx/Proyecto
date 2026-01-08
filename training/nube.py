# ====================================================================
# DETECTOR DE SEGURIDAD COMPLETO PARA TIENDAS
# Detecta: Personas, Manos, Botellas, Robo, Armas
# ====================================================================

# ====================================================================
# PASO 1: INSTALACIÓN
# ====================================================================
print("=" * 70)
print("INSTALANDO DEPENDENCIAS")
print("=" * 70)

!pip install -q ultralytics roboflow PyYAML

from google.colab import drive
import os
import shutil
import yaml
from roboflow import Roboflow
from ultralytics import YOLO
import torch
from datetime import datetime

print("✓ Dependencias instaladas\n")

# ====================================================================
# PASO 2: MONTAR GOOGLE DRIVE
# ====================================================================
print("=" * 70)
print(" MONTANDO GOOGLE DRIVE")
print("=" * 70)

drive.mount('/content/drive')
print("✓ Google Drive montado\n")

# ====================================================================
# PASO 3: CONFIGURACIÓN
# ====================================================================
print("=" * 70)
print(" CONFIGURACIÓN")
print("=" * 70)

# CAMBIA ESTO CON TU API KEY
API_KEY = "TU_API_KEY_AQUI"

CONFIG = {
    'epochs': 100,
    'img_size': 640,
    'batch_size': 16,
    'model_size': 'yolov8m.pt',  # Usamos medium para mejor precisión
    'patience': 15
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✓ Dispositivo: {device.upper()}")
print(f"✓ Configuración lista\n")

# ====================================================================
# PASO 4: DESCARGAR TODOS LOS DATASETS
# ====================================================================
def download_dataset_safe(rf, workspace, project, version, name):
    """Descarga dataset si no existe"""
    dataset_folder = f"{project}-{version}"
    
    if os.path.exists(dataset_folder) and os.path.exists(f"{dataset_folder}/data.yaml"):
        print(f"✓ '{name}' ya existe\n")
        return dataset_folder
    
    print(f"⬇ Descargando '{name}'...")
    try:
        proj = rf.workspace(workspace).project(project)
        dataset = proj.version(version).download("yolov8")
        print(f"✓ Descargado: {dataset.location}\n")
        return dataset.location
    except Exception as e:
        print(f"x Error: {e}\n")
        return None

print("=" * 70)
print(" DESCARGANDO 6 DATASETS")
print("=" * 70)

rf = Roboflow(api_key=API_KEY)

# Dataset 1: Botellas
print("[1/6]  BOTELLAS")
print("-" * 70)
dataset_bottles = download_dataset_safe(
    rf, "test-n0rkj", "bottles-ca5xq", 1, "Botellas"
)

# Dataset 2: Manos
print("[2/6]  MANOS")
print("-" * 70)
dataset_hands = download_dataset_safe(
    rf, "faces-e24ww", "hands-clfdp", 1, "Manos"
)

# Dataset 3: Personas
print("[3/6]  PERSONAS (19k imágenes)")
print("-" * 70)
dataset_people = download_dataset_safe(
    rf, "leo-ueno", "people-detection-o4rdr", 11, "Personas"
)

# Dataset 4: Robo (shoplifting)
print("[4/6]  ROBO/SHOPLIFTING (7.7k imágenes)")
print("-" * 70)
dataset_shoplifting = download_dataset_safe(
    rf, "allen-1e0do", "shoplifting-detection-erald", 1, "Robo"
)

# Dataset 5: Armas completo
print("[5/6]  ARMAS (9.7k imágenes)")
print("-" * 70)
dataset_weapons = download_dataset_safe(
    rf, "yolov7test-u13vc", "weapon-detection-m7qso", 1, "Armas"
)

# Dataset 6: Gun & Knife (adicional)
print("[6/6]  GUN & KNIFE (8.4k imágenes)")
print("-" * 70)
dataset_gun_knife = download_dataset_safe(
    rf, "mahad-ahmed", "gun-and-knife-detection", 1, "Gun-Knife"
)

# Verificar descargas
datasets = [
    dataset_bottles, dataset_hands, dataset_people,
    dataset_shoplifting, dataset_weapons, dataset_gun_knife
]

if not all(datasets):
    print("x Error en descarga")
    raise Exception("Revisa tu API key")

print("✓ Todos los datasets descargados\n")

# ====================================================================
# PASO 5: COMBINAR DATASETS
# ====================================================================
def process_split(dataset_path, output_path, split, prefix, offset):
    """Procesa un split y ajusta las clases"""
    img_src = f'{dataset_path}/{split}/images'
    lbl_src = f'{dataset_path}/{split}/labels'
    
    if not os.path.exists(img_src):
        return 0
    
    img_dst = f'{output_path}/{split}/images'
    lbl_dst = f'{output_path}/{split}/labels'
    
    count = 0
    for img_file in os.listdir(img_src):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        base_name = os.path.splitext(img_file)[0]
        ext = os.path.splitext(img_file)[1]
        new_name = f'{prefix}_{count:06d}'
        
        # Copiar imagen
        shutil.copy(f'{img_src}/{img_file}', f'{img_dst}/{new_name}{ext}')
        
        # Copiar y ajustar labels
        label_file = f'{base_name}.txt'
        if os.path.exists(f'{lbl_src}/{label_file}'):
            with open(f'{lbl_src}/{label_file}', 'r') as f:
                lines = f.readlines()
            
            adjusted = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    old_class = int(parts[0])
                    new_class = old_class + offset
                    parts[0] = str(new_class)
                    adjusted.append(' '.join(parts) + '\n')
            
            with open(f'{lbl_dst}/{new_name}.txt', 'w') as f:
                f.writelines(adjusted)
        
        count += 1
    
    return count

print("=" * 70)
print(" COMBINANDO LOS 6 DATASETS")
print("=" * 70)

output_path = '/content/combined_security_detection'

# Crear directorios
for split in ['train', 'valid', 'test']:
    os.makedirs(f'{output_path}/{split}/images', exist_ok=True)
    os.makedirs(f'{output_path}/{split}/labels', exist_ok=True)

# Leer todas las clases
all_classes = []
class_offset = 0

dataset_configs = [
    (dataset_bottles, "Botellas"),
    (dataset_hands, "Manos"),
    (dataset_people, "Personas"),
    (dataset_shoplifting, "Robo"),
    (dataset_weapons, "Armas"),
    (dataset_gun_knife, "Gun-Knife")
]

print("\n CLASES POR DATASET:")
for ds_path, ds_name in dataset_configs:
    with open(f'{ds_path}/data.yaml', 'r') as f:
        data = yaml.safe_load(f)
    classes = data['names']
    all_classes.extend(classes)
    print(f"  • {ds_name}: {classes}")

print(f"\n TOTAL DE CLASES: {len(all_classes)}")
print(f"   {all_classes}\n")

# Combinar todos los datasets
total_images = 0
class_offset = 0

for split in ['train', 'valid', 'test']:
    print(f"📁 [{split.upper()}]")
    print("-" * 70)
    
    current_offset = 0
    split_total = 0
    
    for i, (ds_path, ds_name) in enumerate(dataset_configs, 1):
        with open(f'{ds_path}/data.yaml', 'r') as f:
            data = yaml.safe_load(f)
        num_classes = len(data['names'])
        
        count = process_split(
            ds_path, output_path, split, 
            f'd{i}', current_offset
        )
        
        print(f"  [{i}/6] {ds_name:15} → {count:6,} imágenes")
        
        current_offset += num_classes
        split_total += count
    
    total_images += split_total
    print(f"  ✓ Total split: {split_total:,} imágenes\n")

# Crear data.yaml final
new_data = {
    'path': output_path,
    'train': 'train/images',
    'val': 'valid/images',
    'test': 'test/images',
    'nc': len(all_classes),
    'names': all_classes
}

with open(f'{output_path}/data.yaml', 'w') as f:
    yaml.dump(new_data, f, default_flow_style=False)

print("=" * 70)
print("✓ COMBINACIÓN COMPLETADA")
print("=" * 70)
print(f" Total imágenes: {total_images:,}")
print(f" Total clases: {len(all_classes)}")
print(f" Ubicación: {output_path}\n")

# ====================================================================
# PASO 6: ENTRENAR MODELO
# ====================================================================
print("=" * 70)
print(" INICIANDO ENTRENAMIENTO")
print("=" * 70)
print(f" Tiempo estimado: 2-3 horas con GPU T4\n")

model = YOLO(CONFIG['model_size'])

try:
    results = model.train(
        data=f'{output_path}/data.yaml',
        epochs=CONFIG['epochs'],
        imgsz=CONFIG['img_size'],
        batch=CONFIG['batch_size'],
        device=device,
        workers=4,
        patience=CONFIG['patience'],
        save=True,
        save_period=10,
        project='/content/runs/detect',
        name='security_detector',
        exist_ok=True,
        plots=True,
        verbose=True,
        amp=True,
    )
    
    print("\n" + "=" * 70)
    print("✓ ENTRENAMIENTO COMPLETADO")
    print("=" * 70)
    
except Exception as e:
    print(f"\nx Error: {e}")
    raise

# ====================================================================
# PASO 7: VALIDAR Y GUARDAR
# ====================================================================
print("\n Validando modelo...")
metrics = model.val()

print("=" * 70)
print(" MÉTRICAS FINALES")
print("=" * 70)
print(f"mAP50:      {metrics.box.map50:.3f}")
print(f"mAP50-95:   {metrics.box.map:.3f}")
print(f"Precisión:  {metrics.box.mp:.3f}")
print(f"Recall:     {metrics.box.mr:.3f}")
print("=" * 70)

# Guardar en Drive
drive_folder = '/content/drive/MyDrive/YOLOv8_Security'
os.makedirs(drive_folder, exist_ok=True)

best_model = '/content/runs/detect/security_detector/weights/best.pt'
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_name = f'security_full_{timestamp}.pt'

shutil.copy(best_model, f'{drive_folder}/{model_name}')

print(f"\n Modelo guardado: {drive_folder}/{model_name}\n")

# ====================================================================
# PASO 8: PROBAR MODELO
# ====================================================================
print("=" * 70)
print("🧪 PROBANDO MODELO")
print("=" * 70)

trained_model = YOLO(best_model)

val_images_dir = f'{output_path}/valid/images'
val_images = os.listdir(val_images_dir)

if val_images:
    test_img = f'{val_images_dir}/{val_images[0]}'
    results = trained_model(test_img)
    
    output_img = '/content/test_prediction.jpg'
    results[0].save(output_img)
    
    print("\n Detecciones:")
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = results[0].names[cls_id]
        print(f"   • {class_name}: {conf:.1%}")
    
    from IPython.display import Image, display
    display(Image(output_img))

# ====================================================================
# RESUMEN FINAL
# ====================================================================
print("\n" + "=" * 70)
print("🎉 ¡DETECTOR DE SEGURIDAD COMPLETO!")
print("=" * 70)
print(f"""
ESTADÍSTICAS:
   • Imágenes totales: {total_images:,}
   • Clases detectadas: {len(all_classes)}
   • Épocas: {CONFIG['epochs']}

CLASES QUE DETECTA:
   0-2:   Objetos (botellas, manos, personas)
   3-4:   Conducta (normal, shoplifting)
   5-30+: Armas (pistolas, cuchillos, rifles, etc.)

USO DEL MODELO:
""")

print("""
from ultralytics import YOLO

# Cargar modelo
model = YOLO('security_full.pt')

# Detectar en video en tiempo real
model.predict(
    source=0,  # Webcam
    show=True,
    conf=0.5,
    save=True
)

# Detectar en imagen
results = model('tienda.jpg')
results[0].show()

# Obtener alertas
for box in results[0].boxes:
    cls = results[0].names[int(box.cls[0])]
    conf = float(box.conf[0])
    
    if 'weapon' in cls or 'gun' in cls or 'knife' in cls:
        print(f" ALERTA: {cls} detectado ({conf:.0%})")
    elif 'shoplifting' in cls:
        print(f"¡ SOSPECHOSO: Posible robo ({conf:.0%})")
""")

print("=" * 70)
print("✓ ¡Listo para detectar amenazas en tu tienda!")
print("=" * 70)
