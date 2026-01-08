import os
import shutil
import yaml

# Caonfiguración de datasets
DATASET_BOTTLES = "bottles-ca5xq-1"
DATASET_HANDS = "hands-clfdp-1"
DATASET_PEOPLE = "people-detection-o4rdr-11"
OUTPUT_FOLDER = "combined_bottles_hands_people"

# Función para procesar cada split
def process_split(dataset_path, output_path, split, prefix, offset):
    """
    Procesa un split (train/valid/test) de un dataset
    
    Args:
        dataset_path: Ruta al dataset original
        output_path: Ruta al dataset combinado
        split: 'train', 'valid' o 'test'
        prefix: Prefijo para nombrar archivos (ej: 'd1', 'd2', 'd3')
        offset: Offset para ajustar los IDs de clase
    
    Returns:
        Número de imágenes procesadas
    """
    img_src = f'{dataset_path}/{split}/images'
    lbl_src = f'{dataset_path}/{split}/labels'
    
    # Si el split no existe, saltar
    if not os.path.exists(img_src):
        print(f"    ¡  {prefix}: Split '{split}' no existe, saltando...")
        return 0
    
    img_dst = f'{output_path}/{split}/images'
    lbl_dst = f'{output_path}/{split}/labels'
    
    count = 0
    for img_file in os.listdir(img_src):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        base_name = os.path.splitext(img_file)[0]
        ext = os.path.splitext(img_file)[1]
        
        # Crear nuevo nombre único
        new_name = f'{prefix}_{count:06d}'
        
        # Copiar imagen
        shutil.copy(
            f'{img_src}/{img_file}',
            f'{img_dst}/{new_name}{ext}'
        )
        
        # Copiar y ajustar labels
        label_file = f'{base_name}.txt'
        if os.path.exists(f'{lbl_src}/{label_file}'):
            with open(f'{lbl_src}/{label_file}', 'r') as f:
                lines = f.readlines()
            
            # Ajustar IDs de clase
            adjusted = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:  # class x y w h
                    old_class = int(parts[0])
                    new_class = old_class + offset
                    parts[0] = str(new_class)
                    adjusted.append(' '.join(parts) + '\n')
            
            # Guardar label ajustado
            with open(f'{lbl_dst}/{new_name}.txt', 'w') as f:
                f.writelines(adjusted)
        
        count += 1
    
    print(f"    {prefix}: {count} imágenes procesadas")
    return count


def combine_datasets(dataset1_path, dataset2_path, dataset3_path, output_path):
    """
    Combina tres datasets de YOLO en uno solo
    
    Args:
        dataset1_path: Ruta al dataset de botellas
        dataset2_path: Ruta al dataset de manos
        dataset3_path: Ruta al dataset de personas
        output_path: Ruta donde guardar el dataset combinado
    """
    print("\n" + "=" * 70)
    print("COMBINANDO DATASETS")
    print("=" * 70)
    
    # Verificar que existan los datasets
    for name, path in [("Botellas", dataset1_path), 
                       ("Manos", dataset2_path), 
                       ("Personas", dataset3_path)]:
        if not os.path.exists(path):
            print(f"x Error: Dataset '{name}' no encontrado en: {path}")
            return None
    
    print("✓ Todos los datasets encontrados\n")
    
    # Crear estructura de directorios
    print("Creando estructura de directorios...")
    for split in ['train', 'valid', 'test']:
        os.makedirs(f'{output_path}/{split}/images', exist_ok=True)
        os.makedirs(f'{output_path}/{split}/labels', exist_ok=True)
    print("✓ Directorios creados\n")
    
    # Leer archivos data.yaml
    print("Leyendo configuraciones de datasets...")
    with open(f'{dataset1_path}/data.yaml', 'r') as f:
        data1 = yaml.safe_load(f)
    with open(f'{dataset2_path}/data.yaml', 'r') as f:
        data2 = yaml.safe_load(f)
    with open(f'{dataset3_path}/data.yaml', 'r') as f:
        data3 = yaml.safe_load(f)
    
    # Obtener nombres de clases
    classes1 = data1['names']  # ['bottle']
    classes2 = data2['names']  # ['hand']
    classes3 = data3['names']  # ['people'] o ['person']
    
    # Combinar clases
    combined_classes = classes1 + classes2 + classes3
    
    print(f"  Dataset 1: {classes1}")
    print(f"  Dataset 2: {classes2}")
    print(f"  Dataset 3: {classes3}")
    print(f"  Total de clases: {len(combined_classes)}")
    print(f"  Clases combinadas: {combined_classes}\n")
    
    # Procesar cada split
    print("Procesando y combinando imágenes...\n")
    total_images = 0
    
    for split in ['train', 'valid', 'test']:
        print(f"  [{split.upper()}]")
        print("  " + "-" * 66)
        
        # Dataset 1 (botellas - clase ID: 0)
        count1 = process_split(dataset1_path, output_path, split, 'd1', offset=0)
        
        # Dataset 2 (manos - clase ID: len(classes1))
        count2 = process_split(dataset2_path, output_path, split, 'd2', offset=len(classes1))
        
        # Dataset 3 (personas - clase ID: len(classes1) + len(classes2))
        count3 = process_split(dataset3_path, output_path, split, 'd3', 
                              offset=len(classes1) + len(classes2))
        
        split_total = count1 + count2 + count3
        total_images += split_total
        print(f"    ✓ Total en {split}: {split_total} imágenes\n")
    
    # Crear archivo data.yaml combinado
    print("Creando archivo data.yaml...")
    new_data = {
        'path': os.path.abspath(output_path),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': len(combined_classes),
        'names': combined_classes
    }
    
    with open(f'{output_path}/data.yaml', 'w') as f:
        yaml.dump(new_data, f, default_flow_style=False)
    
    print("✓ Archivo data.yaml creado\n")
    
    # Resumen final
    print("=" * 70)
    print("¡COMBINACIÓN COMPLETADA!")
    print("=" * 70)
    print(f"Total de imágenes combinadas: {total_images}")
    print(f"Total de clases: {len(combined_classes)}")
    print(f"Clases: {combined_classes}")
    print(f"Dataset guardado en: {os.path.abspath(output_path)}")
    print("=" * 70)
    
    return output_path


# Ejecutar combinación
if __name__ == "__main__":
    print("\nIniciando combinación de datasets...\n")
    
    result = combine_datasets(
        dataset1_path=DATASET_BOTTLES,
        dataset2_path=DATASET_HANDS,
        dataset3_path=DATASET_PEOPLE,
        output_path=OUTPUT_FOLDER
    )
    
    if result:
        print("\n¡Listo para entrenar!")
        print("\n SIGUIENTE PASO:")
        print(f"   Ejecuta: python train_model.py")
        print(f"\n   O manualmente:")
        print(f"   from ultralytics import YOLO")
        print(f"   model = YOLO('yolov8n.pt')")
        print(f"   model.train(data='{OUTPUT_FOLDER}/data.yaml', epochs=50)")
    else:
        print("\n x La combinación falló. Verifica los nombres de las carpetas.")
