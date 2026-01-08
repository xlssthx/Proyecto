from roboflow import Roboflow
import os

# API Key de Roboflow (reemplazar con tu propia API key si es necesario)
API_KEY = "QOL8UwOKMB97adVbFAU3"

# Inicializar Roboflow
rf = Roboflow(api_key=API_KEY)

# Descargar dataset de Roboflow con verificación previa
def download_dataset_safe(workspace, project, version, name):
    """
    Descarga un dataset solo si no existe ya
    
    Args:
        workspace: Nombre del workspace en Roboflow
        project: Nombre del proyecto
        version: Versión del dataset
        name: Nombre descriptivo para mostrar en consola
    
    Returns:
        Ruta del dataset (descargado o existente)
    """
    # Roboflow descarga con este formato de nombre
    dataset_folder = f"{project}-{version}"
    
    # Verificar si ya existe el dataset Y tiene el archivo data.yaml
    if os.path.exists(dataset_folder) and os.path.exists(f"{dataset_folder}/data.yaml"):
        print(f"✓ Dataset '{name}' ya existe en: {dataset_folder}")
        print(f"  → Saltando descarga, pasando al siguiente...\n")
        return dataset_folder
    
    # Si no existe, descargarlo
    print(f" ⬇ Descargando dataset '{name}'...")
    try:
        proj = rf.workspace(workspace).project(project)
        dataset = proj.version(version).download("yolov8")
        print(f"✓ Dataset '{name}' descargado en: {dataset.location}\n")
        return dataset.location
    except Exception as e:
        print(f"x Error al descargar '{name}': {e}\n")
        return None

# Descargar los datasets necesarios (posible expansión futura)
print("=" * 70)
print("DESCARGANDO DATASETS DESDE ROBOFLOW")
print("=" * 70)
print()

# Dataset de Botellas
print("[1/3] Dataset de Botellas (800 imágenes)")
print("-" * 70)
dataset_bottles = download_dataset_safe(
    workspace="test-n0rkj",
    project="bottles-ca5xq",
    version=1,
    name="Botellas"
)

# Dataset de Manos
print("[2/3] Dataset de Manos (920 imágenes)")
print("-" * 70)
dataset_hands = download_dataset_safe(
    workspace="faces-e24ww",
    project="hands-clfdp",
    version=1,
    name="Manos"
)

# Dataset de Personas
print("[3/3] Dataset de Personas (19,233 imágenes - 89.4% mAP)")
print("-" * 70)
dataset_people = download_dataset_safe(
    workspace="leo-ueno",
    project="people-detection-o4rdr",
    version=11,
    name="Personas"
)

# Aquí se verifica si se descargaron correctamente
print("=" * 70)
print("RESUMEN DE DESCARGA")
print("=" * 70)

all_ok = True

if dataset_bottles:
    print(f"✓ Botellas: {dataset_bottles}")
else:
    print(f"x Botellas: ERROR")
    all_ok = False

if dataset_hands:
    print(f"✓ Manos: {dataset_hands}")
else:
    print(f"x Manos: ERROR")
    all_ok = False

if dataset_people:
    print(f"✓ Personas: {dataset_people}")
else:
    print(f"x Personas: ERROR")
    all_ok = False

print("=" * 70)

if all_ok:
    print("\n¡Todos los datasets descargados exitosamente!")
    print("\nESTADÍSTICAS:")
    print(f"   • Total aproximado: ~20,953 imágenes")
    print(f"   • Botellas: 800 imágenes")
    print(f"   • Manos: 920 imágenes")
    print(f"   • Personas: 19,233 imágenes")
    print("\nSIGUIENTE PASO:")
    print("   Ejecuta el script 'combinar.py' para combinarlos")
else:
    print("\¡Algunos datasets no se pudieron descargar!")
    print("   Verifica tu conexión a internet y tu API key.")
