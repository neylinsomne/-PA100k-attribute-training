"""
Script Orquestador Completo - PA-100k Attribute Training
Ejecuta todo el pipeline automáticamente en una nueva máquina

Uso:
    python setup_and_train.py --all                    # Todo el pipeline
    python setup_and_train.py --download-only          # Solo descargar dataset
    python setup_and_train.py --train-only             # Solo entrenar (si ya está configurado)
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

CURRENT_DIR = Path(__file__).parent.absolute()

def run_command(cmd, description, cwd=None):
    """Ejecutar comando y mostrar progreso"""
    print(f"\n{'='*70}")
    print(f"[{description}]")
    print(f"{'='*70}")
    print(f"Comando: {cmd}\n")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            cwd=cwd or CURRENT_DIR,
            text=True
        )
        print(f"✅ {description} - Completado")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Error: {e}")
        return False

def step_1_download_dataset():
    """Paso 1: Descargar dataset PA-100k"""
    print("\n" + "="*70)
    print("PASO 1: Descargar Dataset PA-100k")
    print("="*70)

    # Verificar si ya existe
    if (CURRENT_DIR / "annotation.mat").exists():
        print("Dataset ya descargado. Saltando...")
        return True

    print("""
IMPORTANTE: Debes descargar manualmente el dataset PA-100k:

1. Visita: https://github.com/xh-liu/HydraPlus-Net
2. Descarga:
   - annotation.zip
   - data.zip

3. Coloca los archivos en:
   {}

4. Ejecuta nuevamente este script.
""".format(CURRENT_DIR))

    # Esperar a que el usuario descargue
    input("Presiona ENTER cuando hayas descargado los archivos...")

    # Verificar
    if not (CURRENT_DIR / "annotation.zip").exists():
        print("❌ annotation.zip no encontrado")
        return False

    if not (CURRENT_DIR / "data.zip").exists():
        print("❌ data.zip no encontrado")
        return False

    # Descomprimir
    print("\nDescomprimiendo archivos...")

    if not run_command("unzip -q annotation.zip", "Descomprimir annotation.zip"):
        return False

    if not run_command("unzip -q data.zip", "Descomprimir data.zip"):
        return False

    print("✅ Dataset descargado y descomprimido")
    return True

def step_2_add_male_attribute():
    """Paso 2: Agregar atributo Male al dataset"""
    print("\n" + "="*70)
    print("PASO 2: Agregar Atributo Male")
    print("="*70)

    if (CURRENT_DIR / "annotation_27attr.mat").exists():
        print("Atributo Male ya agregado. Saltando...")
        return True

    return run_command(
        "python add_male_attribute.py",
        "Agregar atributo Male"
    )

def step_3_convert_dataset():
    """Paso 3: Convertir dataset a formato PaddleDetection"""
    print("\n" + "="*70)
    print("PASO 3: Convertir Dataset a Formato PyTorch")
    print("="*70)

    if (CURRENT_DIR / "paddle_format" / "train.txt").exists():
        print("Dataset ya convertido. Saltando...")
        return True

    return run_command(
        "python convert_to_paddle.py --use-27attr",
        "Convertir dataset"
    )

def step_4_install_pytorch():
    """Paso 4: Instalar PyTorch con CUDA"""
    print("\n" + "="*70)
    print("PASO 4: Instalar PyTorch con CUDA 12.8")
    print("="*70)

    # Verificar si PyTorch ya está instalado
    try:
        import torch
        if torch.cuda.is_available():
            print(f"PyTorch {torch.__version__} ya instalado con CUDA")
            print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
            return True
    except ImportError:
        pass

    print("Instalando PyTorch nightly con CUDA 12.8...")

    return run_command(
        "pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 --user",
        "Instalar PyTorch"
    )

def step_5_install_dependencies():
    """Paso 5: Instalar dependencias adicionales"""
    print("\n" + "="*70)
    print("PASO 5: Instalar Dependencias")
    print("="*70)

    deps = [
        "numpy",
        "pillow",
        "scipy",
        "opencv-python",
        "tqdm"
    ]

    for dep in deps:
        if not run_command(f"pip install {dep} --user -q", f"Instalar {dep}"):
            return False

    return True

def step_6_train_model():
    """Paso 6: Entrenar modelo"""
    print("\n" + "="*70)
    print("PASO 6: Entrenar Modelo (60 epochs, ~12 horas)")
    print("="*70)

    print("""
OPCIONES DE ENTRENAMIENTO:

1. Entrenar ahora (bloqueará la terminal ~12 horas)
2. Entrenar en segundo plano
3. Saltar entrenamiento (ejecutar manualmente después)

""")

    choice = input("Selecciona una opción (1/2/3): ").strip()

    if choice == "1":
        return run_command(
            "python train_pytorch.py",
            "Entrenar modelo (primer plano)"
        )
    elif choice == "2":
        print("\nEjecutando en segundo plano...")
        print("Para monitorear: tail -f training.log")

        # Windows PowerShell
        if sys.platform == "win32":
            cmd = 'Start-Process python -ArgumentList "train_pytorch.py" -WorkingDirectory "{}" -RedirectStandardOutput "training.log" -RedirectStandardError "training_error.log" -NoNewWindow'.format(CURRENT_DIR)
            run_command(cmd, "Entrenar modelo (segundo plano)")
        else:
            # Linux/Mac
            run_command(
                "nohup python train_pytorch.py > training.log 2>&1 &",
                "Entrenar modelo (segundo plano)"
            )

        print(f"\nEntrenamiento iniciado en segundo plano")
        print(f"Monitorear: tail -f {CURRENT_DIR}/training.log")
        return True
    else:
        print("Saltando entrenamiento. Ejecuta manualmente:")
        print(f"  cd {CURRENT_DIR}")
        print("  python train_pytorch.py")
        return True

def step_7_export_onnx():
    """Paso 7: Exportar modelo a ONNX"""
    print("\n" + "="*70)
    print("PASO 7: Exportar Modelo a ONNX")
    print("="*70)

    # Verificar si el entrenamiento terminó
    if not (CURRENT_DIR / "output_pytorch" / "checkpoints" / "best_model.pth").exists():
        print("❌ Modelo entrenado no encontrado")
        print("El entrenamiento debe completarse primero (60 epochs)")
        return False

    print("Modelo entrenado encontrado. Exportando a ONNX...")

    # El script train_pytorch.py ya exporta automáticamente
    # Pero podemos verificar
    onnx_path = CURRENT_DIR / "output_pytorch" / "human_attr_pytorch.onnx"

    if onnx_path.exists():
        print(f"✅ Modelo ONNX ya existe: {onnx_path}")
        print(f"Tamaño: {onnx_path.stat().st_size / 1024 / 1024:.2f} MB")
        return True

    return run_command(
        "python train_pytorch.py --export-only",
        "Exportar modelo a ONNX"
    )

def main():
    parser = argparse.ArgumentParser(description="Pipeline completo PA-100k")
    parser.add_argument("--all", action="store_true", help="Ejecutar todo el pipeline")
    parser.add_argument("--download-only", action="store_true", help="Solo descargar y preparar dataset")
    parser.add_argument("--train-only", action="store_true", help="Solo entrenar (dataset ya preparado)")
    parser.add_argument("--export-only", action="store_true", help="Solo exportar a ONNX")

    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════════╗
║       PA-100k Attribute Training - Setup Automático             ║
╚══════════════════════════════════════════════════════════════════╝
""")

    # Flujo completo
    if args.all or (not args.download_only and not args.train_only and not args.export_only):
        steps = [
            ("Descargar Dataset", step_1_download_dataset),
            ("Agregar Male Attribute", step_2_add_male_attribute),
            ("Convertir Dataset", step_3_convert_dataset),
            ("Instalar PyTorch", step_4_install_pytorch),
            ("Instalar Dependencias", step_5_install_dependencies),
            ("Entrenar Modelo", step_6_train_model),
            ("Exportar ONNX", step_7_export_onnx),
        ]

        for i, (name, step_func) in enumerate(steps, 1):
            print(f"\n[PASO {i}/{len(steps)}] {name}")
            if not step_func():
                print(f"\n❌ Pipeline detenido en: {name}")
                sys.exit(1)

    # Solo descarga
    elif args.download_only:
        step_1_download_dataset()
        step_2_add_male_attribute()
        step_3_convert_dataset()

    # Solo entrenamiento
    elif args.train_only:
        step_4_install_pytorch()
        step_5_install_dependencies()
        step_6_train_model()

    # Solo exportar
    elif args.export_only:
        step_7_export_onnx()

    print("""
╔══════════════════════════════════════════════════════════════════╗
║                   Pipeline Completado                            ║
╚══════════════════════════════════════════════════════════════════╝

Archivos generados:
  - output_pytorch/checkpoints/best_model.pth (Modelo PyTorch)
  - output_pytorch/human_attr_pytorch.onnx (Modelo ONNX para DeepStream)

Para usar en DeepStream:
  1. Copiar: output_pytorch/human_attr_pytorch.onnx
  2. A: Computer_vision/inference/weights/human_attr/
  3. Actualizar config en app_config.py
""")

if __name__ == "__main__":
    main()
