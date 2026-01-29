# PA-100k Human Attribute Recognition - Training Pipeline

Fine-tuning de reconocimiento de atributos humanos usando el dataset PA-100k (100,000 imágenes) con PyTorch + CUDA.

> 📖 **[Ver flujo completo paso a paso](WORKFLOW.md)** - Documentación detallada desde cero hasta prueba con video

## 🎯 Características

- **27 Atributos** detectados:
  - **Género**: Female, Male
  - **Edad**: AgeOver60, Age18-60, AgeLess18
  - **Bolsos**: HandBag, ShoulderBag, Backpack
  - **Ropa**: Hat, Glasses, ShortSleeve, LongSleeve, etc.

- **GPU Optimization**: Soporte para NVIDIA RTX 50xx (Blackwell) con PyTorch 2.11+
- **Export ONNX**: Listo para DeepStream/TensorRT

## 📦 Requisitos

### Hardware
- GPU NVIDIA con compute capability >= 5.0
- 16 GB VRAM recomendado (puede funcionar con 8 GB reduciendo batch size)
- 100 GB de espacio en disco

### Software
- Python 3.11+
- CUDA 12.8+
- PyTorch 2.11+ (nightly con soporte sm_120)

## 🚀 Setup Rápido

### Método 1: Script Automático (Recomendado)

```bash
# Clonar repositorio
git clone https://github.com/Orjuelosky8/PA100k-attribute-training.git
cd PA100k-attribute-training

# Ejecutar pipeline completo
python setup_and_train.py --all
```

El script automáticamente:
1. Solicita descarga del dataset PA-100k
2. Agrega atributo "Male" (27 atributos total)
3. Convierte a formato PyTorch
4. Instala PyTorch + dependencias
5. Entrena el modelo (60 epochs)
6. Exporta a ONNX

### Método 2: Manual

#### 1. Descargar Dataset

Descarga manualmente de [PA-100k](https://github.com/xh-liu/HydraPlus-Net):
- `annotation.zip` (~330 KB)
- `data.zip` (~430 MB)

Coloca en la raíz del proyecto y descomprime:
```bash
unzip annotation.zip
unzip data.zip
```

#### 2. Preparar Dataset

```bash
# Agregar atributo Male
python add_male_attribute.py

# Convertir a formato PyTorch
python convert_to_paddle.py --use-27attr
```

#### 3. Instalar Dependencias

```bash
# PyTorch nightly con CUDA 12.8 (para RTX 50xx)
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# Otras dependencias
pip install numpy pillow scipy opencv-python tqdm
```

#### 4. Entrenar

```bash
python train_pytorch.py
```

**Tiempo estimado**: 3-12 horas dependiendo de la GPU

## 📊 Configuración de Entrenamiento

| Parámetro | Valor | Modificable en |
|-----------|-------|----------------|
| Epochs | 60 | `train_pytorch.py:29` |
| Batch Size | 64 | `train_pytorch.py:30` |
| Learning Rate | 0.001 | `train_pytorch.py:32` |
| Input Size | 256x192 | `train_pytorch.py:33` |
| Backbone | ResNet-50 | `train_pytorch.py:92` |
| Loss | BCEWithLogitsLoss | `train_pytorch.py:299` |

## 📁 Estructura del Proyecto

```
PA-110k/
├── setup_and_train.py           # Orquestador automático
├── train_pytorch.py             # Script de entrenamiento
├── add_male_attribute.py        # Agregar atributo Male
├── convert_to_paddle.py         # Convertir dataset
├── download_pphuman.py          # Descargar modelo PP-Human (baseline)
├── export_onnx.py               # Convertir modelos Paddle a ONNX
├── download_test_video.py       # Descargar video de prueba
├── test_attributes_cpu.py       # Test con CPU
├── .gitignore                   # Excluye modelos, videos, ZIPs
├── README_REPO.md               # Este archivo
│
├── annotation_27attr.mat        # Dataset con 27 atributos
├── paddle_format/               # Dataset convertido
│   ├── train.txt                # 80,000 samples
│   ├── val.txt                  # 10,000 samples
│   ├── test.txt                 # 10,000 samples
│   └── attributes.txt           # 27 nombres de atributos
│
├── test_videos/                 # Videos de prueba
│   └── attributes_sim.mp4       # Video descargado de YouTube
│
└── output_pytorch/              # Outputs (generado al entrenar)
    ├── checkpoints/
    │   ├── best_model.pth       # Mejor modelo
    │   ├── final_model.pth      # Modelo final
    │   └── checkpoint_epoch_*.pth
    └── human_attr_pytorch.onnx  # ✅ Modelo ONNX
```

## 🎮 Uso

### Descargar modelo PP-Human (opcional, para comparación)

```bash
python download_pphuman.py
```

Descarga el modelo PP-Human pre-entrenado (PPLCNet x1.0, 26 atributos) para comparar con el modelo fine-tuned de PA-100k (27 atributos).

### Descargar video de prueba

```bash
python download_test_video.py
```

### Entrenar desde cero

```bash
python train_pytorch.py
```

### Reanudar desde checkpoint

```bash
python train_pytorch.py --resume output_pytorch/checkpoints/checkpoint_epoch_30.pth
```

### Solo evaluación

```bash
python train_pytorch.py --eval-only --resume output_pytorch/checkpoints/best_model.pth
```

### Exportar a ONNX

```bash
python train_pytorch.py --export-only
```

### Probar modelo entrenado

```bash
# Descargar video de prueba (si no lo has hecho)
python download_test_video.py

# Probar con CPU
python test_attributes_cpu.py
```

## 🐳 Docker (Alternativo)

```bash
# Construir imagen
docker-compose build

# Entrenar
docker-compose up pa100k-training

# Monitorear
docker-compose logs -f pa100k-training
```

## 🔧 Troubleshooting

### GPU no detectada

```bash
# Verificar CUDA
nvidia-smi

# Verificar PyTorch
python -c "import torch; print(torch.cuda.is_available())"
```

Si usas RTX 50xx (Blackwell), necesitas PyTorch 2.11+:
```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

### Out of Memory (OOM)

Reduce `BATCH_SIZE` en `train_pytorch.py`:
```python
BATCH_SIZE = 32  # o 16 si persiste
```

### Convergencia lenta

El modelo debería alcanzar:
- **Epoch 10**: Loss ~0.25, Acc ~85%
- **Epoch 30**: Loss ~0.15, Acc ~88%
- **Epoch 60**: Loss ~0.10, Acc ~90%

Si no converge, verifica:
- Dataset correctamente preparado
- PyTorch instalado con CUDA
- GPU funcionando (99% uso)

## 📤 Integración con DeepStream

Después del entrenamiento:

1. **Copiar modelo ONNX**:
   ```bash
   cp output_pytorch/human_attr_pytorch.onnx \
      ../Computer_vision/inference/weights/human_attr/
   ```

2. **Actualizar config** en `app_config.py`:
   ```python
   HUMAN_ATTR_ONNX = "weights/human_attr/human_attr_pytorch.onnx"
   ```

3. **Probar**:
   ```bash
   cd ../Computer_vision/inference
   python test_attributes_cpu.py
   ```

## 📊 Dataset PA-100k

- **Total**: 100,000 imágenes de peatones
- **Train**: 80,000 (80%)
- **Val**: 10,000 (10%)
- **Test**: 10,000 (10%)
- **Atributos**: 27 (26 originales + Male)

### Estadísticas

| Atributo | Positivos | Porcentaje |
|----------|-----------|------------|
| Age18-60 | 74,721 | 93.4% |
| Trousers | 56,719 | 70.9% |
| ShortSleeve | 46,878 | 58.6% |
| Male | 43,508 | 54.4% |
| Female | 36,492 | 45.6% |
| boots | 495 | 0.6% |

## 🤝 Contribuciones

Pull requests son bienvenidos. Para cambios mayores, por favor abre un issue primero.

## 📄 Licencia

Este proyecto es de código abierto bajo licencia MIT.

## 🙏 Créditos

- **Dataset**: [PA-100k](https://github.com/xh-liu/HydraPlus-Net) por Xihui Liu et al.
- **Arquitectura**: ResNet-50 pre-entrenado en ImageNet
- **Framework**: PyTorch 2.11+ con CUDA 12.8

## 📞 Soporte

Para problemas o preguntas:
- Abrir un [Issue](https://github.com/Orjuelosky8/PA100k-attribute-training/issues)
- Email: daniel.orju8@gmail.com

---

**Desarrollado por**: Orjuelosky8
**Última actualización**: Enero 2026
