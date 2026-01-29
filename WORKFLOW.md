# 🔄 Flujo Completo de Entrenamiento PA-100k

Este documento describe el flujo completo paso a paso para entrenar un modelo PPHuman de atributos desde cero y probarlo con un video de YouTube.

---

## 📋 Índice

1. [Pre-requisitos](#pre-requisitos)
2. [Flujo Automático (Recomendado)](#flujo-automático-recomendado)
3. [Flujo Manual (Paso a Paso)](#flujo-manual-paso-a-paso)
4. [Prueba del Modelo](#prueba-del-modelo)
5. [Verificación](#verificación)
6. [Integración con DeepStream](#integración-con-deepstream)

---

## 🛠️ Pre-requisitos

### Hardware
- **GPU**: NVIDIA con compute capability >= 5.0
- **VRAM**: Mínimo 8 GB (recomendado 16 GB)
- **Disco**: 100 GB libres
- **RAM**: 16 GB

### Software
- **Python**: 3.11+
- **CUDA**: 12.8+
- **Git**: Para clonar el repositorio

---

## ⚡ Flujo Automático (Recomendado)

### Paso 1: Clonar Repositorio

```bash
git clone https://github.com/neylinsomne/-PA100k-attribute-training.git
cd -PA100k-attribute-training
```

### Paso 2: Ejecutar Pipeline Completo

```bash
python setup_and_train.py --all
```

Este script automáticamente:
1. ✅ Solicita descarga del dataset PA-100k (necesitas descargarlo manualmente)
2. ✅ Añade atributo "Male" (27 atributos en total)
3. ✅ Convierte a formato PyTorch
4. ✅ Instala PyTorch + CUDA + dependencias
5. ✅ Entrena el modelo (60 epochs)
6. ✅ Exporta a ONNX

**Tiempo estimado**: 3-12 horas dependiendo de tu GPU

### Paso 3: Descargar Video de Prueba

```bash
python download_test_video.py
```

Este script:
- Instala `yt-dlp` automáticamente si no está disponible
- Descarga video de YouTube: https://www.youtube.com/shorts/hxeudw4U8Cw
- Lo guarda en: `test_videos/attributes_sim.mp4`

### Paso 4: Probar Modelo

```bash
python test_attributes_cpu.py
```

---

## 🔧 Flujo Manual (Paso a Paso)

Si prefieres control total sobre cada paso:

### 1. Preparar Dataset

#### 1.1 Descargar PA-100k

Descarga manualmente de [PA-100k GitHub](https://github.com/xh-liu/HydraPlus-Net):

- **annotation.zip** (~330 KB) - Anotaciones del dataset
- **data.zip** (~430 MB) - 100,000 imágenes

Coloca ambos archivos en la raíz del proyecto:

```
PA-100k/
├── annotation.zip
├── data.zip
└── ...
```

#### 1.2 Descomprimir Dataset

```bash
# Windows PowerShell
Expand-Archive annotation.zip -DestinationPath .
Expand-Archive data.zip -DestinationPath .

# Linux/Mac
unzip annotation.zip
unzip data.zip
```

#### 1.3 Añadir Atributo "Male"

El dataset original tiene 26 atributos, pero falta "Male" como atributo independiente:

```bash
python add_male_attribute.py
```

**Salida esperada**:
- Crea: `annotation_27attr.mat` (con 27 atributos)
- Confirma: "Male attribute added successfully"

#### 1.4 Convertir a Formato PyTorch

```bash
python convert_to_paddle.py --use-27attr
```

**Salida esperada**:
```
paddle_format/
├── train.txt      # 80,000 muestras
├── val.txt        # 10,000 muestras
├── test.txt       # 10,000 muestras
└── attributes.txt # 27 nombres de atributos
```

### 2. Configurar Entorno Python

#### 2.1 Instalar PyTorch con CUDA

**Para RTX 50xx (Blackwell) - PyTorch Nightly**:
```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

**Para RTX 40xx/30xx - PyTorch Stable**:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

#### 2.2 Instalar Dependencias

```bash
pip install numpy pillow scipy opencv-python tqdm onnxruntime
```

#### 2.3 Verificar GPU

```bash
# PowerShell
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

**Salida esperada**:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 5090
```

### 3. Entrenar Modelo

#### 3.1 Entrenar desde Cero

```bash
python train_pytorch.py
```

**Configuración por defecto**:
- Epochs: 60
- Batch Size: 64
- Learning Rate: 0.001
- Input Size: 256x192
- Backbone: ResNet-50 (pre-entrenado en ImageNet)

**Tiempo estimado por epoch**:
- RTX 5090: ~3-5 minutos
- RTX 4090: ~5-8 minutos
- RTX 3090: ~8-12 minutos

**Total**: 3-12 horas

#### 3.2 Monitorear Entrenamiento

El script muestra progreso en tiempo real:

```
Epoch 1/60
Train Loss: 0.308, Train Acc: 84.23%
Val Loss: 0.245, Val Acc: 86.78%
✓ New best model saved!

Epoch 10/60
Train Loss: 0.189, Train Acc: 88.45%
Val Loss: 0.167, Val Acc: 89.23%
✓ New best model saved!
```

**Benchmarks esperados**:
| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 10    | ~0.25     | ~85%      | ~0.22    | ~87%    |
| 30    | ~0.15     | ~88%      | ~0.14    | ~90%    |
| 60    | ~0.10     | ~90%      | ~0.12    | ~91%    |

#### 3.3 Reanudar desde Checkpoint (Opcional)

Si el entrenamiento se interrumpe:

```bash
python train_pytorch.py --resume output_pytorch/checkpoints/checkpoint_epoch_30.pth
```

#### 3.4 Solo Evaluación (Opcional)

Para evaluar un modelo ya entrenado sin entrenar más:

```bash
python train_pytorch.py --eval-only --resume output_pytorch/checkpoints/best_model.pth
```

#### 3.5 Exportar a ONNX

El entrenamiento exporta automáticamente a ONNX al finalizar. Para exportar manualmente:

```bash
python train_pytorch.py --export-only
```

**Salida esperada**:
```
output_pytorch/
├── checkpoints/
│   ├── best_model.pth      # Mejor modelo (menor val loss)
│   ├── final_model.pth     # Modelo del último epoch
│   └── checkpoint_epoch_*.pth
└── human_attr_pytorch.onnx # ✅ Modelo ONNX
```

### 4. Descargar Video de Prueba

```bash
python download_test_video.py
```

**Detalles**:
- **URL**: https://www.youtube.com/shorts/hxeudw4U8Cw
- **Formato**: MP4 (mejor calidad disponible)
- **Destino**: `test_videos/attributes_sim.mp4`
- **Dependencia**: `yt-dlp` (se instala automáticamente)

**Interactividad**:
- Si el video ya existe, pregunta si quieres descargarlo de nuevo
- Si `yt-dlp` no está instalado, pregunta si quieres instalarlo

### 5. Probar Modelo con Video

```bash
python test_attributes_cpu.py
```

**Qué hace**:
1. Carga el modelo ONNX: `output_pytorch/human_attr_pytorch.onnx`
2. Abre el video: `test_videos/attributes_sim.mp4`
3. Detecta personas usando HOG (detector simple)
4. Para cada persona detectada, predice 27 atributos
5. Muestra resultados en consola

**Salida esperada**:

```
======================================================================
Test de Atributos Humanos - CPU Only
======================================================================

Modelo: output_pytorch\human_attr_pytorch.onnx
Video: test_videos\attributes_sim.mp4

Cargando modelo ONNX (CPU)...
Modelo cargado: ['CPUExecutionProvider']

Abriendo video...
FPS: 30, Total frames: 150
Duración: 5.0s

Procesando frames...

Frame 10: 2 persona(s) detectada(s)
  Persona 1:
    Género: Hombre (F:0.12 M:0.88)
    Edad: 18-60 años (0.94)
    Otros: LongSleeve(0.87), Trousers(0.92)

  Persona 2:
    Género: Mujer (F:0.89 M:0.11)
    Edad: 18-60 años (0.91)
    Bolsos: ShoulderBag(0.78)
    Otros: LongSleeve(0.82), Skirt&Dress(0.85)

...

======================================================================
Procesamiento completo
Total detecciones: 15
======================================================================

Resumen:
  Mujeres detectadas: 8
  Hombres detectados: 7
```

---

## ✅ Verificación

### Verificar Estructura de Directorios

```
PA-100k/
├── annotation_27attr.mat          ✅ Dataset con 27 atributos
├── paddle_format/                 ✅ Dataset convertido
│   ├── train.txt
│   ├── val.txt
│   ├── test.txt
│   └── attributes.txt
├── test_videos/                   ✅ Videos de prueba
│   └── attributes_sim.mp4
├── output_pytorch/                ✅ Modelos entrenados
│   ├── checkpoints/
│   │   ├── best_model.pth
│   │   └── final_model.pth
│   └── human_attr_pytorch.onnx   ✅ Modelo ONNX exportado
└── scripts/
    ├── download_test_video.py    ✅ Descargar video
    ├── test_attributes_cpu.py    ✅ Probar modelo
    ├── train_pytorch.py          ✅ Entrena modelo
    └── setup_and_train.py        ✅ Pipeline automático
```

### Verificar Modelo ONNX

```bash
python -c "import onnxruntime as ort; sess = ort.InferenceSession('output_pytorch/human_attr_pytorch.onnx'); print('Input:', sess.get_inputs()[0].name, sess.get_inputs()[0].shape); print('Output:', sess.get_outputs()[0].name, sess.get_outputs()[0].shape)"
```

**Salida esperada**:
```
Input: input ['batch', 3, 256, 192]
Output: output ['batch', 27]
```

### Verificar Video Descargado

```bash
python -c "import cv2; cap = cv2.VideoCapture('test_videos/attributes_sim.mp4'); print('FPS:', cap.get(cv2.CAP_PROP_FPS)); print('Frames:', cap.get(cv2.CAP_PROP_FRAME_COUNT)); print('Resolution:', cap.get(cv2.CAP_PROP_FRAME_WIDTH), 'x', cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()"
```

---

## 🚀 Integración con DeepStream

Una vez entrenado el modelo, intégralo con tu pipeline de DeepStream:

### 1. Copiar Modelo ONNX

```bash
# Copiar a directorio de DeepStream
cp output_pytorch/human_attr_pytorch.onnx ../Computer_vision/inference/weights/human_attr/

# Verificar
ls ../Computer_vision/inference/weights/human_attr/
```

### 2. Actualizar Configuración

Edita `Computer_vision/inference/app_config.py`:

```python
# Configuración de atributos humanos
HUMAN_ATTR_ONNX = "weights/human_attr/human_attr_pytorch.onnx"
HUMAN_ATTR_ATTRIBUTES = [
    "Female", "AgeOver60", "Age18-60", "AgeLess18",
    "Front", "Side", "Back",
    "Hat", "Glasses",
    "HandBag", "ShoulderBag", "Backpack", "HoldObjectsInFront",
    "ShortSleeve", "LongSleeve",
    "UpperStride", "UpperLogo", "UpperPlaid", "UpperSplice",
    "LowerStripe", "LowerPattern",
    "LongCoat", "Trousers", "Shorts", "Skirt&Dress", "boots",
    "Male"  # Atributo 27
]
```

### 3. Probar en DeepStream

```bash
cd ../Computer_vision/inference
python test_attributes_cpu.py
```

---

## 🐛 Troubleshooting

### Error: GPU no detectada

**Síntoma**:
```
CUDA available: False
```

**Solución**:
1. Verificar drivers: `nvidia-smi`
2. Reinstalar PyTorch con CUDA:
   ```bash
   pip uninstall torch torchvision
   pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
   ```

### Error: Out of Memory (OOM)

**Síntoma**:
```
RuntimeError: CUDA out of memory
```

**Solución**:
Reducir `BATCH_SIZE` en `train_pytorch.py`:
```python
BATCH_SIZE = 32  # o 16 si persiste
```

### Error: Video no se descarga

**Síntoma**:
```
ERROR downloading video
```

**Solución**:
1. Instalar manualmente `yt-dlp`:
   ```bash
   pip install yt-dlp
   ```
2. Verificar URL del video (puede haber cambiado)
3. Descargar manualmente:
   ```bash
   yt-dlp -f "best[ext=mp4]" -o "test_videos/attributes_sim.mp4" https://www.youtube.com/shorts/hxeudw4U8Cw
   ```

### Error: Modelo no converge

**Síntoma**:
- Loss no baja después de 10 epochs
- Accuracy se mantiene en ~50%

**Solución**:
1. Verificar dataset:
   ```bash
   python explore_dataset.py
   ```
2. Verificar GPU está siendo usada:
   ```bash
   nvidia-smi
   ```
   (Debería mostrar ~99% GPU utilization)
3. Limpiar cache de PyTorch:
   ```bash
   rm -rf output_pytorch/checkpoints/*
   ```

---

## 📊 Atributos Detectados (27 total)

| # | Atributo | Categoría | Descripción |
|---|----------|-----------|-------------|
| 0 | Female | Género | Es mujer |
| 1 | AgeOver60 | Edad | Mayor de 60 años |
| 2 | Age18-60 | Edad | Entre 18 y 60 años |
| 3 | AgeLess18 | Edad | Menor de 18 años |
| 4 | Front | Orientación | Vista frontal |
| 5 | Side | Orientación | Vista lateral |
| 6 | Back | Orientación | Vista trasera |
| 7 | Hat | Accesorios | Sombrero/gorra |
| 8 | Glasses | Accesorios | Lentes |
| 9 | HandBag | Bolsos | Bolso de mano |
| 10 | ShoulderBag | Bolsos | Bolso de hombro |
| 11 | Backpack | Bolsos | Mochila |
| 12 | HoldObjectsInFront | Objetos | Sostiene objetos |
| 13 | ShortSleeve | Ropa Superior | Manga corta |
| 14 | LongSleeve | Ropa Superior | Manga larga |
| 15 | UpperStride | Ropa Superior | Rayas en parte superior |
| 16 | UpperLogo | Ropa Superior | Logo en parte superior |
| 17 | UpperPlaid | Ropa Superior | Cuadros en parte superior |
| 18 | UpperSplice | Ropa Superior | Empalme en parte superior |
| 19 | LowerStripe | Ropa Inferior | Rayas en parte inferior |
| 20 | LowerPattern | Ropa Inferior | Patrón en parte inferior |
| 21 | LongCoat | Ropa | Abrigo largo |
| 22 | Trousers | Ropa Inferior | Pantalones |
| 23 | Shorts | Ropa Inferior | Shorts |
| 24 | Skirt&Dress | Ropa Inferior | Falda o vestido |
| 25 | boots | Calzado | Botas |
| 26 | Male | Género | Es hombre |

---

## 📝 Notas Importantes

1. **Dataset PA-100k**: No está incluido en el repo (es muy grande). Debe descargarse manualmente.
2. **Modelos entrenados**: Tampoco están en el repo por tamaño. Se generan al entrenar.
3. **Videos de prueba**: Se descargan automáticamente con `download_test_video.py`.
4. **ONNX Export**: El modelo se exporta automáticamente al finalizar el entrenamiento.
5. **GPU Required**: El entrenamiento requiere GPU. La inferencia puede hacerse en CPU.

---

## 🤝 Contribuir

Si encuentras errores o tienes mejoras:

1. Fork el repositorio
2. Crea una rama: `git checkout -b feature/mejora`
3. Commit tus cambios: `git commit -am 'Añadir mejora'`
4. Push a la rama: `git push origin feature/mejora`
5. Abre un Pull Request

---

## 📞 Soporte

¿Problemas? Abre un [Issue](https://github.com/neylinsomne/-PA100k-attribute-training/issues)

---

**Última actualización**: Enero 2026
