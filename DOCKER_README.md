# PA-100k Attribute Training - Docker Setup

Entrenamiento de atributos de personas con PyTorch + CUDA en Docker.

## 📋 Prerequisitos

- Docker Desktop con soporte GPU (WSL2)
- NVIDIA Container Toolkit
- RTX 5060 Ti (o GPU compatible)

## 🚀 Uso rápido

### 1. Construir imagen

```bash
cd e:/Proyects/FIneData/PA-110k
docker-compose build
```

### 2. Entrenar

```bash
# Iniciar entrenamiento (60 epochs)
docker-compose up pa100k-training

# En segundo plano
docker-compose up -d pa100k-training
```

### 3. Monitorear

```bash
# Ver logs en tiempo real
docker-compose logs -f pa100k-training

# Ver últimas 100 líneas
docker-compose logs --tail=100 pa100k-training
```

### 4. TensorBoard (opcional)

```bash
# Iniciar TensorBoard
docker-compose --profile monitoring up tensorboard

# Abrir en navegador
http://localhost:6006
```

## 📁 Estructura

```
PA-110k/
├── Dockerfile                    # Imagen con PyTorch + CUDA 12.8
├── docker-compose.yml            # Orquestación
├── train_pytorch.py              # Script de entrenamiento
├── paddle_format/                # Dataset convertido (read-only)
│   ├── train.txt
│   ├── val.txt
│   └── attributes.txt
├── release_data/release_data/    # Imágenes (read-only)
└── output_pytorch/               # Outputs (read-write)
    ├── checkpoints/
    │   ├── best_model.pth
    │   ├── checkpoint_epoch_5.pth
    │   └── ...
    └── human_attr_pytorch.onnx
```

## ⚙️ Comandos útiles

### Entrenamiento

```bash
# Entrenar desde cero
docker-compose up pa100k-training

# Reanudar desde checkpoint
docker-compose run pa100k-training python3 train_pytorch.py --resume /workspace/output_pytorch/checkpoints/checkpoint_epoch_30.pth

# Solo evaluación
docker-compose run pa100k-training python3 train_pytorch.py --eval-only --resume /workspace/output_pytorch/checkpoints/best_model.pth
```

### Gestión

```bash
# Detener entrenamiento
docker-compose down

# Ver uso de GPU
nvidia-smi -l 5

# Entrar al contenedor
docker-compose exec pa100k-training bash

# Limpiar todo
docker-compose down -v
docker system prune -a
```

## 📊 Monitoreo

### Logs

```bash
# Tail continuo
docker-compose logs -f pa100k-training | grep "Epoch"

# Buscar mejor accuracy
docker-compose logs pa100k-training | grep "New best"
```

### GPU

```bash
# Watch GPU usage
watch -n 1 nvidia-smi
```

### Outputs

```bash
# Listar checkpoints
ls -lh output_pytorch/checkpoints/

# Ver modelo ONNX
ls -lh output_pytorch/*.onnx
```

## 🔧 Configuración avanzada

### Cambiar batch size

Edita `train_pytorch.py`:
```python
BATCH_SIZE = 32  # Reducir si hay OOM
```

### Cambiar epochs

```python
EPOCHS = 100  # Más epochs para mejor accuracy
```

### Múltiples GPUs

Edita `docker-compose.yml`:
```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0,1  # Usar GPU 0 y 1
```

## 🐛 Troubleshooting

### OOM (Out of Memory)

```bash
# Reducir batch size
BATCH_SIZE = 32  # En train_pytorch.py

# O usar GPU con más VRAM
```

### GPU no detectada

```bash
# Verificar NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi

# Si falla, reinstalar:
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

### Permisos en output/

```bash
# Dar permisos
chmod -R 777 output_pytorch/
```

## 📤 Exportar modelo entrenado

Al finalizar, el modelo se exporta automáticamente a ONNX:

```bash
# Copiar a inference
cp output_pytorch/human_attr_pytorch.onnx \
   ../../Computer_vision/inference/weights/human_attr/

# O desde Docker
docker-compose run pa100k-training \
  cp /workspace/output_pytorch/human_attr_pytorch.onnx \
     /workspace/output_pytorch/human_attr_finetuned.onnx
```

## 🎯 Integración con DeepStream

Después del entrenamiento:

1. Copiar ONNX a inference:
   ```bash
   cp output_pytorch/human_attr_pytorch.onnx \
      e:/Proyects/Computer_vision/inference/weights/human_attr/
   ```

2. Actualizar config en `app_config.py`

3. Probar con video:
   ```bash
   cd e:/Proyects/Computer_vision/inference
   python main_ds.py
   ```

## 📊 Tiempos esperados

| Epochs | RTX 5060 Ti | RTX 4090 | A100 |
|--------|-------------|----------|------|
| 10 | 30-45 min | 20-30 min | 15-20 min |
| 60 | 3-5 horas | 2-3 horas | 1.5-2 horas |

## 🔗 Links útiles

- [PyTorch Docker](https://hub.docker.com/r/pytorch/pytorch)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [Docker Compose GPU](https://docs.docker.com/compose/gpu-support/)
