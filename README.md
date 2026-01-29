# PA-100k Attribute Recognition Fine-Tuning Pipeline

Complete pipeline for fine-tuning PP-Human Attribute model on PA-100k dataset.

## Dataset Overview

**PA-100k (Pedestrian Attribute Recognition)**
- 100,000 pedestrian images
- 26 attributes (gender, age, bags, clothing, etc.)
- Train/Val/Test split: 80k/10k/10k

### 26 Attributes

| Category | Attributes |
|----------|------------|
| **Demographics** | Female, AgeOver60, Age18-60, AgeLess18 |
| **Orientation** | Front, Side, Back |
| **Accessories** | Hat, Glasses, HandBag, ShoulderBag, Backpack, HoldObjectsInFront |
| **Upper Body** | ShortSleeve, LongSleeve, UpperStride, UpperLogo, UpperPlaid, UpperSplice, LongCoat |
| **Lower Body** | LowerStripe, LowerPattern, Trousers, Shorts, Skirt&Dress |
| **Footwear** | boots |

**Perfect match**: These 26 attributes are identical to PP-Human Attribute model, enabling direct fine-tuning without architecture changes.

## Directory Structure

```
PA-110k/
├── README.md                    (This file)
├── annotation.mat               (Original annotations)
├── release_data/
│   └── release_data/           (100,000 images: 000001.jpg - 100000.jpg)
├── paddle_format/               (Converted dataset)
│   ├── train.txt               (80,000 samples)
│   ├── val.txt                 (10,000 samples)
│   ├── test.txt                (10,000 samples)
│   ├── attributes.txt          (26 attribute names)
│   └── paddle_config_reference.yml
├── output/                      (Training outputs)
│   ├── pa100k_attribute/       (Checkpoints)
│   └── onnx_export/            (ONNX models)
├── explore_dataset.py           (Dataset explorer)
├── convert_to_paddle.py         (Format converter)
└── finetune_attributes.py       (Training pipeline)
```

## Quick Start

### 1. Explore Dataset

```bash
python explore_dataset.py
```

Shows:
- 26 attributes with names
- Train/val/test split sizes
- Attribute statistics (positive/negative counts)

### 2. Convert to PaddleDetection Format

```bash
python convert_to_paddle.py
```

Creates `paddle_format/` with:
- `train.txt`, `val.txt`, `test.txt` (image paths + 26 labels)
- `attributes.txt` (attribute names)

Format: `path/to/image.jpg 1 0 1 0 ... (26 binary labels)`

### 3. Fine-Tune Model

**Prerequisites:**
- PaddleDetection installed (see `Computer_vision/training/paddle_detection/download_models.py --clone-repo --install-deps`)
- GPU with CUDA (recommended)

**Start training:**

```bash
python finetune_attributes.py
```

**Resume training:**

```bash
python finetune_attributes.py --resume
```

**Evaluate only:**

```bash
python finetune_attributes.py --eval
```

**Export to ONNX only:**

```bash
python finetune_attributes.py --export-only
```

### 4. Deploy to DeepStream

After training completes, copy the ONNX model:

```bash
cp output/onnx_export/human_attr_finetuned.onnx \
   ../../Computer_vision/inference/weights/human_attr/
```

Then update `inference/config/app_config.py` if needed.

## Training Configuration

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Epochs** | 60 | Total training epochs |
| **Batch Size** | 64 | Images per batch |
| **Learning Rate** | 0.001 | Initial LR (decays at epoch 30, 45) |
| **Optimizer** | Adam | With L2 regularization (0.0001) |
| **Input Size** | 256x192 | Height x Width |
| **Warmup Steps** | 500 | Linear warmup iterations |

### Data Augmentation

- RandomCrop (prob=0.5)
- RandomFlip (prob=0.5)
- Normalization (ImageNet mean/std)

### Model Architecture

- **Backbone**: PPLCNet_x1_0 (lightweight CNN)
- **Loss**: Multi-label Binary Cross-Entropy
- **Metric**: mA (mean Accuracy), Precision, Recall, F1

## Transfer Learning with PyTorch (.pth) Model

### market1501_AGW.pth

Located in parent directory (`../market1501_AGW.pth`), this is a pre-trained **Person Re-Identification** model from [ReID-Survey](https://github.com/mangye16/ReID-Survey).

**Architecture**: ResNet-50 backbone trained on Market-1501 dataset.

### How to Use for Transfer Learning

#### Option 1: Feature Extractor (Recommended)

Use the pre-trained ResNet-50 as a feature extractor for attribute recognition:

```python
import torch
import torch.nn as nn

# Load PyTorch checkpoint
checkpoint = torch.load('../market1501_AGW.pth')

# Extract backbone weights (ResNet-50)
backbone_state = {k: v for k, v in checkpoint['state_dict'].items()
                  if k.startswith('backbone')}

# Create model with pre-trained backbone
class AttributeModel(nn.Module):
    def __init__(self, num_attributes=26):
        super().__init__()
        # Load pre-trained ResNet-50
        from torchvision.models import resnet50
        self.backbone = resnet50(pretrained=False)
        self.backbone.load_state_dict(backbone_state, strict=False)

        # Replace final layer for 26 attributes
        self.backbone.fc = nn.Linear(2048, num_attributes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        features = self.backbone(x)
        return self.sigmoid(features)

model = AttributeModel(num_attributes=26)
```

#### Option 2: Convert PyTorch → PaddlePaddle

If you want to use the .pth model within PaddleDetection:

```python
import torch
import paddle

# Load PyTorch model
pt_checkpoint = torch.load('../market1501_AGW.pth')
pt_state = pt_checkpoint['state_dict']

# Convert to PaddlePaddle format
paddle_state = {}
for k, v in pt_state.items():
    # Convert tensor
    paddle_state[k] = paddle.to_tensor(v.cpu().numpy())

# Save as PaddlePaddle checkpoint
paddle.save(paddle_state, 'market1501_backbone.pdparams')
```

Then load in PaddleDetection config:

```yaml
pretrain_weights: path/to/market1501_backbone.pdparams
```

#### Option 3: Fine-tune PyTorch Model Directly

Train a PyTorch model instead of PaddleDetection:

1. Use market1501_AGW.pth as backbone initialization
2. Add multi-label classification head (26 outputs)
3. Train on PA-100k using PyTorch
4. Export to ONNX for deployment

```bash
# Example training script
python train_pytorch_attributes.py \
    --pretrained ../market1501_AGW.pth \
    --dataset paddle_format/ \
    --num-classes 26 \
    --epochs 60
```

### When to Use Each Option

| Approach | Use Case | Complexity |
|----------|----------|------------|
| **PaddleDetection** (current) | Quick fine-tuning, PP-Human compatible | Low |
| **PyTorch Feature Extractor** | Custom architecture, full control | Medium |
| **PyTorch Fine-Tuning** | Pure PyTorch workflow | Medium |
| **Convert PT→PD** | Use ReID backbone in PaddleDetection | High |

## Dataset Format Details

### train.txt / val.txt / test.txt

```
images/train/000001.jpg 1 0 1 0 1 0 0 0 0 0 1 0 1 1 0 0 0 0 0 0 0 0 1 0 0 0
images/train/000002.jpg 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0
...
```

Format: `<relative_image_path> <26_binary_labels>`
- Labels: 0 (absent) or 1 (present)
- Order matches attributes.txt

### attributes.txt

```
Female
AgeOver60
Age18-60
...
boots
```

One attribute name per line (26 total).

## Evaluation Metrics

After training, you'll see metrics like:

```
mA (mean Accuracy): 0.85
Precision: 0.82
Recall: 0.80
F1-Score: 0.81
```

**Per-attribute accuracy** shows which attributes are hardest to recognize (e.g., boots, HoldObjectsInFront have low frequency).

## Troubleshooting

### Out of Memory (OOM)

Reduce batch size in `finetune_attributes.py`:

```python
batch_size: 32  # was 64
```

### Slow Training

- Use mixed precision training (FP16)
- Reduce image size to 192x128
- Use fewer data augmentations

### Poor Accuracy

- Train longer (increase epochs to 100)
- Use stronger backbone (ResNet-50 instead of PPLCNet)
- Add more data augmentation

### Can't Create Symlinks on Windows

Images will be loaded directly from `release_data/release_data/` using absolute paths. This is already handled in the config.

## Next Steps

1. **Integrate with DeepStream**:
   - Copy ONNX model to `inference/weights/human_attr/`
   - Create attribute processor in `inference/processors/`
   - Add to pipeline in `main_ds.py`

2. **Test with Video**:
   - Download test video: `python simulator/download_videos.py`
   - Use `attributes_sim.mp4` for testing

3. **Compare Models**:
   - Baseline: Pre-trained PP-Human Attribute
   - Fine-tuned: Your PA-100k model
   - Measure improvement in accuracy

## References

- [PA-100k Paper](https://arxiv.org/abs/1709.09930) - HydraPlus-Net
- [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)
- [ReID-Survey](https://github.com/mangye16/ReID-Survey) - PyTorch .pth model source
- [PP-Human Docs](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.8.1/deploy/pipeline)

## License

PA-100k dataset: Please cite the original paper if you use this dataset.
