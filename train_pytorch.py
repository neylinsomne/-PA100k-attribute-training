"""
PA-100k Attribute Fine-Tuning with PyTorch
Optimized for RTX 5060 Ti (16GB VRAM)
"""
import os
import sys
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np

# Paths
CURRENT_DIR = Path(__file__).parent.absolute()
PADDLE_FORMAT_DIR = CURRENT_DIR / "paddle_format"
IMAGE_DIR = CURRENT_DIR / "release_data" / "release_data"
OUTPUT_DIR = CURRENT_DIR / "output_pytorch"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

# Training config
BATCH_SIZE = 64  # RTX 5060 Ti can handle this
NUM_WORKERS = 4
EPOCHS = 60
LEARNING_RATE = 0.001
NUM_CLASSES = 27
INPUT_SIZE = (256, 192)  # H x W

class PA100kDataset(Dataset):
    """PA-100k multi-label attribute dataset"""

    def __init__(self, anno_file, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.samples = []

        # Read annotations
        with open(anno_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 28:  # path + 27 labels
                    continue

                img_path = parts[0]
                # Convert relative path to absolute
                if img_path.startswith('images/'):
                    img_path = img_path.replace('images/train/', '')
                    img_path = img_path.replace('images/val/', '')
                    img_path = img_path.replace('images/test/', '')

                labels = [int(x) for x in parts[1:28]]
                self.samples.append((img_path, labels))

        print(f"Loaded {len(self.samples)} samples from {anno_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, labels = self.samples[idx]

        # Load image
        full_path = self.image_dir / img_path
        try:
            image = Image.open(full_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {full_path}: {e}")
            # Return black image as fallback
            image = Image.new('RGB', (256, 192), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        labels = torch.FloatTensor(labels)
        return image, labels

class AttributeModel(nn.Module):
    """Multi-label attribute classification model"""

    def __init__(self, num_classes=27, pretrained=True):
        super(AttributeModel, self).__init__()

        # Use ResNet-50 as backbone (similar performance to PPLCNet)
        self.backbone = models.resnet50(pretrained=pretrained)

        # Replace final FC layer for multi-label classification
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calculate_accuracy(outputs, labels, threshold=0.5):
    """Calculate multi-label accuracy"""
    preds = (torch.sigmoid(outputs) > threshold).float()
    correct = (preds == labels).float()
    acc = correct.mean()
    return acc.item()

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()

    losses = AverageMeter()
    accuracies = AverageMeter()

    start_time = time.time()

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        acc = calculate_accuracy(outputs, labels)
        losses.update(loss.item(), images.size(0))
        accuracies.update(acc, images.size(0))

        # Log every 100 batches
        if (batch_idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch}/{EPOCHS}] "
                  f"Batch [{batch_idx+1}/{len(train_loader)}] "
                  f"Loss: {losses.avg:.4f} "
                  f"Acc: {accuracies.avg:.4f} "
                  f"Time: {elapsed:.1f}s")

    return losses.avg, accuracies.avg

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()

    losses = AverageMeter()
    accuracies = AverageMeter()

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            acc = calculate_accuracy(outputs, labels)
            losses.update(loss.item(), images.size(0))
            accuracies.update(acc, images.size(0))

    return losses.avg, accuracies.avg

def save_checkpoint(model, optimizer, epoch, best_acc, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def export_to_onnx(model, save_path):
    """Export model to ONNX format"""
    model.eval()
    dummy_input = torch.randn(1, 3, INPUT_SIZE[0], INPUT_SIZE[1]).to(next(model.parameters()).device)

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"ONNX model exported: {save_path}")

def main():
    parser = argparse.ArgumentParser(description='PA-100k Attribute Training')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--eval-only', action='store_true', help='Evaluation only')
    args = parser.parse_args()

    # Setup
    OUTPUT_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print("PA-100k Attribute Training with PyTorch")
    print("=" * 70)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print()

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Datasets
    train_dataset = PA100kDataset(
        PADDLE_FORMAT_DIR / 'train.txt',
        IMAGE_DIR,
        transform=train_transform
    )

    val_dataset = PA100kDataset(
        PADDLE_FORMAT_DIR / 'val.txt',
        IMAGE_DIR,
        transform=val_transform
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print()

    # Model
    model = AttributeModel(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[30, 45],
        gamma=0.1
    )

    # Resume from checkpoint
    start_epoch = 0
    best_acc = 0.0

    if args.resume:
        if os.path.exists(args.resume):
            print(f"Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']
            print(f"Resumed from epoch {start_epoch}, best acc: {best_acc:.4f}")
        else:
            print(f"Checkpoint not found: {args.resume}")

    # Evaluation only
    if args.eval_only:
        print("Evaluation mode")
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        return

    # Training loop
    print("=" * 70)
    print("Starting training...")
    print("=" * 70)

    for epoch in range(start_epoch, EPOCHS):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch + 1
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Learning rate step
        scheduler.step()

        epoch_time = time.time() - epoch_start

        print(f"\nEpoch [{epoch+1}/{EPOCHS}] Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"  Time: {epoch_time:.1f}s")
        print("-" * 70)

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = CHECKPOINT_DIR / f"checkpoint_epoch_{epoch+1}.pth"
            save_checkpoint(model, optimizer, epoch, best_acc, checkpoint_path)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_path = CHECKPOINT_DIR / "best_model.pth"
            save_checkpoint(model, optimizer, epoch, best_acc, best_path)
            print(f"New best model! Acc: {best_acc:.4f}")

    # Save final model
    final_path = CHECKPOINT_DIR / "final_model.pth"
    save_checkpoint(model, optimizer, EPOCHS - 1, best_acc, final_path)

    # Export to ONNX
    print("\n" + "=" * 70)
    print("Exporting to ONNX...")
    onnx_path = OUTPUT_DIR / "human_attr_pytorch.onnx"
    export_to_onnx(model, onnx_path)

    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    print(f"Best validation accuracy: {best_acc:.4f}")
    print(f"Checkpoints saved to: {CHECKPOINT_DIR}")
    print(f"ONNX model: {onnx_path}")
    print(f"\nTo use in DeepStream:")
    print(f"  cp {onnx_path}")
    print(f"     ../../Computer_vision/inference/weights/human_attr/")

if __name__ == "__main__":
    main()
