"""
PA-100k Attribute Fine-Tuning Pipeline for PaddleDetection
Trains PP-Human Attribute model on PA-100k dataset
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

# Paths
CURRENT_DIR = Path(__file__).parent.absolute()
PADDLE_FORMAT_DIR = CURRENT_DIR / "paddle_format"
IMAGE_DIR = CURRENT_DIR / "release_data" / "release_data"
PADDLEDET_DIR = CURRENT_DIR.parent.parent / "Computer_vision" / "training" / "paddle_detection" / "PaddleDetection"
OUTPUT_DIR = CURRENT_DIR / "output"

# Training configuration
CONFIG_TEMPLATE = """
# PP-Human Attribute Fine-Tuning on PA-100k
# Based on PPLCNet_x1_0 StrongBaseline

use_gpu: true
log_iter: 100
save_dir: output/pa100k_attribute
snapshot_epoch: 5
weights: https://bj.bcebos.com/v1/paddledet/models/pipeline/PPLCNet_x1_0_person_attribute_945_infer.zip
epoch: 60
num_classes: 27

# Learning rate
LearningRate:
  base_lr: 0.001
  schedulers:
  - !PiecewiseDecay
    milestones: [30, 45]
    gamma: 0.1
  - !LinearWarmup
    start_factor: 0.1
    steps: 500

# Optimizer
OptimizerBuilder:
  optimizer:
    type: Adam
  regularizer:
    type: L2
    factor: 0.0001

# Dataset
TrainDataset:
  !MultiLabelDataset
    image_dir: {image_dir}
    anno_path: {anno_path_train}
    label_list: {label_list}
    batch_transforms:
    - Decode: {{}}
    - RandomCrop: {{prob: 0.5}}
    - RandomFlip: {{prob: 0.5}}
    - Resize: {{target_size: [256, 192], interp: 1}}
    - NormalizeImage: {{mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}}
    - Permute: {{}}
  batch_size: 64
  shuffle: true
  drop_last: true

EvalDataset:
  !MultiLabelDataset
    image_dir: {image_dir}
    anno_path: {anno_path_val}
    label_list: {label_list}
    batch_transforms:
    - Decode: {{}}
    - Resize: {{target_size: [256, 192], interp: 1}}
    - NormalizeImage: {{mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}}
    - Permute: {{}}
  batch_size: 64

TestDataset:
  !MultiLabelDataset
    image_dir: {image_dir}
    anno_path: {anno_path_test}
    label_list: {label_list}
    batch_transforms:
    - Decode: {{}}
    - Resize: {{target_size: [256, 192], interp: 1}}
    - NormalizeImage: {{mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}}
    - Permute: {{}}
  batch_size: 64

# Model Architecture
architecture: StrongBaseline
pretrain_weights: https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x1_0_pretrained.pdparams

StrongBaseline:
  backbone: PPLCNet
  num_classes: 27
  loss: MultiLabelLoss

PPLCNet:
  scale: 1.0
  feature_maps: [7]

# Evaluation
metric: MultiLabelMetric
num_classes: 27
"""

def setup_environment():
    """Check and setup PaddleDetection environment"""
    print("=" * 70)
    print("PA-100k Attribute Fine-Tuning Pipeline")
    print("=" * 70)

    print("\n[1/6] Checking environment...")

    # Check if PaddleDetection exists
    if not PADDLEDET_DIR.exists():
        print(f"\nERROR: PaddleDetection not found at {PADDLEDET_DIR}")
        print("\nPlease run the download script first:")
        print(f"  cd {Path(__file__).parent.parent.parent / 'Computer_vision' / 'training' / 'paddle_detection'}")
        print("  python download_models.py --clone-repo --install-deps")
        sys.exit(1)

    print(f"       PaddleDetection found: {PADDLEDET_DIR}")

    # Check dataset
    if not (PADDLE_FORMAT_DIR / "train.txt").exists():
        print(f"\nERROR: Dataset not converted yet")
        print("Please run: python convert_to_paddle.py")
        sys.exit(1)

    print(f"       Dataset found: {PADDLE_FORMAT_DIR}")

    # Check images
    sample_img = IMAGE_DIR / "000001.jpg"
    if not sample_img.exists():
        print(f"\nERROR: Images not found at {IMAGE_DIR}")
        sys.exit(1)

    print(f"       Images found: {IMAGE_DIR}")

    return True

def create_config():
    """Create PaddleDetection config file"""
    print("\n[2/6] Creating training configuration...")

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Fill template
    config_content = CONFIG_TEMPLATE.format(
        image_dir=str(IMAGE_DIR).replace('\\', '/'),
        anno_path_train=str(PADDLE_FORMAT_DIR / "train.txt").replace('\\', '/'),
        anno_path_val=str(PADDLE_FORMAT_DIR / "val.txt").replace('\\', '/'),
        anno_path_test=str(PADDLE_FORMAT_DIR / "test.txt").replace('\\', '/'),
        label_list=str(PADDLE_FORMAT_DIR / "attributes.txt").replace('\\', '/')
    )

    # Write config
    config_path = OUTPUT_DIR / "pa100k_attribute_config.yml"
    with open(config_path, 'w') as f:
        f.write(config_content)

    print(f"       Config created: {config_path}")
    return config_path

def train(config_path, resume=False, eval_only=False):
    """Run training"""

    if eval_only:
        print("\n[3/6] Running evaluation only...")
        cmd = [
            "python",
            str(PADDLEDET_DIR / "tools" / "eval.py"),
            "-c", str(config_path),
            "--use_gpu", "true"
        ]
    else:
        print(f"\n[3/6] Starting training (resume={resume})...")
        cmd = [
            "python",
            str(PADDLEDET_DIR / "tools" / "train.py"),
            "-c", str(config_path),
            "--use_gpu", "true",
            "--eval"
        ]

        if resume:
            cmd.extend(["-r", str(OUTPUT_DIR / "pa100k_attribute" / "model_final")])

    print(f"\n       Running command:")
    print(f"       {' '.join(cmd)}\n")

    # Change to PaddleDetection directory
    os.chdir(PADDLEDET_DIR)

    # Run training
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Training failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(0)

def export_to_onnx(weights_path=None):
    """Export trained model to ONNX"""
    print("\n[4/6] Exporting model to ONNX...")

    if weights_path is None:
        weights_path = OUTPUT_DIR / "pa100k_attribute" / "model_final"

    if not Path(weights_path).exists():
        print(f"\nWARNING: Weights not found at {weights_path}")
        print("Skipping ONNX export")
        return None

    export_dir = OUTPUT_DIR / "onnx_export"
    export_dir.mkdir(exist_ok=True)

    cmd = [
        "python",
        str(PADDLEDET_DIR / "tools" / "export_model.py"),
        "-c", str(OUTPUT_DIR / "pa100k_attribute_config.yml"),
        "-o", f"weights={weights_path}",
        "--output_dir", str(export_dir)
    ]

    print(f"       Running: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True, cwd=PADDLEDET_DIR)

        # Convert to ONNX using paddle2onnx
        print("\n[5/6] Converting to ONNX format...")

        onnx_cmd = [
            "paddle2onnx",
            "--model_dir", str(export_dir / "model"),
            "--model_filename", "model.pdmodel",
            "--params_filename", "model.pdiparams",
            "--save_file", str(export_dir / "human_attr_finetuned.onnx"),
            "--opset_version", "11"
        ]

        subprocess.run(onnx_cmd, check=True)

        print(f"\n       ONNX model saved: {export_dir / 'human_attr_finetuned.onnx'}")
        return export_dir / "human_attr_finetuned.onnx"

    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Export failed with exit code {e.returncode}")
        return None

def print_summary(onnx_path=None):
    """Print training summary"""
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)

    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"  - Model weights: {OUTPUT_DIR / 'pa100k_attribute' / 'model_final.pdparams'}")

    if onnx_path:
        print(f"  - ONNX model: {onnx_path}")
        print(f"\nTo use in DeepStream:")
        print(f"  1. Copy {onnx_path.name}")
        print(f"     to: Computer_vision/inference/weights/human_attr/")
        print(f"  2. Update config to use new model")

    print("\nTo resume training:")
    print(f"  python {Path(__file__).name} --resume")

    print("\nTo evaluate:")
    print(f"  python {Path(__file__).name} --eval")

    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="PA-100k Attribute Fine-Tuning")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--eval", action="store_true", help="Evaluation only")
    parser.add_argument("--export-only", action="store_true", help="Export to ONNX only")
    args = parser.parse_args()

    # Setup
    setup_environment()

    # Create config
    config_path = create_config()

    if args.export_only:
        # Export only
        onnx_path = export_to_onnx()
        if onnx_path:
            print(f"\nONNX export complete: {onnx_path}")
    else:
        # Train/Eval
        train(config_path, resume=args.resume, eval_only=args.eval)

        # Export
        if not args.eval:
            onnx_path = export_to_onnx()
            print_summary(onnx_path)

if __name__ == "__main__":
    main()
