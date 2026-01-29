"""
PA-100k to PaddleDetection Format Converter
Converts PA-100k dataset to PaddleDetection multi-label attribute classification format
"""
import scipy.io
import os
import shutil
from pathlib import Path
import numpy as np

# Paths
ANNOTATION_FILE_26 = "annotation.mat"
ANNOTATION_FILE_27 = "annotation_27attr.mat"
IMAGE_DIR = "release_data/release_data"
OUTPUT_DIR = "paddle_format"

def convert_pa100k(use_27attr=True):
    print("=" * 70)
    print("PA-100k to PaddleDetection Format Converter")
    print("=" * 70)

    # Choose annotation file
    annotation_file = ANNOTATION_FILE_27 if use_27attr else ANNOTATION_FILE_26
    num_attrs = 27 if use_27attr else 26

    # Load annotations
    print(f"\n[1/5] Loading {annotation_file}...")
    print(f"       Using {num_attrs} attributes")
    mat_data = scipy.io.loadmat(annotation_file)

    # Extract attributes
    attributes = mat_data['attributes']
    attr_names = [attr[0][0] for attr in attributes]

    print(f"       Found {len(attr_names)} attributes")

    # Create output directory structure
    print(f"\n[2/5] Creating output directory: {OUTPUT_DIR}/")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/images/train", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/images/val", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/images/test", exist_ok=True)

    # Write attribute names file
    attr_file = f"{OUTPUT_DIR}/attributes.txt"
    print(f"\n[3/5] Writing attribute names to {attr_file}")
    with open(attr_file, 'w') as f:
        for attr in attr_names:
            f.write(f"{attr}\n")

    # Process splits
    splits = {
        'train': ('train_images_name', 'train_label'),
        'val': ('val_images_name', 'val_label'),
        'test': ('test_images_name', 'test_label')
    }

    print(f"\n[4/5] Processing dataset splits...")

    for split_name, (img_key, label_key) in splits.items():
        print(f"\n       Processing {split_name.upper()} split...")

        image_names = mat_data[img_key]
        labels = mat_data[label_key]

        # Create annotation file
        anno_file = f"{OUTPUT_DIR}/{split_name}.txt"

        with open(anno_file, 'w') as f:
            for idx, (img_name_arr, label_vec) in enumerate(zip(image_names, labels)):
                # Extract image name
                img_name = img_name_arr[0][0] if isinstance(img_name_arr[0], np.ndarray) else str(img_name_arr[0])

                # Convert label vector to space-separated string
                label_str = ' '.join(map(str, label_vec))

                # Write: relative_path label1 label2 ... label26
                f.write(f"images/{split_name}/{img_name} {label_str}\n")

                # Copy image (optional - can be slow for 100k images)
                # Uncomment if you want to reorganize images
                # src = f"{IMAGE_DIR}/{img_name}"
                # dst = f"{OUTPUT_DIR}/images/{split_name}/{img_name}"
                # if os.path.exists(src) and not os.path.exists(dst):
                #     shutil.copy(src, dst)

        print(f"       -> {anno_file}: {len(labels):,} samples")

    # Create symlink to original images (faster than copying)
    print(f"\n[5/5] Creating symlink to original images...")

    for split_name in ['train', 'val', 'test']:
        link_src = os.path.abspath(IMAGE_DIR)
        link_dst = os.path.abspath(f"{OUTPUT_DIR}/images/{split_name}_source")

        # Remove existing symlink if present
        if os.path.exists(link_dst):
            if os.path.islink(link_dst):
                os.unlink(link_dst)
            elif os.path.isdir(link_dst):
                print(f"       Warning: {link_dst} already exists as directory")

        try:
            # Create symlink (works on Windows with admin rights or Developer Mode)
            os.symlink(link_src, link_dst, target_is_directory=True)
            print(f"       Created symlink: {link_dst} -> {link_src}")
        except OSError as e:
            print(f"       Warning: Could not create symlink ({e})")
            print(f"       You can manually copy images from {IMAGE_DIR}/")
            print(f"       to {OUTPUT_DIR}/images/{split_name}/")

    # Create PaddleDetection config reference
    config_template = f"""
# PaddleDetection Configuration for PA-100k Attribute Recognition
# Place this in PaddleDetection/configs/pedestrian_attribute/

# Model: StrongBaseline (PPLCNet_x1_0)
architecture: StrongBaseline

# Dataset
TrainDataset:
  name: MultiLabelDataset
  image_dir: {os.path.abspath(OUTPUT_DIR)}/images/train_source
  anno_path: {os.path.abspath(OUTPUT_DIR)}/train.txt
  label_list: {os.path.abspath(OUTPUT_DIR)}/attributes.txt

ValDataset:
  name: MultiLabelDataset
  image_dir: {os.path.abspath(OUTPUT_DIR)}/images/val_source
  anno_path: {os.path.abspath(OUTPUT_DIR)}/val.txt
  label_list: {os.path.abspath(OUTPUT_DIR)}/attributes.txt

TestDataset:
  name: MultiLabelDataset
  image_dir: {os.path.abspath(OUTPUT_DIR)}/images/test_source
  anno_path: {os.path.abspath(OUTPUT_DIR)}/test.txt
  label_list: {os.path.abspath(OUTPUT_DIR)}/attributes.txt

# Training hyperparameters
epoch: 60
batch_size: 64
learning_rate: 0.001
num_classes: 26

# Image preprocessing
image_shape: [3, 256, 192]
"""

    config_file = f"{OUTPUT_DIR}/paddle_config_reference.yml"
    with open(config_file, 'w') as f:
        f.write(config_template)

    print(f"\n       Created config reference: {config_file}")

    # Summary
    print("\n" + "=" * 70)
    print("Conversion Complete!")
    print("=" * 70)
    print(f"\nOutput structure:")
    print(f"  {OUTPUT_DIR}/")
    print(f"  ├── attributes.txt          (26 attribute names)")
    print(f"  ├── train.txt               (80,000 samples)")
    print(f"  ├── val.txt                 (10,000 samples)")
    print(f"  ├── test.txt                (10,000 samples)")
    print(f"  ├── paddle_config_reference.yml")
    print(f"  └── images/")
    print(f"      ├── train_source -> {IMAGE_DIR}")
    print(f"      ├── val_source -> {IMAGE_DIR}")
    print(f"      └── test_source -> {IMAGE_DIR}")
    print("\nNext steps:")
    print("  1. Use train.txt/val.txt for PaddleDetection fine-tuning")
    print("  2. See paddle_config_reference.yml for config example")
    print("=" * 70)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-27attr", action="store_true", default=True,
                        help="Use 27 attributes (with Male)")
    parser.add_argument("--use-26attr", dest="use_27attr", action="store_false",
                        help="Use original 26 attributes (Female only)")
    args = parser.parse_args()

    convert_pa100k(use_27attr=args.use_27attr)
