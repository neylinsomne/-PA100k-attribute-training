"""
PA-100k Dataset Explorer
Reads annotation.mat and displays dataset structure
"""
import scipy.io
import numpy as np

def explore_pa100k():
    print("=" * 60)
    print("PA-100k Dataset Explorer")
    print("=" * 60)

    # Load annotation.mat
    mat_path = "annotation.mat"
    print(f"\nLoading {mat_path}...")

    try:
        mat_data = scipy.io.loadmat(mat_path)
    except Exception as e:
        print(f"ERROR: Error loading .mat file: {e}")
        print("\nInstall scipy if needed: pip install scipy")
        return

    print("Loaded successfully!\n")

    # Display all keys
    print("Keys in annotation.mat:")
    for key in mat_data.keys():
        if not key.startswith('__'):
            print(f"   - {key}: {type(mat_data[key])}, shape: {mat_data[key].shape if hasattr(mat_data[key], 'shape') else 'N/A'}")

    print("\n" + "=" * 60)

    # Extract attributes
    if 'attributes' in mat_data:
        attributes = mat_data['attributes']
        print(f"\n26 Attributes in PA-100k:")
        print("-" * 60)
        for idx, attr in enumerate(attributes, 1):
            attr_name = attr[0][0] if isinstance(attr[0], np.ndarray) else str(attr[0])
            print(f"   {idx:2d}. {attr_name}")

    print("\n" + "=" * 60)

    # Dataset splits
    print("\nDataset Splits:")
    print("-" * 60)

    splits = {
        'train': ('train_images_name', 'train_label'),
        'val': ('val_images_name', 'val_label'),
        'test': ('test_images_name', 'test_label')
    }

    total_images = 0
    for split_name, (img_key, label_key) in splits.items():
        if img_key in mat_data and label_key in mat_data:
            num_images = len(mat_data[img_key])
            num_labels = mat_data[label_key].shape[0]
            total_images += num_images
            print(f"   {split_name.upper():5s}: {num_images:6,} images, {num_labels:6,} labels")

    print(f"   {'TOTAL':5s}: {total_images:6,} images")

    print("\n" + "=" * 60)

    # Sample data inspection
    if 'train_label' in mat_data:
        train_labels = mat_data['train_label']
        print(f"\nSample Training Data:")
        print("-" * 60)
        print(f"   Label matrix shape: {train_labels.shape}")
        print(f"   Data type: {train_labels.dtype}")
        print(f"\n   First 3 samples (26 attributes each):")
        for i in range(min(3, train_labels.shape[0])):
            print(f"   Sample {i+1}: {train_labels[i]}")

        # Attribute statistics
        print(f"\nAttribute Statistics (Training Set):")
        print("-" * 60)
        if 'attributes' in mat_data:
            for idx, attr in enumerate(attributes):
                attr_name = attr[0][0] if isinstance(attr[0], np.ndarray) else str(attr[0])
                positive_count = np.sum(train_labels[:, idx] == 1)
                negative_count = np.sum(train_labels[:, idx] == 0)
                total = len(train_labels)
                positive_pct = (positive_count / total) * 100
                print(f"   {idx+1:2d}. {attr_name:20s}: {positive_count:6,} pos ({positive_pct:5.1f}%), {negative_count:6,} neg")

    print("\n" + "=" * 60)
    print("Exploration complete!")
    print("=" * 60)

if __name__ == "__main__":
    explore_pa100k()
