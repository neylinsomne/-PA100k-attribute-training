"""
Add Male attribute to PA-100k dataset
Creates 27-attribute version: Female, Male, and 25 others
"""
import scipy.io
import numpy as np

def add_male_attribute():
    print("=" * 70)
    print("Adding Male Attribute to PA-100k")
    print("=" * 70)

    # Load original annotation
    print("\n[1/3] Loading annotation.mat...")
    mat_data = scipy.io.loadmat("annotation.mat")

    # Extract original data
    attributes = mat_data['attributes']
    train_label = mat_data['train_label']
    val_label = mat_data['val_label']
    test_label = mat_data['test_label']

    print(f"       Original: {train_label.shape[1]} attributes")

    # Create new attribute list with Male as 2nd attribute
    print("\n[2/3] Creating 27 attributes (adding Male)...")

    # Original attributes as list
    attr_names = [attr[0][0] for attr in attributes]

    # New order: Female, Male, then rest
    new_attr_names = ['Female', 'Male'] + attr_names[1:]  # Skip original Female

    print(f"       New attributes: {len(new_attr_names)}")
    print("\n       Order:")
    print("          1. Female (original)")
    print("          2. Male (derived from Female)")
    for i, name in enumerate(new_attr_names[2:], 3):
        print(f"         {i:2d}. {name}")

    # Derive Male labels (inverse of Female)
    def add_male_column(labels):
        """Add Male column as inverse of Female"""
        female_col = labels[:, 0:1]  # First column (Female)
        male_col = 1 - female_col     # Inverse (Male)

        # Concatenate: [Female, Male, rest of attributes]
        return np.concatenate([female_col, male_col, labels[:, 1:]], axis=1)

    train_label_27 = add_male_column(train_label)
    val_label_27 = add_male_column(val_label)
    test_label_27 = add_male_column(test_label)

    print(f"\n       New label shapes:")
    print(f"          Train: {train_label_27.shape}")
    print(f"          Val:   {val_label_27.shape}")
    print(f"          Test:  {test_label_27.shape}")

    # Verify Male is inverse of Female
    print("\n[3/3] Verifying Male attribute...")
    female_pos = np.sum(train_label_27[:, 0] == 1)
    male_pos = np.sum(train_label_27[:, 1] == 1)
    total = len(train_label_27)

    print(f"       Female: {female_pos:6,} pos ({female_pos/total*100:5.1f}%)")
    print(f"       Male:   {male_pos:6,} pos ({male_pos/total*100:5.1f}%)")
    print(f"       Total:  {female_pos + male_pos:6,} (should equal {total:,})")

    assert female_pos + male_pos == total, "ERROR: Female + Male should cover all samples"

    # Save new annotation file
    print("\n       Saving annotation_27attr.mat...")

    # Create new attributes array
    new_attributes = np.array([[name] for name in new_attr_names], dtype=object)

    # Save to new file
    new_mat_data = {
        'attributes': new_attributes,
        'train_images_name': mat_data['train_images_name'],
        'val_images_name': mat_data['val_images_name'],
        'test_images_name': mat_data['test_images_name'],
        'train_label': train_label_27.astype(np.uint8),
        'val_label': val_label_27.astype(np.uint8),
        'test_label': test_label_27.astype(np.uint8)
    }

    scipy.io.savemat("annotation_27attr.mat", new_mat_data)

    print("\n" + "=" * 70)
    print("Complete!")
    print("=" * 70)
    print("\nCreated: annotation_27attr.mat")
    print("  - 27 attributes (Female, Male, + 25 others)")
    print("  - Male derived as inverse of Female")
    print("\nNext steps:")
    print("  1. Run: python convert_to_paddle.py --use-27attr")
    print("  2. Train with 27 attributes")
    print("=" * 70)

if __name__ == "__main__":
    add_male_attribute()
