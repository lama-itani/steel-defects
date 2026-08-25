"""
Dataset Split Script.

Merges the original train/val folders and creates a new stratified 80/20 split.
Stratification uses the *most-frequent defect class* in each image, so
multi-defect images no longer collapse to their first <object> entry. Run once
before training.
"""
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import random
from sklearn.model_selection import train_test_split


def get_dominant_defect_class(xml_path):
    """
    Return the most-frequent object class name in the XML annotation.
    Falls back to 'unknown' if the file has no <object> or is malformed.
    Ties are broken by first-seen order (Counter behavior).
    """
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return 'unknown'

    root = tree.getroot()
    names = []
    for obj in root.findall('object'):
        name_el = obj.find('name')
        if name_el is not None and name_el.text:
            names.append(name_el.text)

    if not names:
        return 'unknown'
    return Counter(names).most_common(1)[0][0]


def resplit_dataset(data_root, train_ratio=0.80, seed=42):
    """
    Resplit dataset into train_ratio / (1 - train_ratio) with stratification.

    Args:
        data_root: Path to data/raw directory (must contain train_images,
        train_annotations, valid_images, valid_annotations).
        train_ratio: Training set ratio (default 0.80).
        seed: Random seed for reproducibility.
    """
    random.seed(seed)
    data_root = Path(data_root)

    print("=" * 70)
    print("Dataset Resplit Utility")
    print("=" * 70)

    # 1. Collect all image-annotation pairs
    print("\n1. Collecting all images and annotations...")
    all_pairs = []

    train_img_dir = data_root / "train_images"
    train_ann_dir = data_root / "train_annotations"
    val_img_dir = data_root / "valid_images"
    val_ann_dir = data_root / "valid_annotations"

    for source_img_dir, source_ann_dir in (
        (train_img_dir, train_ann_dir),
        (val_img_dir, val_ann_dir),
    ):
        if not source_img_dir.exists():
            continue
        for img_path in source_img_dir.glob("*.jpg"):
            ann_path = source_ann_dir / f"{img_path.stem}.xml"
            if ann_path.exists():
                all_pairs.append((img_path, ann_path))

    print(f"   Total samples found: {len(all_pairs)}")
    if len(all_pairs) == 0:
        raise RuntimeError(
            f"No image/annotation pairs found under {data_root}. "
            "Check that train_images/, train_annotations/, valid_images/, "
            "valid_annotations/ exist and contain matching .jpg/.xml files."
        )

    # 2. Extract dominant defect classes for stratification
    print("\n2. Extracting dominant defect class per image for stratification...")
    defect_classes = []
    class_counts = defaultdict(int)

    for _img_path, ann_path in all_pairs:
        defect_class = get_dominant_defect_class(ann_path)
        defect_classes.append(defect_class)
        class_counts[defect_class] += 1

    print("   Class distribution (by dominant class):")
    for cls, count in sorted(class_counts.items()):
        print(f"   - {cls:<20}: {count:>4} samples")

    # 3. Perform stratified split
    print(f"\n3. Performing stratified split ({train_ratio:.0%}/{1 - train_ratio:.0%})...")
    train_pairs, val_pairs = train_test_split(
        all_pairs,
        test_size=1 - train_ratio,
        stratify=defect_classes,
        random_state=seed,
    )

    print(f"   Train set: {len(train_pairs)} samples")
    print(f"   Val set:   {len(val_pairs)} samples")

    # 4. Back up existing folders
    print("\n4. Creating backup of existing folders...")
    backup_dir = data_root / "backup_original_split"
    backup_dir.mkdir(exist_ok=True)

    for folder in ['train_images', 'train_annotations', 'valid_images', 'valid_annotations']:
        src = data_root / folder
        dst = backup_dir / folder
        if src.exists() and not dst.exists():
            shutil.copytree(src, dst)

    print(f"   Backup saved to: {backup_dir}")

    # 5. Clear existing folders
    print("\n5. Clearing existing train/val folders...")
    for folder in ['train_images', 'train_annotations', 'valid_images', 'valid_annotations']:
        folder_path = data_root / folder
        if folder_path.exists():
            shutil.rmtree(folder_path)
        folder_path.mkdir(exist_ok=True)

    # 6. Copy files to new split (single pass — the earlier duplicated block was removed)
    print("\n6. Copying files to new split...")

    for img_path, ann_path in train_pairs:
        img_src = backup_dir / img_path.parent.name / img_path.name
        ann_src = backup_dir / ann_path.parent.name / ann_path.name
        shutil.copy(img_src, train_img_dir / img_path.name)
        shutil.copy(ann_src, train_ann_dir / ann_path.name)

    for img_path, ann_path in val_pairs:
        img_src = backup_dir / img_path.parent.name / img_path.name
        ann_src = backup_dir / ann_path.parent.name / ann_path.name
        shutil.copy(img_src, val_img_dir / img_path.name)
        shutil.copy(ann_src, val_ann_dir / ann_path.name)

    # 7. Verify new split
    print("\n7. Verifying new split...")
    train_count = len(list(train_img_dir.glob("*.jpg")))
    val_count = len(list(val_img_dir.glob("*.jpg")))

    print(f"   Train images: {train_count}")
    print(f"   Val images:   {val_count}")
    print(f"   Total:        {train_count + val_count}")

    # 8. Class distribution in new split
    print("\n8. Class distribution in new split:")

    train_classes = defaultdict(int)
    for ann_path in train_ann_dir.glob("*.xml"):
        train_classes[get_dominant_defect_class(ann_path)] += 1

    val_classes = defaultdict(int)
    for ann_path in val_ann_dir.glob("*.xml"):
        val_classes[get_dominant_defect_class(ann_path)] += 1

    print("\n   TRAIN SET:")
    for cls in sorted(train_classes.keys()):
        print(f"   - {cls:<20}: {train_classes[cls]:>4} samples")

    print("\n   VAL SET:")
    for cls in sorted(val_classes.keys()):
        print(f"   - {cls:<20}: {val_classes[cls]:>4} samples")

    print("\n" + "=" * 70)
    print("Dataset resplit completed.")
    print("=" * 70)
    print(f"\nOriginal split backed up to: {backup_dir}")

    # 9. Verification checks
    print("\n" + "=" * 70)
    print("VERIFICATION CHECK")
    print("=" * 70)
    expected_train = int(len(all_pairs) * train_ratio)
    expected_val = len(all_pairs) - expected_train

    checks = [
        ("Train count", train_count == expected_train, f"{train_count} == {expected_train}"),
        ("Val count", val_count == expected_val, f"{val_count} == {expected_val}"),
        ("Train annotations", len(list(train_ann_dir.glob("*.xml"))) == train_count, "Images match annotations"),
        ("Val annotations", len(list(val_ann_dir.glob("*.xml"))) == val_count, "Images match annotations"),
        (
            "No overlap",
            set(p.name for p in train_img_dir.glob("*.jpg")).isdisjoint(
                set(p.name for p in val_img_dir.glob("*.jpg"))
            ),
            "No duplicate images",
        ),
    ]

    all_passed = True
    for check_name, passed, detail in checks:
        status = "PASS >>> OK" if passed else "FAIL >>> NOK"
        print(f"{status} - {check_name}: {detail}")
        if not passed:
            all_passed = False

    print("=" * 70)
    if all_passed:
        print("ALL CHECKS PASSED - Split is valid")
    else:
        print("SOME CHECKS FAILED - Review output above")
    print("=" * 70)


if __name__ == "__main__":
    import sys

    # Script lives at <project_root>/src/utils/split_dataset.py, so go up TWO
    # levels (src/utils/ -> src/ -> project root). The previous version used
    # Path(__file__).parent, which resolved to src/utils/ and produced the
    # wrong data_root.
    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / "data" / "raw"

    if not data_root.exists():
        print(f"Error: Data directory not found at {data_root}")
        print("Please adjust the path in the script.")
        sys.exit(1)

    resplit_dataset(data_root, train_ratio=0.80, seed=42)
