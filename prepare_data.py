import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm


def prepare_data(
        raw_train_dir: str = "data/cifar100/train",
        raw_test_dir: str = "data/cifar100/test",
        processed_dir: str = "data/processed",
        val_ratio: float = 0.1,
        seed: int = 42
):
    random.seed(seed)
    raw_train_path = Path(raw_train_dir)
    raw_test_path = Path(raw_test_dir)
    processed_path = Path(processed_dir)

    classes = sorted([d for d in os.listdir(raw_train_path) if os.path.isdir(raw_train_path / d)])
    print(f"✅ Found {len(classes)} categories: {classes}")

    for split in ['train', 'val', 'test']:
        for category in classes:
            (processed_path / split / category).mkdir(parents=True, exist_ok=True)

    train_files = {}
    val_files = {}
    test_files = {}

    total_files = 0

    # Train + Val
    for category in classes:
        img_list = [f for f in os.listdir(raw_train_path / category)
                    if f.lower().endswith('.png')]
        random.shuffle(img_list)
        val_size = int(len(img_list) * val_ratio)

        train_files[category] = img_list[val_size:]
        val_files[category] = img_list[:val_size]

        total_files += len(train_files[category]) + len(val_files[category])

    # Test
    for category in classes:
        img_list = [f for f in os.listdir(raw_test_path / category)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        test_files[category] = img_list
        total_files += len(img_list)

    pbar = tqdm(total=total_files, desc="🚀 Preparing data", unit="file")

    for category in classes:
        copy_files(train_files[category], raw_train_path / category,
                   processed_path / 'train' / category, pbar)
        copy_files(val_files[category], raw_train_path / category,
                   processed_path / 'val' / category, pbar)

    for category in classes:
        copy_files(test_files[category], raw_test_path / category,
                   processed_path / 'test' / category, pbar)

    pbar.close()

    train_total = sum(len(list((processed_path / 'train' / c).glob('*'))) for c in classes)
    val_total = sum(len(list((processed_path / 'val' / c).glob('*'))) for c in classes)
    test_total = sum(len(list((processed_path / 'test' / c).glob('*'))) for c in classes)

    tqdm.write(f"🎉 Data preparation complete!")
    tqdm.write(f"Training set: {train_total} images")
    tqdm.write(f"Validation set: {val_total} images")
    tqdm.write(f"Test set: {test_total} images")


def copy_files(file_list, src_dir, dst_dir, pbar=None):
    for file in file_list:
        shutil.copy2(src_dir / file, dst_dir / file)
        if pbar is not None:
            pbar.update(1)


if __name__ == "__main__":
    prepare_data()
