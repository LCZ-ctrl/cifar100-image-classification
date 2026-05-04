from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import config

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                             std=[0.2675, 0.2565, 0.2761]),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                             std=[0.2675, 0.2565, 0.2761]),
    ])
}


def get_train_val_loaders():
    train_tf = data_transforms['train']
    val_tf = data_transforms['val']

    train_dataset = datasets.ImageFolder(root="data/processed/train", transform=train_tf)
    val_dataset = datasets.ImageFolder(root="data/processed/val", transform=val_tf)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=config.NUM_WORKERS, pin_memory=True)

    print(f"Training set sample: {len(train_dataset)}")
    print(f"Validation set sample: {len(val_dataset)}")
    return train_loader, val_loader


def get_test_loader():
    test_tf = data_transforms['val']
    test_dataset = datasets.ImageFolder(root="data/processed/test", transform=test_tf)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                             num_workers=config.NUM_WORKERS, pin_memory=True)
    print(f"Test set sample: {len(test_dataset)}")
    return test_loader
