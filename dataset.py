# This file handles loading STL-10 for training our CNN.

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import yaml

with open("configs/default.yaml", "r") as f:
    config = yaml.safe_load(f)
MEAN = config["MEAN"]
STD = config["STD"]

train_transform_stl10 = transforms.Compose([
    transforms.RandomCrop(88),
    transforms.Resize(96),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])
val_transform_stl10 = transforms.Compose([
    transforms.Resize(96),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

def get_dataloaders_stl10(data_dir="./data", batch_size=64, num_workers=2):
    train_dataset = datasets.STL10(root=data_dir, split="train", download=True, transform=train_transform_stl10)
    val_dataset = datasets.STL10(root=data_dir, split="test", download=True, transform=val_transform_stl10)

    print(f"Training samples:   {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader




train_transform_imagenette = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])
val_transform_imagenette = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

def get_dataloaders_imagenette(data_dir="./data", batch_size=64, num_workers=2):
    train_dataset = datasets.Imagenette(root=data_dir, split="train", size="full", download=True, transform=train_transform_imagenette)
    val_dataset = datasets.Imagenette(root=data_dir, split="val", size="full", download=True, transform=val_transform_imagenette)

    print(f"Training samples:   {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader