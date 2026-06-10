#This file train CNN on dataset and saves the model.

import argparse

import torch
import torch.nn as nn
import os

from cnn import Classifier
from dataset import get_dataloaders_imagenette, get_dataloaders_stl10
from utils import get_device

from tqdm import tqdm
import yaml

parser = argparse.ArgumentParser(description="Neural Style Transfer")
# parser.add_argument("--data-dir", help="Path to the STL-10 dataset", default="./data")
parser.add_argument("--data", help="Dataset to use", required=True)
# parser.add_argument("--config", help="Which config file to use", default="default")
args = parser.parse_args()
DATASET = args.data


with open("configs/default.yaml", "r") as f:
    config = yaml.safe_load(f)
BATCH_SIZE     = config["BATCH_SIZE"]
NUM_EPOCHS     = config["NUM_EPOCHS"]
LEARNING_RATE  = config["LEARNING_RATE"]


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss    = 0
    correct       = 0
    total_samples = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)
    
    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    return avg_loss, accuracy


def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss    = 0
    correct       = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    
    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    return avg_loss, accuracy


def save_backbone(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.backbone.state_dict(), path)
    print(f"\nBackbone weights saved to: {path}")
    print("You can now run main.py to use these weights for NST.")


def main():
    print("=" * 60)
    print("  STEP 1: TRAIN CNN")
    print("=" * 60)
    print("This trains the feature extractor so its weights are meaningful.")
    print("Run this ONCE. Then run main.py for style transfer.\n")

    device = get_device()
    print(f"Using device: {device}\n")
    if DATASET.lower() == "stl10":
        print("Loading STL-10 dataset (downloads ~2.5 GB on first run)...")
        train_loader, val_loader = get_dataloaders_stl10(
            data_dir    = "./data",
            batch_size  = BATCH_SIZE,
            num_workers = 2
        )
        CHECKPOINT_PATH = "./checkpoints/stl10_backbone.pth"
    elif DATASET.lower() == "imagenette":
        print("Loading Imagenette dataset...(1.56 GB on first run)")
        train_loader, val_loader = get_dataloaders_imagenette(
            data_dir    = "./data",
            batch_size  = BATCH_SIZE,
            num_workers = 2
        )
        CHECKPOINT_PATH = "./checkpoints/imagenette_backbone.pth"
    model = Classifier(num_classes=10).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=8, gamma=0.5
    )

    best_val_accuracy = 0.0
    print(f"\nTraining for {NUM_EPOCHS} epochs...")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7} | {'LR':>8}")
    print("-" * 65)

    for epoch in range(1, NUM_EPOCHS + 1):

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device
        )

        val_loss, val_acc = validate(
            model, val_loader, loss_fn, device
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>8.1%} | "
              f"{val_loss:>8.4f} | {val_acc:>6.1%} | {current_lr:>8.2e}")

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            save_backbone(model, CHECKPOINT_PATH)
            print(f"          ↑ New best val accuracy: {val_acc:.1%}")

    print("\n" + "=" * 60)
    print(f"Training complete! Best validation accuracy: {best_val_accuracy:.1%}")
    print(f"Backbone weights saved to: {CHECKPOINT_PATH}")
    print("Run  python main.py  to start style transfer.")
    print("=" * 60)


if __name__ == "__main__":
    main()
