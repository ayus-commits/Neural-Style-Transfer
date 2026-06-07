# This file handles moving images in and out of PyTorch.

import torch
from torchvision import transforms
from PIL import Image

import yaml

with open("configs/default.yaml", "r") as f:
    config = yaml.safe_load(f)
MEAN = config["MEAN"]
STD = config["STD"]

def load_image(path, size=None):
    img = Image.open(path).convert("RGB")
    steps = []
    if size is not None:
        steps.append(transforms.Resize(size))
    steps.append(transforms.ToTensor())
    steps.append(transforms.Normalize(mean=MEAN, std=STD))
    transform = transforms.Compose(steps)
    return transform(img).unsqueeze(0)

def save_image(tensor, path):
    tensor = tensor.squeeze(0).cpu()
    mean_t = torch.tensor(MEAN).view(-1, 1, 1)
    std_t = torch.tensor(STD).view(-1, 1, 1)
    tensor = tensor * std_t + mean_t
    tensor = torch.clamp(tensor, 0, 1)
    to_pil = transforms.ToPILImage()
    img = to_pil(tensor)
    img.save(path)
    print(f"Saved image to: {path}")

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")  #for macOS
    else:
        return torch.device("cpu")