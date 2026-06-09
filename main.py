import torch
import os
import yaml
from utils import load_image, save_image, get_device
from cnn import FeatureExtractor
from style_transfer import run_style_transfer


with open("configs/default.yaml", "r") as f:
    config = yaml.safe_load(f)
CONTENT_IMAGE_PATH = config["CONTENT_IMAGE_PATH"]
STYLE_IMAGE_PATH   = config["STYLE_IMAGE_PATH"]
OUTPUT_DIR         = config["OUTPUT_DIR"]
CHECKPOINT_PATH     = config["CHECKPOINT_PATH"]
IMAGE_SIZE        = config["IMAGE_SIZE"]
ALPHA             = config["ALPHA"]
BETA              = config["BETA"]
NUM_STEPS         = config["NUM_STEPS"]
SAVE_EVERY        = config["SAVE_EVERY"]
CONTENT_LAYER     = config["CONTENT_LAYER"]
STYLE_LAYERS      = config["STYLE_LAYERS"]

def main():
    print("=" * 60)
    print("  NEURAL STYLE TRANSFER — FROM SCRATCH")
    print("=" * 60)
    device = get_device()
    print(f"Using device: {device}")


    if not os.path.exists(CONTENT_IMAGE_PATH):
        print(f"\nERROR: Content image not found at '{CONTENT_IMAGE_PATH}'")
        print("Please put a JPEG image at that path and try again.")
        return

    if not os.path.exists(STYLE_IMAGE_PATH):
        print(f"\nERROR: Style image not found at '{STYLE_IMAGE_PATH}'")
        print("Please put a JPEG image at that path and try again.")
        return

#Load content and style images
    print(f"\nLoading images...")
    content_tensor = load_image(CONTENT_IMAGE_PATH, size=IMAGE_SIZE).to(device)
    style_tensor   = load_image(STYLE_IMAGE_PATH,   size=IMAGE_SIZE).to(device)
    print(f"  Content image: {CONTENT_IMAGE_PATH}  →  tensor shape: {content_tensor.shape}")
    print(f"  Style image:   {STYLE_IMAGE_PATH}  →  tensor shape: {style_tensor.shape}")

#Initialize the feature extractor model
    model = FeatureExtractor().to(device)
    if os.path.exists(CHECKPOINT_PATH):
        print(f"  Loading pretrained weights from: {CHECKPOINT_PATH}")
        state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print("  Pretrained weights loaded successfully.")
    else:
        print(f"  WARNING: No checkpoint found at '{CHECKPOINT_PATH}'.")
        print("  The CNN has RANDOM weights — output quality will be poor.")
        print("  Run 'python classifier.py' first to train the feature extractor.")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Total CNN parameters: {num_params:,}")


    print(f"\nHyperparameters:")
    print(f"  Image size:     {IMAGE_SIZE}px")
    print(f"  Alpha (content): {ALPHA}")
    print(f"  Beta  (style):   {BETA:.0e}")
    print(f"  Steps:           {NUM_STEPS}")
    print(f"  Content layer:   {CONTENT_LAYER}")
    print(f"  Style layers:    {STYLE_LAYERS}")


    os.makedirs(OUTPUT_DIR, exist_ok=True)

#Run style transfer
    final_image = run_style_transfer(
        model          = model,
        content_tensor = content_tensor,
        style_tensor   = style_tensor,
        alpha          = ALPHA,
        beta           = BETA,
        num_steps      = NUM_STEPS,
        save_every     = SAVE_EVERY,
        output_dir     = OUTPUT_DIR,
        device         = device,
        content_layer  = CONTENT_LAYER,
        style_layers   = STYLE_LAYERS,
    )


    final_path = os.path.join(OUTPUT_DIR, "final_output.jpg")
    save_image(final_image, final_path)

    print(f"\nDone! Results saved to '{OUTPUT_DIR}/'")
    print(f"Final output: {final_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()