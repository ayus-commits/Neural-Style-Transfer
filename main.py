import torch
import os
import yaml
from utils import load_image, save_image, get_device
from cnn import FeatureExtractor
from style_transfer import run_style_transfer
import argparse


def main():
    print("=" * 60)
    print("  NEURAL STYLE TRANSFER — FROM SCRATCH")
    print("=" * 60)
    device = get_device()
    print(f"Using device: {device}")


    if not os.path.exists(CONTENT_IMAGE_PATH):
        print(f"\nERROR: Content image not found at '{CONTENT_IMAGE_PATH}'")
        # print("Please put a JPEG image at that path and try again.")
        return

    if not os.path.exists(STYLE_IMAGE_PATH):
        print(f"\nERROR: Style image not found at '{STYLE_IMAGE_PATH}'")
        # print("Please put a JPEG image at that path and try again.")
        return

#Load content and style images
    print(f"\nLoading images...")
    content_tensor = load_image(CONTENT_IMAGE_PATH, size=IMAGE_SIZE).to(device)
    *_, dynamic_height, dynamic_width = content_tensor.shape
    style_tensor   = load_image(STYLE_IMAGE_PATH,   size=(dynamic_height, dynamic_width)).to(device)
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
        print("  Run 'python train.py' first to train the feature extractor.")
        return

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Total CNN parameters: {num_params:,}")


    print(f"\nHyperparameters:")
    print(f"  Image size:     {IMAGE_SIZE}px")
    print(f"  Alpha (content): {ALPHA}")
    print(f"  Beta  (style):   {BETA:.0e}")
    print(f"  Gamma (variation): {GAMMA:.0e}")
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
        gamma          = GAMMA,
        num_steps      = NUM_STEPS,
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

    parser = argparse.ArgumentParser(description="Neural Style Transfer")
    parser.add_argument("--content", help="Path to the content image" ,required=True)
    parser.add_argument("--style", help="Path to the style image", required=True)
    parser.add_argument("--name", help="Name for the output image folder", required=True)

    parser.add_argument("--config", help="Which config file to use", default="default")
    parser.add_argument("--output-dir", help="Directory to save outputs", default="./outputs")
    parser.add_argument("--checkpoint", help="Checkpoint to use", default="imagenette")
    args = parser.parse_args()
    CONTENT_IMAGE_PATH = args.content
    STYLE_IMAGE_PATH   = args.style
    OUTPUT_DIR         = args.output_dir + "/" + args.name
    CHECKPOINT_PATH     = "./checkpoints/" + args.checkpoint + "_backbone.pth"

    with open("configs/" + args.config + ".yaml", "r") as f:
        config = yaml.safe_load(f)
    IMAGE_SIZE        = config["IMAGE_SIZE"]
    ALPHA             = config["ALPHA"]
    BETA              = config['BETA']
    GAMMA             = config['GAMMA']
    NUM_STEPS         = config["NUM_STEPS"]
    CONTENT_LAYER     = config["CONTENT_LAYER"]
    STYLE_LAYERS      = config["STYLE_LAYERS"]

    main()



#function for streamlit app to run style transfer and display results
def generate_style_transfer(
    content_path,
    style_path,
    output_name,
    config_name="default",
    output_dir="./outputs",
    checkpoint="imagenette",
    image_size=None,
    alpha=None,
    beta=None,
    gamma=None,
    num_steps=None
):
    global CONTENT_IMAGE_PATH
    global STYLE_IMAGE_PATH
    global OUTPUT_DIR
    global CHECKPOINT_PATH
    global IMAGE_SIZE
    global ALPHA
    global BETA
    global GAMMA
    global NUM_STEPS
    global CONTENT_LAYER
    global STYLE_LAYERS

    if config_name != "custom":
        with open(f"configs/{config_name}.yaml", "r") as f:
            config = yaml.safe_load(f)
        cfg_image_size = config["IMAGE_SIZE"]
        cfg_alpha = config["ALPHA"]
        cfg_beta = config["BETA"]
        cfg_gamma = config["GAMMA"]
        cfg_num_steps = config["NUM_STEPS"]
        cfg_content_layer = config["CONTENT_LAYER"]
        cfg_style_layers = config["STYLE_LAYERS"]
    else:
        cfg_image_size = image_size
        cfg_alpha = alpha
        cfg_beta = beta
        cfg_gamma = gamma
        cfg_num_steps = num_steps
        cfg_content_layer = "stage4"
        cfg_style_layers = ["stage1", "stage2", "stage3"]
    OUTPUT_DIR = os.path.join(output_dir, output_name)
    CHECKPOINT_PATH = f"./checkpoints/{checkpoint}_backbone.pth"
    CONTENT_IMAGE_PATH = content_path
    STYLE_IMAGE_PATH = style_path
    IMAGE_SIZE = cfg_image_size
    ALPHA = cfg_alpha
    BETA = cfg_beta
    GAMMA = cfg_gamma
    NUM_STEPS = cfg_num_steps
    CONTENT_LAYER = cfg_content_layer
    STYLE_LAYERS = cfg_style_layers

    main()
    final_path = os.path.join(OUTPUT_DIR, "final_output.jpg")
    return final_path