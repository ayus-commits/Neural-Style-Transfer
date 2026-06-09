#this file uses the trained CNN to perform style transfer on input images.

import torch
import os
from losses import content_loss, style_loss, total_loss
from utils import save_image


def run_style_transfer(
    model,
    content_tensor,
    style_tensor,
    alpha,
    beta,
    num_steps,
    save_every,
    output_dir,
    device,
    content_layer,
    style_layers,
):
    # Extract features from content and style images
    with torch.no_grad():
        content_features = model(content_tensor)[content_layer]
        style_features   = {layer: model(style_tensor)[layer] for layer in style_layers}

    # Initialize generated image as a copy of the content image
    generated = content_tensor.clone().requires_grad_(True).to(device)

    optimizer = torch.optim.Adam([generated], lr=0.01)

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()

        gen_features = model(generated)
        gen_content_features = gen_features[content_layer]
        gen_style_features   = {layer: gen_features[layer] for layer in style_layers}

        c_loss = content_loss(gen_content_features, content_features)
        s_loss = sum(style_loss(gen_style_features[layer], style_features[layer]) for layer in style_layers)
        t_loss = total_loss(c_loss, s_loss, alpha, beta)

        t_loss.backward()
        optimizer.step()

        if step % save_every == 0 or step == num_steps:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"generated_step_{step}.png")
            save_image(generated, output_path)
            print(f"Step [{step}/{num_steps}], Total Loss: {t_loss.item():.4f}, Saved: {output_path}")