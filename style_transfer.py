#this file uses the trained CNN to perform style transfer on input images.

import torch
import os
from losses import content_loss, style_loss, total_loss
from utils import save_image
import cv2
from tqdm import tqdm

def run_style_transfer(
    model,
    content_tensor,
    style_tensor,
    alpha,
    beta,
    num_steps,
    output_dir,
    device,
    content_layer,
    style_layers,
):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    with torch.no_grad():
        content_features = model(content_tensor)
        style_features   = model(style_tensor)

    content_target = content_features[content_layer]
    style_targets = {layer: style_features[layer] for layer in style_layers}

    generated = content_tensor.clone()
    generated.requires_grad_(True)
    optimizer = torch.optim.LBFGS([generated], lr=1.0)

    print(f"\nStarting optimization for {num_steps} steps...")
    print(f"Content layer: {content_layer}")
    print(f"Style layers: {style_layers}")
    print("-" * 50)

    latest_losses = {"content": None, "style": None, "total": None}

    def closure():
    
        with torch.no_grad():
            generated.clamp_(-3.0, 3.0)

        optimizer.zero_grad()
        generated_features = model(generated)
        c_loss = content_loss(
            generated_features[content_layer],
            content_target
        )

        s_loss = torch.tensor(0.0, device=device)
        for layer in style_layers:
            s_loss = s_loss + style_loss(
                generated_features[layer],
                style_targets[layer]
            )

        loss = total_loss(c_loss, s_loss, alpha, beta)
        loss.backward()

        latest_losses["content"] = c_loss.item()
        latest_losses["style"] = s_loss.item()
        latest_losses["total"] = loss.item()
        return loss
    
    pbar = tqdm(range(1, num_steps + 1), desc="Stylizing Image", unit="step")

    for step in pbar:
        optimizer.step(closure)

        pbar.set_postfix(
            {
                "Loss": f"{latest_losses['total']:.2f}",
                "Content": f"{latest_losses['content']:.2f}",
                "Style": f"{latest_losses['style']:.4f}",
            }
        )

        _save_intermediate(generated, output_dir, step)
        save_image(generated, os.path.join(output_dir, "latest.jpg"))
        img_to_show = cv2.imread(os.path.join(output_dir, "latest.jpg"))
        cv2.putText(
            img_to_show,
            f"Step {step}/{num_steps}  Loss={latest_losses['total']:.4f}",
            (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255,255,255),
            1
        )
        cv2.imshow(f"Live Output", img_to_show)
        cv2.waitKey(200)

    print("-" * 50)
    print("Optimization complete!")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return generated.detach()

def _save_intermediate(generated, output_dir, step):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"step_{step:03d}.jpg")
    save_image(generated.detach(), path)