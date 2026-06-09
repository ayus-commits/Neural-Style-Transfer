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

    step_counter = [0]

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

        step_counter[0] += 1
        step = step_counter[0]

        if step % 50 == 0 or step == 1:
            print(f"  Step {step:4d}/{num_steps} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Content: {c_loss.item():.4f} | "
                  f"Style: {s_loss.item():.6f}")
            
        if step % save_every == 0:
            _save_intermediate(generated, output_dir, step)
        return loss
    
    for _ in range(num_steps):
        optimizer.step(closure)

    print("-" * 50)
    print("Optimization complete!")
    return generated.detach()

def _save_intermediate(generated, output_dir, step):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"step_{step:04d}.jpg")
    save_image(generated.detach(), path)