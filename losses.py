#this file contains loss functions for training CNN.

import torch
import torch.nn.functional as F


def content_loss(generated_features, content_features):
    return F.mse_loss(generated_features, content_features)


def gram_matrix(feature_map):
    B, C, H, W = feature_map.shape
    features = feature_map.view(B, C, H * W)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (C * H * W)


def style_loss(generated_features, style_features):
    gram_generated = gram_matrix(generated_features)
    gram_style     = gram_matrix(style_features)
    return F.mse_loss(gram_generated, gram_style)


def total_loss(content_loss_val, style_loss_val, alpha, beta):
    return alpha * content_loss_val + beta * style_loss_val