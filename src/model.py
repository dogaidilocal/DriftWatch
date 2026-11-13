"""Pretrained CIFAR-10 classifier, inference only.

The model is never trained or modified anywhere in this project.
"""
import numpy as np
import torch
import torch.nn as nn

from config import BATCH_SIZE, MODEL_NAME, NORM_MEAN, NORM_STD


def load_model(pretrained=True):
    """Load a pretrained CIFAR-10 classifier in eval mode on CPU.

    pretrained=False returns random weights and is for pipeline testing only;
    accuracy figures from it are meaningless.
    """
    if pretrained:
        model = torch.hub.load(
            "chenyaofo/pytorch-cifar-models", MODEL_NAME, pretrained=True, trust_repo=True
        )
    else:
        model = _TinyNet()
    model.eval()
    return model


def _preprocess(images):
    """uint8 (N, 32, 32, 3) -> normalised float tensor (N, 3, 32, 32)."""
    x = torch.from_numpy(np.ascontiguousarray(images)).float().div_(255.0)
    x = x.permute(0, 3, 1, 2)
    mean = torch.tensor(NORM_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(NORM_STD).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def predict(model, images, batch_size=BATCH_SIZE, return_probs=False):
    """Run inference over uint8 images.

    Returns predicted class indices, and softmax probabilities if asked.
    Probabilities are needed later for the ATC baseline.
    """
    preds, probs = [], []
    for i in range(0, len(images), batch_size):
        logits = model(_preprocess(images[i:i + batch_size]))
        preds.append(logits.argmax(dim=1).numpy())
        if return_probs:
            probs.append(torch.softmax(logits, dim=1).numpy())
    preds = np.concatenate(preds)
    if return_probs:
        return preds, np.concatenate(probs)
    return preds


def accuracy(model, images, labels, batch_size=BATCH_SIZE):
    """Top-1 accuracy as a float in [0, 1]."""
    preds = predict(model, images, batch_size=batch_size)
    return float((preds == labels).mean())


class _TinyNet(nn.Module):
    """Stand-in used only to smoke-test the pipeline without downloading weights."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))
