import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F


def train_epoch(
    model,
    loader,
    optimizer,
    device,
):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model.classifier(images)

        loss = F.cross_entropy(
            logits,
            labels,
        )

        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach())
        correct += int(
            (logits.argmax(dim=1) == labels).sum()
        )
        total += labels.size(0)

    return {
        "loss": total_loss / max(len(loader), 1),
        "accuracy": correct / max(total, 1),
    }
