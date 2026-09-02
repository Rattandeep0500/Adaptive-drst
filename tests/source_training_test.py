import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.domain_adapter import ImageDRLModel
from experiments.visda.train_source import train_epoch


def main():
    torch.manual_seed(42)

    images = torch.randn(4, 3, 224, 224)
    labels = torch.randint(0, 12, (4,))

    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=2)

    model = ImageDRLModel(num_classes=12)

    optimizer = torch.optim.Adam(
        model.classifier.parameters(),
        lr=1e-4,
    )

    before = [
        p.detach().clone()
        for p in model.classifier.parameters()
    ]

    metrics = train_epoch(
        model,
        loader,
        optimizer,
        torch.device("cpu"),
    )

    changed = any(
        not torch.allclose(old, new.detach())
        for old, new in zip(
            before,
            model.classifier.parameters(),
        )
    )

    print("Loss:", metrics["loss"])
    print("Accuracy:", metrics["accuracy"])
    print("Parameters updated:", changed)

    assert changed

    print("Source training test OK")


if __name__ == "__main__":
    main()
