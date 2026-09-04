import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(42)

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

    mnist = datasets.MNIST(
        root="data/digits",
        train=True,
        download=True,
        transform=transform,
    )

    usps = datasets.USPS(
        root="data/digits",
        train=True,
        download=True,
        transform=transform,
    )

    source = Subset(mnist, list(range(2000)))
    target = Subset(usps, list(range(1800)))

    source_loader = DataLoader(
        source,
        batch_size=16,
        shuffle=True,
        drop_last=True,
    )

    target_loader = DataLoader(
        target,
        batch_size=16,
        shuffle=True,
        drop_last=True,
    )

    batches = []

    for epoch in range(5):
        source_order = list(source_loader.sampler)
        target_order = list(target_loader.sampler)

        steps = min(
            len(source_order) // 16,
            len(target_order) // 16,
        )

        for i in range(steps):
            batches.append({
                "epoch": epoch,
                "batch": i,
                "source": source_order[
                    i * 16:(i + 1) * 16
                ],
                "target": target_order[
                    i * 16:(i + 1) * 16
                ],
            })

    torch.save(
        batches,
        "data/locked_digits_batches.pt",
    )

    print(
        f"Saved {len(batches)} locked batches"
    )


if __name__ == "__main__":
    main()