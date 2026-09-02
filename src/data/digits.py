import torch
from torchvision import datasets, transforms


def get_datasets(root="data/digits"):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

    mnist = datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=transform,
    )

    usps = datasets.USPS(
        root=root,
        train=True,
        download=True,
        transform=transform,
    )

    return mnist, usps
