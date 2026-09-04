import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision import datasets, transforms

from src.drl.faithful_trainer import BetaDigits


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()

    correct = 0
    total = 0
    source_probs = []
    target_probs = []

    with torch.no_grad():
        for x, y in loader:
            x = x.reshape(x.size(0), -1).to(device)
            y = y.to(device)

            logits = model(x)
            probs = F.softmax(logits, dim=1)

            pred = probs.argmax(dim=1)

            correct += (
                pred == y
            ).sum().item()

            total += y.size(0)

            source_mask = y == 0
            target_mask = y == 1

            if source_mask.any():
                source_probs.append(
                    probs[source_mask, 0].cpu()
                )

            if target_mask.any():
                target_probs.append(
                    probs[target_mask, 1].cpu()
                )

    accuracy = 100.0 * correct / total

    source_prob = torch.cat(source_probs)
    target_prob = torch.cat(target_probs)

    return (
        accuracy,
        source_prob,
        target_prob,
    )


def main():
    seed = 42
    set_seed(seed)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

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

    source_dataset = Subset(
        mnist,
        list(range(2000)),
    )

    target_dataset = Subset(
        usps,
        list(range(1800)),
    )

    source_domain_labels = torch.zeros(
        len(source_dataset),
        dtype=torch.long,
    )

    target_domain_labels = torch.ones(
        len(target_dataset),
        dtype=torch.long,
    )

    domain_dataset = torch.utils.data.TensorDataset(
        torch.cat([
            torch.stack([
                source_dataset[i][0]
                for i in range(len(source_dataset))
            ]),
            torch.stack([
                target_dataset[i][0]
                for i in range(len(target_dataset))
            ]),
        ]),
        torch.cat([
            source_domain_labels,
            target_domain_labels,
        ]),
    )

    domain_loader = DataLoader(
        domain_dataset,
        batch_size=16,
        shuffle=True,
        drop_last=True,
    )

    evaluation_loader = DataLoader(
        domain_dataset,
        batch_size=128,
        shuffle=False,
    )

    model = BetaDigits(
        input_dim=784,
        hidden_dim=256,
    ).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-3,
        momentum=0.9,
        weight_decay=5e-4,
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(5):
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in domain_loader:
            x = x.reshape(
                x.size(0),
                -1,
            ).to(device)

            y = y.to(device)

            logits = model(x)

            loss = loss_fn(
                logits,
                y,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            optimizer.step()

            total_loss += float(
                loss.detach()
            )

            correct += (
                logits.argmax(dim=1) == y
            ).sum().item()

            total += y.size(0)

        epoch_accuracy = (
            100.0 * correct / total
        )

        print(
            f"Epoch {epoch + 1}/5 | "
            f"Domain Loss {total_loss / len(domain_loader):.4f} | "
            f"Domain Accuracy {epoch_accuracy:.2f}%"
        )

    accuracy, source_prob, target_prob = evaluate(
        model,
        evaluation_loader,
        device,
    )

    source_prob = source_prob.clamp(
        min=1e-8,
        max=1.0,
    )

    target_prob = target_prob.clamp(
        min=1e-8,
        max=1.0,
    )

    source_ratio = (
        source_prob / (1.0 - source_prob)
    )

    target_ratio = (
        (1.0 - target_prob) / target_prob
    )

    print()
    print(
        f"Final Domain Accuracy: "
        f"{accuracy:.2f}%"
    )

    print(
        f"Source P(domain=source) Mean: "
        f"{source_prob.mean().item():.4f}"
    )

    print(
        f"Target P(domain=target) Mean: "
        f"{target_prob.mean().item():.4f}"
    )

    print(
        f"Source Ratio Mean: "
        f"{source_ratio.mean().item():.4f}"
    )

    print(
        f"Source Ratio Median: "
        f"{source_ratio.median().item():.4f}"
    )

    print(
        f"Source Ratio Min: "
        f"{source_ratio.min().item():.4f}"
    )

    print(
        f"Source Ratio Max: "
        f"{source_ratio.max().item():.4f}"
    )

    print(
        f"Target Inverse Ratio Mean: "
        f"{target_ratio.mean().item():.4f}"
    )


if __name__ == "__main__":
    main()