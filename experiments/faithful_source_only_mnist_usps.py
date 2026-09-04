import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.drl.faithful_trainer import AlphaDigits


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_batch(dataset, indices, device):
    subset_indices = dataset.indices
    base_dataset = dataset.dataset

    xs = []
    ys = []

    for index in indices:
        real_index = subset_indices[int(index)]
        x, y = base_dataset[real_index]
        xs.append(x.reshape(-1))
        ys.append(y)

    x = torch.stack(xs, dim=0).to(device)
    y = torch.tensor(
        ys,
        dtype=torch.long,
        device=device,
    )

    return x, y


def evaluate(model, loader, device, num_classes=10):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.reshape(x.size(0), -1).to(device)
            y = y.to(device)

            y_onehot = torch.zeros(
                x.size(0),
                num_classes,
                device=device,
            )

            y_onehot.scatter_(
                1,
                y.unsqueeze(1),
                1.0,
            )

            density_ratio = torch.ones(
                x.size(0),
                1,
                device=device,
            )

            logits = model(
                x,
                y_onehot,
                density_ratio,
            )

            pred = logits.argmax(dim=1)

            correct += (
                pred == y
            ).sum().item()

            total += y.size(0)

    return 100.0 * correct / total


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

    usps_train = datasets.USPS(
        root="data/digits",
        train=True,
        download=True,
        transform=transform,
    )

    usps_test = datasets.USPS(
        root="data/digits",
        train=False,
        download=True,
        transform=transform,
    )

    source_dataset = Subset(
        mnist,
        list(range(2000)),
    )

    target_dataset = Subset(
        usps_train,
        list(range(1800)),
    )

    source_eval_loader = DataLoader(
        source_dataset,
        batch_size=128,
        shuffle=False,
    )

    target_eval_loader = DataLoader(
        usps_test,
        batch_size=128,
        shuffle=False,
    )

    locked_batches = torch.load(
        "data/locked_digits_batches.pt",
        map_location="cpu",
    )

    model = AlphaDigits(
        input_dim=784,
        hidden_dim=256,
        num_classes=10,
    ).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-3,
        momentum=0.9,
        weight_decay=5e-4,
    )

    for epoch in range(5):
        model.train()

        epoch_batches = [
            batch
            for batch in locked_batches
            if int(batch["epoch"]) == epoch
        ]

        total_correct = 0
        total_samples = 0

        for batch in epoch_batches:
            source_x, source_y = get_batch(
                source_dataset,
                batch["source"],
                device,
            )

            if source_x.ndim != 2:
                raise RuntimeError(
                    f"Expected source_x to have shape [N, 784], got {tuple(source_x.shape)}"
                )

            if source_x.shape[1] != 784:
                raise RuntimeError(
                    f"Expected 784 features, got {source_x.shape[1]}"
                )

            y_onehot = torch.zeros(
                source_y.size(0),
                10,
                device=device,
            )

            y_onehot.scatter_(
                1,
                source_y.unsqueeze(1),
                1.0,
            )

            density_ratio = torch.ones(
                source_y.size(0),
                1,
                device=device,
            )

            logits = model(
                source_x,
                y_onehot,
                density_ratio,
            )

            loss = torch.sum(logits)

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            optimizer.step()

            total_correct += (
                logits.argmax(dim=1) == source_y
            ).sum().item()

            total_samples += source_y.size(0)

        source_batch_accuracy = (
            100.0
            * total_correct
            / total_samples
        )

        target_accuracy = evaluate(
            model,
            target_eval_loader,
            device,
        )

        print(
            f"Epoch {epoch + 1}/5 | "
            f"Source Accuracy "
            f"{source_batch_accuracy:.2f}% | "
            f"USPS Target Accuracy "
            f"{target_accuracy:.2f}%"
        )

    final_source_accuracy = evaluate(
        model,
        source_eval_loader,
        device,
    )

    final_target_accuracy = evaluate(
        model,
        target_eval_loader,
        device,
    )

    print()

    print(
        f"Final Source Accuracy: "
        f"{final_source_accuracy:.2f}%"
    )

    print(
        f"Final USPS Accuracy: "
        f"{final_target_accuracy:.2f}%"
    )


if __name__ == "__main__":
    main()