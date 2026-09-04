import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.drl.faithful_trainer import FaithfulDRLTrainer


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


def evaluate(trainer, loader, device):
    trainer.alpha.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.reshape(
                x.size(0),
                -1,
            ).to(device)

            y = y.to(device)

            y_onehot = torch.zeros(
                x.size(0),
                10,
                device=device,
            )

            y_onehot.scatter_(
                1,
                y.unsqueeze(1),
                1.0,
            )

            ratio = torch.ones(
                x.size(0),
                1,
                device=device,
            )

            logits = trainer.alpha(
                x,
                y_onehot,
                ratio,
            )

            pred = logits.argmax(
                dim=1
            )

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

    trainer = FaithfulDRLTrainer(
        input_dim=784,
        hidden_dim=256,
        num_classes=10,
        lr=1e-3,
        beta_lr=1e-3,
        momentum=0.9,
        weight_decay=5e-4,
        device=device,
    )

    for epoch in range(5):
        trainer.alpha.train()
        trainer.beta.train()

        epoch_batches = [
            batch
            for batch in locked_batches
            if int(batch["epoch"]) == epoch
        ]

        metrics = []

        for batch in epoch_batches:
            source_x, source_y = get_batch(
                source_dataset,
                batch["source"],
                device,
            )

            target_x, _ = get_batch(
                target_dataset,
                batch["target"],
                device,
            )

            result = trainer.train_step(
                source_x,
                source_y,
                target_x,
                epoch,
                int(batch["batch"]),
            )

            metrics.append(result)

        domain_loss = np.mean([
            m["domain_loss"]
            for m in metrics
        ])

        source_accuracy = np.mean([
            m["source_accuracy"]
            for m in metrics
        ])

        ratio_mean = np.mean([
            m["ratio_mean"]
            for m in metrics
        ])

        ratio_median = np.mean([
            m["ratio_median"]
            for m in metrics
        ])

        ratio_min = np.min([
            m["ratio_min"]
            for m in metrics
        ])

        ratio_max = np.max([
            m["ratio_max"]
            for m in metrics
        ])

        target_accuracy = evaluate(
            trainer,
            target_eval_loader,
            device,
        )

        print(
            f"Epoch {epoch + 1}/5 | "
            f"Domain Loss {domain_loss:.4f} | "
            f"Source Batch Acc {source_accuracy * 100:.2f}% | "
            f"Target Acc {target_accuracy:.2f}% | "
            f"Ratio Mean {ratio_mean:.4f} | "
            f"Ratio Median {ratio_median:.4f} | "
            f"Ratio Min {ratio_min:.4f} | "
            f"Ratio Max {ratio_max:.4f}"
        )

    final_source = evaluate(
        trainer,
        source_eval_loader,
        device,
    )

    final_target = evaluate(
        trainer,
        target_eval_loader,
        device,
    )

    print()
    print(
        f"Final Source Accuracy: "
        f"{final_source:.2f}%"
    )

    print(
        f"Final USPS Accuracy: "
        f"{final_target:.2f}%"
    )


if __name__ == "__main__":
    main()