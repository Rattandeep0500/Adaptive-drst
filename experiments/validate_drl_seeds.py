import random
import numpy as np
import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms

from src.drl.faithful_trainer import AlphaDigits, FaithfulDRLTrainer


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


def evaluate_source_only(model, dataset, device, batch_size=128):
    model.eval()

    correct = 0
    total = 0

    for start in range(0, len(dataset), batch_size):
        indices = range(
            start,
            min(start + batch_size, len(dataset)),
        )

        xs = []
        ys = []

        for index in indices:
            x, y = dataset[index]
            xs.append(x.reshape(-1))
            ys.append(y)

        x = torch.stack(xs, dim=0).to(device)

        y = torch.tensor(
            ys,
            dtype=torch.long,
            device=device,
        )

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

        with torch.no_grad():
            logits = model(
                x,
                y_onehot,
                ratio,
            )

        correct += (
            logits.argmax(dim=1) == y
        ).sum().item()

        total += y.size(0)

    return 100.0 * correct / total


def evaluate_drl(trainer, dataset, device, batch_size=128):
    trainer.alpha.eval()

    correct = 0
    total = 0

    for start in range(0, len(dataset), batch_size):
        indices = range(
            start,
            min(start + batch_size, len(dataset)),
        )

        xs = []
        ys = []

        for index in indices:
            x, y = dataset[index]
            xs.append(x.reshape(-1))
            ys.append(y)

        x = torch.stack(xs, dim=0).to(device)

        y = torch.tensor(
            ys,
            dtype=torch.long,
            device=device,
        )

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

        with torch.no_grad():
            logits = trainer.alpha(
                x,
                y_onehot,
                ratio,
            )

        correct += (
            logits.argmax(dim=1) == y
        ).sum().item()

        total += y.size(0)

    return 100.0 * correct / total


def run_source_only(
    source_dataset,
    target_test,
    locked_batches,
    seed,
    device,
):
    set_seed(seed)

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
        epoch_batches = [
            batch
            for batch in locked_batches
            if int(batch["epoch"]) == epoch
        ]

        model.train()

        for batch in epoch_batches:
            source_x, source_y = get_batch(
                source_dataset,
                batch["source"],
                device,
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

            ratio = torch.ones(
                source_y.size(0),
                1,
                device=device,
            )

            logits = model(
                source_x,
                y_onehot,
                ratio,
            )

            loss = torch.sum(logits)

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            optimizer.step()

    return evaluate_source_only(
        model,
        target_test,
        device,
    )


def run_drl(
    source_dataset,
    target_dataset,
    target_test,
    locked_batches,
    seed,
    device,
):
    set_seed(seed)

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
        epoch_batches = [
            batch
            for batch in locked_batches
            if int(batch["epoch"]) == epoch
        ]

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

            trainer.train_step(
                source_x,
                source_y,
                target_x,
                epoch,
                int(batch["batch"]),
            )

    return evaluate_drl(
        trainer,
        target_test,
        device,
    )


def main():
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

    locked_batches = torch.load(
        "data/locked_digits_batches.pt",
        map_location="cpu",
    )

    seeds = [42, 43, 44, 45, 46]

    results = []

    for seed in seeds:
        source_acc = run_source_only(
            source_dataset,
            usps_test,
            locked_batches,
            seed,
            device,
        )

        drl_acc = run_drl(
            source_dataset,
            target_dataset,
            usps_test,
            locked_batches,
            seed,
            device,
        )

        gain = drl_acc - source_acc

        results.append(
            (
                seed,
                source_acc,
                drl_acc,
                gain,
            )
        )

        print(
            f"Seed {seed} | "
            f"Source-only {source_acc:.2f}% | "
            f"DRL {drl_acc:.2f}% | "
            f"Gain {gain:+.2f} pp"
        )

    source_values = np.array(
        [x[1] for x in results]
    )

    drl_values = np.array(
        [x[2] for x in results]
    )

    gain_values = np.array(
        [x[3] for x in results]
    )

    print()
    print(
        f"Source-only Mean: "
        f"{source_values.mean():.2f}% "
        f"+/- {source_values.std(ddof=1):.2f}"
    )

    print(
        f"DRL Mean: "
        f"{drl_values.mean():.2f}% "
        f"+/- {drl_values.std(ddof=1):.2f}"
    )

    print(
        f"Gain Mean: "
        f"{gain_values.mean():+.2f} pp "
        f"+/- {gain_values.std(ddof=1):.2f}"
    )


if __name__ == "__main__":
    main()