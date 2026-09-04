import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms

from src.drl.faithful_trainer import FaithfulDRLTrainer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_sample(dataset, index, device):
    x, y = dataset[int(index)]

    x = x.reshape(-1).unsqueeze(0).to(device)

    y = torch.tensor(
        [y],
        dtype=torch.long,
        device=device,
    )

    return x, y


def get_locked_batch(dataset, indices, device):
    xs = []
    ys = []

    subset_indices = dataset.indices
    base_dataset = dataset.dataset

    for index in indices:
        real_index = subset_indices[int(index)]
        x, y = base_dataset[real_index]

        xs.append(
            x.reshape(-1)
        )

        ys.append(y)

    x = torch.stack(
        xs,
        dim=0,
    ).to(device)

    y = torch.tensor(
        ys,
        dtype=torch.long,
        device=device,
    )

    return x, y


def evaluate(trainer, dataset, device):
    trainer.alpha.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for start in range(
            0,
            len(dataset),
            128,
        ):
            xs = []
            ys = []

            end = min(
                start + 128,
                len(dataset),
            )

            for i in range(start, end):
                x, y = dataset[i]
                xs.append(
                    x.reshape(-1)
                )
                ys.append(y)

            x = torch.stack(
                xs,
                dim=0,
            ).to(device)

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


def predict_unlabeled(
    trainer,
    target_dataset,
    remaining_indices,
    device,
):
    trainer.alpha.eval()
    trainer.beta.eval()

    if not remaining_indices:
        return (
            torch.empty(0, 10),
            torch.empty(0),
        )

    probabilities = []
    ratios = []

    with torch.no_grad():
        for start in range(
            0,
            len(remaining_indices),
            128,
        ):
            batch_indices = remaining_indices[
                start:start + 128
            ]

            xs = []

            for index in batch_indices:
                x, _ = target_dataset[
                    int(index)
                ]

                xs.append(
                    x.reshape(-1)
                )

            x = torch.stack(
                xs,
                dim=0,
            ).to(device)

            domain_logits = trainer.beta(x)

            domain_probs = F.softmax(
                domain_logits,
                dim=1,
            )

            ratio = (
                domain_probs[:, 0:1]
                /
                domain_probs[:, 1:2].clamp_min(
                    1e-8
                )
            )

            dummy_labels = torch.ones(
                x.size(0),
                10,
                device=device,
            )

            logits = trainer.alpha(
                x,
                dummy_labels,
                ratio,
            )

            probs = F.softmax(
                logits,
                dim=1,
            )

            probabilities.append(
                probs.cpu()
            )

            ratios.append(
                ratio.squeeze(1).cpu()
            )

    return (
        torch.cat(probabilities, dim=0),
        torch.cat(ratios, dim=0),
    )


def select_pseudo_labels(
    probabilities,
    portion,
    min_confidence,
):
    if probabilities.numel() == 0:
        return (
            torch.empty(
                0,
                dtype=torch.long,
            ),
            torch.empty(
                0,
                dtype=torch.long,
            ),
            torch.empty(0),
        )

    confidence, labels = probabilities.max(
        dim=1
    )

    valid = torch.where(
        confidence >= min_confidence
    )[0]

    target_count = int(
        probabilities.size(0) * portion
    )

    target_count = min(
        target_count,
        valid.numel(),
    )

    if target_count == 0:
        return (
            torch.empty(
                0,
                dtype=torch.long,
            ),
            torch.empty(
                0,
                dtype=torch.long,
            ),
            torch.empty(0),
        )

    ranked = valid[
        torch.argsort(
            confidence[valid],
            descending=True,
        )
    ]

    selected = ranked[
        :target_count
    ]

    return (
        selected,
        labels[selected],
        confidence[selected],
    )


def train_pseudo_batch(
    trainer,
    target_dataset,
    indices,
    labels,
    ratios,
    device,
):
    if len(indices) == 0:
        return 0.0

    trainer.alpha.train()

    order = torch.randperm(
        len(indices)
    )

    indices = indices[order]
    labels = labels[order]
    ratios = ratios[order]

    total_correct = 0
    total_samples = 0

    batch_size = 16

    for start in range(
        0,
        len(indices),
        batch_size,
    ):
        batch_indices = indices[
            start:start + batch_size
        ]

        batch_labels = labels[
            start:start + batch_size
        ].to(device)

        batch_ratios = ratios[
            start:start + batch_size
        ].to(device).reshape(-1, 1)

        xs = []

        for index in batch_indices:
            x, _ = target_dataset[
                int(index)
            ]

            xs.append(
                x.reshape(-1)
            )

        x = torch.stack(
            xs,
            dim=0,
        ).to(device)

        y_onehot = torch.zeros(
            batch_labels.size(0),
            10,
            device=device,
        )

        y_onehot.scatter_(
            1,
            batch_labels.unsqueeze(1),
            1.0,
        )

        logits = trainer.alpha(
            x,
            y_onehot,
            batch_ratios,
        )

        loss = torch.sum(logits)

        trainer.optimizer_alpha.zero_grad(
            set_to_none=True
        )

        loss.backward()

        trainer.optimizer_alpha.step()

        total_correct += (
            logits.argmax(dim=1)
            == batch_labels
        ).sum().item()

        total_samples += (
            batch_labels.size(0)
        )

    return (
        100.0
        * total_correct
        / total_samples
        if total_samples > 0
        else 0.0
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

    target_test_dataset = Subset(
        usps_test,
        list(range(len(usps_test))),
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

    remaining_indices = list(
        range(len(target_dataset))
    )

    pseudo_indices = []
    pseudo_labels = []
    pseudo_ratios = []

    portion = 0.20
    min_confidence = 0.70

    for epoch in range(5):
        epoch_batches = [
            batch
            for batch in locked_batches
            if int(batch["epoch"]) == epoch
        ]

        metrics = []

        for batch in epoch_batches:
            source_x, source_y = get_locked_batch(
                source_dataset,
                batch["source"],
                device,
            )

            target_x, _ = get_locked_batch(
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

        probabilities, ratios = predict_unlabeled(
            trainer,
            target_dataset,
            remaining_indices,
            device,
        )

        selected_positions, selected_labels, selected_conf = (
            select_pseudo_labels(
                probabilities,
                portion,
                min_confidence,
            )
        )

        selected_target_indices = [
            remaining_indices[int(position)]
            for position in selected_positions
        ]

        selected_target_indices = torch.tensor(
            selected_target_indices,
            dtype=torch.long,
        )

        selected_ratios = ratios[
            selected_positions
        ]

        pseudo_train_acc = train_pseudo_batch(
            trainer,
            target_dataset,
            selected_target_indices,
            selected_labels,
            selected_ratios,
            device,
        )

        selected_set = set(
            selected_target_indices.tolist()
        )

        remaining_indices = [
            index
            for index in remaining_indices
            if index not in selected_set
        ]

        pseudo_indices.extend(
            selected_target_indices.tolist()
        )

        pseudo_labels.extend(
            selected_labels.tolist()
        )

        pseudo_ratios.extend(
            selected_ratios.tolist()
        )

        source_accuracy = evaluate(
            trainer,
            source_dataset,
            device,
        )

        target_accuracy = evaluate(
            trainer,
            target_test_dataset,
            device,
        )

        domain_loss = np.mean([
            m["domain_loss"]
            for m in metrics
        ])

        ratio_mean = np.mean([
            m["ratio_mean"]
            for m in metrics
        ])

        mean_confidence = (
            selected_conf.mean().item()
            if selected_conf.numel() > 0
            else 0.0
        )

        coverage = (
            100.0
            * len(pseudo_indices)
            / len(target_dataset)
        )

        print(
            f"Epoch {epoch + 1}/5 | "
            f"Domain Loss {domain_loss:.4f} | "
            f"Ratio {ratio_mean:.4f} | "
            f"Selected {len(selected_target_indices)} | "
            f"Remaining {len(remaining_indices)} | "
            f"Total Pseudo {len(pseudo_indices)} | "
            f"Coverage {coverage:.2f}% | "
            f"Confidence {mean_confidence:.4f} | "
            f"Pseudo Train Acc {pseudo_train_acc:.2f}% | "
            f"Source Acc {source_accuracy:.2f}% | "
            f"USPS Acc {target_accuracy:.2f}%"
        )

    final_source = evaluate(
        trainer,
        source_dataset,
        device,
    )

    final_target = evaluate(
        trainer,
        target_test_dataset,
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

    print(
        f"Total Unique Pseudo-labels: "
        f"{len(pseudo_indices)}"
    )

    print(
        f"Remaining Target Samples: "
        f"{len(remaining_indices)}"
    )


if __name__ == "__main__":
    main()