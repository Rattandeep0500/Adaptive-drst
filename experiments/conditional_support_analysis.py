import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torchvision import datasets, transforms

from src.drl.faithful_trainer import FaithfulDRLTrainer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_all_data(dataset):
    xs = []
    ys = []

    for i in range(len(dataset)):
        x, y = dataset[i]

        xs.append(
            x.reshape(-1)
        )

        ys.append(y)

    return (
        torch.stack(xs),
        torch.tensor(
            ys,
            dtype=torch.long,
        ),
    )


def get_locked_batch(
    dataset,
    indices,
    device,
):
    subset_indices = dataset.indices
    base_dataset = dataset.dataset

    xs = []
    ys = []

    for index in indices:
        real_index = subset_indices[
            int(index)
        ]

        x, y = base_dataset[
            real_index
        ]

        xs.append(
            x.reshape(-1)
        )

        ys.append(y)

    return (
        torch.stack(xs).to(device),
        torch.tensor(
            ys,
            dtype=torch.long,
            device=device,
        ),
    )


def train_drl(
    trainer,
    source_dataset,
    target_dataset,
    locked_batches,
    device,
):
    for epoch in range(5):
        epoch_batches = [
            batch
            for batch in locked_batches
            if int(batch["epoch"]) == epoch
        ]

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

            trainer.train_step(
                source_x,
                source_y,
                target_x,
                epoch,
                int(batch["batch"]),
            )


def get_predictions(
    trainer,
    target_x,
    device,
):
    trainer.alpha.eval()
    trainer.beta.eval()

    probabilities = []

    with torch.no_grad():
        for start in range(
            0,
            target_x.size(0),
            128,
        ):
            x = target_x[
                start:start + 128
            ].to(device)

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

            dummy = torch.ones(
                x.size(0),
                10,
                device=device,
            )

            logits = trainer.alpha(
                x,
                dummy,
                ratio,
            )

            probabilities.append(
                F.softmax(
                    logits,
                    dim=1,
                ).cpu()
            )

    return torch.cat(
        probabilities,
        dim=0,
    )


def get_embeddings(
    trainer,
    x,
    device,
):
    trainer.alpha.eval()

    features = []

    with torch.no_grad():
        for start in range(
            0,
            x.size(0),
            128,
        ):
            batch = x[
                start:start + 128
            ].to(device)

            features.append(
                trainer.alpha.features(
                    batch
                ).cpu()
            )

    return torch.cat(
        features,
        dim=0,
    )


def compute_support_margin(
    source_features,
    source_labels,
    target_features,
    predicted_labels,
):
    source_features = F.normalize(
        source_features,
        dim=1,
    )

    target_features = F.normalize(
        target_features,
        dim=1,
    )

    similarities = (
        target_features
        @ source_features.t()
    )

    margins = []

    for i in range(
        target_features.size(0)
    ):
        predicted_class = int(
            predicted_labels[i]
        )

        same_class = (
            source_labels
            == predicted_class
        )

        other_class = (
            source_labels
            != predicted_class
        )

        same_score = similarities[
            i,
            same_class,
        ].max()

        other_score = similarities[
            i,
            other_class,
        ].max()

        margins.append(
            same_score - other_score
        )

    return torch.stack(
        margins
    )


def summarize(
    name,
    confidence,
    support,
    correctness,
    confidence_mask,
):
    selected_support = support[
        confidence_mask
    ]

    selected_correctness = correctness[
        confidence_mask
    ]

    n = selected_support.numel()

    if n == 0:
        print()
        print(name)
        print("No samples")
        return

    overall_error = (
        1.0
        - selected_correctness.float().mean().item()
    )

    order = torch.argsort(
        selected_support
    )

    low10_count = max(
        1,
        int(0.10 * n),
    )

    low20_count = max(
        1,
        int(0.20 * n),
    )

    low10 = order[
        :low10_count
    ]

    low20 = order[
        :low20_count
    ]

    low10_error = (
        1.0
        - selected_correctness[
            low10
        ].float().mean().item()
    )

    low20_error = (
        1.0
        - selected_correctness[
            low20
        ].float().mean().item()
    )

    print()
    print(name)
    print(
        f"Samples: {n}"
    )
    print(
        f"Overall Error: "
        f"{overall_error * 100:.2f}%"
    )
    print(
        f"Lowest-support 10% Error: "
        f"{low10_error * 100:.2f}%"
    )
    print(
        f"Lowest-support 20% Error: "
        f"{low20_error * 100:.2f}%"
    )
    print(
        f"Support Mean: "
        f"{selected_support.mean().item():.4f}"
    )
    print(
        f"Support Median: "
        f"{selected_support.median().item():.4f}"
    )
    print(
        f"Support 10th Percentile: "
        f"{torch.quantile(selected_support, 0.10).item():.4f}"
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

    train_drl(
        trainer,
        source_dataset,
        target_dataset,
        locked_batches,
        device,
    )

    source_x, source_y = get_all_data(
        source_dataset
    )

    target_x, target_y = get_all_data(
        target_dataset
    )

    probabilities = get_predictions(
        trainer,
        target_x,
        device,
    )

    confidence, predicted_labels = (
        probabilities.max(dim=1)
    )

    source_features = get_embeddings(
        trainer,
        source_x,
        device,
    )

    target_features = get_embeddings(
        trainer,
        target_x,
        device,
    )

    support_margin = compute_support_margin(
        source_features,
        source_y,
        target_features,
        predicted_labels,
    )

    correctness = (
        predicted_labels == target_y
    )

    thresholds = [
        0.70,
        0.80,
        0.90,
        0.95,
    ]

    print(
        f"Total Target Samples: "
        f"{len(target_dataset)}"
    )

    print(
        f"Overall Pseudo-label Accuracy: "
        f"{correctness.float().mean().item() * 100:.2f}%"
    )

    for threshold in thresholds:
        mask = confidence >= threshold

        summarize(
            f"Confidence >= {threshold:.2f}",
            confidence,
            support_margin,
            correctness,
            mask,
        )

    high_confidence = confidence >= 0.90

    high_support = (
        support_margin
        >= torch.quantile(
            support_margin[
                high_confidence
            ],
            0.20,
        )
    )

    combined = (
        high_confidence
        & high_support
    )

    summarize(
        "Confidence >= 0.90 AND support above bottom 20%",
        confidence,
        support_margin,
        correctness,
        combined,
    )

    high_confidence_indices = torch.where(
        high_confidence
    )[0]

    high_confidence_support = support_margin[
        high_confidence_indices
    ]

    high_confidence_correctness = correctness[
        high_confidence_indices
    ]

    order = torch.argsort(
        high_confidence_support
    )

    n = len(
        high_confidence_indices
    )

    low10_count = max(
        1,
        int(0.10 * n),
    )

    low20_count = max(
        1,
        int(0.20 * n),
    )

    print()
    print(
        "High-confidence conditional analysis"
    )

    print(
        f"Confidence >= 0.90 Samples: "
        f"{n}"
    )

    print(
        f"Confidence >= 0.90 Accuracy: "
        f"{high_confidence_correctness.float().mean().item() * 100:.2f}%"
    )

    print(
        f"Lowest-support 10% within confidence >= 0.90: "
        f"{high_confidence_correctness[order[:low10_count]].float().mean().item() * 100:.2f}%"
    )

    print(
        f"Lowest-support 20% within confidence >= 0.90: "
        f"{high_confidence_correctness[order[:low20_count]].float().mean().item() * 100:.2f}%"
    )


if __name__ == "__main__":
    main()