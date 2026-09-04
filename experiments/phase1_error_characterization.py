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


def predict(
    trainer,
    target_x,
    device,
):
    trainer.alpha.eval()
    trainer.beta.eval()

    probabilities = []
    features = []

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

            probs = F.softmax(
                logits,
                dim=1,
            )

            feat = trainer.alpha.features(
                x
            )

            probabilities.append(
                probs.cpu()
            )

            features.append(
                feat.cpu()
            )

    return (
        torch.cat(
            probabilities,
            dim=0,
        ),
        torch.cat(
            features,
            dim=0,
        ),
    )


def threshold_report(
    confidence,
    correctness,
    predicted_labels,
    true_labels,
    threshold,
):
    mask = confidence >= threshold

    count = int(mask.sum())

    if count == 0:
        return

    selected_correct = correctness[
        mask
    ]

    accuracy = (
        selected_correct.float().mean().item()
    )

    coverage = (
        100.0
        * count
        / len(confidence)
    )

    error = 1.0 - accuracy

    print()
    print(
        f"Confidence >= {threshold:.2f}"
    )

    print(
        f"Samples: {count}"
    )

    print(
        f"Coverage: {coverage:.2f}%"
    )

    print(
        f"Accuracy: {accuracy * 100:.2f}%"
    )

    print(
        f"Error: {error * 100:.2f}%"
    )

    print(
        "Per-class results:"
    )

    for cls in range(10):
        class_mask = (
            mask
            & (predicted_labels == cls)
        )

        class_count = int(
            class_mask.sum()
        )

        if class_count == 0:
            continue

        class_accuracy = (
            correctness[
                class_mask
            ]
            .float()
            .mean()
            .item()
        )

        true_class_count = int(
            (
                mask
                & (true_labels == cls)
            ).sum()
        )

        print(
            f"Class {cls}: "
            f"n={class_count} | "
            f"pred-accuracy={class_accuracy * 100:.2f}% | "
            f"true-count={true_class_count}"
        )


def oracle_curve(
    confidence,
    correctness,
):
    order = torch.argsort(
        confidence,
        descending=True,
    )

    sorted_correctness = correctness[
        order
    ]

    fractions = [
        1.00,
        0.90,
        0.80,
        0.70,
        0.60,
        0.50,
        0.40,
        0.30,
        0.20,
        0.10,
    ]

    print()
    print(
        "Oracle confidence risk-coverage curve"
    )

    for fraction in fractions:
        count = max(
            1,
            int(
                len(sorted_correctness)
                * fraction
            ),
        )

        selected = sorted_correctness[
            :count
        ]

        accuracy = (
            selected.float()
            .mean()
            .item()
        )

        coverage = (
            100.0
            * count
            / len(sorted_correctness)
        )

        risk = 1.0 - accuracy

        print(
            f"Coverage {coverage:6.2f}% | "
            f"Accuracy {accuracy * 100:6.2f}% | "
            f"Risk {risk * 100:6.2f}%"
        )


def feature_error_analysis(
    features,
    correctness,
):
    features = F.normalize(
        features,
        dim=1,
    )

    correct_features = features[
        correctness
    ]

    incorrect_features = features[
        ~correctness
    ]

    print()
    print(
        "Feature-space error analysis"
    )

    print(
        f"Correct samples: "
        f"{correct_features.size(0)}"
    )

    print(
        f"Incorrect samples: "
        f"{incorrect_features.size(0)}"
    )

    if (
        correct_features.size(0) > 0
        and incorrect_features.size(0) > 0
    ):
        correct_center = (
            correct_features.mean(dim=0)
        )

        incorrect_center = (
            incorrect_features.mean(dim=0)
        )

        center_similarity = F.cosine_similarity(
            correct_center.unsqueeze(0),
            incorrect_center.unsqueeze(0),
        ).item()

        print(
            f"Correct/error centroid cosine: "
            f"{center_similarity:.4f}"
        )

    if incorrect_features.size(0) > 1:
        similarity = (
            incorrect_features
            @ incorrect_features.t()
        )

        mask = ~torch.eye(
            incorrect_features.size(0),
            dtype=torch.bool,
        )

        mean_similarity = (
            similarity[
                mask
            ]
            .mean()
            .item()
        )

        print(
            f"Mean error-error similarity: "
            f"{mean_similarity:.4f}"
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

    target_x, target_y = get_all_data(
        target_dataset
    )

    probabilities, features = predict(
        trainer,
        target_x,
        device,
    )

    confidence, predicted_labels = (
        probabilities.max(dim=1)
    )

    correctness = (
        predicted_labels == target_y
    )

    print(
        f"Total target samples: "
        f"{len(target_y)}"
    )

    print(
        f"Overall prediction accuracy: "
        f"{correctness.float().mean().item() * 100:.2f}%"
    )

    for threshold in [
        0.90,
        0.95,
        0.99,
    ]:
        threshold_report(
            confidence,
            correctness,
            predicted_labels,
            target_y,
            threshold,
        )

    oracle_curve(
        confidence,
        correctness,
    )

    feature_error_analysis(
        features,
        correctness,
    )


if __name__ == "__main__":
    main()