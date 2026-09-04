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
        xs.append(x.reshape(-1))
        ys.append(y)

    return (
        torch.stack(xs),
        torch.tensor(ys, dtype=torch.long),
    )


def get_locked_batch(dataset, indices, device):
    subset_indices = dataset.indices
    base_dataset = dataset.dataset

    xs = []
    ys = []

    for index in indices:
        real_index = subset_indices[int(index)]
        x, y = base_dataset[real_index]

        xs.append(x.reshape(-1))
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


def get_drl_predictions(
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


def get_source_embedding(
    trainer,
    source_x,
    device,
):
    trainer.alpha.eval()

    features = []

    with torch.no_grad():
        for start in range(
            0,
            source_x.size(0),
            128,
        ):
            x = source_x[
                start:start + 128
            ].to(device)

            features.append(
                trainer.alpha.features(x).cpu()
            )

    return torch.cat(
        features,
        dim=0,
    )


def get_target_embedding(
    trainer,
    target_x,
    device,
):
    trainer.alpha.eval()

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

            features.append(
                trainer.alpha.features(x).cpu()
            )

    return torch.cat(
        features,
        dim=0,
    )


def class_support_margin(
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

    scores = []

    for i in range(
        target_features.size(0)
    ):
        predicted_class = int(
            predicted_labels[i]
        )

        predicted_mask = (
            source_labels
            == predicted_class
        )

        competing_mask = (
            source_labels
            != predicted_class
        )

        if not predicted_mask.any():
            scores.append(
                torch.tensor(0.0)
            )
            continue

        predicted_similarity = similarities[
            i,
            predicted_mask,
        ].max()

        if competing_mask.any():
            competing_similarity = similarities[
                i,
                competing_mask,
            ].max()
        else:
            competing_similarity = torch.tensor(
                0.0
            )

        scores.append(
            predicted_similarity
            - competing_similarity
        )

    return torch.stack(scores)


def auroc(scores, correctness):
    scores = scores.detach().cpu().numpy()
    correctness = (
        correctness.detach().cpu().numpy()
    )

    order = np.argsort(
        -scores
    )

    correctness = correctness[order]

    positives = correctness.sum()
    negatives = (
        len(correctness)
        - positives
    )

    if positives == 0 or negatives == 0:
        return float("nan")

    tp = 0.0
    fp = 0.0
    previous_tpr = 0.0
    previous_fpr = 0.0
    area = 0.0

    for value in correctness:
        if value == 1:
            tp += 1
        else:
            fp += 1

        tpr = tp / positives
        fpr = fp / negatives

        area += (
            (fpr - previous_fpr)
            * (tpr + previous_tpr)
            / 2.0
        )

        previous_tpr = tpr
        previous_fpr = fpr

    return area


def precision_at_coverage(
    scores,
    correctness,
    fraction,
):
    count = max(
        1,
        int(
            len(scores)
            * fraction
        ),
    )

    order = torch.argsort(
        scores,
        descending=True,
    )

    selected = correctness[
        order[:count]
    ]

    return selected.float().mean().item()


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

    probabilities = get_drl_predictions(
        trainer,
        target_x,
        device,
    )

    confidence, predicted_labels = (
        probabilities.max(dim=1)
    )

    source_features = get_source_embedding(
        trainer,
        source_x,
        device,
    )

    target_features = get_target_embedding(
        trainer,
        target_x,
        device,
    )

    support_margin = class_support_margin(
        source_features,
        source_y,
        target_features,
        predicted_labels,
    )

    correctness = (
        predicted_labels == target_y
    )

    combined_score = (
        confidence
        * torch.sigmoid(
            support_margin
        )
    )

    confidence_auc = auroc(
        confidence,
        correctness,
    )

    support_auc = auroc(
        support_margin,
        correctness,
    )

    combined_auc = auroc(
        combined_score,
        correctness,
    )

    confidence_precision_20 = (
        precision_at_coverage(
            confidence,
            correctness,
            0.20,
        )
    )

    support_precision_20 = (
        precision_at_coverage(
            support_margin,
            correctness,
            0.20,
        )
    )

    combined_precision_20 = (
        precision_at_coverage(
            combined_score,
            correctness,
            0.20,
        )
    )

    confidence_precision_10 = (
        precision_at_coverage(
            confidence,
            correctness,
            0.10,
        )
    )

    support_precision_10 = (
        precision_at_coverage(
            support_margin,
            correctness,
            0.10,
        )
    )

    combined_precision_10 = (
        precision_at_coverage(
            combined_score,
            correctness,
            0.10,
        )
    )

    print()
    print(
        f"Target Samples: "
        f"{len(target_dataset)}"
    )

    print(
        f"Pseudo-label Accuracy: "
        f"{correctness.float().mean().item() * 100:.2f}%"
    )

    print(
        f"Confidence Mean: "
        f"{confidence.mean().item():.4f}"
    )

    print(
        f"Support Margin Mean: "
        f"{support_margin.mean().item():.4f}"
    )

    print()
    print(
        f"Confidence AUROC: "
        f"{confidence_auc:.4f}"
    )

    print(
        f"Support Margin AUROC: "
        f"{support_auc:.4f}"
    )

    print(
        f"Combined AUROC: "
        f"{combined_auc:.4f}"
    )

    print()
    print(
        f"Top 20% Confidence Precision: "
        f"{confidence_precision_20 * 100:.2f}%"
    )

    print(
        f"Top 20% Support Precision: "
        f"{support_precision_20 * 100:.2f}%"
    )

    print(
        f"Top 20% Combined Precision: "
        f"{combined_precision_20 * 100:.2f}%"
    )

    print()
    print(
        f"Top 10% Confidence Precision: "
        f"{confidence_precision_10 * 100:.2f}%"
    )

    print(
        f"Top 10% Support Precision: "
        f"{support_precision_10 * 100:.2f}%"
    )

    print(
        f"Top 10% Combined Precision: "
        f"{combined_precision_10 * 100:.2f}%"
    )


if __name__ == "__main__":
    main()