import random
import numpy as np
import torch
import torch.nn as nn
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


def extract_target_statistics(
    trainer,
    target_x,
    device,
):
    trainer.alpha.eval()
    trainer.beta.eval()

    all_features = []
    all_probabilities = []
    all_confidences = []
    all_entropies = []
    all_predictions = []
    all_ratios = []

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
                trainer.num_classes,
                device=device,
            )

            logits = trainer.alpha(
                x,
                dummy,
                ratio,
            )

            probabilities = F.softmax(
                logits,
                dim=1,
            )

            confidence, prediction = (
                probabilities.max(dim=1)
            )

            entropy = -torch.sum(
                probabilities
                * torch.log(
                    probabilities.clamp_min(
                        1e-8
                    )
                ),
                dim=1,
            )

            features = trainer.alpha.features(x)

            all_features.append(
                features.cpu()
            )

            all_probabilities.append(
                probabilities.cpu()
            )

            all_confidences.append(
                confidence.cpu()
            )

            all_entropies.append(
                entropy.cpu()
            )

            all_predictions.append(
                prediction.cpu()
            )

            all_ratios.append(
                ratio.squeeze(1).cpu()
            )

    return {
        "features": torch.cat(
            all_features,
            dim=0,
        ),
        "probabilities": torch.cat(
            all_probabilities,
            dim=0,
        ),
        "confidence": torch.cat(
            all_confidences,
            dim=0,
        ),
        "entropy": torch.cat(
            all_entropies,
            dim=0,
        ),
        "predictions": torch.cat(
            all_predictions,
            dim=0,
        ),
        "ratios": torch.cat(
            all_ratios,
            dim=0,
        ),
    }


def build_reliability_features(
    features,
    probabilities,
    confidence,
    entropy,
    predictions,
    num_classes=10,
):
    feature_dim = features.shape[1]

    prediction_onehot = F.one_hot(
        predictions.long(),
        num_classes=num_classes,
    ).float()

    confidence = confidence.unsqueeze(1)

    entropy = entropy.unsqueeze(1)

    probability_statistics = torch.cat(
        [
            probabilities,
            confidence,
            entropy,
        ],
        dim=1,
    )

    return torch.cat(
        [
            features,
            probability_statistics,
            prediction_onehot,
        ],
        dim=1,
    ).float()


class ReliabilityPredictor(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                1,
            ),
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


def train_reliability_model(
    x,
    y,
    device,
):
    model = ReliabilityPredictor(
        input_dim=x.shape[1],
        hidden_dim=128,
    ).to(device)

    x = x.to(device)
    y = y.float().to(device)

    positive = y.sum().item()
    negative = len(y) - positive

    if positive > 0 and negative > 0:
        pos_weight = torch.tensor(
            [negative / positive],
            dtype=torch.float32,
            device=device,
        )
    else:
        pos_weight = None

    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    batch_size = 64
    epochs = 100

    model.train()

    for _ in range(epochs):
        permutation = torch.randperm(
            x.size(0),
            device=device,
        )

        for start in range(
            0,
            x.size(0),
            batch_size,
        ):
            indices = permutation[
                start:start + batch_size
            ]

            xb = x[indices]
            yb = y[indices]

            logits = model(xb)

            loss = loss_fn(
                logits,
                yb,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            optimizer.step()

    return model


def accuracy_at_top_fraction(
    scores,
    correctness,
    fraction,
):
    count = max(
        1,
        int(len(scores) * fraction),
    )

    order = torch.argsort(
        scores,
        descending=True,
    )

    selected = correctness[
        order[:count]
    ]

    return selected.float().mean().item()


def auroc(scores, correctness):
    scores = scores.detach().cpu().numpy()
    correctness = correctness.detach().cpu().numpy()

    order = np.argsort(
        -scores
    )

    labels = correctness[order]

    positives = labels.sum()
    negatives = len(labels) - positives

    if positives == 0 or negatives == 0:
        return float("nan")

    tp = 0.0
    fp = 0.0
    prev_tpr = 0.0
    prev_fpr = 0.0
    auc = 0.0

    for value in labels:
        if value == 1:
            tp += 1
        else:
            fp += 1

        tpr = tp / positives
        fpr = fp / negatives

        auc += (
            (fpr - prev_fpr)
            * (tpr + prev_tpr)
            / 2.0
        )

        prev_tpr = tpr
        prev_fpr = fpr

    return auc


def average_precision(scores, correctness):
    scores = scores.detach().cpu()
    correctness = correctness.detach().cpu()

    order = torch.argsort(
        scores,
        descending=True,
    )

    labels = correctness[
        order
    ].float()

    positives = labels.sum().item()

    if positives == 0:
        return float("nan")

    running_correct = 0.0
    total_precision = 0.0

    for i in range(
        len(labels)
    ):
        if labels[i].item() == 1:
            running_correct += 1.0
            total_precision += (
                running_correct
                / (i + 1)
            )

    return total_precision / positives


def report_score(
    name,
    scores,
    correctness,
):
    print()
    print(name)

    print(
        f"AUROC: "
        f"{auroc(scores, correctness):.4f}"
    )

    print(
        f"AUPRC: "
        f"{average_precision(scores, correctness):.4f}"
    )

    for fraction in [
        0.50,
        0.20,
        0.10,
    ]:
        precision = accuracy_at_top_fraction(
            scores,
            correctness,
            fraction,
        )

        print(
            f"Top {fraction * 100:.0f}% Precision: "
            f"{precision * 100:.2f}%"
        )


def per_class_report(
    scores,
    correctness,
    predictions,
):
    print()
    print(
        "Reliability precision by predicted class"
    )

    for cls in range(10):
        mask = predictions == cls

        count = int(mask.sum())

        if count == 0:
            continue

        cls_scores = scores[mask]
        cls_correctness = correctness[mask]

        precision = cls_correctness.float().mean().item()

        top_count = max(
            1,
            int(0.20 * count),
        )

        order = torch.argsort(
            cls_scores,
            descending=True,
        )

        top_precision = (
            cls_correctness[
                order[:top_count]
            ]
            .float()
            .mean()
            .item()
        )

        print(
            f"Class {cls} | "
            f"n={count} | "
            f"Base Precision={precision * 100:.2f}% | "
            f"Top20 Reliability={top_precision * 100:.2f}%"
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

    statistics = extract_target_statistics(
        trainer,
        target_x,
        device,
    )

    features = statistics["features"]
    probabilities = statistics["probabilities"]
    confidence = statistics["confidence"]
    entropy = statistics["entropy"]
    predictions = statistics["predictions"]
    ratios = statistics["ratios"]

    correctness = (
        predictions == target_y
    )

    reliability_features = build_reliability_features(
        features,
        probabilities,
        confidence,
        entropy,
        predictions,
        num_classes=10,
    )

    calibration_indices = torch.arange(
        0,
        600,
    )

    evaluation_indices = torch.arange(
        600,
        1800,
    )

    calibration_x = reliability_features[
        calibration_indices
    ]

    evaluation_x = reliability_features[
        evaluation_indices
    ]

    calibration_y = correctness[
        calibration_indices
    ]

    evaluation_y = correctness[
        evaluation_indices
    ]

    calibration_model = train_reliability_model(
        calibration_x,
        calibration_y,
        device,
    )

    calibration_model.eval()

    with torch.no_grad():
        reliability_logits = calibration_model(
            evaluation_x.to(device)
        )

        reliability_scores = torch.sigmoid(
            reliability_logits
        ).cpu()

    evaluation_confidence = confidence[
        evaluation_indices
    ]

    evaluation_entropy = entropy[
        evaluation_indices
    ]

    evaluation_ratios = ratios[
        evaluation_indices
    ]

    evaluation_predictions = predictions[
        evaluation_indices
    ]

    print(
        f"Calibration Samples: "
        f"{len(calibration_indices)}"
    )

    print(
        f"Evaluation Samples: "
        f"{len(evaluation_indices)}"
    )

    print(
        f"Calibration Accuracy: "
        f"{calibration_y.float().mean().item() * 100:.2f}%"
    )

    print(
        f"Evaluation Accuracy: "
        f"{evaluation_y.float().mean().item() * 100:.2f}%"
    )

    report_score(
        "Confidence",
        evaluation_confidence,
        evaluation_y,
    )

    report_score(
        "Negative Entropy",
        -evaluation_entropy,
        evaluation_y,
    )

    report_score(
        "Density Ratio",
        evaluation_ratios,
        evaluation_y,
    )

    report_score(
        "Learned Reliability",
        reliability_scores,
        evaluation_y,
    )

    per_class_report(
        reliability_scores,
        evaluation_y,
        evaluation_predictions,
    )

    print()
    print(
        f"Mean Confidence: "
        f"{evaluation_confidence.mean().item():.4f}"
    )

    print(
        f"Mean Reliability: "
        f"{reliability_scores.mean().item():.4f}"
    )

    print(
        f"Mean Density Ratio: "
        f"{evaluation_ratios.mean().item():.4f}"
    )


if __name__ == "__main__":
    main()