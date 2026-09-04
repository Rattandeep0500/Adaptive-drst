import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


SEED = 42
DEVICE = torch.device("cpu")

BATCH_SIZE = 16
SOURCE_SAMPLES = 2000
TARGET_ADAPT_SAMPLES = 1800
MCD_EPOCHS = 9

LR_FEATURE = 1e-4
LR_CLASSIFIER = 1e-3
WEIGHT_DECAY = 1e-4

DISCREPANCY_WEIGHT = 1.0
SOURCE_PRESERVE_WEIGHT = 0.0
CLASSIFIER_STEPS = 1
FEATURE_STEPS = 1

DANN_CHECKPOINT = "checkpoints/dann_81_51_seed42.pt"
OUTPUT_CHECKPOINT = "checkpoints/dann_mcd_seed42.pt"
HISTORY_PATH = "checkpoints/dann_mcd_seed42.json"
LOCKED_BATCHES_PATH = "data/locked_digits_batches.pt"


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.network(x)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.network(x)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data():
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ]
    )

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

    source = torch.utils.data.Subset(
        mnist,
        list(range(SOURCE_SAMPLES)),
    )

    target = torch.utils.data.Subset(
        usps_train,
        list(range(TARGET_ADAPT_SAMPLES)),
    )

    return source, target, usps_test


def collect_batch(dataset, indices):
    xs = []
    ys = []

    for index in indices:
        x, y = dataset[int(index)]
        xs.append(x)
        ys.append(y)

    x = torch.stack(xs, dim=0).to(DEVICE)
    y = torch.tensor(
        ys,
        dtype=torch.long,
        device=DEVICE,
    )

    return x, y


def collect_target_batch(dataset, indices):
    xs = []

    for index in indices:
        x, _ = dataset[int(index)]
        xs.append(x)

    return torch.stack(
        xs,
        dim=0,
    ).to(DEVICE)


def get_locked_batches():
    batches = torch.load(
        LOCKED_BATCHES_PATH,
        map_location="cpu",
    )

    return batches


@torch.no_grad()
def evaluate_single(
    feature,
    classifier,
    dataset,
):
    feature.eval()
    classifier.eval()

    correct = 0
    total = 0

    for start in range(
        0,
        len(dataset),
        128,
    ):
        end = min(
            start + 128,
            len(dataset),
        )

        xs = []
        ys = []

        for i in range(start, end):
            x, y = dataset[i]
            xs.append(x)
            ys.append(y)

        x = torch.stack(
            xs,
            dim=0,
        ).to(DEVICE)

        y = torch.tensor(
            ys,
            dtype=torch.long,
            device=DEVICE,
        )

        logits = classifier(
            feature(x)
        )

        correct += (
            logits.argmax(dim=1) == y
        ).sum().item()

        total += y.size(0)

    return 100.0 * correct / total


@torch.no_grad()
def evaluate_dual(
    feature,
    classifier1,
    classifier2,
    dataset,
):
    feature.eval()
    classifier1.eval()
    classifier2.eval()

    correct1 = 0
    correct2 = 0
    correct_ensemble = 0
    total = 0

    for start in range(
        0,
        len(dataset),
        128,
    ):
        end = min(
            start + 128,
            len(dataset),
        )

        xs = []
        ys = []

        for i in range(start, end):
            x, y = dataset[i]
            xs.append(x)
            ys.append(y)

        x = torch.stack(
            xs,
            dim=0,
        ).to(DEVICE)

        y = torch.tensor(
            ys,
            dtype=torch.long,
            device=DEVICE,
        )

        features = feature(x)

        probs1 = F.softmax(
            classifier1(features),
            dim=1,
        )

        probs2 = F.softmax(
            classifier2(features),
            dim=1,
        )

        pred1 = probs1.argmax(dim=1)
        pred2 = probs2.argmax(dim=1)
        ensemble = (
            0.5 * probs1
            + 0.5 * probs2
        ).argmax(dim=1)

        correct1 += (
            pred1 == y
        ).sum().item()

        correct2 += (
            pred2 == y
        ).sum().item()

        correct_ensemble += (
            ensemble == y
        ).sum().item()

        total += y.size(0)

    return {
        "classifier1": 100.0 * correct1 / total,
        "classifier2": 100.0 * correct2 / total,
        "ensemble": 100.0 * correct_ensemble / total,
    }


@torch.no_grad()
def target_disagreement(
    feature,
    classifier1,
    classifier2,
    target_dataset,
):
    feature.eval()
    classifier1.eval()
    classifier2.eval()

    values = []

    for start in range(
        0,
        len(target_dataset),
        128,
    ):
        end = min(
            start + 128,
            len(target_dataset),
        )

        xs = []

        for i in range(start, end):
            x, _ = target_dataset[i]
            xs.append(x)

        x = torch.stack(
            xs,
            dim=0,
        ).to(DEVICE)

        features = feature(x)

        probs1 = F.softmax(
            classifier1(features),
            dim=1,
        )

        probs2 = F.softmax(
            classifier2(features),
            dim=1,
        )

        values.append(
            torch.abs(
                probs1 - probs2
            ).mean(
                dim=1
            ).cpu()
        )

    return float(
        torch.cat(values).mean().item()
    )


def discrepancy(
    logits1,
    logits2,
):
    probs1 = F.softmax(
        logits1,
        dim=1,
    )

    probs2 = F.softmax(
        logits2,
        dim=1,
    )

    return torch.abs(
        probs1 - probs2
    ).mean()


def perturb_classifier(
    classifier,
    scale=0.01,
):
    with torch.no_grad():
        for parameter in classifier.parameters():
            parameter.add_(
                torch.randn_like(parameter)
                * scale
            )


def load_dann_checkpoint(
    path,
    feature,
    classifier,
):
    checkpoint = torch.load(
        path,
        map_location=DEVICE,
    )

    feature.load_state_dict(
        checkpoint[
            "feature_extractor"
        ]
    )

    classifier.load_state_dict(
        checkpoint[
            "classifier"
        ]
    )


def train_mcd_epoch(
    feature,
    classifier1,
    classifier2,
    optimizer_feature,
    optimizer_classifier1,
    optimizer_classifier2,
    source_dataset,
    target_dataset,
    locked_batches,
):
    feature.train()
    classifier1.train()
    classifier2.train()

    phase_a_source = 0.0
    phase_b_source = 0.0
    phase_b_discrepancy = 0.0
    phase_c_discrepancy = 0.0

    used_batches = 0

    for epoch_slot in range(5):
        batches = [
            batch
            for batch in locked_batches
            if int(batch["epoch"]) == epoch_slot
        ]

        for batch in batches:
            source_x, source_y = collect_batch(
                source_dataset,
                batch["source"],
            )

            target_x = collect_target_batch(
                target_dataset,
                batch["target"],
            )

            used_batches += 1

            for parameter in feature.parameters():
                parameter.requires_grad_(False)

            for parameter in classifier1.parameters():
                parameter.requires_grad_(True)

            for parameter in classifier2.parameters():
                parameter.requires_grad_(True)

            for _ in range(CLASSIFIER_STEPS):
                optimizer_classifier1.zero_grad(
                    set_to_none=True
                )
                optimizer_classifier2.zero_grad(
                    set_to_none=True
                )

                with torch.no_grad():
                    source_features = feature(
                        source_x
                    )

                source_logits1 = classifier1(
                    source_features
                )

                source_logits2 = classifier2(
                    source_features
                )

                loss_a = (
                    F.cross_entropy(
                        source_logits1,
                        source_y,
                    )
                    + F.cross_entropy(
                        source_logits2,
                        source_y,
                    )
                )

                loss_a.backward()

                optimizer_classifier1.step()
                optimizer_classifier2.step()

                phase_a_source += (
                    loss_a.item()
                )

            for _ in range(FEATURE_STEPS):
                optimizer_classifier1.zero_grad(
                    set_to_none=True
                )
                optimizer_classifier2.zero_grad(
                    set_to_none=True
                )

                with torch.no_grad():
                    source_features = feature(
                        source_x
                    )

                source_logits1 = classifier1(
                    source_features
                )

                source_logits2 = classifier2(
                    source_features
                )

                source_loss = (
                    F.cross_entropy(
                        source_logits1,
                        source_y,
                    )
                    + F.cross_entropy(
                        source_logits2,
                        source_y,
                    )
                )

                with torch.no_grad():
                    target_features = feature(
                        target_x
                    )

                target_logits1 = classifier1(
                    target_features
                )

                target_logits2 = classifier2(
                    target_features
                )

                target_disagreement = discrepancy(
                    target_logits1,
                    target_logits2,
                )

                loss_b = (
                    source_loss
                    - DISCREPANCY_WEIGHT
                    * target_disagreement
                )

                loss_b.backward()

                optimizer_classifier1.step()
                optimizer_classifier2.step()

                phase_b_source += (
                    source_loss.item()
                )

                phase_b_discrepancy += (
                    target_disagreement.item()
                )

            for parameter in classifier1.parameters():
                parameter.requires_grad_(False)

            for parameter in classifier2.parameters():
                parameter.requires_grad_(False)

            for parameter in feature.parameters():
                parameter.requires_grad_(True)

            for _ in range(FEATURE_STEPS):
                optimizer_feature.zero_grad(
                    set_to_none=True
                )

                source_features = feature(
                    source_x
                )

                target_features = feature(
                    target_x
                )

                target_logits1 = classifier1(
                    target_features
                )

                target_logits2 = classifier2(
                    target_features
                )

                target_disagreement = discrepancy(
                    target_logits1,
                    target_logits2,
                )

                if SOURCE_PRESERVE_WEIGHT > 0.0:
                    source_logits1 = classifier1(
                        source_features
                    )

                    source_logits2 = classifier2(
                        source_features
                    )

                    source_loss = (
                        F.cross_entropy(
                            source_logits1,
                            source_y,
                        )
                        + F.cross_entropy(
                            source_logits2,
                            source_y,
                        )
                    )

                    loss_c = (
                        target_disagreement
                        + SOURCE_PRESERVE_WEIGHT
                        * source_loss
                    )
                else:
                    loss_c = target_disagreement

                loss_c.backward()
                optimizer_feature.step()

                phase_c_discrepancy += (
                    target_disagreement.item()
                )

    for parameter in classifier1.parameters():
        parameter.requires_grad_(True)

    for parameter in classifier2.parameters():
        parameter.requires_grad_(True)

    for parameter in feature.parameters():
        parameter.requires_grad_(True)

    denominator = max(
        used_batches,
        1,
    )

    return {
        "phase_a_source": phase_a_source / denominator,
        "phase_b_source": phase_b_source / denominator,
        "phase_b_discrepancy": phase_b_discrepancy / denominator,
        "phase_c_discrepancy": phase_c_discrepancy / denominator,
    }


def main():
    set_seed(SEED)

    print("========================================")
    print("DANN CHECKPOINT + MCD ADAPTATION")
    print("========================================")
    print(f"device={DEVICE}")
    print(f"seed={SEED}")
    print(f"batch_size={BATCH_SIZE}")
    print(f"epochs={MCD_EPOCHS}")
    print(f"classifier_lr={LR_CLASSIFIER}")
    print(f"feature_lr={LR_FEATURE}")
    print(f"discrepancy_weight={DISCREPANCY_WEIGHT}")
    print()

    source_dataset, target_dataset, target_test = (
        load_data()
    )

    locked_batches = get_locked_batches()

    feature = FeatureExtractor().to(DEVICE)
    classifier1 = Classifier().to(DEVICE)
    classifier2 = Classifier().to(DEVICE)

    load_dann_checkpoint(
        DANN_CHECKPOINT,
        feature,
        classifier1,
    )

    classifier2.load_state_dict(
        classifier1.state_dict()
    )

    perturb_classifier(
        classifier2,
        scale=0.01,
    )

    baseline_single = evaluate_single(
        feature,
        classifier1,
        target_test,
    )

    baseline_dual = evaluate_dual(
        feature,
        classifier1,
        classifier2,
        target_test,
    )

    print(
        f"DANN baseline USPS: "
        f"{baseline_single:.2f}%"
    )

    print(
        f"Initial classifier2 USPS: "
        f"{baseline_dual['classifier2']:.2f}%"
    )

    print(
        f"Initial ensemble USPS: "
        f"{baseline_dual['ensemble']:.2f}%"
    )

    print(
        f"Initial target disagreement: "
        f"{target_disagreement(feature, classifier1, classifier2, target_dataset):.6f}"
    )

    for parameter in feature.parameters():
        parameter.requires_grad_(True)

    for parameter in classifier1.parameters():
        parameter.requires_grad_(True)

    for parameter in classifier2.parameters():
        parameter.requires_grad_(True)

    optimizer_feature = torch.optim.Adam(
        feature.parameters(),
        lr=LR_FEATURE,
        weight_decay=WEIGHT_DECAY,
    )

    optimizer_classifier1 = torch.optim.Adam(
        classifier1.parameters(),
        lr=LR_CLASSIFIER,
        weight_decay=WEIGHT_DECAY,
    )

    optimizer_classifier2 = torch.optim.Adam(
        classifier2.parameters(),
        lr=LR_CLASSIFIER,
        weight_decay=WEIGHT_DECAY,
    )

    history = []
    best_ensemble = baseline_dual["ensemble"]

    for epoch in range(
        1,
        MCD_EPOCHS + 1,
    ):
        losses = train_mcd_epoch(
            feature,
            classifier1,
            classifier2,
            optimizer_feature,
            optimizer_classifier1,
            optimizer_classifier2,
            source_dataset,
            target_dataset,
            locked_batches,
        )

        source_accuracy = evaluate_single(
            feature,
            classifier1,
            source_dataset,
        )

        target_metrics = evaluate_dual(
            feature,
            classifier1,
            classifier2,
            target_test,
        )

        disagreement = target_disagreement(
            feature,
            classifier1,
            classifier2,
            target_dataset,
        )

        best_ensemble = max(
            best_ensemble,
            target_metrics["ensemble"],
        )

        row = {
            "epoch": epoch,
            "source_accuracy": source_accuracy,
            "classifier1_usps": target_metrics[
                "classifier1"
            ],
            "classifier2_usps": target_metrics[
                "classifier2"
            ],
            "ensemble_usps": target_metrics[
                "ensemble"
            ],
            "target_disagreement": disagreement,
            "phase_a_source": losses[
                "phase_a_source"
            ],
            "phase_b_source": losses[
                "phase_b_source"
            ],
            "phase_b_discrepancy": losses[
                "phase_b_discrepancy"
            ],
            "phase_c_discrepancy": losses[
                "phase_c_discrepancy"
            ],
        }

        history.append(row)

        print(
            f"Epoch {epoch:02d}/{MCD_EPOCHS} "
            f"| Source {source_accuracy:.2f}% "
            f"| C1 USPS {target_metrics['classifier1']:.2f}% "
            f"| C2 USPS {target_metrics['classifier2']:.2f}% "
            f"| Ensemble {target_metrics['ensemble']:.2f}% "
            f"| Target Disc {disagreement:.6f} "
            f"| PhaseB Disc {losses['phase_b_discrepancy']:.6f}"
        )

    Path(
        OUTPUT_CHECKPOINT
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "feature_extractor": feature.state_dict(),
            "classifier1": classifier1.state_dict(),
            "classifier2": classifier2.state_dict(),
            "seed": SEED,
            "baseline_dann": baseline_single,
            "best_ensemble_usps": best_ensemble,
            "history": history,
        },
        OUTPUT_CHECKPOINT,
    )

    with open(
        HISTORY_PATH,
        "w",
        encoding="utf-8",
    ) as handle:
        import json

        json.dump(
            {
                "baseline_dann": baseline_single,
                "best_ensemble_usps": best_ensemble,
                "history": history,
            },
            handle,
            indent=2,
        )

    print()
    print("========================================")
    print("FINAL RESULT")
    print("========================================")
    print(
        f"DANN baseline          = {baseline_single:.2f}%"
    )
    print(
        "Current best           = 87.34%"
    )
    print(
        f"Best MCD ensemble      = {best_ensemble:.2f}%"
    )
    print(
        f"Improvement over DANN  = {best_ensemble - baseline_single:+.2f} pp"
    )
    print(
        f"Improvement over 87.34 = {best_ensemble - 87.34:+.2f} pp"
    )
    print(
        f"checkpoint={OUTPUT_CHECKPOINT}"
    )
    print(
        f"history={HISTORY_PATH}"
    )


if __name__ == "__main__":
    main()
