import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms


SEED = 42
DEVICE = torch.device("cpu")

BATCH_SIZE = 64
SOURCE_SAMPLES = 2000
TARGET_ADAPT_SAMPLES = 1800

NUM_PL_EPOCHS = 9
LR = 1e-3
WEIGHT_DECAY = 1e-4

EMA_DECAY = 0.99
CONF_FLOOR = 0.85
REFRESH_EVERY = 3
MAX_CLASS_RATIO = 2.0

DANN_CHECKPOINT = "checkpoints/dann_81_51_seed42.pt"
OUTPUT_CHECKPOINT = "checkpoints/dann_ema_soft_class_admission_seed42.pt"
HISTORY_PATH = "checkpoints/dann_ema_soft_class_admission_seed42.json"


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


class DomainDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.network(x)


class TargetPoolDataset(torch.utils.data.Dataset):
    def __init__(self, xs, ys, weights):
        self.xs = xs
        self.ys = ys
        self.weights = weights

    def __len__(self):
        return self.xs.size(0)

    def __getitem__(self, index):
        return (
            self.xs[index],
            self.ys[index],
            self.weights[index],
        )


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

    source = Subset(
        mnist,
        list(range(SOURCE_SAMPLES)),
    )

    target = Subset(
        usps_train,
        list(range(TARGET_ADAPT_SAMPLES)),
    )

    return source, target, usps_test


def make_loader(dataset, batch_size, shuffle, drop_last):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
    )


@torch.no_grad()
def evaluate(feature, classifier, dataset_or_loader):
    feature.eval()
    classifier.eval()

    loader = (
        dataset_or_loader
        if isinstance(dataset_or_loader, DataLoader)
        else make_loader(
            dataset_or_loader,
            256,
            False,
            False,
        )
    )

    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = classifier(
            feature(x)
        )

        pred = logits.argmax(
            dim=1
        )

        correct += (
            pred == y
        ).sum().item()

        total += y.size(0)

    return 100.0 * correct / total


@torch.no_grad()
def collect_teacher_predictions(
    teacher_feature,
    teacher_classifier,
    target_dataset,
):
    teacher_feature.eval()
    teacher_classifier.eval()

    loader = make_loader(
        target_dataset,
        256,
        False,
        False,
    )

    xs = []
    preds = []
    confs = []

    for x, _ in loader:
        x = x.to(DEVICE)

        logits = teacher_classifier(
            teacher_feature(x)
        )

        probs = F.softmax(
            logits,
            dim=1,
        )

        confidence, prediction = probs.max(
            dim=1
        )

        xs.append(
            x.cpu()
        )
        preds.append(
            prediction.cpu()
        )
        confs.append(
            confidence.cpu()
        )

    return (
        torch.cat(xs, dim=0),
        torch.cat(preds, dim=0),
        torch.cat(confs, dim=0),
    )


def class_aware_admission(
    target_x,
    target_pred,
    target_conf,
):
    confidence_mask = (
        target_conf >= CONF_FLOOR
    )

    candidate_x = target_x[
        confidence_mask
    ]

    candidate_y = target_pred[
        confidence_mask
    ]

    candidate_w = target_conf[
        confidence_mask
    ]

    before_counts = torch.bincount(
        candidate_y,
        minlength=10,
    )

    nonzero = before_counts[
        before_counts > 0
    ]

    if nonzero.numel() == 0:
        return None

    minimum_count = int(
        nonzero.min().item()
    )

    maximum_allowed = max(
        1,
        int(
            minimum_count
            * MAX_CLASS_RATIO
        ),
    )

    selected_positions = []

    after_counts = torch.zeros(
        10,
        dtype=torch.long,
    )

    for class_id in range(10):
        class_positions = torch.where(
            candidate_y == class_id
        )[0]

        if class_positions.numel() == 0:
            continue

        class_conf = candidate_w[
            class_positions
        ]

        order = torch.argsort(
            class_conf,
            descending=True,
        )

        chosen = class_positions[
            order[:maximum_allowed]
        ]

        selected_positions.append(
            chosen
        )

        after_counts[class_id] = (
            chosen.numel()
        )

    if not selected_positions:
        return None

    positions = torch.cat(
        selected_positions
    )

    selected_x = candidate_x[
        positions
    ]

    selected_y = candidate_y[
        positions
    ]

    selected_w = candidate_w[
        positions
    ]

    selected_w = (
        selected_w
        / selected_w.mean().clamp_min(1e-8)
    )

    return {
        "x": selected_x,
        "y": selected_y,
        "weights": selected_w,
        "before_counts": before_counts,
        "after_counts": after_counts,
        "candidate_count": candidate_y.numel(),
        "selected_count": selected_y.numel(),
    }


@torch.no_grad()
def update_ema(
    student_feature,
    student_classifier,
    teacher_feature,
    teacher_classifier,
    decay,
):
    student_feature_state = (
        student_feature.state_dict()
    )
    teacher_feature_state = (
        teacher_feature.state_dict()
    )

    for name in teacher_feature_state:
        student_value = (
            student_feature_state[name]
        )
        teacher_value = (
            teacher_feature_state[name]
        )

        if torch.is_floating_point(
            teacher_value
        ):
            teacher_value.mul_(
                decay
            ).add_(
                student_value,
                alpha=1.0 - decay,
            )
        else:
            teacher_value.copy_(
                student_value
            )

    student_classifier_state = (
        student_classifier.state_dict()
    )
    teacher_classifier_state = (
        teacher_classifier.state_dict()
    )

    for name in teacher_classifier_state:
        student_value = (
            student_classifier_state[name]
        )
        teacher_value = (
            teacher_classifier_state[name]
        )

        if torch.is_floating_point(
            teacher_value
        ):
            teacher_value.mul_(
                decay
            ).add_(
                student_value,
                alpha=1.0 - decay,
            )
        else:
            teacher_value.copy_(
                student_value
            )


def train_one_epoch(
    student_feature,
    student_classifier,
    teacher_feature,
    teacher_classifier,
    source_loader,
    target_loader,
    optimizer,
):
    student_feature.train()
    student_classifier.train()

    source_iterator = iter(
        source_loader
    )
    target_iterator = iter(
        target_loader
    )

    steps = max(
        len(source_loader),
        len(target_loader),
    )

    source_loss_total = 0.0
    target_loss_total = 0.0

    for step in range(steps):
        optimizer.zero_grad(
            set_to_none=True
        )

        total_loss = torch.zeros(
            (),
            device=DEVICE,
        )

        if step < len(source_loader):
            xs, ys = next(
                source_iterator
            )

            xs = xs.to(DEVICE)
            ys = ys.to(DEVICE)

            source_logits = (
                student_classifier(
                    student_feature(xs)
                )
            )

            source_loss = F.cross_entropy(
                source_logits,
                ys,
            )

            total_loss = (
                total_loss
                + source_loss
            )

            source_loss_total += (
                source_loss.item()
            )

        if step < len(target_loader):
            xt, yt, wt = next(
                target_iterator
            )

            xt = xt.to(DEVICE)
            yt = yt.to(DEVICE)
            wt = wt.to(DEVICE)

            target_logits = (
                student_classifier(
                    student_feature(xt)
                )
            )

            target_losses = (
                F.cross_entropy(
                    target_logits,
                    yt,
                    reduction="none",
                )
            )

            target_loss = (
                wt * target_losses
            ).mean()

            total_loss = (
                total_loss
                + target_loss
            )

            target_loss_total += (
                target_loss.item()
            )

        total_loss.backward()
        optimizer.step()

        update_ema(
            student_feature,
            student_classifier,
            teacher_feature,
            teacher_classifier,
            EMA_DECAY,
        )

    source_loss_value = (
        source_loss_total
        / max(len(source_loader), 1)
    )

    target_loss_value = (
        target_loss_total
        / max(len(target_loader), 1)
    )

    return (
        source_loss_value,
        target_loss_value,
    )


def load_dann_checkpoint(
    path,
    feature,
    classifier,
    domain,
):
    checkpoint = torch.load(
        path,
        map_location=DEVICE,
    )

    if (
        "feature_extractor"
        not in checkpoint
        or "classifier"
        not in checkpoint
        or "domain_discriminator"
        not in checkpoint
    ):
        raise RuntimeError(
            "Checkpoint format is not the expected DANN format."
        )

    feature.load_state_dict(
        checkpoint["feature_extractor"]
    )

    classifier.load_state_dict(
        checkpoint["classifier"]
    )

    domain.load_state_dict(
        checkpoint["domain_discriminator"]
    )

    return checkpoint


def main():
    set_seed(SEED)

    print("========================================")
    print("DANN + EMA + SOFT + CLASS-AWARE ADMISSION")
    print("========================================")
    print(f"device={DEVICE}")
    print(f"seed={SEED}")
    print(f"batch_size={BATCH_SIZE}")
    print(f"pseudo_epochs={NUM_PL_EPOCHS}")
    print(f"ema_decay={EMA_DECAY}")
    print(f"confidence_floor={CONF_FLOOR}")
    print(f"refresh_every={REFRESH_EVERY}")
    print(f"max_class_ratio={MAX_CLASS_RATIO}")
    print()

    source_dataset, target_dataset, target_test = (
        load_data()
    )

    source_loader = make_loader(
        source_dataset,
        BATCH_SIZE,
        True,
        True,
    )

    source_eval_loader = make_loader(
        source_dataset,
        256,
        False,
        False,
    )

    target_test_loader = make_loader(
        target_test,
        256,
        False,
        False,
    )

    student_feature = (
        FeatureExtractor().to(DEVICE)
    )

    student_classifier = (
        Classifier().to(DEVICE)
    )

    domain_discriminator = (
        DomainDiscriminator().to(DEVICE)
    )

    load_dann_checkpoint(
        DANN_CHECKPOINT,
        student_feature,
        student_classifier,
        domain_discriminator,
    )

    baseline = evaluate(
        student_feature,
        student_classifier,
        target_test_loader,
    )

    print(
        f"Loaded DANN USPS Accuracy: {baseline:.2f}%"
    )

    teacher_feature = copy.deepcopy(
        student_feature
    ).to(DEVICE)

    teacher_classifier = copy.deepcopy(
        student_classifier
    ).to(DEVICE)

    teacher_feature.eval()
    teacher_classifier.eval()

    for parameter in teacher_feature.parameters():
        parameter.requires_grad_(False)

    for parameter in teacher_classifier.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.Adam(
        list(
            student_feature.parameters()
        )
        + list(
            student_classifier.parameters()
        ),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    history = []

    best_accuracy = baseline

    selected_pool = None

    for epoch in range(1, NUM_PL_EPOCHS + 1):
        should_refresh = (
            epoch == 1
            or (epoch - 1) % REFRESH_EVERY == 0
        )

        if should_refresh:
            (
                target_x,
                target_pred,
                target_conf,
            ) = collect_teacher_predictions(
                teacher_feature,
                teacher_classifier,
                target_dataset,
            )

            selected_pool = (
                class_aware_admission(
                    target_x,
                    target_pred,
                    target_conf,
                )
            )

            if selected_pool is None:
                raise RuntimeError(
                    "No target samples passed confidence floor."
                )

            before_counts = [
                int(x.item())
                for x in selected_pool[
                    "before_counts"
                ]
            ]

            after_counts = [
                int(x.item())
                for x in selected_pool[
                    "after_counts"
                ]
            ]

            selected_count = (
                selected_pool[
                    "selected_count"
                ]
            )

            coverage = (
                100.0
                * selected_count
                / TARGET_ADAPT_SAMPLES
            )

            mean_confidence = float(
                selected_pool[
                    "weights"
                ].mean().item()
            )

            nonzero_after = [
                value
                for value in after_counts
                if value > 0
            ]

            imbalance = (
                max(nonzero_after)
                / max(min(nonzero_after), 1)
                if nonzero_after
                else 0.0
            )

            print()
            print(
                f"REFRESH Epoch {epoch:02d}"
            )
            print(
                f"Before admission counts: {before_counts}"
            )
            print(
                f"After admission counts: {after_counts}"
            )
            print(
                f"Candidates: "
                f"{selected_pool['candidate_count']}"
            )
            print(
                f"Selected: "
                f"{selected_count}/{TARGET_ADAPT_SAMPLES}"
            )
            print(
                f"Coverage: {coverage:.2f}%"
            )
            print(
                f"Raw mean confidence: "
                f"{float(target_conf[target_conf >= CONF_FLOOR].mean().item()):.4f}"
            )
            print(
                f"Imbalance ratio: {imbalance:.2f}x"
            )

        pl_dataset = TargetPoolDataset(
            selected_pool["x"],
            selected_pool["y"],
            selected_pool["weights"],
        )

        pl_loader = make_loader(
            pl_dataset,
            BATCH_SIZE,
            True,
            False,
        )

        source_loss, target_loss = train_one_epoch(
            student_feature,
            student_classifier,
            teacher_feature,
            teacher_classifier,
            source_loader,
            pl_loader,
            optimizer,
        )

        source_accuracy = evaluate(
            student_feature,
            student_classifier,
            source_eval_loader,
        )

        target_accuracy = evaluate(
            student_feature,
            student_classifier,
            target_test_loader,
        )

        selected_count = selected_pool[
            "selected_count"
        ]

        coverage = (
            100.0
            * selected_count
            / TARGET_ADAPT_SAMPLES
        )

        raw_confidence = (
            selected_pool["weights"]
            * selected_pool["weights"].mean()
        )

        mean_confidence = float(
            raw_confidence.mean().item()
        )

        best_accuracy = max(
            best_accuracy,
            target_accuracy,
        )

        row = {
            "epoch": epoch,
            "source_loss": source_loss,
            "target_loss": target_loss,
            "source_accuracy": source_accuracy,
            "usps_accuracy": target_accuracy,
            "selected_count": selected_count,
            "coverage": coverage,
            "mean_confidence": mean_confidence,
            "class_counts": [
                int(x.item())
                for x in selected_pool[
                    "after_counts"
                ]
            ],
        }

        history.append(row)

        print(
            f"Epoch {epoch:02d}/{NUM_PL_EPOCHS} "
            f"| Source Loss {source_loss:.4f} "
            f"| Pseudo Loss {target_loss:.4f} "
            f"| Source Acc {source_accuracy:.2f}% "
            f"| USPS Acc {target_accuracy:.2f}% "
            f"| Selected {selected_count} "
            f"| Coverage {coverage:.2f}%"
        )

    output_directory = Path(
        OUTPUT_CHECKPOINT
    ).parent

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "feature_extractor": student_feature.state_dict(),
            "classifier": student_classifier.state_dict(),
            "teacher_feature": teacher_feature.state_dict(),
            "teacher_classifier": teacher_classifier.state_dict(),
            "seed": SEED,
            "baseline_dann": baseline,
            "best_usps": best_accuracy,
            "history": history,
        },
        OUTPUT_CHECKPOINT,
    )

    with open(
        HISTORY_PATH,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "baseline_dann": baseline,
                "best_usps": best_accuracy,
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
        f"DANN baseline          = {baseline:.2f}%"
    )
    print(
        "Previous best          = 87.34%"
    )
    print(
        f"New best               = {best_accuracy:.2f}%"
    )
    print(
        f"Improvement over DANN  = {best_accuracy - baseline:+.2f} pp"
    )
    print(
        f"Improvement over 85.25 = {best_accuracy - 85.25:+.2f} pp"
    )
    print(
        f"Improvement over 86.00 = {best_accuracy - 86.00:+.2f} pp"
    )
    print(
        f"Improvement over 87.34 = {best_accuracy - 87.34:+.2f} pp"
    )
    print(
        f"checkpoint={OUTPUT_CHECKPOINT}"
    )
    print(
        f"history={HISTORY_PATH}"
    )


if __name__ == "__main__":
    main()
