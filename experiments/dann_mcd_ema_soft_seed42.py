from pathlib import Path
import copy
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from torchvision import datasets, transforms


SEED = 42
DEVICE = torch.device("cpu")

BATCH_SIZE = 16
SOURCE_SAMPLES = 2000
TARGET_ADAPT_SAMPLES = 1800

NUM_MCD_EPOCHS = 9
NUM_PL_EPOCHS = 9

LR_FEATURE = 1e-4
LR_CLASSIFIER = 1e-3
PL_LR = 5e-4
WEIGHT_DECAY = 1e-4

DISCREPANCY_WEIGHT = 1.0
EMA_DECAY = 0.99
CONF_FLOOR = 0.85
REFRESH_EVERY = 3

DANN_CHECKPOINT = "checkpoints/dann_81_51_seed42.pt"
OUTPUT_CHECKPOINT = "checkpoints/dann_mcd_ema_soft_seed42.pt"
HISTORY_PATH = "checkpoints/dann_mcd_ema_soft_seed42.json"
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


class TargetPoolDataset(Dataset):
    def __init__(self, x, y, w):
        self.x = x
        self.y = y
        self.w = w

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.w[index]


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


@torch.no_grad()
def evaluate(
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
def evaluate_ensemble(
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

        correct1 += (
            probs1.argmax(dim=1) == y
        ).sum().item()

        correct2 += (
            probs2.argmax(dim=1) == y
        ).sum().item()

        ensemble = (
            probs1 + probs2
        ).mul(0.5).argmax(dim=1)

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
def measure_disagreement(
    feature,
    classifier1,
    classifier2,
    target,
):
    feature.eval()
    classifier1.eval()
    classifier2.eval()

    values = []

    for start in range(
        0,
        len(target),
        128,
    ):
        end = min(
            start + 128,
            len(target),
        )

        xs = []

        for i in range(start, end):
            x, _ = target[i]
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
            ).mean(dim=1).cpu()
        )

    return float(
        torch.cat(values).mean().item()
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

    required = [
        "feature_extractor",
        "classifier",
    ]

    missing = [
        key
        for key in required
        if key not in checkpoint
    ]

    if missing:
        raise RuntimeError(
            f"DANN checkpoint missing keys: {missing}"
        )

    feature.load_state_dict(
        checkpoint["feature_extractor"]
    )

    classifier.load_state_dict(
        checkpoint["classifier"]
    )


def locked_batches_for_epoch(
    locked_batches,
    epoch_number,
):
    slot = (
        epoch_number - 1
    ) % 5

    return [
        batch
        for batch in locked_batches
        if int(batch["epoch"]) == slot
    ]


def train_mcd_epoch(
    feature,
    classifier1,
    classifier2,
    feature_optimizer,
    classifier1_optimizer,
    classifier2_optimizer,
    source,
    target,
    locked_batches,
    epoch_number,
):
    batches = locked_batches_for_epoch(
        locked_batches,
        epoch_number,
    )

    feature.train()
    classifier1.train()
    classifier2.train()

    source_total = 0.0
    classifier_dis_total = 0.0
    feature_dis_total = 0.0

    for batch in batches:
        xs, ys = collect_batch(
            source,
            batch["source"],
        )

        xt = collect_target_batch(
            target,
            batch["target"],
        )

        for parameter in feature.parameters():
            parameter.requires_grad_(False)

        for parameter in classifier1.parameters():
            parameter.requires_grad_(True)

        for parameter in classifier2.parameters():
            parameter.requires_grad_(True)

        classifier1_optimizer.zero_grad(
            set_to_none=True
        )

        classifier2_optimizer.zero_grad(
            set_to_none=True
        )

        with torch.no_grad():
            source_features = feature(xs)

        source_logits1 = classifier1(
            source_features
        )

        source_logits2 = classifier2(
            source_features
        )

        source_loss = (
            F.cross_entropy(
                source_logits1,
                ys,
            )
            + F.cross_entropy(
                source_logits2,
                ys,
            )
        )

        source_loss.backward()

        classifier1_optimizer.step()
        classifier2_optimizer.step()

        source_total += (
            source_loss.item()
        )

        classifier1_optimizer.zero_grad(
            set_to_none=True
        )

        classifier2_optimizer.zero_grad(
            set_to_none=True
        )

        with torch.no_grad():
            target_features = feature(xt)

        target_logits1 = classifier1(
            target_features
        )

        target_logits2 = classifier2(
            target_features
        )

        target_probs1 = F.softmax(
            target_logits1,
            dim=1,
        )

        target_probs2 = F.softmax(
            target_logits2,
            dim=1,
        )

        target_disagreement = torch.abs(
            target_probs1
            - target_probs2
        ).mean()

        (
            -DISCREPANCY_WEIGHT
            * target_disagreement
        ).backward()

        classifier1_optimizer.step()
        classifier2_optimizer.step()

        classifier_dis_total += (
            target_disagreement.item()
        )

        for parameter in classifier1.parameters():
            parameter.requires_grad_(False)

        for parameter in classifier2.parameters():
            parameter.requires_grad_(False)

        for parameter in feature.parameters():
            parameter.requires_grad_(True)

        feature_optimizer.zero_grad(
            set_to_none=True
        )

        target_features = feature(xt)

        target_logits1 = classifier1(
            target_features
        )

        target_logits2 = classifier2(
            target_features
        )

        target_probs1 = F.softmax(
            target_logits1,
            dim=1,
        )

        target_probs2 = F.softmax(
            target_logits2,
            dim=1,
        )

        target_disagreement = torch.abs(
            target_probs1
            - target_probs2
        ).mean()

        target_disagreement.backward()

        feature_optimizer.step()

        feature_dis_total += (
            target_disagreement.item()
        )

        for parameter in classifier1.parameters():
            parameter.requires_grad_(True)

        for parameter in classifier2.parameters():
            parameter.requires_grad_(True)

    denominator = max(
        len(batches),
        1,
    )

    return {
        "source_loss": source_total / denominator,
        "classifier_disagreement": (
            classifier_dis_total / denominator
        ),
        "feature_disagreement": (
            feature_dis_total / denominator
        ),
    }


@torch.no_grad()
def update_ema(
    student_feature,
    student_classifier1,
    student_classifier2,
    teacher_feature,
    teacher_classifier1,
    teacher_classifier2,
    decay,
):
    student_states = [
        student_feature.state_dict(),
        student_classifier1.state_dict(),
        student_classifier2.state_dict(),
    ]

    teacher_states = [
        teacher_feature.state_dict(),
        teacher_classifier1.state_dict(),
        teacher_classifier2.state_dict(),
    ]

    for student_state, teacher_state in zip(
        student_states,
        teacher_states,
    ):
        for name in teacher_state:
            student_value = student_state[name]
            teacher_value = teacher_state[name]

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


@torch.no_grad()
def select_soft_pool(
    teacher_feature,
    teacher_classifier1,
    teacher_classifier2,
    target,
):
    teacher_feature.eval()
    teacher_classifier1.eval()
    teacher_classifier2.eval()

    all_x = []
    all_y = []
    all_w = []
    all_disagreement = []

    loader = DataLoader(
        target,
        batch_size=256,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    for x, _ in loader:
        x = x.to(DEVICE)

        features = teacher_feature(x)

        logits1 = teacher_classifier1(
            features
        )

        logits2 = teacher_classifier2(
            features
        )

        probs1 = F.softmax(
            logits1,
            dim=1,
        )

        probs2 = F.softmax(
            logits2,
            dim=1,
        )

        probabilities = (
            probs1 + probs2
        ) * 0.5

        confidence, labels = probabilities.max(
            dim=1
        )

        disagreement = torch.abs(
            probs1 - probs2
        ).mean(dim=1)

        mask = confidence >= CONF_FLOOR

        if mask.any():
            all_x.append(
                x.cpu()[mask.cpu()]
            )
            all_y.append(
                labels.cpu()[mask.cpu()]
            )
            all_w.append(
                confidence.cpu()[mask.cpu()]
            )
            all_disagreement.append(
                disagreement.cpu()[mask.cpu()]
            )

    if not all_x:
        return None

    selected_x = torch.cat(
        all_x,
        dim=0,
    )

    selected_y = torch.cat(
        all_y,
        dim=0,
    )

    selected_w = torch.cat(
        all_w,
        dim=0,
    )

    selected_disagreement = torch.cat(
        all_disagreement,
        dim=0,
    )

    selected_w = (
        selected_w
        / selected_w.mean().clamp_min(1e-8)
    )

    counts = torch.bincount(
        selected_y,
        minlength=10,
    )

    return {
        "x": selected_x,
        "y": selected_y,
        "w": selected_w,
        "disagreement": selected_disagreement,
        "counts": counts,
    }


def main():
    set_seed(SEED)

    print("========================================")
    print("DANN + MCD + EMA + SOFT PSEUDO-LABELING")
    print("========================================")
    print(f"device={DEVICE}")
    print(f"seed={SEED}")
    print(f"batch_size={BATCH_SIZE}")
    print(f"mcd_epochs={NUM_MCD_EPOCHS}")
    print(f"pseudo_epochs={NUM_PL_EPOCHS}")
    print(f"ema_decay={EMA_DECAY}")
    print(f"confidence_floor={CONF_FLOOR}")
    print(f"refresh_every={REFRESH_EVERY}")
    print()

    source, target, target_test = load_data()

    locked_batches = torch.load(
        LOCKED_BATCHES_PATH,
        map_location="cpu",
    )

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

    with torch.no_grad():
        for parameter in classifier2.parameters():
            parameter.add_(
                torch.randn_like(parameter)
                * 0.005
            )

    baseline = evaluate(
        feature,
        classifier1,
        target_test,
    )

    print(
        f"Loaded DANN USPS Accuracy: "
        f"{baseline:.2f}%"
    )

    initial_metrics = evaluate_ensemble(
        feature,
        classifier1,
        classifier2,
        target_test,
    )

    print(
        f"Initial C2 USPS: "
        f"{initial_metrics['classifier2']:.2f}%"
    )

    print(
        f"Initial ensemble USPS: "
        f"{initial_metrics['ensemble']:.2f}%"
    )

    feature_optimizer = torch.optim.Adam(
        feature.parameters(),
        lr=LR_FEATURE,
        weight_decay=WEIGHT_DECAY,
    )

    classifier1_optimizer = torch.optim.Adam(
        classifier1.parameters(),
        lr=LR_CLASSIFIER,
        weight_decay=WEIGHT_DECAY,
    )

    classifier2_optimizer = torch.optim.Adam(
        classifier2.parameters(),
        lr=LR_CLASSIFIER,
        weight_decay=WEIGHT_DECAY,
    )

    mcd_history = []

    best_mcd = initial_metrics[
        "ensemble"
    ]

    print()
    print("========================================")
    print("PHASE 1: MCD ADAPTATION")
    print("========================================")

    for epoch in range(
        1,
        NUM_MCD_EPOCHS + 1,
    ):
        losses = train_mcd_epoch(
            feature,
            classifier1,
            classifier2,
            feature_optimizer,
            classifier1_optimizer,
            classifier2_optimizer,
            source,
            target,
            locked_batches,
            epoch,
        )

        source_accuracy = evaluate(
            feature,
            classifier1,
            source,
        )

        metrics = evaluate_ensemble(
            feature,
            classifier1,
            classifier2,
            target_test,
        )

        disagreement = measure_disagreement(
            feature,
            classifier1,
            classifier2,
            target,
        )

        best_mcd = max(
            best_mcd,
            metrics["ensemble"],
        )

        row = {
            "epoch": epoch,
            "source_accuracy": source_accuracy,
            "classifier1": metrics[
                "classifier1"
            ],
            "classifier2": metrics[
                "classifier2"
            ],
            "ensemble": metrics[
                "ensemble"
            ],
            "target_disagreement": disagreement,
            "source_loss": losses[
                "source_loss"
            ],
            "classifier_disagreement": losses[
                "classifier_disagreement"
            ],
            "feature_disagreement": losses[
                "feature_disagreement"
            ],
        }

        mcd_history.append(row)

        print(
            f"MCD Epoch {epoch:02d}/{NUM_MCD_EPOCHS} "
            f"| Source {source_accuracy:.2f}% "
            f"| C1 {metrics['classifier1']:.2f}% "
            f"| C2 {metrics['classifier2']:.2f}% "
            f"| Ensemble {metrics['ensemble']:.2f}% "
            f"| Target Disc {disagreement:.6f}"
        )

    teacher_feature = copy.deepcopy(
        feature
    ).to(DEVICE)

    teacher_classifier1 = copy.deepcopy(
        classifier1
    ).to(DEVICE)

    teacher_classifier2 = copy.deepcopy(
        classifier2
    ).to(DEVICE)

    teacher_feature.eval()
    teacher_classifier1.eval()
    teacher_classifier2.eval()

    for parameter in teacher_feature.parameters():
        parameter.requires_grad_(False)

    for parameter in teacher_classifier1.parameters():
        parameter.requires_grad_(False)

    for parameter in teacher_classifier2.parameters():
        parameter.requires_grad_(False)

    pseudo_optimizer = torch.optim.Adam(
        list(feature.parameters())
        + list(classifier1.parameters())
        + list(classifier2.parameters()),
        lr=PL_LR,
        weight_decay=WEIGHT_DECAY,
    )

    history = []
    best_combined = best_mcd

    current_pool = None

    print()
    print("========================================")
    print("PHASE 2: EMA + SOFT PSEUDO-LABELING")
    print("========================================")

    for epoch in range(
        1,
        NUM_PL_EPOCHS + 1,
    ):
        refresh = (
            epoch == 1
            or (epoch - 1) % REFRESH_EVERY == 0
        )

        if refresh:
            current_pool = select_soft_pool(
                teacher_feature,
                teacher_classifier1,
                teacher_classifier2,
                target,
            )

            if current_pool is None:
                print("No target samples selected.")
                break

            selected_count = current_pool[
                "x"
            ].size(0)

            coverage = (
                100.0
                * selected_count
                / TARGET_ADAPT_SAMPLES
            )

            raw_confidence = (
                current_pool["w"]
                * current_pool["w"].mean()
            ).mean().item()

            mean_disagreement = current_pool[
                "disagreement"
            ].mean().item()

            class_counts = [
                int(value.item())
                for value in current_pool[
                    "counts"
                ]
            ]

            print(
                f"REFRESH Epoch {epoch:02d} "
                f"| Selected {selected_count}/{TARGET_ADAPT_SAMPLES} "
                f"| Coverage {coverage:.2f}% "
                f"| Mean Conf {raw_confidence:.4f} "
                f"| Mean Disc {mean_disagreement:.6f}"
            )

            print(
                f"Class counts: {class_counts}"
            )

        pool_dataset = TargetPoolDataset(
            current_pool["x"],
            current_pool["y"],
            current_pool["w"],
        )

        pool_loader = DataLoader(
            pool_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )

        source_loader = DataLoader(
            source,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )

        source_iterator = iter(
            source_loader
        )

        pool_iterator = iter(
            pool_loader
        )

        feature.train()
        classifier1.train()
        classifier2.train()

        steps = min(
            len(source_loader),
            len(pool_loader),
        )

        source_loss_total = 0.0
        pseudo_loss_total = 0.0

        for _ in range(steps):
            xs, ys = next(
                source_iterator
            )

            xt, yt, wt = next(
                pool_iterator
            )

            xs = xs.to(DEVICE)
            ys = ys.to(DEVICE)
            xt = xt.to(DEVICE)
            yt = yt.to(DEVICE)
            wt = wt.to(DEVICE)

            pseudo_optimizer.zero_grad(
                set_to_none=True
            )

            source_features = feature(xs)

            source_logits1 = classifier1(
                source_features
            )

            source_logits2 = classifier2(
                source_features
            )

            source_loss = (
                F.cross_entropy(
                    source_logits1,
                    ys,
                )
                + F.cross_entropy(
                    source_logits2,
                    ys,
                )
            )

            target_features = feature(xt)

            target_logits1 = classifier1(
                target_features
            )

            target_logits2 = classifier2(
                target_features
            )

            target_loss1 = (
                wt
                * F.cross_entropy(
                    target_logits1,
                    yt,
                    reduction="none",
                )
            ).mean()

            target_loss2 = (
                wt
                * F.cross_entropy(
                    target_logits2,
                    yt,
                    reduction="none",
                )
            ).mean()

            target_loss = (
                target_loss1
                + target_loss2
            ) * 0.5

            loss = (
                source_loss
                + target_loss
            )

            loss.backward()
            pseudo_optimizer.step()

            update_ema(
                feature,
                classifier1,
                classifier2,
                teacher_feature,
                teacher_classifier1,
                teacher_classifier2,
                EMA_DECAY,
            )

            source_loss_total += (
                source_loss.item()
            )

            pseudo_loss_total += (
                target_loss.item()
            )

        source_accuracy = evaluate(
            feature,
            classifier1,
            source,
        )

        metrics = evaluate_ensemble(
            feature,
            classifier1,
            classifier2,
            target_test,
        )

        selected_count = current_pool[
            "x"
        ].size(0)

        coverage = (
            100.0
            * selected_count
            / TARGET_ADAPT_SAMPLES
        )

        disagreement = measure_disagreement(
            feature,
            classifier1,
            classifier2,
            target,
        )

        best_combined = max(
            best_combined,
            metrics["ensemble"],
        )

        row = {
            "epoch": epoch,
            "source_accuracy": source_accuracy,
            "classifier1": metrics[
                "classifier1"
            ],
            "classifier2": metrics[
                "classifier2"
            ],
            "ensemble": metrics[
                "ensemble"
            ],
            "selected_count": selected_count,
            "coverage": coverage,
            "target_disagreement": disagreement,
            "source_loss": (
                source_loss_total
                / max(steps, 1)
            ),
            "pseudo_loss": (
                pseudo_loss_total
                / max(steps, 1)
            ),
            "class_counts": [
                int(value.item())
                for value in current_pool[
                    "counts"
                ]
            ],
        }

        history.append(row)

        print(
            f"PL Epoch {epoch:02d}/{NUM_PL_EPOCHS} "
            f"| Source {source_accuracy:.2f}% "
            f"| C1 {metrics['classifier1']:.2f}% "
            f"| C2 {metrics['classifier2']:.2f}% "
            f"| Ensemble {metrics['ensemble']:.2f}% "
            f"| Disc {disagreement:.6f} "
            f"| Selected {selected_count} "
            f"| Coverage {coverage:.2f}%"
        )

    output_dir = Path(
        OUTPUT_CHECKPOINT
    ).parent

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "feature_extractor": feature.state_dict(),
            "classifier1": classifier1.state_dict(),
            "classifier2": classifier2.state_dict(),
            "teacher_feature": teacher_feature.state_dict(),
            "teacher_classifier1": teacher_classifier1.state_dict(),
            "teacher_classifier2": teacher_classifier2.state_dict(),
            "seed": SEED,
            "baseline_dann": baseline,
            "best_mcd": best_mcd,
            "best_combined": best_combined,
            "mcd_history": mcd_history,
            "pseudo_history": history,
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
                "best_mcd": best_mcd,
                "best_combined": best_combined,
                "mcd_history": mcd_history,
                "pseudo_history": history,
            },
            handle,
            indent=2,
        )

    print()
    print("========================================")
    print("FINAL RESULT")
    print("========================================")
    print(
        f"DANN baseline        = {baseline:.2f}%"
    )
    print(
        "Previous best         = 87.69%"
    )
    print(
        f"MCD best             = {best_mcd:.2f}%"
    )
    print(
        f"MCD + EMA/soft best  = {best_combined:.2f}%"
    )
    print(
        f"Gain over 87.69      = {best_combined - 87.69:+.2f} pp"
    )
    print(
        f"Gain over DANN       = {best_combined - baseline:+.2f} pp"
    )
    print(
        f"checkpoint={OUTPUT_CHECKPOINT}"
    )
    print(
        f"history={HISTORY_PATH}"
    )


if __name__ == "__main__":
    main()
