import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


DEVICE = torch.device("cpu")

SEEDS = [42, 43, 44, 45, 46]

BATCH_SIZE = 16
SOURCE_SAMPLES = 2000
TARGET_ADAPT_SAMPLES = 1800

MCD_EPOCHS = 9
PL_EPOCHS = 9

MCD_FEATURE_LR = 1e-4
MCD_CLASSIFIER_LR = 1e-3
PL_LR = 5e-4
WEIGHT_DECAY = 1e-4

DISCREPANCY_WEIGHT = 1.0
EMA_DECAY = 0.99
CONF_FLOOR = 0.85
REFRESH_EVERY = 3

DETERMINISTIC_PERTURBATION = 0.005

DANN_CHECKPOINT = "checkpoints/dann_81_51_seed42.pt"
LOCKED_BATCHES_PATH = "data/locked_digits_batches.pt"

RESULTS_DIR = Path("checkpoints/validation_stability")
SUMMARY_PATH = RESULTS_DIR / "summary_fixed_dann_deterministic_mcd.json"


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
    torch.use_deterministic_algorithms(True)

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


def load_locked_batches():
    return torch.load(
        LOCKED_BATCHES_PATH,
        map_location="cpu",
    )


def collect_batch(dataset, indices):
    xs = []
    ys = []

    for index in indices:
        x, y = dataset[int(index)]
        xs.append(x)
        ys.append(y)

    return (
        torch.stack(xs, dim=0).to(DEVICE),
        torch.tensor(
            ys,
            dtype=torch.long,
            device=DEVICE,
        ),
    )


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
def evaluate(feature, classifier, dataset):
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

        pred1 = probs1.argmax(
            dim=1
        )

        pred2 = probs2.argmax(
            dim=1
        )

        ensemble = (
            probs1 + probs2
        ).mul(0.5).argmax(
            dim=1
        )

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


def deterministic_perturbation(classifier):
    with torch.no_grad():
        for parameter in classifier.parameters():
            direction = torch.sign(parameter)
            direction = torch.where(
                direction == 0,
                torch.ones_like(direction),
                direction,
            )
            parameter.add_(
                DETERMINISTIC_PERTURBATION
                * direction
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

    source_total = 0.0
    classifier_dis_total = 0.0
    feature_dis_total = 0.0

    for batch in batches:
        source_x, source_y = collect_batch(
            source,
            batch["source"],
        )

        target_x = collect_target_batch(
            target,
            batch["target"],
        )

        for parameter in feature.parameters():
            parameter.requires_grad_(False)

        classifier1_optimizer.zero_grad(
            set_to_none=True
        )
        classifier2_optimizer.zero_grad(
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

        source_loss.backward()

        classifier1_optimizer.step()
        classifier2_optimizer.step()

        source_total += source_loss.item()

        classifier1_optimizer.zero_grad(
            set_to_none=True
        )
        classifier2_optimizer.zero_grad(
            set_to_none=True
        )

        with torch.no_grad():
            target_features = feature(
                target_x
            )

        target_probs1 = F.softmax(
            classifier1(target_features),
            dim=1,
        )

        target_probs2 = F.softmax(
            classifier2(target_features),
            dim=1,
        )

        classifier_disagreement = torch.abs(
            target_probs1 - target_probs2
        ).mean()

        (
            -DISCREPANCY_WEIGHT
            * classifier_disagreement
        ).backward()

        classifier1_optimizer.step()
        classifier2_optimizer.step()

        classifier_dis_total += (
            classifier_disagreement.item()
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

        target_features = feature(
            target_x
        )

        target_probs1 = F.softmax(
            classifier1(target_features),
            dim=1,
        )

        target_probs2 = F.softmax(
            classifier2(target_features),
            dim=1,
        )

        feature_disagreement = torch.abs(
            target_probs1 - target_probs2
        ).mean()

        feature_disagreement.backward()

        feature_optimizer.step()

        feature_dis_total += (
            feature_disagreement.item()
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
        "source_loss": (
            source_total
            / denominator
        ),
        "classifier_disagreement": (
            classifier_dis_total
            / denominator
        ),
        "feature_disagreement": (
            feature_dis_total
            / denominator
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
                    EMA_DECAY
                ).add_(
                    student_value,
                    alpha=1.0 - EMA_DECAY,
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

    xs = []
    ys = []
    ws = []

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

        probs1 = F.softmax(
            teacher_classifier1(features),
            dim=1,
        )

        probs2 = F.softmax(
            teacher_classifier2(features),
            dim=1,
        )

        probabilities = (
            probs1 + probs2
        ) * 0.5

        confidence, labels = probabilities.max(
            dim=1
        )

        keep = confidence >= CONF_FLOOR

        if keep.any():
            xs.append(
                x.cpu()[keep.cpu()]
            )
            ys.append(
                labels.cpu()[keep.cpu()]
            )
            ws.append(
                confidence.cpu()[keep.cpu()]
            )

    if not xs:
        return None

    selected_x = torch.cat(
        xs,
        dim=0,
    )

    selected_y = torch.cat(
        ys,
        dim=0,
    )

    selected_w = torch.cat(
        ws,
        dim=0,
    )

    selected_w = (
        selected_w
        / selected_w.mean().clamp_min(1e-8)
    )

    return {
        "x": selected_x,
        "y": selected_y,
        "w": selected_w,
    }


def copy_model(model):
    clone = type(model)()
    clone.load_state_dict(
        model.state_dict()
    )
    return clone


def run_seed(
    seed,
    source,
    target,
    target_test,
    locked_batches,
):
    set_seed(seed)

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

    deterministic_perturbation(
        classifier2
    )

    fixed_dann_accuracy = evaluate(
        feature,
        classifier1,
        target_test,
    )

    print(
        f"Seed {seed} | Fixed DANN "
        f"{fixed_dann_accuracy:.2f}%"
    )

    feature_optimizer = torch.optim.Adam(
        feature.parameters(),
        lr=MCD_FEATURE_LR,
        weight_decay=WEIGHT_DECAY,
    )

    classifier1_optimizer = torch.optim.Adam(
        classifier1.parameters(),
        lr=MCD_CLASSIFIER_LR,
        weight_decay=WEIGHT_DECAY,
    )

    classifier2_optimizer = torch.optim.Adam(
        classifier2.parameters(),
        lr=MCD_CLASSIFIER_LR,
        weight_decay=WEIGHT_DECAY,
    )

    mcd_history = []

    for epoch in range(
        1,
        MCD_EPOCHS + 1,
    ):
        train_mcd_epoch(
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

        metrics = evaluate_ensemble(
            feature,
            classifier1,
            classifier2,
            target_test,
        )

        mcd_history.append(
            metrics["ensemble"]
        )

        print(
            f"Seed {seed} | MCD "
            f"Epoch {epoch:02d}/{MCD_EPOCHS} "
            f"| C1 {metrics['classifier1']:.2f}% "
            f"| C2 {metrics['classifier2']:.2f}% "
            f"| Ensemble {metrics['ensemble']:.2f}%"
        )

    teacher_feature = copy_model(
        feature
    )

    teacher_classifier1 = copy_model(
        classifier1
    )

    teacher_classifier2 = copy_model(
        classifier2
    )

    for module in [
        teacher_feature,
        teacher_classifier1,
        teacher_classifier2,
    ]:
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    pseudo_optimizer = torch.optim.Adam(
        list(feature.parameters())
        + list(classifier1.parameters())
        + list(classifier2.parameters()),
        lr=PL_LR,
        weight_decay=WEIGHT_DECAY,
    )

    pool = None
    pseudo_history = []

    for epoch in range(
        1,
        PL_EPOCHS + 1,
    ):
        if (
            epoch == 1
            or (epoch - 1) % REFRESH_EVERY == 0
        ):
            pool = select_soft_pool(
                teacher_feature,
                teacher_classifier1,
                teacher_classifier2,
                target,
            )

            if pool is None:
                raise RuntimeError(
                    f"Seed {seed}: empty pseudo-label pool."
                )

            counts = torch.bincount(
                pool["y"],
                minlength=10,
            ).tolist()

            print(
                f"Seed {seed} | REFRESH {epoch:02d} "
                f"| Selected {pool['x'].size(0)}/1800 "
                f"| Coverage {100.0 * pool['x'].size(0) / 1800:.2f}% "
                f"| Mean Conf {pool['w'].mean().item():.4f}"
            )

            print(
                f"Seed {seed} | Classes {counts}"
            )

        pool_dataset = TargetPoolDataset(
            pool["x"],
            pool["y"],
            pool["w"],
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

        steps = min(
            len(source_loader),
            len(pool_loader),
        )

        feature.train()
        classifier1.train()
        classifier2.train()

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

            target_loss = 0.5 * (
                (
                    wt
                    * F.cross_entropy(
                        target_logits1,
                        yt,
                        reduction="none",
                    )
                ).mean()
                + (
                    wt
                    * F.cross_entropy(
                        target_logits2,
                        yt,
                        reduction="none",
                    )
                ).mean()
            )

            loss = (
                source_loss
                + target_loss
            )

            pseudo_optimizer.zero_grad(
                set_to_none=True
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
            )

        metrics = evaluate_ensemble(
            feature,
            classifier1,
            classifier2,
            target_test,
        )

        pseudo_history.append(
            metrics["ensemble"]
        )

        print(
            f"Seed {seed} | PL "
            f"Epoch {epoch:02d}/{PL_EPOCHS} "
            f"| C1 {metrics['classifier1']:.2f}% "
            f"| C2 {metrics['classifier2']:.2f}% "
            f"| Ensemble {metrics['ensemble']:.2f}% "
            f"| Selected {pool['x'].size(0)}"
        )

    return {
        "seed": seed,
        "fixed_dann_accuracy": fixed_dann_accuracy,
        "mcd_final_accuracy": mcd_history[-1],
        "mcd_best_accuracy": max(mcd_history),
        "pl_final_accuracy": pseudo_history[-1],
        "pl_best_accuracy": max(pseudo_history),
        "mcd_history": mcd_history,
        "pseudo_history": pseudo_history,
    }


def main():
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    source, target, target_test = load_data()
    locked_batches = load_locked_batches()

    print("========================================")
    print("5-SEED STABILITY TEST")
    print("FIXED 81.51% DANN CHECKPOINT")
    print("DETERMINISTIC MCD INITIALIZATION")
    print("========================================")
    print(
        f"seeds={SEEDS}"
    )
    print(
        f"fixed_checkpoint={DANN_CHECKPOINT}"
    )
    print(
        f"deterministic_perturbation={DETERMINISTIC_PERTURBATION}"
    )
    print()

    results = []

    for seed in SEEDS:
        print()
        print("========================================")
        print(
            f"RUNNING SEED {seed}"
        )
        print("========================================")

        result = run_seed(
            seed,
            source,
            target,
            target_test,
            locked_batches,
        )

        results.append(result)

        with open(
            RESULTS_DIR / f"seed_{seed}_fixed_dann.json",
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                result,
                handle,
                indent=2,
            )

        print(
            f"Seed {seed} final USPS: "
            f"{result['pl_final_accuracy']:.2f}%"
        )

    final_values = np.array(
        [
            result["pl_final_accuracy"]
            for result in results
        ],
        dtype=np.float64,
    )

    best_values = np.array(
        [
            result["pl_best_accuracy"]
            for result in results
        ],
        dtype=np.float64,
    )

    summary = {
        "seeds": SEEDS,
        "fixed_dann_accuracy": results[0]["fixed_dann_accuracy"],
        "final_mean": float(
            final_values.mean()
        ),
        "final_std": float(
            final_values.std(ddof=1)
        ),
        "final_best": float(
            final_values.max()
        ),
        "final_worst": float(
            final_values.min()
        ),
        "best_epoch_mean": float(
            best_values.mean()
        ),
        "best_epoch_std": float(
            best_values.std(ddof=1)
        ),
        "best_epoch_best": float(
            best_values.max()
        ),
        "best_epoch_worst": float(
            best_values.min()
        ),
        "previous_final_mean": 87.17,
        "previous_final_std": 2.99,
        "previous_best": 90.23,
        "previous_worst": 83.36,
        "results": results,
    }

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )

    print()
    print("========================================")
    print("STABILITY SUMMARY")
    print("========================================")
    print(
        f"Previous final mean: "
        f"{summary['previous_final_mean']:.2f}%"
    )
    print(
        f"Previous final std: "
        f"{summary['previous_final_std']:.2f}%"
    )
    print(
        f"New final mean: "
        f"{summary['final_mean']:.2f}%"
    )
    print(
        f"New final std: "
        f"{summary['final_std']:.2f}%"
    )
    print(
        f"New final best: "
        f"{summary['final_best']:.2f}%"
    )
    print(
        f"New final worst: "
        f"{summary['final_worst']:.2f}%"
    )
    print(
        f"New best-epoch mean: "
        f"{summary['best_epoch_mean']:.2f}%"
    )
    print(
        f"New best-epoch std: "
        f"{summary['best_epoch_std']:.2f}%"
    )
    print(
        f"Mean change: "
        f"{summary['final_mean'] - summary['previous_final_mean']:+.2f} pp"
    )
    print(
        f"Std change: "
        f"{summary['final_std'] - summary['previous_final_std']:+.2f} pp"
    )
    print(
        f"Summary: {SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()
