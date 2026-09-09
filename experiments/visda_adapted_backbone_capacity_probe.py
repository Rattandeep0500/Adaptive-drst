import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 42

NUM_CLASSES = 12
INPUT_DIM = 2048
HIDDEN_DIM = 512

BATCH_SIZE = 1024
EVAL_BATCH_SIZE = 4096

EPOCHS = 30

LR = 0.01
WEIGHT_DECAY = 1e-4

BASE_CHECKPOINT = Path(
    "checkpoints/visda_geometry_gated_mcd/"
    "geometry_gated_mcd_seed42.pt"
)

CACHE_ROOT = Path(
    "checkpoints/visda_feature_cache"
)

TARGET_CACHE = CACHE_ROOT / "target"

OUTPUT_DIR = Path(
    "checkpoints/visda_adapted_backbone_capacity"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "adapted_backbone_capacity_seed42.json"
)

CLASSES = [
    "aeroplane",
    "bicycle",
    "bus",
    "car",
    "horse",
    "knife",
    "motorcycle",
    "person",
    "plant",
    "skateboard",
    "train",
    "truck",
]


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                INPUT_DIM,
                HIDDEN_DIM
            ),
            nn.BatchNorm1d(
                HIDDEN_DIM
            ),
            nn.ReLU(
                inplace=True
            )
        )

    def forward(self, x):
        if x.ndim != 2:
            raise RuntimeError(
                f"Expected 2-D input, got {tuple(x.shape)}"
            )

        if x.shape[1] != INPUT_DIM:
            raise RuntimeError(
                f"Expected {INPUT_DIM}-D input, got {x.shape[1]}"
            )

        return self.net(x)


class MCDModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.adapter = Adapter()

        self.classifier1 = nn.Linear(
            HIDDEN_DIM,
            NUM_CLASSES
        )

        self.classifier2 = nn.Linear(
            HIDDEN_DIM,
            NUM_CLASSES
        )

    def encode(self, x):
        return self.adapter(x)

    def forward(self, x):
        z = self.adapter(x)

        return (
            self.classifier1(z),
            self.classifier2(z)
        )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cache(cache_dir):
    files = sorted(
        cache_dir.glob("chunk_*.pt")
    )

    if not files:
        raise RuntimeError(
            f"No cache files found in {cache_dir}"
        )

    x_parts = []
    y_parts = []

    for path in files:
        payload = torch.load(
            path,
            map_location="cpu"
        )

        x = payload["features"].float()
        y = payload["labels"].long()

        if x.ndim != 2:
            raise RuntimeError(
                f"Invalid feature tensor in {path}"
            )

        if x.shape[1] != INPUT_DIM:
            raise RuntimeError(
                f"Expected {INPUT_DIM}-D features, "
                f"got {x.shape[1]}"
            )

        if len(x) != len(y):
            raise RuntimeError(
                f"Feature/label mismatch in {path}"
            )

        x_parts.append(x)
        y_parts.append(y)

    x = torch.cat(
        x_parts,
        dim=0
    )

    y = torch.cat(
        y_parts,
        dim=0
    )

    return x, y


def normalize_state_dict(state_dict):
    result = {}

    for key, value in state_dict.items():
        new_key = key

        changed = True

        while changed:
            changed = False

            for prefix in (
                "module.",
                "model.",
                "student.",
                "teacher.",
            ):
                if new_key.startswith(prefix):
                    new_key = new_key[
                        len(prefix):
                    ]
                    changed = True
                    break

        result[new_key] = value

    return result


def load_student_checkpoint():
    if not BASE_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {BASE_CHECKPOINT}"
        )

    payload = torch.load(
        BASE_CHECKPOINT,
        map_location="cpu"
    )

    if "student_state_dict" not in payload:
        raise RuntimeError(
            "student_state_dict missing"
        )

    model = MCDModel()

    model.load_state_dict(
        normalize_state_dict(
            payload[
                "student_state_dict"
            ]
        ),
        strict=True
    )

    model.eval()

    return model


@torch.no_grad()
def collect_adapted_features(
    model,
    features
):
    model.eval()

    loader = DataLoader(
        TensorDataset(
            features
        ),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    parts = []

    for (x,) in loader:
        z = model.encode(
            x
        )

        if z.shape[1] != HIDDEN_DIM:
            raise RuntimeError(
                "Adapted feature dimension is not 512"
            )

        parts.append(
            z.cpu()
        )

    return torch.cat(
        parts,
        dim=0
    )


class LinearProbe(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(
            HIDDEN_DIM,
            NUM_CLASSES
        )

    def forward(self, x):
        return self.linear(x)


class MLPProbe(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                HIDDEN_DIM,
                512
            ),
            nn.ReLU(
                inplace=True
            ),
            nn.Linear(
                512,
                256
            ),
            nn.ReLU(
                inplace=True
            ),
            nn.Linear(
                256,
                NUM_CLASSES
            )
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def evaluate_probe(
    model,
    features,
    labels,
    device
):
    model.eval()

    loader = DataLoader(
        TensorDataset(
            features,
            labels
        ),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    total_correct = 0
    total_count = 0

    class_correct = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long
    )

    class_total = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long
    )

    for x, y in loader:
        x = x.to(
            device
        )

        y = y.to(
            device
        )

        logits = model(x)

        predictions = logits.argmax(
            dim=1
        )

        total_correct += int(
            (
                predictions
                == y
            ).sum().item()
        )

        total_count += len(y)

        for class_id in range(
            NUM_CLASSES
        ):
            mask = (
                y == class_id
            )

            if mask.any():
                class_total[
                    class_id
                ] += int(
                    mask.sum().item()
                )

                class_correct[
                    class_id
                ] += int(
                    (
                        predictions[mask]
                        == y[mask]
                    ).sum().item()
                )

    per_class = (
        100.0
        * class_correct.float()
        / class_total.clamp_min(
            1
        )
    )

    return {
        "overall":
            100.0
            * total_correct
            / max(
                total_count,
                1
            ),
        "mean_class":
            float(
                per_class.mean().item()
            ),
        "per_class":
            per_class.tolist()
    }


def train_probe(
    probe,
    train_features,
    train_labels,
    val_features,
    val_labels,
    device,
    name
):
    probe = probe.to(
        device
    )

    loader = DataLoader(
        TensorDataset(
            train_features,
            train_labels
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=0
    )

    optimizer = torch.optim.SGD(
        probe.parameters(),
        lr=LR,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS
        )
    )

    best_mean_class = -1.0
    best_epoch = 0

    best_state = None

    history = []

    for epoch in range(
        1,
        EPOCHS + 1
    ):
        probe.train()

        loss_sum = 0.0
        count = 0

        for x, y in loader:
            x = x.to(
                device
            )

            y = y.to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = probe(x)

            loss = nn.functional.cross_entropy(
                logits,
                y
            )

            loss.backward()

            optimizer.step()

            loss_sum += (
                loss.item()
                * len(y)
            )

            count += len(y)

        scheduler.step()

        metrics = evaluate_probe(
            probe,
            val_features,
            val_labels,
            device
        )

        mean_loss = (
            loss_sum
            / max(
                count,
                1
            )
        )

        history.append(
            {
                "epoch":
                    epoch,
                "loss":
                    mean_loss,
                "overall":
                    metrics["overall"],
                "mean_class":
                    metrics["mean_class"]
            }
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"{name} | "
            f"Loss {mean_loss:.5f} | "
            f"Overall {metrics['overall']:.2f}% | "
            f"Mean-class {metrics['mean_class']:.2f}%"
        )

        if (
            metrics["mean_class"]
            > best_mean_class
        ):
            best_mean_class = (
                metrics["mean_class"]
            )

            best_epoch = epoch

            best_state = {
                key:
                    value.detach().cpu().clone()
                for key, value in probe.state_dict().items()
            }

    probe.load_state_dict(
        best_state
    )

    final_metrics = evaluate_probe(
        probe,
        val_features,
        val_labels,
        device
    )

    return {
        "best_epoch":
            best_epoch,
        "best_mean_class":
            best_mean_class,
        "best_metrics":
            final_metrics,
        "history":
            history
    }


def main():
    set_seed(
        SEED
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 90)
    print(
        "VISDA-2017 ADAPTED-BACKBONE CAPACITY PROBE"
    )
    print("=" * 90)

    print(
        f"device={device}"
    )

    print(
        f"seed={SEED}"
    )

    print(
        f"epochs={EPOCHS}"
    )

    print()

    print(
        "Loading target cache..."
    )

    target_features, target_labels = load_cache(
        TARGET_CACHE
    )

    print(
        f"Target samples: "
        f"{len(target_features)}"
    )

    print(
        f"Input dimension: "
        f"{target_features.shape[1]}"
    )

    print()

    print(
        "Loading adapted MCD checkpoint..."
    )

    model = load_student_checkpoint()

    print(
        "Checkpoint loaded successfully."
    )

    print()

    print(
        "Collecting adapted target features..."
    )

    adapted_features = collect_adapted_features(
        model,
        target_features
    )

    print(
        f"Adapted features: "
        f"{tuple(adapted_features.shape)}"
    )

    if adapted_features.shape[1] != HIDDEN_DIM:
        raise RuntimeError(
            "Expected adapted dimension 512"
        )

    torch.save(
        adapted_features,
        OUTPUT_DIR
        / "adapted_target_features_seed42.pt"
    )

    generator = torch.Generator()

    generator.manual_seed(
        SEED
    )

    indices = torch.randperm(
        len(adapted_features),
        generator=generator
    )

    train_size = int(
        0.8
        * len(indices)
    )

    train_indices = indices[
        :train_size
    ]

    val_indices = indices[
        train_size:
    ]

    train_features = adapted_features[
        train_indices
    ]

    train_labels = target_labels[
        train_indices
    ]

    val_features = adapted_features[
        val_indices
    ]

    val_labels = target_labels[
        val_indices
    ]

    print()

    print(
        "Train samples: "
        f"{len(train_features)}"
    )

    print(
        "Validation samples: "
        f"{len(val_features)}"
    )

    print()
    print("=" * 90)
    print(
        "TRAINING LINEAR PROBE"
    )
    print("=" * 90)

    linear_result = train_probe(
        LinearProbe(),
        train_features,
        train_labels,
        val_features,
        val_labels,
        device,
        "LINEAR"
    )

    print()
    print(
        "=" * 90
    )

    print(
        "TRAINING MLP PROBE"
    )

    print(
        "=" * 90
    )

    mlp_result = train_probe(
        MLPProbe(),
        train_features,
        train_labels,
        val_features,
        val_labels,
        device,
        "MLP"
    )

    print()
    print("=" * 90)
    print(
        "RESULTS"
    )
    print("=" * 90)

    for name, result in (
        (
            "LINEAR",
            linear_result
        ),
        (
            "MLP",
            mlp_result
        )
    ):
        metrics = result[
            "best_metrics"
        ]

        print()
        print(
            name
        )

        print(
            f"Best epoch: "
            f"{result['best_epoch']}"
        )

        print(
            f"Overall: "
            f"{metrics['overall']:.2f}%"
        )

        print(
            f"Mean-class: "
            f"{metrics['mean_class']:.2f}%"
        )

        print()
        print(
            "Per-class:"
        )

        for class_name, accuracy in zip(
            CLASSES,
            metrics["per_class"]
        ):
            print(
                f"{class_name:12s}: "
                f"{accuracy:.2f}%"
            )

    best_name = (
        "MLP"
        if mlp_result[
            "best_mean_class"
        ]
        > linear_result[
            "best_mean_class"
        ]
        else "LINEAR"
    )

    best_result = (
        mlp_result
        if best_name == "MLP"
        else linear_result
    )

    base_reference = {
        "overall":
            73.95,
        "mean_class":
            73.89,
        "truck":
            4.07
    }

    capacity_gap = (
        best_result[
            "best_mean_class"
        ]
        - base_reference[
            "mean_class"
        ]
    )

    output = {
        "experiment":
            "visda_adapted_backbone_capacity_probe",
        "seed":
            SEED,
        "checkpoint":
            str(BASE_CHECKPOINT),
        "target_samples":
            len(target_features),
        "adapted_dimension":
            HIDDEN_DIM,
        "base_reference":
            base_reference,
        "linear":
            linear_result,
        "mlp":
            mlp_result,
        "best_probe":
            best_name,
        "best_mean_class":
            best_result[
                "best_mean_class"
            ],
        "gap_vs_adapted_mcd":
            capacity_gap
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as handle:
        json.dump(
            output,
            handle,
            indent=2
        )

    print()
    print("=" * 90)
    print(
        "CAPACITY PROBE COMPLETE"
    )
    print("=" * 90)

    print(
        f"Best probe: "
        f"{best_name}"
    )

    print(
        f"Best mean-class: "
        f"{best_result['best_mean_class']:.2f}%"
    )

    print(
        f"Difference vs 73.89% adapted baseline: "
        f"{capacity_gap:+.2f} pp"
    )

    print(
        f"Saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()