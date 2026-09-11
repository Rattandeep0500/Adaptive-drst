import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SEED = 42

NUM_CLASSES = 12
INPUT_DIM = 2048
HIDDEN_DIM = 512

BATCH_SIZE = 512
EVAL_BATCH_SIZE = 4096

EPOCHS = 2

LR_ADAPTER = 0.001
LR_CLASSIFIER = 0.01

MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

PSEUDO_FRACTION = 0.20

TRUCK = 11

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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "checkpoints/visda_rpc_causal_experiment/"
            "rpc_seed42.pt"
        ),
    )

    parser.add_argument(
        "--support-artifact",
        type=Path,
        default=Path(
            "checkpoints/visda_target_class_conditional_support/"
            "target_class_conditional_support_seed42.npz"
        ),
    )

    parser.add_argument(
        "--source-cache",
        type=Path,
        default=Path(
            "checkpoints/visda_feature_cache/source"
        ),
    )

    parser.add_argument(
        "--target-cache",
        type=Path,
        default=Path(
            "checkpoints/visda_feature_cache/target"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "checkpoints/visda_support_gated_target_adaptation"
        ),
    )

    parser.add_argument(
        "--pseudo-fraction",
        type=float,
        default=PSEUDO_FRACTION,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def safe_torch_load(path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def load_feature_cache(
    cache_dir,
    require_labels,
):
    files = sorted(
        cache_dir.glob(
            "chunk_*.pt"
        )
    )

    if not files:
        raise RuntimeError(
            f"No chunk_*.pt files found in {cache_dir}"
        )

    features = []
    labels = [] if require_labels else None

    for path in files:
        payload = safe_torch_load(path)

        if not isinstance(payload, dict):
            raise RuntimeError(
                f"Unexpected cache object in {path}"
            )

        if "features" not in payload:
            raise RuntimeError(
                f"Missing features in {path}"
            )

        x = payload[
            "features"
        ].float().cpu()

        if x.ndim != 2:
            raise RuntimeError(
                f"Invalid feature tensor in {path}: {tuple(x.shape)}"
            )

        if x.shape[1] != INPUT_DIM:
            raise RuntimeError(
                f"Expected feature dimension {INPUT_DIM} "
                f"but found {x.shape[1]} in {path}"
            )

        features.append(x)

        if require_labels:
            if "labels" not in payload:
                raise RuntimeError(
                    f"Missing labels in {path}"
                )

            y = payload[
                "labels"
            ].long().cpu()

            if len(y) != len(x):
                raise RuntimeError(
                    f"Feature/label mismatch in {path}"
                )

            labels.append(y)

    feature_tensor = torch.cat(
        features,
        dim=0,
    )

    if require_labels:
        label_tensor = torch.cat(
            labels,
            dim=0,
        )

        if len(feature_tensor) != len(label_tensor):
            raise RuntimeError(
                "Final feature/label count mismatch"
            )

        return feature_tensor, label_tensor

    return feature_tensor


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                INPUT_DIM,
                HIDDEN_DIM,
            ),
            nn.BatchNorm1d(
                HIDDEN_DIM
            ),
            nn.ReLU(
                inplace=True,
            ),
        )

    def forward(self, x):
        return self.net(x)


class MCDModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.adapter = Adapter()

        self.classifier1 = nn.Linear(
            HIDDEN_DIM,
            NUM_CLASSES,
        )

        self.classifier2 = nn.Linear(
            HIDDEN_DIM,
            NUM_CLASSES,
        )

    def encode(self, x):
        return self.adapter(x)

    def forward(self, x):
        z = self.encode(x)

        return (
            self.classifier1(z),
            self.classifier2(z),
        )


def extract_student_state(payload):
    if not isinstance(payload, dict):
        raise RuntimeError(
            "RPC checkpoint must be a dictionary"
        )

    if "student_state_dict" in payload:
        return (
            payload[
                "student_state_dict"
            ],
            "student_state_dict",
        )

    if "state_dict" in payload:
        return (
            payload[
                "state_dict"
            ],
            "state_dict",
        )

    tensor_keys = [
        key
        for key, value in payload.items()
        if torch.is_tensor(value)
    ]

    if any(
        str(key).startswith(
            "adapter."
        )
        for key in tensor_keys
    ):
        return (
            payload,
            "root",
        )

    raise RuntimeError(
        "Could not locate student_state_dict "
        "or state_dict in RPC checkpoint"
    )


def load_model(path):
    payload = safe_torch_load(path)

    state_dict, state_key = extract_student_state(
        payload
    )

    model = MCDModel()

    model.load_state_dict(
        copy.deepcopy(
            state_dict
        ),
        strict=True,
    )

    return model, state_key


def load_support_artifact(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Support artifact not found: {path}"
        )

    payload = np.load(
        path,
        allow_pickle=False,
    )

    required = [
        "support_top1",
        "support_top1_score",
        "support_margin",
        "support_reliability",
    ]

    missing = [
        key
        for key in required
        if key not in payload.files
    ]

    if missing:
        raise RuntimeError(
            "Support artifact missing required arrays: "
            + ", ".join(missing)
        )

    support_top1 = payload[
        "support_top1"
    ].astype(
        np.int64
    )

    support_top1_score = payload[
        "support_top1_score"
    ].astype(
        np.float64
    )

    support_margin = payload[
        "support_margin"
    ].astype(
        np.float64
    )

    support_reliability = payload[
        "support_reliability"
    ].astype(
        np.float64
    )

    n = len(
        support_top1
    )

    arrays = [
        support_top1_score,
        support_margin,
        support_reliability,
    ]

    for array in arrays:
        if len(array) != n:
            raise RuntimeError(
                "Support artifact arrays have inconsistent lengths"
            )

    if np.any(
        support_top1 < 0
    ) or np.any(
        support_top1 >= NUM_CLASSES
    ):
        raise RuntimeError(
            "support_top1 contains invalid class IDs"
        )

    for name, array in [
        (
            "support_top1_score",
            support_top1_score,
        ),
        (
            "support_margin",
            support_margin,
        ),
        (
            "support_reliability",
            support_reliability,
        ),
    ]:
        if not np.all(
            np.isfinite(array)
        ):
            raise RuntimeError(
                f"{name} contains non-finite values"
            )

    return {
        "support_top1": support_top1,
        "support_top1_score": support_top1_score,
        "support_margin": support_margin,
        "support_reliability": support_reliability,
    }


@torch.no_grad()
def predict_target(
    model,
    target_x,
):
    model.eval()

    predictions = []
    confidences = []

    for start in range(
        0,
        len(target_x),
        EVAL_BATCH_SIZE,
    ):
        end = min(
            start + EVAL_BATCH_SIZE,
            len(target_x),
        )

        x = target_x[
            start:end
        ]

        logits1, logits2 = model(
            x
        )

        p1 = F.softmax(
            logits1,
            dim=1,
        )

        p2 = F.softmax(
            logits2,
            dim=1,
        )

        probabilities = (
            p1 + p2
        ) / 2.0

        confidence, prediction = (
            probabilities.max(
                dim=1
            )
        )

        predictions.append(
            prediction.cpu()
        )

        confidences.append(
            confidence.cpu()
        )

    return (
        torch.cat(
            predictions
        ).numpy().astype(
            np.int64
        ),
        torch.cat(
            confidences
        ).numpy().astype(
            np.float64
        ),
    )


def make_epoch_batches(
    n_source,
    n_target,
    epoch,
    seed,
):
    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        seed
        + epoch * 100003
    )

    source_perm = torch.randperm(
        n_source,
        generator=generator,
    )

    target_perm = torch.randperm(
        n_target,
        generator=generator,
    )

    source_steps = (
        n_source // BATCH_SIZE
    )

    target_steps = (
        n_target // BATCH_SIZE
    )

    if source_steps == 0:
        raise RuntimeError(
            "Source dataset smaller than batch size"
        )

    if target_steps == 0:
        raise RuntimeError(
            "Target dataset smaller than batch size"
        )

    source_perm = source_perm[
        :source_steps * BATCH_SIZE
    ]

    target_perm = target_perm[
        :target_steps * BATCH_SIZE
    ]

    source_batches = source_perm.view(
        source_steps,
        BATCH_SIZE,
    )

    target_batches = target_perm.view(
        target_steps,
        BATCH_SIZE,
    )

    return (
        source_batches,
        target_batches,
    )


def select_pseudo_labels(
    teacher_prediction,
    teacher_confidence,
    support_top1,
    support_margin,
    support_reliability,
    mode,
    count,
):
    n = len(
        teacher_prediction
    )

    count = max(
        1,
        min(
            int(count),
            n,
        ),
    )

    if mode == "vanilla":
        score = teacher_confidence.copy()

    elif mode == "support_gated":
        agreement = (
            teacher_prediction
            == support_top1
        )

        score = (
            0.50
            * teacher_confidence
            + 0.30
            * support_reliability
            + 0.20
            * support_margin
        )

        score = score.astype(
            np.float64,
            copy=True,
        )

        score[
            ~agreement
        ] -= 1.0

    elif mode == "support_margin":
        agreement = (
            teacher_prediction
            == support_top1
        )

        score = (
            0.50
            * teacher_confidence
            + 0.50
            * support_margin
        )

        score = score.astype(
            np.float64,
            copy=True,
        )

        score[
            ~agreement
        ] -= 1.0

    else:
        raise ValueError(
            f"Unknown mode: {mode}"
        )

    if not np.all(
        np.isfinite(score)
    ):
        raise RuntimeError(
            f"Non-finite pseudo-label score in mode {mode}"
        )

    order = np.argsort(
        -score,
        kind="mergesort",
    )

    selected = order[
        :count
    ]

    return selected.astype(
        np.int64
    )


def source_update(
    model,
    optimizer_adapter,
    source_x,
    source_y,
):
    model.train()

    optimizer_adapter.zero_grad(
        set_to_none=True
    )

    logits1, logits2 = model(
        source_x
    )

    loss = (
        F.cross_entropy(
            logits1,
            source_y,
        )
        + F.cross_entropy(
            logits2,
            source_y,
        )
    )

    loss.backward()

    optimizer_adapter.step()

    return float(
        loss.detach().item()
    )


def classifier_update(
    model,
    optimizer_classifier,
    source_x,
    source_y,
    target_x,
):
    model.train()

    optimizer_classifier.zero_grad(
        set_to_none=True
    )

    source_z = model.encode(
        source_x
    )

    with torch.no_grad():
        target_z = model.encode(
            target_x
        )

    source_logits1 = (
        model.classifier1(
            source_z
        )
    )

    source_logits2 = (
        model.classifier2(
            source_z
        )
    )

    target_logits1 = (
        model.classifier1(
            target_z
        )
    )

    target_logits2 = (
        model.classifier2(
            target_z
        )
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

    target_p1 = F.softmax(
        target_logits1,
        dim=1,
    )

    target_p2 = F.softmax(
        target_logits2,
        dim=1,
    )

    discrepancy = (
        target_p1
        - target_p2
    ).abs().mean()

    objective = (
        source_loss
        - discrepancy
    )

    objective.backward()

    optimizer_classifier.step()

    return (
        float(
            source_loss.detach().item()
        ),
        float(
            discrepancy.detach().item()
        ),
    )


def target_discrepancy_update(
    model,
    optimizer_adapter,
    target_x,
):
    model.train()

    optimizer_adapter.zero_grad(
        set_to_none=True
    )

    z = model.encode(
        target_x
    )

    logits1 = model.classifier1(
        z
    )

    logits2 = model.classifier2(
        z
    )

    p1 = F.softmax(
        logits1,
        dim=1,
    )

    p2 = F.softmax(
        logits2,
        dim=1,
    )

    discrepancy = (
        p1 - p2
    ).abs().mean()

    discrepancy.backward()

    optimizer_adapter.step()

    return float(
        discrepancy.detach().item()
    )


def pseudo_label_update(
    model,
    optimizer_adapter,
    optimizer_classifier,
    pseudo_x,
    pseudo_y,
):
    model.train()

    optimizer_adapter.zero_grad(
        set_to_none=True
    )

    optimizer_classifier.zero_grad(
        set_to_none=True
    )

    logits1, logits2 = model(
        pseudo_x
    )

    loss = (
        F.cross_entropy(
            logits1,
            pseudo_y,
        )
        + F.cross_entropy(
            logits2,
            pseudo_y,
        )
    )

    loss.backward()

    optimizer_adapter.step()
    optimizer_classifier.step()

    return float(
        loss.detach().item()
    )


def build_optimizers(model):
    optimizer_adapter = torch.optim.SGD(
        model.adapter.parameters(),
        lr=LR_ADAPTER,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    optimizer_classifier = torch.optim.SGD(
        list(
            model.classifier1.parameters()
        )
        + list(
            model.classifier2.parameters()
        ),
        lr=LR_CLASSIFIER,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    return (
        optimizer_adapter,
        optimizer_classifier,
    )


@torch.no_grad()
def evaluate(
    model,
    target_x,
    target_y,
):
    model.eval()

    correct_total = 0
    total = 0

    class_correct = np.zeros(
        NUM_CLASSES,
        dtype=np.int64,
    )

    class_total = np.zeros(
        NUM_CLASSES,
        dtype=np.int64,
    )

    for start in range(
        0,
        len(target_x),
        EVAL_BATCH_SIZE,
    ):
        end = min(
            start + EVAL_BATCH_SIZE,
            len(target_x),
        )

        x = target_x[
            start:end
        ]

        y = target_y[
            start:end
        ]

        logits1, logits2 = model(
            x
        )

        p1 = F.softmax(
            logits1,
            dim=1,
        )

        p2 = F.softmax(
            logits2,
            dim=1,
        )

        prediction = (
            (p1 + p2)
            / 2.0
        ).argmax(
            dim=1
        )

        correct = (
            prediction
            == y
        )

        correct_total += int(
            correct.sum()
        )

        total += len(
            y
        )

        prediction_np = (
            prediction.cpu().numpy()
        )

        y_np = (
            y.cpu().numpy()
        )

        for class_id in range(
            NUM_CLASSES
        ):
            mask = (
                y_np
                == class_id
            )

            class_total[
                class_id
            ] += int(
                mask.sum()
            )

            class_correct[
                class_id
            ] += int(
                (
                    prediction_np[
                        mask
                    ]
                    == class_id
                ).sum()
            )

    per_class = (
        class_correct
        / np.maximum(
            class_total,
            1,
        )
        * 100.0
    )

    return {
        "overall": float(
            correct_total
            / max(
                total,
                1,
            )
            * 100.0
        ),
        "mca": float(
            per_class.mean()
        ),
        "per_class": {
            CLASSES[class_id]: float(
                per_class[class_id]
            )
            for class_id in range(
                NUM_CLASSES
            )
        },
    }


def main():
    args = parse_args()

    set_seed(
        args.seed
    )

    if not 0.0 < args.pseudo_fraction <= 1.0:
        raise RuntimeError(
            "--pseudo-fraction must be in (0, 1]"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print(
        "VISDA-2017 SUPPORT-GATED TARGET ADAPTATION"
    )
    print("=" * 100)
    print(
        "device=cpu"
    )
    print(
        f"seed={args.seed}"
    )
    print(
        f"epochs={EPOCHS}"
    )
    print(
        f"pseudo_fraction={args.pseudo_fraction}"
    )

    print(
        "\nLoading source feature cache..."
    )

    source_x, source_y = load_feature_cache(
        args.source_cache,
        require_labels=True,
    )

    print(
        f"source_samples={len(source_x)}"
    )

    print(
        "\nLoading target feature cache WITHOUT labels..."
    )

    target_x = load_feature_cache(
        args.target_cache,
        require_labels=False,
    )

    print(
        f"target_samples={len(target_x)}"
    )

    print(
        "\nLoading frozen RPC checkpoint..."
    )

    base_model, state_key = load_model(
        args.checkpoint
    )

    initial_state = copy.deepcopy(
        base_model.state_dict()
    )

    print(
        f"rpc_state_key={state_key}"
    )

    del base_model

    print(
        "\nLoading frozen independent-support artifact..."
    )

    support = load_support_artifact(
        args.support_artifact
    )

    support_n = len(
        support[
            "support_top1"
        ]
    )

    if support_n != len(
        target_x
    ):
        raise RuntimeError(
            "Support artifact target count does not match "
            "target feature cache"
        )

    print(
        f"support_samples={support_n}"
    )

    print(
        "\nVerifying frozen support arrays..."
    )

    stored_agreement = (
        support[
            "support_top1"
        ]
    )

    print(
        f"support_reliability_mean="
        f"{support['support_reliability'].mean():.6f}"
    )

    print(
        f"support_margin_mean="
        f"{support['support_margin'].mean():.6f}"
    )

    print(
        f"support_top1_score_mean="
        f"{support['support_top1_score'].mean():.6f}"
    )

    print(
        "\nNo target labels have been loaded."
    )

    modes = [
        "vanilla",
        "support_gated",
        "support_margin",
    ]

    results = {}

    for mode_index, mode in enumerate(
        modes
    ):
        print(
            "\n"
            + "=" * 100
        )

        print(
            f"ARM {mode}"
        )

        print(
            "=" * 100
        )

        set_seed(
            args.seed
            + mode_index
            * 1000
        )

        model = MCDModel()

        model.load_state_dict(
            copy.deepcopy(
                initial_state
            ),
            strict=True,
        )

        (
            optimizer_adapter,
            optimizer_classifier,
        ) = build_optimizers(
            model
        )

        history = []

        for epoch in range(
            1,
            EPOCHS + 1,
        ):
            prediction, confidence = (
                predict_target(
                    model,
                    target_x,
                )
            )

            pseudo_count = max(
                1,
                int(
                    round(
                        args.pseudo_fraction
                        * len(
                            target_x
                        )
                    )
                ),
            )

            selected = select_pseudo_labels(
                prediction,
                confidence,
                support[
                    "support_top1"
                ],
                support[
                    "support_margin"
                ],
                support[
                    "support_reliability"
                ],
                mode,
                pseudo_count,
            )

            selected_t = torch.from_numpy(
                selected
            ).long()

            pseudo_x = target_x[
                selected_t
            ]

            pseudo_y = torch.from_numpy(
                prediction[
                    selected
                ]
            ).long()

            selected_agreement = (
                prediction[
                    selected
                ]
                == support[
                    "support_top1"
                ][
                    selected
                ]
            )

            selected_support = (
                support[
                    "support_reliability"
                ][
                    selected
                ]
            )

            selected_margin = (
                support[
                    "support_margin"
                ][
                    selected
                ]
            )

            selected_confidence = (
                confidence[
                    selected
                ]
            )

            (
                source_batches,
                target_batches,
            ) = make_epoch_batches(
                len(source_x),
                len(target_x),
                epoch,
                args.seed
                + mode_index * 1000,
            )

            steps = int(
                source_batches.shape[0]
            )

            source_loss_total = 0.0
            classifier_loss_total = 0.0
            classifier_disc_total = 0.0
            target_disc_total = 0.0
            pseudo_loss_total = 0.0

            for step in range(
                steps
            ):
                source_indices = (
                    source_batches[
                        step
                    ]
                )

                target_indices = (
                    target_batches[
                        step
                        % int(
                            target_batches.shape[0]
                        )
                    ]
                )

                sx = source_x[
                    source_indices
                ]

                sy = source_y[
                    source_indices
                ]

                tx = target_x[
                    target_indices
                ]

                source_loss_total += (
                    source_update(
                        model,
                        optimizer_adapter,
                        sx,
                        sy,
                    )
                )

                (
                    source_loss_value,
                    classifier_disc,
                ) = classifier_update(
                    model,
                    optimizer_classifier,
                    sx,
                    sy,
                    tx,
                )

                classifier_loss_total += (
                    source_loss_value
                )

                classifier_disc_total += (
                    classifier_disc
                )

                target_disc_total += (
                    target_discrepancy_update(
                        model,
                        optimizer_adapter,
                        tx,
                    )
                )

                if len(
                    pseudo_x
                ) > 0:
                    generator = torch.Generator(
                        device="cpu"
                    )

                    generator.manual_seed(
                        args.seed
                        + mode_index * 100000
                        + epoch * 1000
                        + step
                    )

                    permutation = torch.randperm(
                        len(
                            pseudo_x
                        ),
                        generator=generator,
                    )

                    take = min(
                        BATCH_SIZE,
                        len(
                            permutation
                        ),
                    )

                    pseudo_indices = (
                        permutation[
                            :take
                        ]
                    )

                    px = pseudo_x[
                        pseudo_indices
                    ]

                    py = pseudo_y[
                        pseudo_indices
                    ]

                    pseudo_loss_total += (
                        pseudo_label_update(
                            model,
                            optimizer_adapter,
                            optimizer_classifier,
                            px,
                            py,
                        )
                    )

            current_prediction, current_confidence = (
                predict_target(
                    model,
                    target_x,
                )
            )

            history_entry = {
                "epoch": epoch,
                "pseudo_count": int(
                    len(selected)
                ),
                "pseudo_fraction": float(
                    len(selected)
                    / len(target_x)
                ),
                "selected_mean_confidence": float(
                    selected_confidence.mean()
                ),
                "selected_mean_support_reliability": float(
                    selected_support.mean()
                ),
                "selected_mean_support_margin": float(
                    selected_margin.mean()
                ),
                "selected_support_agreement": float(
                    selected_agreement.mean()
                ),
                "post_epoch_prediction_distribution": {
                    CLASSES[class_id]: int(
                        np.sum(
                            current_prediction
                            == class_id
                        )
                    )
                    for class_id in range(
                        NUM_CLASSES
                    )
                },
                "post_epoch_mean_confidence": float(
                    current_confidence.mean()
                ),
                "source_loss_mean": float(
                    source_loss_total
                    / max(
                        steps,
                        1,
                    )
                ),
                "classifier_source_loss_mean": float(
                    classifier_loss_total
                    / max(
                        steps,
                        1,
                    )
                ),
                "classifier_discrepancy_mean": float(
                    classifier_disc_total
                    / max(
                        steps,
                        1,
                    )
                ),
                "target_discrepancy_mean": float(
                    target_disc_total
                    / max(
                        steps,
                        1,
                    )
                ),
                "pseudo_loss_mean": float(
                    pseudo_loss_total
                    / max(
                        steps,
                        1,
                    )
                ),
            }

            history.append(
                history_entry
            )

            print(
                f"{mode:18s} "
                f"epoch={epoch}/{EPOCHS} "
                f"pseudo={len(selected):6d} "
                f"agreement={selected_agreement.mean() * 100:6.2f}% "
                f"conf={selected_confidence.mean():.4f} "
                f"support={selected_support.mean():.4f} "
                f"margin={selected_margin.mean():.4f}"
            )

        print(
            "\nTraining complete for arm."
        )

        print(
            "Target labels are now loaded for this arm's final evaluation."
        )

        target_y = load_feature_cache(
            args.target_cache,
            require_labels=True,
        )[1]

        final_metrics = evaluate(
            model,
            target_x,
            target_y,
        )

        results[
            mode
        ] = {
            "history": history,
            "final_metrics": final_metrics,
            "final_state_dict": copy.deepcopy(
                model.state_dict()
            ),
        }

        print(
            f"{mode} FINAL "
            f"MCA={final_metrics['mca']:.2f}% "
            f"OA={final_metrics['overall']:.2f}% "
            f"truck={final_metrics['per_class']['truck']:.2f}% "
            f"car={final_metrics['per_class']['car']:.2f}% "
            f"bus={final_metrics['per_class']['bus']:.2f}% "
            f"train={final_metrics['per_class']['train']:.2f}%"
        )

        del target_y
        del model

    vanilla = results[
        "vanilla"
    ][
        "final_metrics"
    ]

    gated = results[
        "support_gated"
    ][
        "final_metrics"
    ]

    margin = results[
        "support_margin"
    ][
        "final_metrics"
    ]

    print(
        "\n"
        + "=" * 100
    )

    print(
        "FINAL COMPARISON"
    )

    print(
        "=" * 100
    )

    for mode in modes:
        metrics = results[
            mode
        ][
            "final_metrics"
        ]

        print(
            f"{mode:18s} "
            f"MCA={metrics['mca']:.2f}% "
            f"OA={metrics['overall']:.2f}% "
            f"truck={metrics['per_class']['truck']:.2f}% "
            f"car={metrics['per_class']['car']:.2f}% "
            f"bus={metrics['per_class']['bus']:.2f}% "
            f"train={metrics['per_class']['train']:.2f}%"
        )

    print(
        "\nMCA DELTAS VS VANILLA"
    )

    print(
        f"support_gated={gated['mca'] - vanilla['mca']:+.2f} pp"
    )

    print(
        f"support_margin={margin['mca'] - vanilla['mca']:+.2f} pp"
    )

    print(
        "\nOA DELTAS VS VANILLA"
    )

    print(
        f"support_gated={gated['overall'] - vanilla['overall']:+.2f} pp"
    )

    print(
        f"support_margin={margin['overall'] - vanilla['overall']:+.2f} pp"
    )

    print(
        "\nCLASS DELTAS VS VANILLA"
    )

    for class_name in CLASSES:
        vanilla_value = vanilla[
            "per_class"
        ][
            class_name
        ]

        gated_value = gated[
            "per_class"
        ][
            class_name
        ]

        margin_value = margin[
            "per_class"
        ][
            class_name
        ]

        print(
            f"{class_name:12s} "
            f"gated={gated_value - vanilla_value:+.2f} pp "
            f"margin={margin_value - vanilla_value:+.2f} pp"
        )

    if (
        gated["mca"]
        > vanilla["mca"]
        and margin["mca"]
        > vanilla["mca"]
    ):
        decision = (
            "support_gating_improves_adaptation"
        )

    elif (
        gated["mca"]
        > vanilla["mca"]
        or margin["mca"]
        > vanilla["mca"]
    ):
        decision = (
            "support_gating_shows_partial_adaptation_benefit"
        )

    else:
        decision = (
            "support_gating_does_not_improve_adaptation"
        )

    print(
        f"\nSCIENTIFIC_DECISION={decision}"
    )

    output_npz = (
        args.output_dir
        / f"support_gated_target_adaptation_seed{args.seed}.npz"
    )

    np.savez_compressed(
        output_npz,
        vanilla_mca=np.asarray(
            vanilla["mca"]
        ),
        gated_mca=np.asarray(
            gated["mca"]
        ),
        margin_mca=np.asarray(
            margin["mca"]
        ),
        vanilla_oa=np.asarray(
            vanilla["overall"]
        ),
        gated_oa=np.asarray(
            gated["overall"]
        ),
        margin_oa=np.asarray(
            margin["overall"]
        ),
    )

    output_json = (
        args.output_dir
        / f"support_gated_target_adaptation_seed{args.seed}.json"
    )

    summary = {
        "experiment": (
            "visda_support_gated_target_adaptation"
        ),
        "seed": args.seed,
        "device": "cpu",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "pseudo_fraction": args.pseudo_fraction,
        "constraints": {
            "target_labels_used_during_training": False,
            "target_labels_used_for_selector_construction": False,
            "target_labels_used_for_support_signal": False,
            "support_artifact_frozen": True,
            "rpc_checkpoint_frozen_at_initialization": True,
            "training_model_uses_current_predictions": True,
        },
        "inputs": {
            "checkpoint": str(
                args.checkpoint
            ),
            "support_artifact": str(
                args.support_artifact
            ),
            "source_cache": str(
                args.source_cache
            ),
            "target_cache": str(
                args.target_cache
            ),
        },
        "support_artifact": {
            "samples": int(
                support_n
            ),
            "support_reliability_mean": float(
                support[
                    "support_reliability"
                ].mean()
            ),
            "support_margin_mean": float(
                support[
                    "support_margin"
                ].mean()
            ),
            "support_top1_score_mean": float(
                support[
                    "support_top1_score"
                ].mean()
            ),
        },
        "results": {
            mode: {
                "history": results[
                    mode
                ][
                    "history"
                ],
                "final_metrics": results[
                    mode
                ][
                    "final_metrics"
                ],
            }
            for mode in modes
        },
        "comparison": {
            "support_gated_MCA_delta_vs_vanilla": float(
                gated["mca"]
                - vanilla["mca"]
            ),
            "support_margin_MCA_delta_vs_vanilla": float(
                margin["mca"]
                - vanilla["mca"]
            ),
            "support_gated_OA_delta_vs_vanilla": float(
                gated["overall"]
                - vanilla["overall"]
            ),
            "support_margin_OA_delta_vs_vanilla": float(
                margin["overall"]
                - vanilla["overall"]
            ),
            "scientific_decision": decision,
        },
        "outputs": {
            "npz": str(
                output_npz
            ),
            "json": str(
                output_json
            ),
        },
    }

    with open(
        output_json,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )

    print(
        "\nOUTPUTS"
    )

    print(
        f"npz={output_npz}"
    )

    print(
        f"json={output_json}"
    )

    print(
        "\nSUPPORT-GATED TARGET ADAPTATION COMPLETE"
    )


if __name__ == "__main__":
    main()