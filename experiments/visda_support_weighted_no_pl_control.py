import argparse
import copy
import hashlib
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

LR_ADAPTER = 1e-3
LR_CLASSIFIER = 1e-2
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

PSEUDO_FRACTION = 0.20
EPS = 1e-12

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

MODES = [
    "no_pseudo_label",
    "vanilla",
    "norm_matched_global_control",
    "confidence_weighted",
    "confidence_support_weighted",
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "checkpoints/visda_rpc_causal_experiment/rpc_seed42.pt"
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
            "checkpoints/visda_support_weighted_no_pl_control"
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


def safe_load(path):
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


def load_cache(cache_dir, require_labels):
    files = sorted(
        cache_dir.glob("chunk_*.pt")
    )

    if not files:
        raise RuntimeError(
            f"No cache chunks found in {cache_dir}"
        )

    feature_parts = []
    label_parts = []

    for path in files:
        payload = safe_load(path)

        if not isinstance(payload, dict):
            raise RuntimeError(
                f"Invalid cache object: {path}"
            )

        if "features" not in payload:
            raise RuntimeError(
                f"Missing features in {path}"
            )

        x = payload["features"].float().cpu()

        if x.ndim != 2:
            raise RuntimeError(
                f"Invalid feature shape in {path}: {tuple(x.shape)}"
            )

        if x.shape[1] != INPUT_DIM:
            raise RuntimeError(
                f"Expected {INPUT_DIM}-D features, "
                f"found {x.shape[1]} in {path}"
            )

        feature_parts.append(x)

        if require_labels:
            if "labels" not in payload:
                raise RuntimeError(
                    f"Missing labels in {path}"
                )

            y = payload["labels"].long().cpu()

            if len(x) != len(y):
                raise RuntimeError(
                    f"Feature/label mismatch in {path}"
                )

            label_parts.append(y)

    features = torch.cat(
        feature_parts,
        dim=0,
    )

    if require_labels:
        labels = torch.cat(
            label_parts,
            dim=0,
        )

        if len(features) != len(labels):
            raise RuntimeError(
                "Final feature/label count mismatch"
            )

        return features, labels

    return features


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
                inplace=True
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


def load_rpc_model(path):
    payload = safe_load(path)

    if not isinstance(payload, dict):
        raise RuntimeError(
            "RPC checkpoint is not a dictionary"
        )

    if "student_state_dict" in payload:
        state = payload[
            "student_state_dict"
        ]
        state_key = "student_state_dict"

    elif "state_dict" in payload:
        state = payload[
            "state_dict"
        ]
        state_key = "state_dict"

    else:
        state = payload
        state_key = "root"

    model = MCDModel()

    model.load_state_dict(
        copy.deepcopy(state),
        strict=True,
    )

    return model, state_key


def load_support(path, n_target):
    payload = np.load(
        path,
        allow_pickle=False,
    )

    required = [
        "support_probabilities",
        "support_top1",
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
            "Missing support arrays: "
            + ", ".join(missing)
        )

    probabilities = np.asarray(
        payload["support_probabilities"],
        dtype=np.float64,
    )

    top1 = np.asarray(
        payload["support_top1"],
        dtype=np.int64,
    )

    margin = np.asarray(
        payload["support_margin"],
        dtype=np.float64,
    )

    reliability = np.asarray(
        payload["support_reliability"],
        dtype=np.float64,
    )

    expected = (
        n_target,
        NUM_CLASSES,
    )

    if probabilities.shape != expected:
        raise RuntimeError(
            f"support_probabilities shape "
            f"{probabilities.shape} != {expected}"
        )

    for name, value in [
        ("support_top1", top1),
        ("support_margin", margin),
        ("support_reliability", reliability),
    ]:
        if len(value) != n_target:
            raise RuntimeError(
                f"{name} length mismatch"
            )

        if not np.all(
            np.isfinite(value)
        ):
            raise RuntimeError(
                f"{name} contains non-finite values"
            )

    if np.any(
        top1 < 0
    ) or np.any(
        top1 >= NUM_CLASSES
    ):
        raise RuntimeError(
            "support_top1 contains invalid class IDs"
        )

    return {
        "probabilities": probabilities,
        "top1": top1,
        "margin": margin,
        "reliability": reliability,
    }


@torch.no_grad()
def predict_target(
    model,
    target_x,
):
    model.eval()

    probability_parts = []

    for start in range(
        0,
        len(target_x),
        EVAL_BATCH_SIZE,
    ):
        end = min(
            start + EVAL_BATCH_SIZE,
            len(target_x),
        )

        logits1, logits2 = model(
            target_x[start:end]
        )

        p1 = F.softmax(
            logits1,
            dim=1,
        )

        p2 = F.softmax(
            logits2,
            dim=1,
        )

        probability_parts.append(
            (
                (p1 + p2)
                / 2.0
            ).cpu()
        )

    probabilities = torch.cat(
        probability_parts,
        dim=0,
    )

    confidence, prediction = probabilities.max(
        dim=1
    )

    return (
        probabilities.numpy().astype(
            np.float64
        ),
        prediction.numpy().astype(
            np.int64
        ),
        confidence.numpy().astype(
            np.float64
        ),
    )


def make_batches(
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

    if source_steps <= 0:
        raise RuntimeError(
            "Source dataset too small"
        )

    if target_steps <= 0:
        raise RuntimeError(
            "Target dataset too small"
        )

    source_perm = source_perm[
        :source_steps * BATCH_SIZE
    ]

    target_perm = target_perm[
        :target_steps * BATCH_SIZE
    ]

    return (
        source_perm.view(
            source_steps,
            BATCH_SIZE,
        ),
        target_perm.view(
            target_steps,
            BATCH_SIZE,
        ),
    )


def source_update(
    model,
    optimizer,
    source_x,
    source_y,
):
    model.train()

    optimizer.zero_grad(
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

    optimizer.step()

    return float(
        loss.detach()
    )


def classifier_update(
    model,
    optimizer,
    source_x,
    source_y,
    target_x,
):
    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    source_z = model.encode(
        source_x
    )

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

    loss = (
        source_loss
        - discrepancy
    )

    loss.backward()

    optimizer.step()

    return (
        float(
            source_loss.detach()
        ),
        float(
            discrepancy.detach()
        ),
    )


def target_update(
    model,
    optimizer,
    target_x,
):
    model.train()

    optimizer.zero_grad(
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
        p1
        - p2
    ).abs().mean()

    discrepancy.backward()

    optimizer.step()

    return float(
        discrepancy.detach()
    )


def pseudo_loss(
    model,
    pseudo_x,
    pseudo_y,
    weights,
):
    logits1, logits2 = model(
        pseudo_x
    )

    loss1 = F.cross_entropy(
        logits1,
        pseudo_y,
        reduction="none",
    )

    loss2 = F.cross_entropy(
        logits2,
        pseudo_y,
        reduction="none",
    )

    losses = (
        loss1 + loss2
    )

    vanilla = losses.mean()

    if weights is None:
        weighted = vanilla

    else:
        w = weights.detach()

        weighted = (
            losses
            * w
        ).sum() / (
            w.sum().clamp_min(
                EPS
            )
        )

    return (
        vanilla,
        weighted,
    )


def parameters_all(model):
    return [
        p
        for p in model.parameters()
        if p.requires_grad
    ]


def grad_norm(
    gradients
):
    total = 0.0

    for gradient in gradients:
        if gradient is not None:
            total += float(
                gradient.detach().pow(2).sum().item()
            )

    return float(
        np.sqrt(total)
    )


def install_gradients(
    model,
    gradients,
):
    for parameter, gradient in zip(
        parameters_all(model),
        gradients,
    ):
        if gradient is None:
            parameter.grad = None
        else:
            parameter.grad = gradient.detach().clone()


def pseudo_update_weighted(
    model,
    adapter_optimizer,
    classifier_optimizer,
    pseudo_x,
    pseudo_y,
    weights,
):
    model.train()

    adapter_optimizer.zero_grad(
        set_to_none=True
    )

    classifier_optimizer.zero_grad(
        set_to_none=True
    )

    _, weighted = pseudo_loss(
        model,
        pseudo_x,
        pseudo_y,
        weights,
    )

    parameters = parameters_all(
        model
    )

    gradients = torch.autograd.grad(
        weighted,
        parameters,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )

    install_gradients(
        model,
        gradients,
    )

    adapter_ids = {
        id(parameter)
        for parameter in model.adapter.parameters()
    }

    adapter_squared = 0.0
    classifier_squared = 0.0

    for parameter in parameters:
        if parameter.grad is None:
            continue

        value = float(
            parameter.grad.detach().pow(2).sum().item()
        )

        if id(parameter) in adapter_ids:
            adapter_squared += value
        else:
            classifier_squared += value

    adapter_norm = float(
        np.sqrt(adapter_squared)
    )

    classifier_norm = float(
        np.sqrt(classifier_squared)
    )

    total_norm = float(
        np.sqrt(
            adapter_norm ** 2
            + classifier_norm ** 2
        )
    )

    adapter_optimizer.step()
    classifier_optimizer.step()

    return {
        "loss": float(
            weighted.detach()
        ),
        "adapter_norm": adapter_norm,
        "classifier_norm": classifier_norm,
        "total_norm": total_norm,
    }


def pseudo_update_norm_matched(
    model,
    adapter_optimizer,
    classifier_optimizer,
    pseudo_x,
    pseudo_y,
    treatment_weights,
):
    model.train()

    adapter_optimizer.zero_grad(
        set_to_none=True
    )

    classifier_optimizer.zero_grad(
        set_to_none=True
    )

    vanilla_loss, weighted_loss = pseudo_loss(
        model,
        pseudo_x,
        pseudo_y,
        treatment_weights,
    )

    parameters = parameters_all(
        model
    )

    vanilla_gradients = torch.autograd.grad(
        vanilla_loss,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )

    weighted_gradients = torch.autograd.grad(
        weighted_loss,
        parameters,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )

    vanilla_norm = grad_norm(
        vanilla_gradients
    )

    weighted_norm = grad_norm(
        weighted_gradients
    )

    scale = (
        weighted_norm
        / max(
            vanilla_norm,
            EPS,
        )
    )

    scaled_vanilla = []

    for gradient in vanilla_gradients:
        if gradient is None:
            scaled_vanilla.append(
                None
            )
        else:
            scaled_vanilla.append(
                gradient.detach()
                * scale
            )

    install_gradients(
        model,
        scaled_vanilla,
    )

    adapter_ids = {
        id(parameter)
        for parameter in model.adapter.parameters()
    }

    adapter_squared = 0.0
    classifier_squared = 0.0

    for parameter in parameters:
        if parameter.grad is None:
            continue

        value = float(
            parameter.grad.detach().pow(2).sum().item()
        )

        if id(parameter) in adapter_ids:
            adapter_squared += value
        else:
            classifier_squared += value

    adapter_norm = float(
        np.sqrt(adapter_squared)
    )

    classifier_norm = float(
        np.sqrt(classifier_squared)
    )

    applied_norm = float(
        np.sqrt(
            adapter_norm ** 2
            + classifier_norm ** 2
        )
    )

    adapter_optimizer.step()
    classifier_optimizer.step()

    return {
        "loss": float(
            vanilla_loss.detach()
        ),
        "counterfactual_weighted_loss": float(
            weighted_loss.detach()
        ),
        "vanilla_norm": vanilla_norm,
        "counterfactual_weighted_norm": weighted_norm,
        "applied_norm": applied_norm,
        "scale": float(
            scale
        ),
        "match_error": float(
            abs(
                applied_norm
                - weighted_norm
            )
        ),
        "adapter_norm": adapter_norm,
        "classifier_norm": classifier_norm,
    }


def make_weights(
    mode,
    prediction,
    confidence,
    support_probabilities,
):
    n = len(
        prediction
    )

    if mode in (
        "vanilla",
        "norm_matched_global_control",
    ):
        raw = np.ones(
            n,
            dtype=np.float64,
        )

    elif mode == "confidence_weighted":
        raw = np.clip(
            confidence,
            1e-8,
            None,
        )

    elif mode == "confidence_support_weighted":
        rows = np.arange(n)

        current_support = (
            support_probabilities[
                rows,
                prediction,
            ]
        )

        raw = (
            confidence
            * np.clip(
                current_support,
                1e-8,
                None,
            )
        )

    else:
        raise RuntimeError(
            f"Unknown weighting mode: {mode}"
        )

    mean_raw = float(
        raw.mean()
    )

    if mean_raw <= EPS:
        return np.ones(
            n,
            dtype=np.float32,
        )

    return (
        raw / mean_raw
    ).astype(
        np.float32
    )


def run_arm(
    mode,
    initial_state,
    source_x,
    source_y,
    target_x,
    support,
    args,
):
    set_seed(
        args.seed
    )

    model = MCDModel()

    model.load_state_dict(
        copy.deepcopy(
            initial_state
        ),
        strict=True,
    )

    adapter_optimizer = torch.optim.SGD(
        model.adapter.parameters(),
        lr=LR_ADAPTER,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    classifier_optimizer = torch.optim.SGD(
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

    history = []

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        (
            teacher_probabilities,
            teacher_prediction,
            teacher_confidence,
        ) = predict_target(
            model,
            target_x,
        )

        pseudo_count = max(
            1,
            int(
                round(
                    args.pseudo_fraction
                    * len(target_x)
                )
            ),
        )

        order = np.argsort(
            -teacher_confidence,
            kind="mergesort",
        )

        selected = order[
            :pseudo_count
        ]

        selected_tensor = (
            torch.from_numpy(
                selected
            ).long()
        )

        pseudo_x = target_x[
            selected_tensor
        ]

        pseudo_y = torch.from_numpy(
            teacher_prediction[
                selected
            ]
        ).long()

        selected_prediction = (
            teacher_prediction[
                selected
            ]
        )

        selected_confidence = (
            teacher_confidence[
                selected
            ]
        )

        selected_support_probabilities = (
            support[
                "probabilities"
            ][
                selected
            ]
        )

        if mode == "no_pseudo_label":
            weights = np.empty(
                0,
                dtype=np.float32,
            )

        else:
            weights = make_weights(
                mode,
                selected_prediction,
                selected_confidence,
                selected_support_probabilities,
            )

        (
            source_batches,
            target_batches,
        ) = make_batches(
            len(source_x),
            len(target_x),
            epoch,
            args.seed,
        )

        pseudo_generator = torch.Generator(
            device="cpu"
        )

        pseudo_generator.manual_seed(
            args.seed
            + epoch * 100003
        )

        pseudo_order = torch.randperm(
            pseudo_count,
            generator=pseudo_generator,
        )

        steps = int(
            source_batches.shape[0]
        )

        source_loss_sum = 0.0
        classifier_loss_sum = 0.0
        classifier_disc_sum = 0.0
        target_disc_sum = 0.0

        pseudo_loss_sum = 0.0
        pseudo_grad_sum = 0.0

        adapter_grad_sum = 0.0
        classifier_grad_sum = 0.0

        counterfactual_grad_sum = 0.0
        norm_error_sum = 0.0
        norm_scale_sum = 0.0

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
                    % len(target_batches)
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

            source_loss_sum += (
                source_update(
                    model,
                    adapter_optimizer,
                    sx,
                    sy,
                )
            )

            (
                classifier_loss,
                classifier_disc,
            ) = classifier_update(
                model,
                classifier_optimizer,
                sx,
                sy,
                tx,
            )

            classifier_loss_sum += (
                classifier_loss
            )

            classifier_disc_sum += (
                classifier_disc
            )

            target_disc_sum += (
                target_update(
                    model,
                    adapter_optimizer,
                    tx,
                )
            )

            if mode == "no_pseudo_label":
                continue

            take = min(
                BATCH_SIZE,
                pseudo_count,
            )

            start = (
                step
                * take
            ) % pseudo_count

            batch_indices = (
                pseudo_order[
                    start:
                    start + take
                ]
            )

            if len(
                batch_indices
            ) < take:
                remaining = (
                    take
                    - len(
                        batch_indices
                    )
                )

                batch_indices = torch.cat(
                    [
                        batch_indices,
                        pseudo_order[
                            :remaining
                        ],
                    ]
                )

            batch_indices_np = (
                batch_indices.numpy()
            )

            px = pseudo_x[
                batch_indices
            ]

            py = pseudo_y[
                batch_indices
            ]

            pw = torch.from_numpy(
                weights[
                    batch_indices_np
                ]
            ).float()

            if mode == "norm_matched_global_control":
                treatment_weights = torch.from_numpy(
                    make_weights(
                        "confidence_support_weighted",
                        selected_prediction[
                            batch_indices_np
                        ],
                        selected_confidence[
                            batch_indices_np
                        ],
                        selected_support_probabilities[
                            batch_indices_np
                        ],
                    )
                ).float()

                result = pseudo_update_norm_matched(
                    model,
                    adapter_optimizer,
                    classifier_optimizer,
                    px,
                    py,
                    treatment_weights,
                )

                counterfactual_grad_sum += (
                    result[
                        "counterfactual_weighted_norm"
                    ]
                )

                norm_error_sum += (
                    result[
                        "match_error"
                    ]
                )

                norm_scale_sum += (
                    result[
                        "scale"
                    ]
                )

            else:
                result = pseudo_update_weighted(
                    model,
                    adapter_optimizer,
                    classifier_optimizer,
                    px,
                    py,
                    pw,
                )

            pseudo_loss_sum += (
                result[
                    "loss"
                ]
            )

            pseudo_grad_sum += (
                result[
                    "applied_norm"
                ]
                if "applied_norm" in result
                else result[
                    "total_norm"
                ]
            )

            adapter_grad_sum += (
                result[
                    "adapter_norm"
                ]
            )

            classifier_grad_sum += (
                result[
                    "classifier_norm"
                ]
            )

        selected_support = (
            support[
                "probabilities"
            ][
                selected,
                selected_prediction,
            ]
            if len(selected) > 0
            else np.empty(
                0,
                dtype=np.float64,
            )
        )

        selected_margin = (
            support[
                "margin"
            ][
                selected
            ]
            if len(selected) > 0
            else np.empty(
                0,
                dtype=np.float64,
            )
        )

        selected_agreement = (
            selected_prediction
            == support[
                "top1"
            ][
                selected
            ]
            if len(selected) > 0
            else np.empty(
                0,
                dtype=bool,
            )
        )

        history.append(
            {
                "epoch": epoch,
                "pseudo_count": int(
                    pseudo_count
                ),
                "selected_sha256": hashlib.sha256(
                    selected.astype(
                        np.int64
                    ).tobytes()
                ).hexdigest(),
                "selected_mean_confidence": float(
                    selected_confidence.mean()
                ),
                "selected_mean_current_class_support": float(
                    selected_support.mean()
                ),
                "selected_mean_support_margin": float(
                    selected_margin.mean()
                ),
                "selected_support_agreement": float(
                    selected_agreement.mean()
                ),
                "weight_mean": float(
                    weights.mean()
                    if len(weights)
                    else 0.0
                ),
                "weight_std": float(
                    weights.std()
                    if len(weights)
                    else 0.0
                ),
                "weight_min": float(
                    weights.min()
                    if len(weights)
                    else 0.0
                ),
                "weight_max": float(
                    weights.max()
                    if len(weights)
                    else 0.0
                ),
                "source_loss_mean": float(
                    source_loss_sum
                    / max(
                        steps,
                        1,
                    )
                ),
                "classifier_loss_mean": float(
                    classifier_loss_sum
                    / max(
                        steps,
                        1,
                    )
                ),
                "classifier_discrepancy_mean": float(
                    classifier_disc_sum
                    / max(
                        steps,
                        1,
                    )
                ),
                "target_discrepancy_mean": float(
                    target_disc_sum
                    / max(
                        steps,
                        1,
                    )
                ),
                "pseudo_loss_mean": float(
                    pseudo_loss_sum
                    / max(
                        steps,
                        1,
                    )
                ),
                "pseudo_gradient_norm_mean": float(
                    pseudo_grad_sum
                    / max(
                        steps,
                        1,
                    )
                ),
                "pseudo_adapter_gradient_norm_mean": float(
                    adapter_grad_sum
                    / max(
                        steps,
                        1,
                    )
                ),
                "pseudo_classifier_gradient_norm_mean": float(
                    classifier_grad_sum
                    / max(
                        steps,
                        1,
                    )
                ),
                "counterfactual_weighted_gradient_norm_mean": float(
                    counterfactual_grad_sum
                    / max(
                        steps,
                        1,
                    )
                ),
                "norm_match_error_mean": float(
                    norm_error_sum
                    / max(
                        steps,
                        1,
                    )
                ),
                "norm_match_scale_mean": float(
                    norm_scale_sum
                    / max(
                        steps,
                        1,
                    )
                ),
            }
        )

        print(
            f"{mode:34s} "
            f"epoch={epoch}/{EPOCHS} "
            f"pseudo={pseudo_count:6d} "
            f"conf={selected_confidence.mean():.6f} "
            f"support={selected_support.mean():.6f} "
            f"margin={selected_margin.mean():.6f} "
            f"wmean={weights.mean() if len(weights) else 0.0:.6f}"
        )

    return {
        "history": history,
        "state_dict": copy.deepcopy(
            model.state_dict()
        ),
    }


def expected_calibration_error(
    probabilities,
    labels,
    bins=15,
):
    confidence = probabilities.max(
        axis=1
    )

    prediction = probabilities.argmax(
        axis=1
    )

    correctness = (
        prediction
        == labels
    ).astype(
        np.float64
    )

    ece = 0.0

    for index in range(
        bins
    ):
        lower = index / bins
        upper = (index + 1) / bins

        if index == bins - 1:
            mask = (
                (confidence >= lower)
                & (confidence <= upper)
            )
        else:
            mask = (
                (confidence >= lower)
                & (confidence < upper)
            )

        if not np.any(mask):
            continue

        ece += (
            mask.mean()
            * abs(
                confidence[
                    mask
                ].mean()
                - correctness[
                    mask
                ].mean()
            )
        )

    return float(ece)


def evaluate_model(
    model,
    target_x,
    target_y,
):
    probabilities, prediction, _ = (
        predict_target(
            model,
            target_x,
        )
    )

    labels = target_y.numpy().astype(
        np.int64
    )

    correctness = (
        prediction
        == labels
    )

    per_class = {}

    for class_id, class_name in enumerate(
        CLASSES
    ):
        mask = (
            labels == class_id
        )

        per_class[
            class_name
        ] = float(
            np.mean(
                prediction[
                    mask
                ]
                == class_id
            )
            * 100.0
        )

    one_hot = np.zeros_like(
        probabilities
    )

    one_hot[
        np.arange(
            len(labels)
        ),
        labels,
    ] = 1.0

    brier = float(
        np.mean(
            np.sum(
                (
                    probabilities
                    - one_hot
                ) ** 2,
                axis=1,
            )
        )
    )

    true_probability = probabilities[
        np.arange(
            len(labels)
        ),
        labels,
    ]

    nll = float(
        -np.mean(
            np.log(
                np.clip(
                    true_probability,
                    1e-12,
                    None,
                )
            )
        )
    )

    confidence = probabilities.max(
        axis=1
    )

    entropy = float(
        -np.mean(
            np.sum(
                probabilities
                * np.log(
                    np.clip(
                        probabilities,
                        1e-12,
                        None,
                    )
                ),
                axis=1,
            )
        )
    )

    return {
        "overall": float(
            correctness.mean()
            * 100.0
        ),
        "mca": float(
            np.mean(
                list(
                    per_class.values()
                )
            )
        ),
        "ece": expected_calibration_error(
            probabilities,
            labels,
        ),
        "brier": brier,
        "nll": nll,
        "mean_confidence": float(
            confidence.mean()
        ),
        "confidence_accuracy_gap": float(
            confidence.mean()
            - correctness.mean()
        ),
        "mean_entropy": entropy,
        "per_class": per_class,
    }


def main():
    args = parse_args()

    set_seed(
        args.seed
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print(
        "VISDA-2017 PSEUDO-LABEL CAUSAL CONTROL COMPLETION"
    )
    print("=" * 100)
    print(
        f"device=cpu"
    )
    print(
        f"seed={args.seed}"
    )
    print(
        f"epochs={EPOCHS}"
    )
    print(
        f"batch_size={BATCH_SIZE}"
    )
    print(
        f"pseudo_fraction={args.pseudo_fraction}"
    )

    print(
        "\nLoading source cache..."
    )

    source_x, source_y = load_cache(
        args.source_cache,
        True,
    )

    print(
        f"source_features={tuple(source_x.shape)}"
    )

    print(
        "\nLoading target features WITHOUT accessing labels..."
    )

    target_x = load_cache(
        args.target_cache,
        False,
    )

    print(
        f"target_features={tuple(target_x.shape)}"
    )

    print(
        "\nLoading frozen RPC checkpoint..."
    )

    base_model, state_key = load_rpc_model(
        args.checkpoint
    )

    initial_state = copy.deepcopy(
        base_model.state_dict()
    )

    del base_model

    print(
        f"rpc_state_key={state_key}"
    )

    print(
        "\nLoading frozen independent-support artifact..."
    )

    support = load_support(
        args.support_artifact,
        len(target_x),
    )

    print(
        f"support_samples={len(target_x)}"
    )

    print(
        "\nAll training quantities are label-free."
    )

    results = {}

    for mode in MODES:
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

        results[
            mode
        ] = run_arm(
            mode,
            initial_state,
            source_x,
            source_y,
            target_x,
            support,
            args,
        )

    print(
        "\nAll training arms are frozen."
    )

    print(
        "Loading target labels for evaluation only..."
    )

    _, target_y = load_cache(
        args.target_cache,
        True,
    )

    final_metrics = {}

    print(
        "\n"
        + "=" * 100
    )

    print(
        "FINAL FIXED-EPOCH EVALUATION"
    )

    print(
        "=" * 100
    )

    for mode in MODES:
        model = MCDModel()

        model.load_state_dict(
            results[
                mode
            ][
                "state_dict"
            ],
            strict=True,
        )

        metrics = evaluate_model(
            model,
            target_x,
            target_y,
        )

        final_metrics[
            mode
        ] = metrics

        print(
            f"{mode:34s} "
            f"MCA={metrics['mca']:.2f}% "
            f"OA={metrics['overall']:.2f}% "
            f"ECE={metrics['ece']:.6f} "
            f"Brier={metrics['brier']:.6f} "
            f"NLL={metrics['nll']:.6f} "
            f"conf={metrics['mean_confidence']:.6f} "
            f"gap={metrics['confidence_accuracy_gap']:.6f}"
        )

    no_pl = final_metrics[
        "no_pseudo_label"
    ]

    vanilla = final_metrics[
        "vanilla"
    ]

    norm_control = final_metrics[
        "norm_matched_global_control"
    ]

    confidence = final_metrics[
        "confidence_weighted"
    ]

    confidence_support = final_metrics[
        "confidence_support_weighted"
    ]

    print(
        "\nMCA DELTAS"
    )

    print(
        f"vanilla_vs_no_PL="
        f"{vanilla['mca'] - no_pl['mca']:+.2f} pp"
    )

    print(
        f"confidence_vs_no_PL="
        f"{confidence['mca'] - no_pl['mca']:+.2f} pp"
    )

    print(
        f"confidence_support_vs_no_PL="
        f"{confidence_support['mca'] - no_pl['mca']:+.2f} pp"
    )

    print(
        f"confidence_support_vs_vanilla="
        f"{confidence_support['mca'] - vanilla['mca']:+.2f} pp"
    )

    print(
        f"confidence_support_vs_norm_control="
        f"{confidence_support['mca'] - norm_control['mca']:+.2f} pp"
    )

    print(
        "\nOA DELTAS"
    )

    print(
        f"vanilla_vs_no_PL="
        f"{vanilla['overall'] - no_pl['overall']:+.2f} pp"
    )

    print(
        f"confidence_support_vs_no_PL="
        f"{confidence_support['overall'] - no_pl['overall']:+.2f} pp"
    )

    print(
        f"confidence_support_vs_vanilla="
        f"{confidence_support['overall'] - vanilla['overall']:+.2f} pp"
    )

    print(
        "\nCLASS DELTAS: CONFIDENCE-SUPPORT VS VANILLA"
    )

    for class_name in CLASSES:
        delta = (
            confidence_support[
                "per_class"
            ][
                class_name
            ]
            - vanilla[
                "per_class"
            ][
                class_name
            ]
        )

        print(
            f"{class_name:12s} "
            f"{delta:+.2f} pp"
        )

    print(
        "\nNORM-MATCH CONTROL"
    )

    for row in results[
        "norm_matched_global_control"
    ][
        "history"
    ]:
        print(
            f"epoch={row['epoch']} "
            f"scale={row['norm_match_scale_mean']:.8f} "
            f"error={row['norm_match_error_mean']:.8e}"
        )

    if (
        confidence_support["mca"]
        > no_pl["mca"]
        and confidence_support["mca"]
        > vanilla["mca"]
        and confidence_support["mca"]
        > norm_control["mca"]
    ):
        decision = (
            "confidence_support_has_positive_net_adaptation_value"
        )

    elif (
        confidence_support["mca"]
        > vanilla["mca"]
    ):
        decision = (
            "confidence_support_reduces_pseudo_label_damage_but_is_below_no_PL"
        )

    else:
        decision = (
            "confidence_support_does_not_improve_adaptation"
        )

    print(
        f"\nSCIENTIFIC_DECISION={decision}"
    )

    output_npz = (
        args.output_dir
        / f"support_weighted_no_pl_control_seed{args.seed}.npz"
    )

    np.savez_compressed(
        output_npz,
        no_pl_mca=no_pl["mca"],
        vanilla_mca=vanilla["mca"],
        norm_control_mca=norm_control["mca"],
        confidence_mca=confidence["mca"],
        confidence_support_mca=confidence_support["mca"],
        no_pl_oa=no_pl["overall"],
        vanilla_oa=vanilla["overall"],
        norm_control_oa=norm_control["overall"],
        confidence_oa=confidence["overall"],
        confidence_support_oa=confidence_support["overall"],
    )

    output_json = (
        args.output_dir
        / f"support_weighted_no_pl_control_seed{args.seed}.json"
    )

    summary = {
        "experiment": (
            "visda_support_weighted_no_pl_control"
        ),
        "seed": int(args.seed),
        "device": "cpu",
        "epochs": int(EPOCHS),
        "batch_size": int(BATCH_SIZE),
        "pseudo_fraction": float(
            args.pseudo_fraction
        ),
        "checkpoint": str(
            args.checkpoint
        ),
        "support_artifact": str(
            args.support_artifact
        ),
        "training_target_labels_accessed": False,
        "target_labels_loaded_only_after_training": True,
        "support_artifact_frozen": True,
        "arms": MODES,
        "results": {
            mode: {
                "history": results[
                    mode
                ][
                    "history"
                ],
                "final_metrics": final_metrics[
                    mode
                ],
            }
            for mode in MODES
        },
        "deltas": {
            "vanilla_vs_no_PL_MCA": float(
                vanilla["mca"]
                - no_pl["mca"]
            ),
            "confidence_vs_no_PL_MCA": float(
                confidence["mca"]
                - no_pl["mca"]
            ),
            "confidence_support_vs_no_PL_MCA": float(
                confidence_support["mca"]
                - no_pl["mca"]
            ),
            "confidence_support_vs_vanilla_MCA": float(
                confidence_support["mca"]
                - vanilla["mca"]
            ),
            "confidence_support_vs_norm_control_MCA": float(
                confidence_support["mca"]
                - norm_control["mca"]
            ),
            "confidence_support_vs_no_PL_OA": float(
                confidence_support["overall"]
                - no_pl["overall"]
            ),
            "confidence_support_vs_vanilla_OA": float(
                confidence_support["overall"]
                - vanilla["overall"]
            ),
        },
        "scientific_decision": decision,
        "outputs": {
            "npz": str(output_npz),
            "json": str(output_json),
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
        "\nPSEUDO-LABEL CAUSAL CONTROL COMPLETION COMPLETE"
    )


if __name__ == "__main__":
    main()