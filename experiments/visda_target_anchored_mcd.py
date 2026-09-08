import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


SEED = 42

CACHE_ROOT = Path("checkpoints/visda_feature_cache")
SOURCE_CACHE = CACHE_ROOT / "source"
TARGET_CACHE = CACHE_ROOT / "target"

OUTPUT_DIR = Path(
    "checkpoints/visda_target_anchored_mcd"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

NUM_CLASSES = 12
INPUT_DIM = 2048
HIDDEN_DIM = 512

BATCH_SIZE = 512
EVAL_BATCH = 4096
PROTO_BATCH = 4096

EPOCHS = 20
WARMUP_EPOCHS = 3

LR_ADAPTER = 0.001
LR_CLASSIFIER = 0.01

MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

EMA_DECAY = 0.97

HEAD_PERTURBATION = 0.005

PL_THRESHOLD_START = 0.65
PL_THRESHOLD_END = 0.90

GLOBAL_PL_FRACTION = 0.25

PL_WEIGHT = 0.20
ANCHOR_WEIGHT = 0.10
MARGINAL_WEIGHT = 0.05

PROTO_TEMPERATURE = 0.10
EPS = 1e-8

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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_cache(cache_dir):
    files = sorted(
        cache_dir.glob("chunk_*.pt")
    )

    if not files:
        raise RuntimeError(
            f"No cache chunks found in {cache_dir}"
        )

    features = []
    labels = []

    for path in files:
        payload = torch.load(
            path,
            map_location="cpu",
        )

        features.append(
            payload["features"].float()
        )

        labels.append(
            payload["labels"].long()
        )

    features = torch.cat(
        features,
        dim=0,
    )

    labels = torch.cat(
        labels,
        dim=0,
    )

    if features.ndim != 2:
        raise RuntimeError(
            f"Expected 2D features, got {features.shape}"
        )

    if features.shape[1] != INPUT_DIM:
        raise RuntimeError(
            f"Expected feature dimension {INPUT_DIM}, "
            f"got {features.shape[1]}"
        )

    if len(features) != len(labels):
        raise RuntimeError(
            "Feature and label counts do not match."
        )

    return features, labels


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                INPUT_DIM,
                HIDDEN_DIM,
            ),
            nn.LayerNorm(
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

    def forward(self, x):
        z = self.adapter(x)

        return (
            self.classifier1(z),
            self.classifier2(z),
        )

    def encode(self, x):
        return self.adapter(x)


def initialize_heads(model):
    with torch.no_grad():
        model.classifier2.weight.copy_(
            model.classifier1.weight
        )

        model.classifier2.bias.copy_(
            model.classifier1.bias
        )

        values = torch.arange(
            model.classifier2.weight.numel(),
            dtype=model.classifier2.weight.dtype,
        ).reshape_as(
            model.classifier2.weight
        )

        signs = torch.where(
            values.remainder(2) == 0,
            torch.ones_like(values),
            -torch.ones_like(values),
        )

        model.classifier2.weight.add_(
            HEAD_PERTURBATION * signs
        )


def clone_model(model):
    teacher = MCDModel()

    teacher.load_state_dict(
        model.state_dict()
    )

    teacher.eval()

    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    return teacher


@torch.no_grad()
def update_ema(
    teacher,
    student,
    decay,
):
    for teacher_parameter, student_parameter in zip(
        teacher.parameters(),
        student.parameters(),
    ):
        teacher_parameter.mul_(decay)

        teacher_parameter.add_(
            student_parameter,
            alpha=1.0 - decay,
        )


def discrepancy(
    logits1,
    logits2,
):
    p1 = F.softmax(
        logits1,
        dim=1,
    )

    p2 = F.softmax(
        logits2,
        dim=1,
    )

    return torch.mean(
        torch.abs(
            p1 - p2
        )
    )


def threshold_for_epoch(epoch):
    if epoch <= WARMUP_EPOCHS:
        return float("inf")

    span = max(
        EPOCHS - WARMUP_EPOCHS,
        1,
    )

    progress = (
        epoch - WARMUP_EPOCHS
    ) / span

    progress = min(
        max(progress, 0.0),
        1.0,
    )

    return (
        PL_THRESHOLD_START
        + (
            PL_THRESHOLD_END
            - PL_THRESHOLD_START
        ) * progress
    )


@torch.no_grad()
def predict_distribution(
    model,
    x,
):
    logits1, logits2 = model(x)

    p1 = F.softmax(
        logits1,
        dim=1,
    )

    p2 = F.softmax(
        logits2,
        dim=1,
    )

    p = (
        p1 + p2
    ) / 2.0

    confidence, prediction = p.max(
        dim=1
    )

    return (
        p,
        confidence,
        prediction,
    )


@torch.no_grad()
def compute_fixed_source_prototypes(
    model,
    source_features,
    source_labels,
    device,
):
    model.eval()

    sums = torch.zeros(
        NUM_CLASSES,
        HIDDEN_DIM,
        dtype=torch.float64,
    )

    counts = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
    )

    loader = DataLoader(
        TensorDataset(
            source_features,
            source_labels,
        ),
        batch_size=PROTO_BATCH,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    for x, y in loader:
        x = x.to(device)

        z = model.encode(x)

        z = F.normalize(
            z,
            dim=1,
        )

        for class_id in range(
            NUM_CLASSES
        ):
            mask = (
                y == class_id
            )

            if mask.any():
                class_z = z[
                    mask.to(device)
                ]

                sums[
                    class_id
                ] += class_z.double().sum(
                    dim=0
                ).cpu()

                counts[
                    class_id
                ] += int(
                    mask.sum().item()
                )

    prototypes = (
        sums
        / counts.clamp_min(
            1
        ).double().unsqueeze(1)
    )

    prototypes = prototypes.float()

    prototypes = F.normalize(
        prototypes,
        dim=1,
    )

    return prototypes.to(
        device
    )


def prototype_logits(
    features,
    prototypes,
):
    features = F.normalize(
        features,
        dim=1,
    )

    prototypes = F.normalize(
        prototypes,
        dim=1,
    )

    return (
        features
        @ prototypes.t()
    ) / PROTO_TEMPERATURE


def prototype_anchor_loss(
    features,
    labels,
    prototypes,
):
    logits = prototype_logits(
        features,
        prototypes,
    )

    return F.cross_entropy(
        logits,
        labels,
    )


@torch.no_grad()
def compute_target_marginal(
    teacher,
    target_features,
    device,
):
    teacher.eval()

    loader = DataLoader(
        TensorDataset(
            target_features
        ),
        batch_size=EVAL_BATCH,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    total_mass = torch.zeros(
        NUM_CLASSES,
        dtype=torch.float64,
    )

    total = 0

    for (x,) in loader:
        x = x.to(device)

        p, _, _ = predict_distribution(
            teacher,
            x,
        )

        total_mass += p.sum(
            dim=0
        ).double().cpu()

        total += x.size(0)

    marginal = (
        total_mass
        / max(total, 1)
    )

    marginal = marginal.float()

    marginal = marginal.clamp_min(
        EPS
    )

    marginal = (
        marginal
        / marginal.sum()
    )

    return marginal


def marginal_stability_loss(
    current_mass,
    reference_mass,
):
    current_mass = current_mass.clamp_min(
        EPS
    )

    reference_mass = reference_mass.clamp_min(
        EPS
    )

    current_mass = (
        current_mass
        / current_mass.sum()
    )

    reference_mass = (
        reference_mass
        / reference_mass.sum()
    )

    return F.kl_div(
        current_mass.log(),
        reference_mass,
        reduction="sum",
    )


def select_top_confident(
    confidence,
    threshold,
    max_fraction,
):
    n = len(confidence)

    eligible = (
        confidence >= threshold
    )

    indices = torch.nonzero(
        eligible,
        as_tuple=False,
    ).flatten()

    if len(indices) == 0:
        return torch.zeros(
            n,
            dtype=torch.bool,
        )

    max_count = max(
        1,
        int(
            n
            * max_fraction
        ),
    )

    if len(indices) > max_count:
        order = torch.argsort(
            confidence[
                indices
            ],
            descending=True,
        )

        indices = indices[
            order[:max_count]
        ]

    selected = torch.zeros(
        n,
        dtype=torch.bool,
    )

    selected[
        indices
    ] = True

    return selected


def train_epoch(
    student,
    teacher,
    source_loader,
    target_loader,
    optimizer_adapter,
    optimizer_classifier,
    source_prototypes,
    reference_marginal,
    epoch,
    device,
):
    student.train()
    teacher.eval()

    source_iter = iter(
        source_loader
    )

    target_iter = iter(
        target_loader
    )

    steps = len(
        source_loader
    )

    threshold = threshold_for_epoch(
        epoch
    )

    source_correct = 0
    source_total = 0

    source_loss_sum = 0.0
    discrepancy_sum = 0.0
    pseudo_loss_sum = 0.0
    anchor_loss_sum = 0.0
    marginal_loss_sum = 0.0

    pseudo_selected = 0
    pseudo_total = 0
    pseudo_confidence_sum = 0.0

    for _ in range(
        steps
    ):
        try:
            source_x, source_y = next(
                source_iter
            )
        except StopIteration:
            source_iter = iter(
                source_loader
            )

            source_x, source_y = next(
                source_iter
            )

        try:
            target_x = next(
                target_iter
            )[0]
        except StopIteration:
            target_iter = iter(
                target_loader
            )

            target_x = next(
                target_iter
            )[0]

        source_x = source_x.to(
            device
        )

        source_y = source_y.to(
            device
        )

        target_x = target_x.to(
            device
        )

        optimizer_adapter.zero_grad(
            set_to_none=True
        )

        source_logits1, source_logits2 = student(
            source_x
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

        optimizer_adapter.step()

        source_predictions = (
            (
                source_logits1
                + source_logits2
            )
            / 2.0
        ).argmax(
            dim=1
        )

        source_correct += int(
            (
                source_predictions
                == source_y
            ).sum().item()
        )

        source_total += source_y.size(0)

        source_loss_sum += (
            source_loss.item()
        )

        optimizer_classifier.zero_grad(
            set_to_none=True
        )

        with torch.no_grad():
            source_features = student.encode(
                source_x
            )

            target_features = student.encode(
                target_x
            )

        source_logits1 = student.classifier1(
            source_features
        )

        source_logits2 = student.classifier2(
            source_features
        )

        target_logits1 = student.classifier1(
            target_features
        )

        target_logits2 = student.classifier2(
            target_features
        )

        source_classifier_loss = (
            F.cross_entropy(
                source_logits1,
                source_y,
            )
            + F.cross_entropy(
                source_logits2,
                source_y,
            )
        )

        target_discrepancy = discrepancy(
            target_logits1,
            target_logits2,
        )

        classifier_loss = (
            source_classifier_loss
            - target_discrepancy
        )

        classifier_loss.backward()

        optimizer_classifier.step()

        discrepancy_sum += (
            target_discrepancy.item()
        )

        optimizer_adapter.zero_grad(
            set_to_none=True
        )

        for parameter in student.classifier1.parameters():
            parameter.requires_grad_(False)

        for parameter in student.classifier2.parameters():
            parameter.requires_grad_(False)

        target_features = student.encode(
            target_x
        )

        target_logits1 = student.classifier1(
            target_features
        )

        target_logits2 = student.classifier2(
            target_features
        )

        generator_discrepancy = discrepancy(
            target_logits1,
            target_logits2,
        )

        generator_discrepancy.backward()

        optimizer_adapter.step()

        for parameter in student.classifier1.parameters():
            parameter.requires_grad_(True)

        for parameter in student.classifier2.parameters():
            parameter.requires_grad_(True)

        update_ema(
            teacher,
            student,
            EMA_DECAY,
        )

        with torch.no_grad():
            (
                target_probabilities,
                target_confidence,
                target_predictions,
            ) = predict_distribution(
                teacher,
                target_x,
            )

        if epoch <= WARMUP_EPOCHS:
            selected = torch.zeros(
                len(target_x),
                dtype=torch.bool,
                device=device,
            )
        else:
            selected = select_top_confident(
                target_confidence.cpu(),
                threshold,
                GLOBAL_PL_FRACTION,
            ).to(device)

        selected_count = int(
            selected.sum().item()
        )

        pseudo_selected += (
            selected_count
        )

        pseudo_total += target_x.size(0)

        if selected_count > 0:
            selected_x = target_x[
                selected
            ]

            selected_y = target_predictions[
                selected
            ]

            selected_confidence = target_confidence[
                selected
            ].detach()

            optimizer_adapter.zero_grad(
                set_to_none=True
            )

            selected_features = student.encode(
                selected_x
            )

            logits1 = student.classifier1(
                selected_features
            )

            logits2 = student.classifier2(
                selected_features
            )

            loss1 = F.cross_entropy(
                logits1,
                selected_y,
                reduction="none",
            )

            loss2 = F.cross_entropy(
                logits2,
                selected_y,
                reduction="none",
            )

            confidence_weighted_loss = (
                (
                    loss1
                    + loss2
                )
                * selected_confidence
            ).mean()

            anchor_loss = prototype_anchor_loss(
                selected_features,
                selected_y,
                source_prototypes,
            )

            pseudo_total_loss = (
                PL_WEIGHT
                * confidence_weighted_loss
                + ANCHOR_WEIGHT
                * anchor_loss
            )

            pseudo_total_loss.backward()

            optimizer_adapter.step()

            optimizer_classifier.zero_grad(
                set_to_none=True
            )

            with torch.no_grad():
                selected_features = student.encode(
                    selected_x
                )

            classifier_logits1 = student.classifier1(
                selected_features
            )

            classifier_logits2 = student.classifier2(
                selected_features
            )

            classifier_pseudo_loss = (
                F.cross_entropy(
                    classifier_logits1,
                    selected_y,
                    reduction="none",
                )
                + F.cross_entropy(
                    classifier_logits2,
                    selected_y,
                    reduction="none",
                )
            )

            classifier_pseudo_loss = (
                classifier_pseudo_loss
                * selected_confidence
            ).mean()

            (
                PL_WEIGHT
                * classifier_pseudo_loss
            ).backward()

            optimizer_classifier.step()

            update_ema(
                teacher,
                student,
                EMA_DECAY,
            )

            pseudo_loss_sum += (
                confidence_weighted_loss.item()
            )

            anchor_loss_sum += (
                anchor_loss.item()
            )

            pseudo_confidence_sum += (
                selected_confidence.sum().item()
            )

    current_marginal = compute_target_marginal(
        teacher,
        target_features,
        device,
    )

    if reference_marginal is None:
        reference_marginal = current_marginal.clone()

    marginal_loss = marginal_stability_loss(
        current_marginal,
        reference_marginal,
    )

    if epoch > WARMUP_EPOCHS:
        optimizer_adapter.zero_grad(
            set_to_none=True
        )

        target_subset = target_features[
            :min(
                len(target_features),
                EVAL_BATCH * 4,
            )
        ].to(device)

        current_features = student.encode(
            target_subset
        )

        current_logits1 = student.classifier1(
            current_features
        )

        current_logits2 = student.classifier2(
            current_features
        )

        current_probabilities = (
            F.softmax(
                current_logits1,
                dim=1,
            )
            + F.softmax(
                current_logits2,
                dim=1,
            )
        ) / 2.0

        current_batch_marginal = (
            current_probabilities.mean(
                dim=0
            )
        )

        reference_device = reference_marginal.to(
            device
        )

        batch_marginal_loss = marginal_stability_loss(
            current_batch_marginal,
            reference_device,
        )

        (
            MARGINAL_WEIGHT
            * batch_marginal_loss
        ).backward()

        optimizer_adapter.step()

        marginal_loss_sum = (
            batch_marginal_loss.item()
        )

    else:
        marginal_loss_sum = 0.0

    return {
        "threshold":
            threshold,
        "source_accuracy":
            100.0
            * source_correct
            / max(
                source_total,
                1,
            ),
        "source_loss":
            source_loss_sum
            / max(
                steps,
                1,
            ),
        "discrepancy":
            discrepancy_sum
            / max(
                steps,
                1,
            ),
        "pseudo_loss":
            pseudo_loss_sum
            / max(
                steps,
                1,
            ),
        "anchor_loss":
            anchor_loss_sum
            / max(
                steps,
                1,
            ),
        "marginal_loss":
            marginal_loss_sum,
        "pseudo_selected":
            pseudo_selected,
        "pseudo_total":
            pseudo_total,
        "pseudo_coverage":
            100.0
            * pseudo_selected
            / max(
                pseudo_total,
                1,
            ),
        "pseudo_confidence":
            pseudo_confidence_sum
            / max(
                pseudo_selected,
                1,
            ),
        "target_marginal":
            current_marginal.tolist(),
        "reference_marginal":
            reference_marginal.tolist(),
        "reference_marginal_tensor":
            reference_marginal,
    }


def evaluate(
    model,
    features,
    labels,
    device,
):
    model.eval()

    loader = DataLoader(
        TensorDataset(
            features,
            labels,
        ),
        batch_size=EVAL_BATCH,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    total_correct = 0
    total_examples = 0
    confidence_sum = 0.0

    class_correct = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
    )

    class_total = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
    )

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            logits1, logits2 = model(x)

            p1 = F.softmax(
                logits1,
                dim=1,
            )

            p2 = F.softmax(
                logits2,
                dim=1,
            )

            p = (
                p1 + p2
            ) / 2.0

            confidence, predictions = p.max(
                dim=1
            )

            y_device = y.to(device)

            total_correct += int(
                (
                    predictions
                    == y_device
                ).sum().item()
            )

            total_examples += y.size(0)

            confidence_sum += (
                confidence.sum().item()
            )

            for class_id in range(
                NUM_CLASSES
            ):
                mask = (
                    y
                    == class_id
                )

                if mask.any():
                    mask_device = mask.to(
                        device
                    )

                    class_total[
                        class_id
                    ] += int(
                        mask.sum().item()
                    )

                    class_correct[
                        class_id
                    ] += int(
                        (
                            predictions[
                                mask_device
                            ]
                            == y_device[
                                mask_device
                            ]
                        ).sum().item()
                    )

    per_class = (
        100.0
        * class_correct.float()
        / class_total.clamp_min(1)
    )

    return {
        "overall_accuracy":
            100.0
            * total_correct
            / max(
                total_examples,
                1,
            ),
        "mean_class_accuracy":
            float(
                per_class.mean().item()
            ),
        "per_class_accuracy":
            per_class.tolist(),
        "mean_confidence":
            confidence_sum
            / max(
                total_examples,
                1,
            ),
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
        "VISDA-2017 TARGET-ANCHORED FLOW-STABLE MCD"
    )
    print("=" * 90)

    print(
        f"device={device}"
    )

    print(
        f"seed={SEED}"
    )

    print(
        f"batch_size={BATCH_SIZE}"
    )

    print(
        f"epochs={EPOCHS}"
    )

    print(
        f"warmup_epochs={WARMUP_EPOCHS}"
    )

    print(
        f"ema_decay={EMA_DECAY}"
    )

    print(
        f"pl_threshold={PL_THRESHOLD_START}"
        f"->{PL_THRESHOLD_END}"
    )

    print(
        f"global_pl_fraction={GLOBAL_PL_FRACTION}"
    )

    print(
        f"pl_weight={PL_WEIGHT}"
    )

    print(
        f"anchor_weight={ANCHOR_WEIGHT}"
    )

    print(
        f"marginal_weight={MARGINAL_WEIGHT}"
    )

    print()
    print(
        "Loading source cache..."
    )

    source_features, source_labels = load_cache(
        SOURCE_CACHE
    )

    print(
        f"Source features: "
        f"{tuple(source_features.shape)}"
    )

    print()
    print(
        "Loading target cache..."
    )

    target_features, target_labels = load_cache(
        TARGET_CACHE
    )

    print(
        f"Target features: "
        f"{tuple(target_features.shape)}"
    )

    source_dataset = TensorDataset(
        source_features,
        source_labels,
    )

    target_dataset = TensorDataset(
        target_features
    )

    generator = torch.Generator()

    generator.manual_seed(
        SEED
    )

    source_loader = DataLoader(
        source_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=generator,
    )

    target_loader = DataLoader(
        target_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=generator,
    )

    student = MCDModel().to(
        device
    )

    initialize_heads(
        student
    )

    teacher = clone_model(
        student
    ).to(device)

    print()
    print(
        "Running warm-up MCD..."
    )

    optimizer_adapter = torch.optim.SGD(
        student.adapter.parameters(),
        lr=LR_ADAPTER,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    optimizer_classifier = torch.optim.SGD(
        list(
            student.classifier1.parameters()
        )
        + list(
            student.classifier2.parameters()
        ),
        lr=LR_CLASSIFIER,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler_adapter = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_adapter,
            T_max=EPOCHS,
        )
    )

    scheduler_classifier = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_classifier,
            T_max=EPOCHS,
        )
    )

    source_prototypes = None
    reference_marginal = None

    history = []

    start_time = time.perf_counter()

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        epoch_start = time.perf_counter()

        if (
            epoch == WARMUP_EPOCHS + 1
            and source_prototypes is None
        ):
            source_prototypes = (
                compute_fixed_source_prototypes(
                    student,
                    source_features,
                    source_labels,
                    device,
                )
            )

            reference_marginal = (
                compute_target_marginal(
                    teacher,
                    target_features,
                    device,
                )
            )

            print()
            print(
                "Anchor initialization complete."
            )

            print(
                "Reference target marginal:"
            )

            for class_name, value in zip(
                CLASSES,
                reference_marginal.tolist(),
            ):
                print(
                    f"{class_name:12s}: "
                    f"{value:.6f}"
                )

        if source_prototypes is None:
            source_prototypes = torch.zeros(
                NUM_CLASSES,
                HIDDEN_DIM,
                device=device,
            )

        metrics = train_epoch(
            student,
            teacher,
            source_loader,
            target_loader,
            optimizer_adapter,
            optimizer_classifier,
            source_prototypes,
            reference_marginal,
            epoch,
            device,
        )

        reference_marginal = (
            metrics[
                "reference_marginal_tensor"
            ].clone()
        )

        scheduler_adapter.step()
        scheduler_classifier.step()

        student_metrics = evaluate(
            student,
            target_features,
            target_labels,
            device,
        )

        teacher_metrics = evaluate(
            teacher,
            target_features,
            target_labels,
            device,
        )

        epoch_seconds = (
            time.perf_counter()
            - epoch_start
        )

        record = {
            "epoch":
                epoch,
            "threshold":
                metrics["threshold"]
                if np.isfinite(
                    metrics["threshold"]
                )
                else None,
            "source_accuracy":
                metrics["source_accuracy"],
            "source_loss":
                metrics["source_loss"],
            "mcd_discrepancy":
                metrics["discrepancy"],
            "pseudo_loss":
                metrics["pseudo_loss"],
            "anchor_loss":
                metrics["anchor_loss"],
            "marginal_loss":
                metrics["marginal_loss"],
            "pseudo_selected":
                metrics["pseudo_selected"],
            "pseudo_total":
                metrics["pseudo_total"],
            "pseudo_coverage":
                metrics["pseudo_coverage"],
            "pseudo_confidence":
                metrics["pseudo_confidence"],
            "student_target_overall":
                student_metrics[
                    "overall_accuracy"
                ],
            "student_target_mean_class":
                student_metrics[
                    "mean_class_accuracy"
                ],
            "teacher_target_overall":
                teacher_metrics[
                    "overall_accuracy"
                ],
            "teacher_target_mean_class":
                teacher_metrics[
                    "mean_class_accuracy"
                ],
            "student_target_confidence":
                student_metrics[
                    "mean_confidence"
                ],
            "teacher_target_confidence":
                teacher_metrics[
                    "mean_confidence"
                ],
            "target_marginal":
                metrics[
                    "target_marginal"
                ],
            "reference_marginal":
                metrics[
                    "reference_marginal"
                ],
            "epoch_seconds":
                epoch_seconds,
        }

        history.append(
            record
        )

        threshold_text = (
            "WARMUP"
            if not np.isfinite(
                metrics["threshold"]
            )
            else f"{metrics['threshold']:.3f}"
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Thr {threshold_text:>7s} | "
            f"Source "
            f"{metrics['source_accuracy']:.2f}% | "
            f"Student "
            f"{student_metrics['mean_class_accuracy']:.2f}% | "
            f"Teacher "
            f"{teacher_metrics['mean_class_accuracy']:.2f}% | "
            f"PL "
            f"{metrics['pseudo_coverage']:.2f}% | "
            f"PLconf "
            f"{metrics['pseudo_confidence']:.4f} | "
            f"Anchor "
            f"{metrics['anchor_loss']:.4f} | "
            f"Marginal "
            f"{metrics['marginal_loss']:.6f} | "
            f"{epoch_seconds:.2f}s"
        )

    total_seconds = (
        time.perf_counter()
        - start_time
    )

    final_student = evaluate(
        student,
        target_features,
        target_labels,
        device,
    )

    final_teacher = evaluate(
        teacher,
        target_features,
        target_labels,
        device,
    )

    best_student = max(
        history,
        key=lambda x:
            x[
                "student_target_mean_class"
            ],
    )

    best_teacher = max(
        history,
        key=lambda x:
            x[
                "teacher_target_mean_class"
            ],
    )

    print()
    print("=" * 90)
    print(
        "FINAL RESULT"
    )
    print("=" * 90)

    print(
        f"Student target overall: "
        f"{final_student['overall_accuracy']:.2f}%"
    )

    print(
        f"Student target mean-class: "
        f"{final_student['mean_class_accuracy']:.2f}%"
    )

    print(
        f"Teacher target overall: "
        f"{final_teacher['overall_accuracy']:.2f}%"
    )

    print(
        f"Teacher target mean-class: "
        f"{final_teacher['mean_class_accuracy']:.2f}%"
    )

    print()
    print(
        f"Best student epoch: "
        f"{best_student['epoch']}"
    )

    print(
        f"Best student mean-class: "
        f"{best_student['student_target_mean_class']:.2f}%"
    )

    print(
        f"Best teacher epoch: "
        f"{best_teacher['epoch']}"
    )

    print(
        f"Best teacher mean-class: "
        f"{best_teacher['teacher_target_mean_class']:.2f}%"
    )

    print()
    print(
        "Student per-class accuracy:"
    )

    for class_name, accuracy in zip(
        CLASSES,
        final_student[
            "per_class_accuracy"
        ],
    ):
        print(
            f"{class_name:12s}: "
            f"{accuracy:.2f}%"
        )

    print()
    print(
        "Teacher per-class accuracy:"
    )

    for class_name, accuracy in zip(
        CLASSES,
        final_teacher[
            "per_class_accuracy"
        ],
    ):
        print(
            f"{class_name:12s}: "
            f"{accuracy:.2f}%"
        )

    print()
    print(
        "Final target marginal:"
    )

    final_marginal = history[-1][
        "target_marginal"
    ]

    for class_name, value in zip(
        CLASSES,
        final_marginal,
    ):
        print(
            f"{class_name:12s}: "
            f"{value:.6f}"
        )

    print()
    print(
        f"Total training time: "
        f"{total_seconds:.2f}s"
    )

    checkpoint_path = (
        OUTPUT_DIR
        / "target_anchored_flow_seed42.pt"
    )

    history_path = (
        OUTPUT_DIR
        / "target_anchored_flow_seed42.json"
    )

    torch.save(
        {
            "student_state_dict":
                student.state_dict(),
            "teacher_state_dict":
                teacher.state_dict(),
            "source_prototypes":
                source_prototypes.cpu()
                if source_prototypes is not None
                else None,
            "reference_marginal":
                reference_marginal.tolist()
                if reference_marginal is not None
                else None,
            "seed":
                SEED,
            "classes":
                CLASSES,
            "final_student":
                final_student,
            "final_teacher":
                final_teacher,
            "best_student_epoch":
                best_student["epoch"],
            "best_student_mean_class":
                best_student[
                    "student_target_mean_class"
                ],
            "best_teacher_epoch":
                best_teacher["epoch"],
            "best_teacher_mean_class":
                best_teacher[
                    "teacher_target_mean_class"
                ],
            "history":
                history,
        },
        checkpoint_path,
    )

    report = {
        "experiment":
            "visda_target_anchored_flow_constrained_mcd",
        "seed":
            SEED,
        "backbone":
            "ImageNet-pretrained ResNet-50 frozen",
        "feature_dimension":
            INPUT_DIM,
        "adapter_dimension":
            HIDDEN_DIM,
        "target_labels_used_for_training":
            False,
        "target_labels_used_for_evaluation":
            True,
        "epochs":
            EPOCHS,
        "warmup_epochs":
            WARMUP_EPOCHS,
        "ema_decay":
            EMA_DECAY,
        "pl_threshold_start":
            PL_THRESHOLD_START,
        "pl_threshold_end":
            PL_THRESHOLD_END,
        "global_pl_fraction":
            GLOBAL_PL_FRACTION,
        "pl_weight":
            PL_WEIGHT,
        "anchor_weight":
            ANCHOR_WEIGHT,
        "marginal_weight":
            MARGINAL_WEIGHT,
        "final_student":
            final_student,
        "final_teacher":
            final_teacher,
        "best_student_epoch":
            best_student["epoch"],
        "best_student_mean_class":
            best_student[
                "student_target_mean_class"
            ],
        "best_teacher_epoch":
            best_teacher["epoch"],
        "best_teacher_mean_class":
            best_teacher[
                "teacher_target_mean_class"
            ],
        "history":
            history,
    }

    with open(
        history_path,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
        )

    print()
    print("=" * 90)
    print(
        "EXPERIMENT COMPLETE"
    )
    print("=" * 90)

    print(
        f"checkpoint={checkpoint_path}"
    )

    print(
        f"history={history_path}"
    )


if __name__ == "__main__":
    main()