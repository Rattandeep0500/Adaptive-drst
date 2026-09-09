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

TRUCK_ID = 11
CAR_ID = 3
BUS_ID = 2
TRAIN_ID = 10

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

SOURCE_CACHE = Path(
    "checkpoints/visda_feature_cache/source"
)

TARGET_CACHE = Path(
    "checkpoints/visda_feature_cache/target"
)

RPC_CHECKPOINT = Path(
    "checkpoints/visda_rpc_causal_experiment/rpc_seed42.pt"
)

GRAPH_OUTPUTS = Path(
    "checkpoints/visda_graph_semantic_diffusion/"
    "graph_semantic_outputs_seed42.pt"
)

OUTPUT_DIR = Path(
    "checkpoints/visda_truck_car_local_boundary"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "truck_car_local_boundary_seed42.json"
)

OUTPUT_TENSOR = (
    OUTPUT_DIR
    / "truck_car_local_boundary_seed42.pt"
)

BATCH_SIZE = 4096

SOURCE_PER_CLASS = 12000

TRAIN_EPOCHS = 150

TRAIN_LR = 0.01

WEIGHT_DECAY = 1e-4

TOP_K = 3

PAIR_THRESHOLD = 0.5

TEMPERATURES = [
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
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


class PairHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(
            HIDDEN_DIM,
            1
        )

    def forward(self, x):
        return self.linear(x).squeeze(1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_load(path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu"
        )


def load_cache(cache_dir):
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
        payload = safe_load(
            path
        )

        if "features" not in payload:
            raise RuntimeError(
                f"Missing features in {path}"
            )

        if "labels" not in payload:
            raise RuntimeError(
                f"Missing labels in {path}"
            )

        features = payload[
            "features"
        ].float()

        labels = payload[
            "labels"
        ].long()

        if features.ndim != 2:
            raise RuntimeError(
                f"Invalid features in {path}"
            )

        if features.shape[1] != INPUT_DIM:
            raise RuntimeError(
                f"Expected {INPUT_DIM}-D features, "
                f"got {features.shape[1]}"
            )

        feature_parts.append(
            features
        )

        label_parts.append(
            labels
        )

    features = torch.cat(
        feature_parts,
        dim=0
    )

    labels = torch.cat(
        label_parts,
        dim=0
    )

    return features, labels


def normalize_state_dict(
    state_dict
):
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
                "teacher."
            ):
                if new_key.startswith(
                    prefix
                ):
                    new_key = new_key[
                        len(prefix):
                    ]
                    changed = True
                    break

        result[
            new_key
        ] = value

    return result


def extract_state(payload):
    candidates = [
        "rpc_student_state_dict",
        "student_state_dict",
        "baseline_student_state_dict",
        "state_dict"
    ]

    for key in candidates:
        if key in payload:
            return payload[
                key
            ], key

    if all(
        isinstance(key, str)
        and (
            key.startswith("adapter.")
            or key.startswith("classifier1.")
            or key.startswith("classifier2.")
        )
        for key in payload.keys()
    ):
        return payload, "root"

    raise RuntimeError(
        "No compatible state dict found"
    )


def load_rpc_model():
    payload = safe_load(
        RPC_CHECKPOINT
    )

    state_dict, name = extract_state(
        payload
    )

    state_dict = normalize_state_dict(
        state_dict
    )

    model = MCDModel()

    expected = set(
        model.state_dict().keys()
    )

    actual = set(
        state_dict.keys()
    )

    missing = sorted(
        expected - actual
    )

    unexpected = sorted(
        actual - expected
    )

    if missing:
        raise RuntimeError(
            f"Missing checkpoint keys: {missing}"
        )

    if unexpected:
        raise RuntimeError(
            f"Unexpected checkpoint keys: {unexpected}"
        )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.eval()

    print(
        f"Loaded state: {name}"
    )

    return model


@torch.no_grad()
def collect_adapted_features(
    model,
    features,
    device
):
    model.eval()

    parts = []

    for start in range(
        0,
        len(features),
        BATCH_SIZE
    ):
        end = min(
            start + BATCH_SIZE,
            len(features)
        )

        x = features[
            start:end
        ].to(
            device
        )

        z = model.encode(
            x
        )

        z = F.normalize(
            z,
            dim=1
        )

        parts.append(
            z.cpu()
        )

    return torch.cat(
        parts,
        dim=0
    )


def balanced_pair(
    features,
    labels
):
    truck_indices = torch.nonzero(
        labels == TRUCK_ID,
        as_tuple=False
    ).flatten()

    car_indices = torch.nonzero(
        labels == CAR_ID,
        as_tuple=False
    ).flatten()

    generator = torch.Generator()
    generator.manual_seed(
        SEED
    )

    n = min(
        len(truck_indices),
        len(car_indices),
        SOURCE_PER_CLASS
    )

    truck_perm = torch.randperm(
        len(truck_indices),
        generator=generator
    )[:n]

    car_perm = torch.randperm(
        len(car_indices),
        generator=generator
    )[:n]

    truck_indices = truck_indices[
        truck_perm
    ]

    car_indices = car_indices[
        car_perm
    ]

    x = torch.cat(
        [
            features[
                truck_indices
            ],
            features[
                car_indices
            ]
        ],
        dim=0
    )

    y = torch.cat(
        [
            torch.ones(
                n,
                dtype=torch.float32
            ),
            torch.zeros(
                n,
                dtype=torch.float32
            )
        ],
        dim=0
    )

    permutation = torch.randperm(
        len(x),
        generator=generator
    )

    return (
        x[
            permutation
        ],
        y[
            permutation
        ]
    )


def train_pair_head(
    x,
    y,
    device
):
    head = PairHead().to(
        device
    )

    optimizer = torch.optim.Adam(
        head.parameters(),
        lr=TRAIN_LR,
        weight_decay=WEIGHT_DECAY
    )

    x = x.to(
        device
    )

    y = y.to(
        device
    )

    best_loss = float(
        "inf"
    )

    best_state = None

    for _ in range(
        TRAIN_EPOCHS
    ):
        head.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = head(
            x
        )

        loss = F.binary_cross_entropy_with_logits(
            logits,
            y
        )

        loss.backward()

        optimizer.step()

        value = float(
            loss.item()
        )

        if value < best_loss:
            best_loss = value

            best_state = {
                key:
                    value.detach().cpu().clone()
                for key, value in head.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError(
            "Pair head failed to train"
        )

    head.load_state_dict(
        best_state
    )

    head.eval()

    return head.cpu()


@torch.no_grad()
def predict(
    head,
    features,
    device
):
    head = head.to(
        device
    )

    head.eval()

    results = []

    for start in range(
        0,
        len(features),
        BATCH_SIZE
    ):
        end = min(
            start + BATCH_SIZE,
            len(features)
        )

        x = features[
            start:end
        ].to(
            device
        )

        results.append(
            torch.sigmoid(
                head(x)
            ).cpu()
        )

    head.cpu()

    return torch.cat(
        results,
        dim=0
    )


def auc(
    scores,
    labels,
    positive_id,
    negative_id,
    mask=None
):
    if mask is not None:
        scores = scores[
            mask
        ]

        labels = labels[
            mask
        ]

    keep = (
        (labels == positive_id)
        | (
            labels == negative_id
        )
    )

    scores = scores[
        keep
    ]

    labels = labels[
        keep
    ]

    pos = scores[
        labels == positive_id
    ]

    neg = scores[
        labels == negative_id
    ]

    if len(pos) == 0:
        return None

    if len(neg) == 0:
        return None

    greater = (
        pos.unsqueeze(1)
        > neg.unsqueeze(0)
    ).float()

    equal = (
        pos.unsqueeze(1)
        == neg.unsqueeze(0)
    ).float()

    return float(
        (
            greater.mean()
            + 0.5 * equal.mean()
        ).item()
    )


def pair_metrics(
    scores,
    labels,
    negative_id,
    mask=None
):
    if mask is not None:
        scores = scores[
            mask
        ]

        labels = labels[
            mask
        ]

    keep = (
        (labels == TRUCK_ID)
        | (
            labels == negative_id
        )
    )

    scores = scores[
        keep
    ]

    labels = labels[
        keep
    ]

    if len(scores) == 0:
        return {
            "count": 0,
            "auc": None,
            "truck_recall": None,
            "negative_fpr": None
        }

    prediction = (
        scores
        >= PAIR_THRESHOLD
    )

    truck_mask = (
        labels
        == TRUCK_ID
    )

    negative_mask = (
        labels
        == negative_id
    )

    truck_recall = (
        prediction[
            truck_mask
        ]
        .float()
        .mean()
        .item()
        if truck_mask.any()
        else 0.0
    )

    negative_fpr = (
        prediction[
            negative_mask
        ]
        .float()
        .mean()
        .item()
        if negative_mask.any()
        else 0.0
    )

    return {
        "count":
            int(
                len(scores)
            ),
        "auc":
            auc(
                scores,
                labels,
                TRUCK_ID,
                negative_id
            ),
        "truck_recall":
            100.0
            * truck_recall,
        "negative_fpr":
            100.0
            * negative_fpr
    }


def graph_candidate_masks(
    graph_scores
):
    topk = torch.topk(
        graph_scores,
        k=TOP_K,
        dim=1,
        largest=True,
        sorted=True
    ).indices

    truck_present = (
        topk
        == TRUCK_ID
    ).any(
        dim=1
    )

    car_present = (
        topk
        == CAR_ID
    ).any(
        dim=1
    )

    bus_present = (
        topk
        == BUS_ID
    ).any(
        dim=1
    )

    train_present = (
        topk
        == TRAIN_ID
    ).any(
        dim=1
    )

    return {
        "topk":
            topk,
        "truck_top3":
            truck_present,
        "truck_car":
            truck_present
            & car_present,
        "truck_bus":
            truck_present
            & bus_present,
        "truck_train":
            truck_present
            & train_present
    }


def candidate_score(
    graph_scores,
    truck_probability,
    temperature
):
    graph_logits = torch.log(
        graph_scores.clamp_min(
            1e-12
        )
    )

    pair_logit = torch.log(
        truck_probability.clamp(
            1e-6,
            1.0 - 1e-6
        )
        / (
            1.0
            - truck_probability.clamp(
                1e-6,
                1.0 - 1e-6
            )
        )
    )

    adjusted = (
        graph_logits[
            :,
            TRUCK_ID
        ]
        + temperature
        * pair_logit
    )

    return adjusted


def apply_local_rescue(
    graph_scores,
    pairwise_scores,
    candidate_mask,
    temperature
):
    graph_predictions = graph_scores.argmax(
        dim=1
    )

    final_predictions = (
        graph_predictions.clone()
    )

    truck_score = (
        graph_scores[
            :,
            TRUCK_ID
        ]
        * 0.0
    )

    car_pair = pairwise_scores[
        CAR_ID
    ]

    bus_pair = pairwise_scores[
        BUS_ID
    ]

    train_pair = pairwise_scores[
        TRAIN_ID
    ]

    truck_score = (
        torch.log(
            graph_scores[
                :,
                TRUCK_ID
            ].clamp_min(
                1e-12
            )
        )
        + temperature
        * torch.log(
            car_pair.clamp(
                1e-6,
                1.0 - 1e-6
            )
            / (
                1.0
                - car_pair.clamp(
                    1e-6,
                    1.0 - 1e-6
                )
            )
        )
    )

    competitor_scores = torch.stack(
        [
            torch.log(
                graph_scores[
                    :,
                    CAR_ID
                ].clamp_min(
                    1e-12
                )
            ),
            torch.log(
                graph_scores[
                    :,
                    BUS_ID
                ].clamp_min(
                    1e-12
                )
            ),
            torch.log(
                graph_scores[
                    :,
                    TRAIN_ID
                ].clamp_min(
                    1e-12
                )
            )
        ],
        dim=1
    )

    best_competitor_score = (
        competitor_scores.max(
            dim=1
        ).values
    )

    pair_support = torch.stack(
        [
            car_pair,
            bus_pair,
            train_pair
        ],
        dim=1
    )

    strongest_pair_support = (
        pair_support.min(
            dim=1
        ).values
    )

    winner = (
        truck_score
        > best_competitor_score
    )

    strong_pair = (
        strongest_pair_support
        >= PAIR_THRESHOLD
    )

    rescue = (
        candidate_mask
        & winner
        & strong_pair
        & (
            graph_predictions
            != TRUCK_ID
        )
    )

    final_predictions[
        rescue
    ] = TRUCK_ID

    return (
        final_predictions,
        rescue
    )


def multiclass_metrics(
    predictions,
    labels
):
    correct = (
        predictions
        == labels
    )

    overall = (
        100.0
        * correct.float().mean().item()
    )

    per_class = []

    for class_id in range(
        NUM_CLASSES
    ):
        mask = (
            labels
            == class_id
        )

        if mask.any():
            value = (
                100.0
                * correct[
                    mask
                ]
                .float()
                .mean()
                .item()
            )
        else:
            value = 0.0

        per_class.append(
            value
        )

    return {
        "overall":
            overall,
        "mean_class":
            float(
                np.mean(
                    per_class
                )
            ),
        "per_class":
            per_class
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

    print("=" * 95)
    print(
        "VISDA-2017 TRUCK-CAR LOCAL BOUNDARY PROBE"
    )
    print("=" * 95)

    print(
        f"device={device}"
    )

    print(
        f"seed={SEED}"
    )

    print(
        f"pairwise_epochs={TRAIN_EPOCHS}"
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

    print()

    print(
        "Loading RPC checkpoint..."
    )

    model = load_rpc_model().to(
        device
    )

    print()

    print(
        "Collecting adapted source features..."
    )

    adapted_source = collect_adapted_features(
        model,
        source_features,
        device
    )

    print(
        f"Adapted source: "
        f"{tuple(adapted_source.shape)}"
    )

    print()

    print(
        "Collecting adapted target features..."
    )

    adapted_target = collect_adapted_features(
        model,
        target_features,
        device
    )

    print(
        f"Adapted target: "
        f"{tuple(adapted_target.shape)}"
    )

    print()

    print(
        "Loading graph diffusion outputs..."
    )

    graph_payload = safe_load(
        GRAPH_OUTPUTS
    )

    graph_scores = graph_payload[
        "diffused_probabilities"
    ].float()

    if graph_scores.shape != (
        len(target_labels),
        NUM_CLASSES
    ):
        raise RuntimeError(
            "Graph score shape mismatch"
        )

    graph_predictions = graph_scores.argmax(
        dim=1
    )

    graph_metrics = multiclass_metrics(
        graph_predictions,
        target_labels
    )

    print(
        "GRAPH BASELINE"
    )

    print(
        f"Overall={graph_metrics['overall']:.2f}%"
    )

    print(
        f"MCA={graph_metrics['mean_class']:.2f}%"
    )

    print(
        f"Truck={graph_metrics['per_class'][TRUCK_ID]:.2f}%"
    )

    print()

    candidates = graph_candidate_masks(
        graph_scores
    )

    truck_top3_count = int(
        candidates[
            "truck_top3"
        ].sum().item()
    )

    truck_true_count = int(
        (
            candidates[
                "truck_top3"
            ]
            & (
                target_labels
                == TRUCK_ID
            )
        ).sum().item()
    )

    print(
        "TRUCK TOP-3 REGION"
    )

    print(
        f"candidate_count={truck_top3_count}"
    )

    print(
        f"true_truck={truck_true_count}"
    )

    print(
        f"candidate_precision="
        f"{100.0 * truck_true_count / max(truck_top3_count, 1):.2f}%"
    )

    print(
        f"candidate_recall="
        f"{100.0 * truck_true_count / max(int((target_labels == TRUCK_ID).sum().item()), 1):.2f}%"
    )

    print()

    print(
        "TRAINING SOURCE TRUCK-CAR PAIRWISE HEAD"
    )

    pair_x, pair_y = balanced_pair(
        adapted_source,
        source_labels
    )

    pair_head = train_pair_head(
        pair_x,
        pair_y,
        device
    )

    pair_score = predict(
        pair_head,
        adapted_target,
        device
    )

    pairwise_scores = {
        CAR_ID:
            pair_score
    }

    global_metrics = pair_metrics(
        pair_score,
        target_labels,
        CAR_ID
    )

    candidate_metrics = pair_metrics(
        pair_score,
        target_labels,
        CAR_ID,
        candidates[
            "truck_car"
        ]
    )

    print()
    print(
        "TRUCK VS CAR"
    )

    print(
        f"global_auc={global_metrics['auc']:.6f}"
    )

    print(
        f"candidate_auc={candidate_metrics['auc']:.6f}"
    )

    print(
        f"candidate_truck_recall="
        f"{candidate_metrics['truck_recall']:.2f}%"
    )

    print(
        f"candidate_FPR="
        f"{candidate_metrics['negative_fpr']:.2f}%"
    )

    print()

    print(
        "TRAINING SOURCE TRUCK-BUS PAIRWISE HEAD"
    )

    bus_x, bus_y = balanced_pair_against_class(
        adapted_source,
        source_labels,
        BUS_ID
    )

    bus_head = train_pair_head(
        bus_x,
        bus_y,
        device
    )

    bus_score = predict(
        bus_head,
        adapted_target,
        device
    )

    pairwise_scores[
        BUS_ID
    ] = bus_score

    bus_global = pair_metrics(
        bus_score,
        target_labels,
        BUS_ID
    )

    bus_candidate = pair_metrics(
        bus_score,
        target_labels,
        BUS_ID,
        candidates[
            "truck_bus"
        ]
    )

    print(
        f"truck vs bus | "
        f"global_auc={bus_global['auc']:.6f} | "
        f"candidate_auc={bus_candidate['auc']:.6f} | "
        f"candidate_recall={bus_candidate['truck_recall']:.2f}% | "
        f"candidate_FPR={bus_candidate['negative_fpr']:.2f}%"
    )

    print()

    print(
        "TRAINING SOURCE TRUCK-TRAIN PAIRWISE HEAD"
    )

    train_x, train_y = balanced_pair_against_class(
        adapted_source,
        source_labels,
        TRAIN_ID
    )

    train_head = train_pair_head(
        train_x,
        train_y,
        device
    )

    train_score = predict(
        train_head,
        adapted_target,
        device
    )

    pairwise_scores[
        TRAIN_ID
    ] = train_score

    train_global = pair_metrics(
        train_score,
        target_labels,
        TRAIN_ID
    )

    train_candidate = pair_metrics(
        train_score,
        target_labels,
        TRAIN_ID,
        candidates[
            "truck_train"
        ]
    )

    print(
        f"truck vs train | "
        f"global_auc={train_global['auc']:.6f} | "
        f"candidate_auc={train_candidate['auc']:.6f} | "
        f"candidate_recall={train_candidate['truck_recall']:.2f}% | "
        f"candidate_FPR={train_candidate['negative_fpr']:.2f}%"
    )

    print()

    print(
        "TRUCK-CAR LOCAL MARGIN"
    )

    truck_mask = (
        target_labels
        == TRUCK_ID
    )

    car_mask = (
        target_labels
        == CAR_ID
    )

    truck_pair_values = pair_score[
        truck_mask
    ]

    car_pair_values = pair_score[
        car_mask
    ]

    candidate_pair_values = pair_score[
        candidates[
            "truck_car"
        ]
    ]

    print(
        f"truck mean probability="
        f"{truck_pair_values.mean().item():.6f}"
    )

    print(
        f"car mean probability="
        f"{car_pair_values.mean().item():.6f}"
    )

    print(
        f"truck-car candidate mean="
        f"{candidate_pair_values.mean().item():.6f}"
    )

    print()

    print(
        "TEMPERATURE / LOCAL MARGIN SWEEP"
    )

    temperature_results = {}

    for temperature in TEMPERATURES:
        adjusted_truck = (
            torch.log(
                graph_scores[
                    :,
                    TRUCK_ID
                ].clamp_min(
                    1e-12
                )
            )
            + temperature
            * torch.log(
                pair_score.clamp(
                    1e-6,
                    1.0 - 1e-6
                )
                / (
                    1.0
                    - pair_score.clamp(
                        1e-6,
                        1.0 - 1e-6
                    )
                )
            )
        )

        competitor = torch.stack(
            [
                torch.log(
                    graph_scores[
                        :,
                        CAR_ID
                    ].clamp_min(
                        1e-12
                    )
                ),
                torch.log(
                    graph_scores[
                        :,
                        BUS_ID
                    ].clamp_min(
                        1e-12
                    )
                ),
                torch.log(
                    graph_scores[
                        :,
                        TRAIN_ID
                    ].clamp_min(
                        1e-12
                    )
                )
            ],
            dim=1
        ).max(
            dim=1
        ).values

        pair_support = torch.stack(
            [
                pair_score,
                bus_score,
                train_score
            ],
            dim=1
        ).min(
            dim=1
        ).values

        rescue = (
            candidates[
                "truck_top3"
            ]
            & (
                adjusted_truck
                > competitor
            )
            & (
                pair_support
                >= PAIR_THRESHOLD
            )
            & (
                graph_predictions
                != TRUCK_ID
            )
        )

        rescued_predictions = (
            graph_predictions.clone()
        )

        rescued_predictions[
            rescue
        ] = TRUCK_ID

        metrics = multiclass_metrics(
            rescued_predictions,
            target_labels
        )

        correct_rescues = int(
            (
                target_labels[
                    rescue
                ]
                == TRUCK_ID
            ).sum().item()
        )

        rescue_count = int(
            rescue.sum().item()
        )

        precision = (
            correct_rescues
            / max(
                rescue_count,
                1
            )
        )

        recall = (
            correct_rescues
            / max(
                int(
                    truck_mask.sum().item()
                ),
                1
            )
        )

        temperature_results[
            str(temperature)
        ] = {
            "rescue_count":
                rescue_count,
            "correct_rescues":
                correct_rescues,
            "precision":
                precision,
            "recall":
                recall,
            "overall":
                metrics["overall"],
            "mean_class":
                metrics["mean_class"],
            "truck":
                metrics["per_class"][TRUCK_ID],
            "car":
                metrics["per_class"][CAR_ID],
            "bus":
                metrics["per_class"][BUS_ID],
            "train":
                metrics["per_class"][TRAIN_ID]
        }

        print(
            f"T={temperature:.2f} | "
            f"rescues={rescue_count} | "
            f"precision={100.0 * precision:.2f}% | "
            f"recall={100.0 * recall:.2f}% | "
            f"MCA={metrics['mean_class']:.2f}% | "
            f"OA={metrics['overall']:.2f}% | "
            f"truck={metrics['per_class'][TRUCK_ID]:.2f}%"
        )

    best_temperature = max(
        temperature_results.items(),
        key=lambda item: (
            item[1]["mean_class"],
            item[1]["precision"],
            item[1]["recall"]
        )
    )

    print()
    print(
        "BEST OFFLINE LOCAL CALIBRATION"
    )

    print(
        f"temperature={best_temperature[0]}"
    )

    print(
        f"MCA={best_temperature[1]['mean_class']:.2f}%"
    )

    print(
        f"OA={best_temperature[1]['overall']:.2f}%"
    )

    print(
        f"truck={best_temperature[1]['truck']:.2f}%"
    )

    print(
        f"car={best_temperature[1]['car']:.2f}%"
    )

    print(
        f"bus={best_temperature[1]['bus']:.2f}%"
    )

    print(
        f"train={best_temperature[1]['train']:.2f}%"
    )

    torch.save(
        {
            "graph_scores":
                graph_scores,
            "graph_top3":
                candidates[
                    "topk"
                ],
            "truck_top3":
                candidates[
                    "truck_top3"
                ],
            "truck_car_mask":
                candidates[
                    "truck_car"
                ],
            "truck_bus_mask":
                candidates[
                    "truck_bus"
                ],
            "truck_train_mask":
                candidates[
                    "truck_train"
                ],
            "truck_car_pair_score":
                pair_score,
            "truck_bus_pair_score":
                bus_score,
            "truck_train_pair_score":
                train_score,
            "target_labels":
                target_labels
        },
        OUTPUT_TENSOR
    )

    result = {
        "experiment":
            "visda_truck_car_local_boundary_probe",
        "seed":
            SEED,
        "device":
            str(device),
        "source_samples":
            len(source_features),
        "target_samples":
            len(target_features),
        "adapted_dimension":
            HIDDEN_DIM,
        "graph_baseline":
            graph_metrics,
        "candidate_pool": {
            "count":
                truck_top3_count,
            "true_truck":
                truck_true_count,
            "precision":
                truck_true_count
                / max(
                    truck_top3_count,
                    1
                ),
            "recall":
                truck_true_count
                / max(
                    int(
                        truck_mask.sum().item()
                    ),
                    1
                )
        },
        "truck_car": {
            "global":
                global_metrics,
            "candidate":
                candidate_metrics
        },
        "truck_bus": {
            "global":
                bus_global,
            "candidate":
                bus_candidate
        },
        "truck_train": {
            "global":
                train_global,
            "candidate":
                train_candidate
        },
        "pair_probability": {
            "truck_mean":
                float(
                    truck_pair_values.mean().item()
                ),
            "car_mean":
                float(
                    car_pair_values.mean().item()
                ),
            "candidate_mean":
                float(
                    candidate_pair_values.mean().item()
                )
        },
        "temperature_results":
            temperature_results,
        "best_temperature":
            best_temperature[0],
        "best_result":
            best_temperature[1]
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as handle:
        json.dump(
            result,
            handle,
            indent=2
        )

    print()
    print("=" * 95)
    print(
        "TRUCK-CAR LOCAL BOUNDARY PROBE COMPLETE"
    )
    print("=" * 95)

    print(
        f"Saved: {OUTPUT_JSON}"
    )


def balanced_pair_against_class(
    features,
    labels,
    negative_id
):
    positive_indices = torch.nonzero(
        labels == TRUCK_ID,
        as_tuple=False
    ).flatten()

    negative_indices = torch.nonzero(
        labels == negative_id,
        as_tuple=False
    ).flatten()

    generator = torch.Generator()
    generator.manual_seed(
        SEED
        + negative_id
    )

    n = min(
        len(positive_indices),
        len(negative_indices),
        SOURCE_PER_CLASS
    )

    positive_order = torch.randperm(
        len(positive_indices),
        generator=generator
    )[:n]

    negative_order = torch.randperm(
        len(negative_indices),
        generator=generator
    )[:n]

    positive_indices = positive_indices[
        positive_order
    ]

    negative_indices = negative_indices[
        negative_order
    ]

    x = torch.cat(
        [
            features[
                positive_indices
            ],
            features[
                negative_indices
            ]
        ],
        dim=0
    )

    y = torch.cat(
        [
            torch.ones(
                n,
                dtype=torch.float32
            ),
            torch.zeros(
                n,
                dtype=torch.float32
            )
        ],
        dim=0
    )

    permutation = torch.randperm(
        len(x),
        generator=generator
    )

    return (
        x[
            permutation
        ],
        y[
            permutation
        ]
    )


if __name__ == "__main__":
    main()