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

PAIR_IDS = [
    CAR_ID,
    BUS_ID,
    TRAIN_ID
]

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
    "checkpoints/visda_pairwise_truck_disambiguation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "pairwise_truck_disambiguation_seed42.json"
)

OUTPUT_TENSOR = (
    OUTPUT_DIR
    / "pairwise_truck_disambiguation_seed42.pt"
)

BATCH_SIZE = 4096
SOURCE_PER_CLASS = 10000

TRAIN_EPOCHS = 100
TRAIN_LR = 0.02
WEIGHT_DECAY = 1e-4

TOP_K = 3
PAIR_THRESHOLD = 0.5


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


class PairwiseHead(nn.Module):
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
                f"Invalid feature tensor in {path}"
            )

        if features.shape[1] != INPUT_DIM:
            raise RuntimeError(
                f"Expected {INPUT_DIM}-D features, "
                f"got {features.shape[1]}"
            )

        if len(features) != len(labels):
            raise RuntimeError(
                f"Feature/label mismatch in {path}"
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

    if features.shape[1] != INPUT_DIM:
        raise RuntimeError(
            "Final feature dimension mismatch"
        )

    if len(features) != len(labels):
        raise RuntimeError(
            "Final feature/label mismatch"
        )

    return features, labels


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
                "teacher."
            ):
                if new_key.startswith(prefix):
                    new_key = new_key[
                        len(prefix):
                    ]
                    changed = True
                    break

        result[
            new_key
        ] = value

    return result


def extract_student_state_dict(payload):
    preferred = [
        "rpc_student_state_dict",
        "student_state_dict",
        "baseline_student_state_dict",
        "state_dict"
    ]

    for key in preferred:
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
        "No compatible student state dict found"
    )


def load_rpc_model():
    if not RPC_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"RPC checkpoint not found:\n"
            f"{RPC_CHECKPOINT}"
        )

    payload = safe_load(
        RPC_CHECKPOINT
    )

    state_dict, state_name = (
        extract_student_state_dict(
            payload
        )
    )

    state_dict = normalize_state_dict(
        state_dict
    )

    model = MCDModel()

    expected_keys = set(
        model.state_dict().keys()
    )

    actual_keys = set(
        state_dict.keys()
    )

    missing = sorted(
        expected_keys
        - actual_keys
    )

    unexpected = sorted(
        actual_keys
        - expected_keys
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
        f"Loaded student state: {state_name}"
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

        if z.shape[1] != HIDDEN_DIM:
            raise RuntimeError(
                f"Expected {HIDDEN_DIM}-D adapted features"
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


def build_balanced_pair(
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
        * 100
    )

    if len(positive_indices) > SOURCE_PER_CLASS:
        permutation = torch.randperm(
            len(positive_indices),
            generator=generator
        )

        positive_indices = positive_indices[
            permutation[
                :SOURCE_PER_CLASS
            ]
        ]

    if len(negative_indices) > SOURCE_PER_CLASS:
        permutation = torch.randperm(
            len(negative_indices),
            generator=generator
        )

        negative_indices = negative_indices[
            permutation[
                :SOURCE_PER_CLASS
            ]
        ]

    positive_x = features[
        positive_indices
    ]

    negative_x = features[
        negative_indices
    ]

    x = torch.cat(
        [
            positive_x,
            negative_x
        ],
        dim=0
    )

    y = torch.cat(
        [
            torch.ones(
                len(positive_x)
            ),
            torch.zeros(
                len(negative_x)
            )
        ],
        dim=0
    ).float()

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


def train_pairwise_head(
    x,
    y,
    device,
    competitor_id
):
    head = PairwiseHead().to(
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

        current_loss = float(
            loss.item()
        )

        if current_loss < best_loss:
            best_loss = current_loss

            best_state = {
                key:
                    value.detach().cpu().clone()
                for key, value in head.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError(
            "Pairwise training produced no state"
        )

    head.load_state_dict(
        best_state,
        strict=True
    )

    head.eval()

    with torch.no_grad():
        probabilities = torch.sigmoid(
            head(
                x
            )
        )

        predictions = (
            probabilities
            >= PAIR_THRESHOLD
        ).float()

        accuracy = (
            predictions
            == y
        ).float().mean().item()

    print(
        f"truck vs {CLASSES[competitor_id]:5s} | "
        f"source loss={best_loss:.8f} | "
        f"source accuracy={100.0 * accuracy:.2f}%"
    )

    head.cpu()

    return head


@torch.no_grad()
def predict_pairwise(
    head,
    features,
    device
):
    head = head.to(
        device
    )

    head.eval()

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

        probability = torch.sigmoid(
            head(
                x
            )
        )

        parts.append(
            probability.cpu()
        )

    head.cpu()

    return torch.cat(
        parts,
        dim=0
    )


def binary_auc(
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

    positive_scores = scores[
        labels == positive_id
    ]

    negative_scores = scores[
        labels == negative_id
    ]

    if (
        len(positive_scores) == 0
        or len(negative_scores) == 0
    ):
        return None

    greater = (
        positive_scores.unsqueeze(1)
        > negative_scores.unsqueeze(0)
    ).float()

    ties = (
        positive_scores.unsqueeze(1)
        == negative_scores.unsqueeze(0)
    ).float()

    return float(
        (
            greater.mean()
            + 0.5 * ties.mean()
        ).item()
    )


def pairwise_metrics(
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
            "truck_count": 0,
            "negative_count": 0,
            "accuracy": None,
            "truck_recall": None,
            "negative_false_positive_rate": None,
            "auc": None
        }

    y = (
        labels
        == TRUCK_ID
    ).long()

    predictions = (
        scores
        >= PAIR_THRESHOLD
    ).long()

    truck_mask = (
        y == 1
    )

    negative_mask = (
        y == 0
    )

    accuracy = (
        predictions
        == y
    ).float().mean().item()

    truck_recall = (
        predictions[
            truck_mask
        ]
        .float()
        .mean()
        .item()
        if truck_mask.any()
        else 0.0
    )

    negative_fpr = (
        predictions[
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
            int(len(scores)),
        "truck_count":
            int(truck_mask.sum().item()),
        "negative_count":
            int(negative_mask.sum().item()),
        "accuracy":
            100.0 * accuracy,
        "truck_recall":
            100.0 * truck_recall,
        "negative_false_positive_rate":
            100.0 * negative_fpr,
        "auc":
            binary_auc(
                scores,
                labels,
                TRUCK_ID,
                negative_id
            )
    }


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
            labels == class_id
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


def candidate_statistics(
    graph_scores,
    labels
):
    topk = torch.topk(
        graph_scores,
        k=TOP_K,
        dim=1,
        largest=True,
        sorted=True
    ).indices

    truck_candidate = (
        topk == TRUCK_ID
    ).any(
        dim=1
    )

    true_truck = (
        labels == TRUCK_ID
    )

    candidate_count = int(
        truck_candidate.sum().item()
    )

    true_count = int(
        (
            truck_candidate
            & true_truck
        ).sum().item()
    )

    truck_total = int(
        true_truck.sum().item()
    )

    precision = (
        true_count
        / max(
            candidate_count,
            1
        )
    )

    recall = (
        true_count
        / max(
            truck_total,
            1
        )
    )

    pair_masks = {}

    for competitor_id in PAIR_IDS:
        competitor_present = (
            topk
            == competitor_id
        ).any(
            dim=1
        )

        pair_masks[
            competitor_id
        ] = (
            truck_candidate
            & competitor_present
        )

    return {
        "topk":
            topk,
        "truck_candidate":
            truck_candidate,
        "pair_masks":
            pair_masks,
        "count":
            candidate_count,
        "true_count":
            true_count,
        "precision":
            precision,
        "recall":
            recall
    }


def conservative_truck_rescue(
    graph_scores,
    pairwise_scores,
    labels
):
    topk = torch.topk(
        graph_scores,
        k=TOP_K,
        dim=1,
        largest=True,
        sorted=True
    ).indices

    graph_predictions = graph_scores.argmax(
        dim=1
    )

    truck_candidate = (
        topk == TRUCK_ID
    ).any(
        dim=1
    )

    rescue_mask = torch.zeros(
        len(labels),
        dtype=torch.bool
    )

    win_counts = torch.zeros(
        len(labels),
        dtype=torch.long
    )

    skipped_missing_head = 0

    for sample_index in torch.nonzero(
        truck_candidate,
        as_tuple=False
    ).flatten().tolist():

        candidates = topk[
            sample_index
        ].tolist()

        competitors = [
            class_id
            for class_id in candidates
            if class_id != TRUCK_ID
        ]

        if not competitors:
            continue

        if any(
            competitor not in pairwise_scores
            for competitor in competitors
        ):
            skipped_missing_head += 1
            continue

        wins = 0

        for competitor in competitors:
            probability = pairwise_scores[
                competitor
            ][
                sample_index
            ].item()

            if probability >= PAIR_THRESHOLD:
                wins += 1

        win_counts[
            sample_index
        ] = wins

        if (
            wins == len(competitors)
            and graph_predictions[
                sample_index
            ].item()
            != TRUCK_ID
        ):
            rescue_mask[
                sample_index
            ] = True

    rescued_predictions = (
        graph_predictions.clone()
    )

    rescued_predictions[
        rescue_mask
    ] = TRUCK_ID

    correct = (
        labels[
            rescue_mask
        ]
        == TRUCK_ID
    )

    rescue_count = int(
        rescue_mask.sum().item()
    )

    correct_count = int(
        correct.sum().item()
    )

    truck_total = int(
        (
            labels == TRUCK_ID
        ).sum().item()
    )

    return {
        "topk":
            topk,
        "truck_candidate":
            truck_candidate,
        "rescue_mask":
            rescue_mask,
        "win_counts":
            win_counts,
        "rescued_predictions":
            rescued_predictions,
        "rescue_count":
            rescue_count,
        "correct_count":
            correct_count,
        "precision":
            correct_count
            / max(
                rescue_count,
                1
            ),
        "recall":
            correct_count
            / max(
                truck_total,
                1
            ),
        "skipped_missing_head":
            skipped_missing_head
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
        "VISDA-2017 SOURCE-TRAINED PAIRWISE TRUCK DISAMBIGUATION"
    )
    print("=" * 90)

    print(
        f"device={device}"
    )

    print(
        f"seed={SEED}"
    )

    print(
        f"pairwise epochs={TRAIN_EPOCHS}"
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
        "Loading RPC model..."
    )

    model = load_rpc_model().to(
        device
    )

    model.eval()

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
        "Loading graph semantic scores..."
    )

    graph_payload = safe_load(
        GRAPH_OUTPUTS
    )

    if "diffused_probabilities" not in graph_payload:
        raise RuntimeError(
            "Graph output missing diffused_probabilities"
        )

    graph_scores = graph_payload[
        "diffused_probabilities"
    ].float()

    if graph_scores.shape != (
        len(target_labels),
        NUM_CLASSES
    ):
        raise RuntimeError(
            f"Graph score shape mismatch: "
            f"{tuple(graph_scores.shape)}"
        )

    graph_predictions = graph_scores.argmax(
        dim=1
    )

    graph_metrics = multiclass_metrics(
        graph_predictions,
        target_labels
    )

    print()
    print(
        "GRAPH BASELINE"
    )

    print(
        f"Overall: "
        f"{graph_metrics['overall']:.2f}%"
    )

    print(
        f"MCA: "
        f"{graph_metrics['mean_class']:.2f}%"
    )

    print(
        f"Truck: "
        f"{graph_metrics['per_class'][TRUCK_ID]:.2f}%"
    )

    print()

    candidate_info = candidate_statistics(
        graph_scores,
        target_labels
    )

    print(
        "TRUCK TOP-3 CANDIDATE POOL"
    )

    print(
        f"Candidate count: "
        f"{candidate_info['count']}"
    )

    print(
        f"True truck candidates: "
        f"{candidate_info['true_count']}"
    )

    print(
        f"Candidate precision: "
        f"{100.0 * candidate_info['precision']:.2f}%"
    )

    print(
        f"Candidate recall: "
        f"{100.0 * candidate_info['recall']:.2f}%"
    )

    print()

    print(
        "TRAINING SOURCE PAIRWISE HEADS"
    )

    pairwise_heads = {}
    pairwise_scores = {}
    pairwise_results = {}

    for competitor_id in PAIR_IDS:
        print()

        print(
            f"Training truck vs "
            f"{CLASSES[competitor_id]}"
        )

        x, y = build_balanced_pair(
            adapted_source,
            source_labels,
            competitor_id
        )

        head = train_pairwise_head(
            x,
            y,
            device,
            competitor_id
        )

        pairwise_heads[
            competitor_id
        ] = head

        scores = predict_pairwise(
            head,
            adapted_target,
            device
        )

        pairwise_scores[
            competitor_id
        ] = scores

        global_metrics = pairwise_metrics(
            scores,
            target_labels,
            competitor_id
        )

        candidate_mask = candidate_info[
            "pair_masks"
        ][
            competitor_id
        ]

        candidate_metrics = pairwise_metrics(
            scores,
            target_labels,
            competitor_id,
            candidate_mask
        )

        pairwise_results[
            CLASSES[competitor_id]
        ] = {
            "global":
                global_metrics,
            "candidate_region":
                candidate_metrics,
            "candidate_count":
                int(
                    candidate_mask.sum().item()
                )
        }

        print(
            f"Global AUC="
            f"{global_metrics['auc']:.6f} | "
            f"Candidate AUC="
            f"{candidate_metrics['auc']:.6f} | "
            f"Candidate truck recall="
            f"{candidate_metrics['truck_recall']:.2f}% | "
            f"Candidate FPR="
            f"{candidate_metrics['negative_false_positive_rate']:.2f}%"
        )

    print()
    print(
        "PAIRWISE CANDIDATE-REGION RESULTS"
    )

    for competitor_id in PAIR_IDS:
        result = pairwise_results[
            CLASSES[competitor_id]
        ]

        print(
            f"truck vs "
            f"{CLASSES[competitor_id]:5s} | "
            f"global_auc="
            f"{result['global']['auc']:.6f} | "
            f"candidate_auc="
            f"{result['candidate_region']['auc']:.6f} | "
            f"candidate_recall="
            f"{result['candidate_region']['truck_recall']:.2f}% | "
            f"candidate_FPR="
            f"{result['candidate_region']['negative_false_positive_rate']:.2f}%"
        )

    print()
    print(
        "CONSERVATIVE TRUCK TOURNAMENT"
    )

    rescue = conservative_truck_rescue(
        graph_scores,
        pairwise_scores,
        target_labels
    )

    rescued_predictions = rescue[
        "rescued_predictions"
    ]

    rescued_metrics = multiclass_metrics(
        rescued_predictions,
        target_labels
    )

    print(
        f"Rescue count: "
        f"{rescue['rescue_count']}"
    )

    print(
        f"Correct rescues: "
        f"{rescue['correct_count']}"
    )

    print(
        f"Rescue precision: "
        f"{100.0 * rescue['precision']:.2f}%"
    )

    print(
        f"Rescue recall: "
        f"{100.0 * rescue['recall']:.2f}%"
    )

    print(
        f"Skipped candidates missing a trained pairwise head: "
        f"{rescue['skipped_missing_head']}"
    )

    print()
    print(
        "RESCUED MULTICLASS PERFORMANCE"
    )

    print(
        f"Overall: "
        f"{rescued_metrics['overall']:.2f}%"
    )

    print(
        f"MCA: "
        f"{rescued_metrics['mean_class']:.2f}%"
    )

    print(
        f"Truck: "
        f"{rescued_metrics['per_class'][TRUCK_ID]:.2f}%"
    )

    print(
        f"Car: "
        f"{rescued_metrics['per_class'][CAR_ID]:.2f}%"
    )

    print(
        f"Bus: "
        f"{rescued_metrics['per_class'][BUS_ID]:.2f}%"
    )

    print(
        f"Train: "
        f"{rescued_metrics['per_class'][TRAIN_ID]:.2f}%"
    )

    print()
    print(
        "CLASS DELTAS"
    )

    class_deltas = {}

    for class_id in range(
        NUM_CLASSES
    ):
        before = graph_metrics[
            "per_class"
        ][class_id]

        after = rescued_metrics[
            "per_class"
        ][class_id]

        delta = (
            after
            - before
        )

        class_deltas[
            CLASSES[class_id]
        ] = delta

        print(
            f"{CLASSES[class_id]:12s} | "
            f"before={before:7.2f}% | "
            f"after={after:7.2f}% | "
            f"delta={delta:+7.2f} pp"
        )

    print()
    print(
        "TRUE TRUCK FLOW"
    )

    truck_mask = (
        target_labels
        == TRUCK_ID
    )

    before_flow = torch.bincount(
        graph_predictions[
            truck_mask
        ],
        minlength=NUM_CLASSES
    )

    after_flow = torch.bincount(
        rescued_predictions[
            truck_mask
        ],
        minlength=NUM_CLASSES
    )

    print(
        "Before:"
    )

    for class_id in range(
        NUM_CLASSES
    ):
        count = int(
            before_flow[
                class_id
            ].item()
        )

        if count > 0:
            print(
                f"{CLASSES[class_id]:12s}: "
                f"{count}"
            )

    print()
    print(
        "After:"
    )

    for class_id in range(
        NUM_CLASSES
    ):
        count = int(
            after_flow[
                class_id
            ].item()
        )

        if count > 0:
            print(
                f"{CLASSES[class_id]:12s}: "
                f"{count}"
            )

    print()
    print(
        "TRUCK PAIRWISE SCORE DISTRIBUTIONS"
    )

    pairwise_distribution = {}

    for competitor_id in PAIR_IDS:
        competitor_name = CLASSES[
            competitor_id
        ]

        scores = pairwise_scores[
            competitor_id
        ]

        truck_values = scores[
            truck_mask
        ]

        competitor_mask = (
            target_labels
            == competitor_id
        )

        competitor_values = scores[
            competitor_mask
        ]

        candidate_mask = candidate_info[
            "pair_masks"
        ][
            competitor_id
        ]

        candidate_values = scores[
            candidate_mask
        ]

        pairwise_distribution[
            competitor_name
        ] = {
            "truck_mean":
                float(
                    truck_values.mean().item()
                ),
            "truck_median":
                float(
                    truck_values.median().item()
                ),
            "competitor_mean":
                float(
                    competitor_values.mean().item()
                ),
            "competitor_median":
                float(
                    competitor_values.median().item()
                ),
            "candidate_mean":
                float(
                    candidate_values.mean().item()
                )
                if len(candidate_values) > 0
                else None,
            "candidate_median":
                float(
                    candidate_values.median().item()
                )
                if len(candidate_values) > 0
                else None
        }

        print(
            f"truck vs "
            f"{competitor_name:5s} | "
            f"truck_mean="
            f"{truck_values.mean().item():.6f} | "
            f"competitor_mean="
            f"{competitor_values.mean().item():.6f} | "
            f"candidate_mean="
            f"{pairwise_distribution[competitor_name]['candidate_mean']}"
        )

    print()
    print(
        "PAIRWISE SOURCE MODEL SUMMARY"
    )

    source_summary = {}

    for competitor_id in PAIR_IDS:
        competitor_name = CLASSES[
            competitor_id
        ]

        x, y = balanced_pair_data(
            adapted_source,
            source_labels,
            competitor_id
        )

        with torch.no_grad():
            head = pairwise_heads[
                competitor_id
            ].to(
                device
            )

            source_probability = torch.sigmoid(
                head(
                    x.to(device)
                )
            ).cpu()

            head.cpu()

        source_summary[
            competitor_name
        ] = {
            "samples":
                len(x),
            "positive":
                int(
                    (y == 1).sum().item()
                ),
            "negative":
                int(
                    (y == 0).sum().item()
                ),
            "accuracy":
                float(
                    (
                        (
                            source_probability
                            >= PAIR_THRESHOLD
                        )
                        == y
                    )
                    .float()
                    .mean()
                    .item()
                    * 100.0
                )
        }

        print(
            f"truck vs "
            f"{competitor_name:5s} | "
            f"source_samples="
            f"{len(x)} | "
            f"source_accuracy="
            f"{source_summary[competitor_name]['accuracy']:.2f}%"
        )

    torch.save(
        {
            "graph_scores":
                graph_scores,
            "graph_top3":
                candidate_info["topk"],
            "truck_candidate":
                candidate_info["truck_candidate"],
            "pair_masks":
                candidate_info["pair_masks"],
            "pairwise_truck_vs_car":
                pairwise_scores[
                    CAR_ID
                ],
            "pairwise_truck_vs_bus":
                pairwise_scores[
                    BUS_ID
                ],
            "pairwise_truck_vs_train":
                pairwise_scores[
                    TRAIN_ID
                ],
            "rescue_mask":
                rescue[
                    "rescue_mask"
                ],
            "rescued_predictions":
                rescued_predictions,
            "target_labels":
                target_labels
        },
        OUTPUT_TENSOR
    )

    result = {
        "experiment":
            "visda_source_trained_pairwise_truck_disambiguation",
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
        "pairwise_epochs":
            TRAIN_EPOCHS,
        "graph_baseline":
            graph_metrics,
        "truck_candidate_pool": {
            "count":
                candidate_info[
                    "count"
                ],
            "true_count":
                candidate_info[
                    "true_count"
                ],
            "precision":
                candidate_info[
                    "precision"
                ],
            "recall":
                candidate_info[
                    "recall"
                ]
        },
        "pairwise_results":
            pairwise_results,
        "pairwise_distributions":
            pairwise_distribution,
        "source_summary":
            source_summary,
        "rescue": {
            "count":
                rescue[
                    "rescue_count"
                ],
            "correct":
                rescue[
                    "correct_count"
                ],
            "precision":
                rescue[
                    "precision"
                ],
            "recall":
                rescue[
                    "recall"
                ],
            "skipped_missing_head":
                rescue[
                    "skipped_missing_head"
                ]
        },
        "rescued_metrics":
            rescued_metrics,
        "class_deltas":
            class_deltas,
        "before_truck_flow": {
            CLASSES[i]:
                int(
                    before_flow[i].item()
                )
            for i in range(
                NUM_CLASSES
            )
        },
        "after_truck_flow": {
            CLASSES[i]:
                int(
                    after_flow[i].item()
                )
            for i in range(
                NUM_CLASSES
            )
        }
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
    print("=" * 90)
    print(
        "PAIRWISE TRUCK DISAMBIGUATION COMPLETE"
    )
    print("=" * 90)

    print(
        f"Graph MCA: "
        f"{graph_metrics['mean_class']:.2f}%"
    )

    print(
        f"Rescued MCA: "
        f"{rescued_metrics['mean_class']:.2f}%"
    )

    print(
        f"Graph truck: "
        f"{graph_metrics['per_class'][TRUCK_ID]:.2f}%"
    )

    print(
        f"Rescued truck: "
        f"{rescued_metrics['per_class'][TRUCK_ID]:.2f}%"
    )

    print(
        f"Rescue precision: "
        f"{100.0 * rescue['precision']:.2f}%"
    )

    print(
        f"Rescue recall: "
        f"{100.0 * rescue['recall']:.2f}%"
    )

    print(
        f"Saved: {OUTPUT_JSON}"
    )


if __name__ == "__main__":
    main()