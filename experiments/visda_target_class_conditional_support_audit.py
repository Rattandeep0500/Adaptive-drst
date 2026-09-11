import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import average_precision_score, roc_auc_score


SEED = 42
NUM_CLASSES = 12
INPUT_DIM = 2048
HIDDEN_DIM = 512
TRUCK = 11
DEFAULT_SUBPROTOTYPES = 5
DEFAULT_SOURCE_PER_CLASS = 2000
DEFAULT_CHUNK = 512

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
        "--rpc-checkpoint",
        type=Path,
        default=Path(
            "checkpoints/visda_rpc_causal_experiment/rpc_seed42.pt"
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
            "checkpoints/visda_target_class_conditional_support"
        ),
    )

    parser.add_argument(
        "--source-per-class",
        type=int,
        default=DEFAULT_SOURCE_PER_CLASS,
    )

    parser.add_argument(
        "--subprototypes",
        type=int,
        default=DEFAULT_SUBPROTOTYPES,
    )

    parser.add_argument(
        "--target-chunk",
        type=int,
        default=DEFAULT_CHUNK,
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


def load_cache(cache_dir):
    files = sorted(
        cache_dir.glob("chunk_*.pt")
    )

    if not files:
        raise RuntimeError(
            f"No chunk_*.pt files found in {cache_dir}"
        )

    features = []
    labels = []

    have_labels = True

    for path in files:
        payload = safe_load(path)

        if "features" not in payload:
            raise RuntimeError(
                f"Missing features in {path}"
            )

        features.append(
            payload["features"]
            .float()
            .cpu()
        )

        if "labels" in payload:
            labels.append(
                payload["labels"]
                .long()
                .cpu()
            )
        else:
            have_labels = False

    features = torch.cat(
        features,
        dim=0,
    )

    if features.ndim != 2:
        raise RuntimeError(
            f"Expected 2D feature matrix, got {features.shape}"
        )

    if features.shape[1] != INPUT_DIM:
        raise RuntimeError(
            f"Expected feature dimension {INPUT_DIM}, "
            f"got {features.shape[1]}"
        )

    if have_labels:
        labels = torch.cat(
            labels,
            dim=0,
        )
    else:
        labels = None

    return features, labels, files


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


def extract_state_dict(payload):
    if not isinstance(
        payload,
        dict,
    ):
        raise RuntimeError(
            "Checkpoint is not a dictionary"
        )

    if "student_state_dict" in payload:
        return payload[
            "student_state_dict"
        ], "student_state_dict"

    if "state_dict" in payload:
        return payload[
            "state_dict"
        ], "state_dict"

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
        return payload, "root"

    raise RuntimeError(
        "Could not find student state dict"
    )


def load_model(checkpoint):
    payload = safe_load(
        checkpoint
    )

    state_dict, state_key = extract_state_dict(
        payload
    )

    model = MCDModel()

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    return model, state_key


def encode_features(
    model,
    features,
    chunk_size,
):
    outputs = []

    with torch.no_grad():
        for start in range(
            0,
            len(features),
            chunk_size,
        ):
            end = min(
                start + chunk_size,
                len(features),
            )

            batch = features[
                start:end
            ]

            outputs.append(
                model.encode(
                    batch
                ).cpu()
            )

    return torch.cat(
        outputs,
        dim=0,
    )


def deterministic_class_subset(
    features,
    labels,
    class_id,
    count,
):
    indices = torch.nonzero(
        labels == class_id,
        as_tuple=False,
    ).flatten()

    if len(indices) < count:
        count = len(indices)

    if count == 0:
        raise RuntimeError(
            f"No source examples for class {class_id}"
        )

    step = max(
        1,
        len(indices) // count,
    )

    selected = indices[
        ::step
    ]

    if len(selected) > count:
        selected = selected[
            :count
        ]

    if len(selected) < count:
        used = set(
            selected.tolist()
        )

        extra = [
            int(index)
            for index in indices.tolist()
            if int(index) not in used
        ]

        need = count - len(selected)

        if need > 0:
            selected = torch.cat(
                [
                    selected,
                    torch.tensor(
                        extra[:need],
                        dtype=torch.long,
                    ),
                ]
            )

    return selected


def build_source_subprototypes(
    source_features,
    source_labels,
    per_class,
    subprototypes,
    seed,
):
    normalized = F.normalize(
        source_features,
        dim=1,
    ).numpy()

    prototypes = np.zeros(
        (
            NUM_CLASSES,
            subprototypes,
            HIDDEN_DIM,
        ),
        dtype=np.float32,
    )

    supports = np.zeros(
        NUM_CLASSES,
        dtype=np.int64,
    )

    for class_id in range(
        NUM_CLASSES
    ):
        selected = deterministic_class_subset(
            source_features,
            source_labels,
            class_id,
            per_class,
        )

        supports[
            class_id
        ] = len(selected)

        x = normalized[
            selected.numpy()
        ]

        effective_k = min(
            subprototypes,
            len(x),
        )

        if effective_k == 1:
            center = np.mean(
                x,
                axis=0,
                keepdims=True,
            )
            center /= np.maximum(
                np.linalg.norm(
                    center,
                    axis=1,
                    keepdims=True,
                ),
                1e-12,
            )
            prototypes[
                class_id,
                0,
            ] = center[0]

            continue

        kmeans = MiniBatchKMeans(
            n_clusters=effective_k,
            random_state=seed + class_id,
            batch_size=min(
                512,
                len(x),
            ),
            n_init=5,
            max_iter=100,
        )

        assignments = kmeans.fit_predict(
            x
        )

        centers = kmeans.cluster_centers_

        centers /= np.maximum(
            np.linalg.norm(
                centers,
                axis=1,
                keepdims=True,
            ),
            1e-12,
        )

        for prototype_id in range(
            effective_k
        ):
            prototypes[
                class_id,
                prototype_id,
            ] = centers[
                prototype_id
            ]

        if effective_k < subprototypes:
            filler = centers[
                0
            ]

            for prototype_id in range(
                effective_k,
                subprototypes,
            ):
                prototypes[
                    class_id,
                    prototype_id,
                ] = filler

    return (
        prototypes,
        supports,
    )


def compute_support_scores(
    target_features,
    prototypes,
    chunk_size,
):
    prototype_tensor = torch.from_numpy(
        prototypes
    ).float()

    prototype_tensor = F.normalize(
        prototype_tensor,
        dim=2,
    )

    class_supports = []

    with torch.no_grad():
        for start in range(
            0,
            len(target_features),
            chunk_size,
        ):
            end = min(
                start + chunk_size,
                len(target_features),
            )

            batch = target_features[
                start:end
            ].float()

            batch = F.normalize(
                batch,
                dim=1,
            )

            similarity = torch.einsum(
                "bd,ckd->bck",
                batch,
                prototype_tensor,
            )

            class_score = similarity.max(
                dim=2
            ).values

            class_supports.append(
                class_score.cpu()
            )

    return torch.cat(
        class_supports,
        dim=0,
    )


def softmax_support(
    support_scores,
    temperature,
):
    return torch.softmax(
        support_scores
        / temperature,
        dim=1,
    )


def entropy(probabilities):
    eps = 1e-12

    return -torch.sum(
        probabilities
        * torch.log(
            probabilities.clamp_min(eps)
        ),
        dim=1,
    )


def normalized_entropy(probabilities):
    return (
        entropy(
            probabilities
        )
        / np.log(
            NUM_CLASSES
        )
    )


def load_teacher_predictions(
    graph_path,
    target_count,
):
    payload = safe_load(
        graph_path
    )

    arrays = collect_arrays(
        payload
    )

    candidates = []

    for name, value in arrays.items():
        value = np.asarray(
            value
        )

        if value.ndim != 2:
            continue

        if value.shape[0] != target_count:
            continue

        if value.shape[1] != NUM_CLASSES:
            continue

        normalized = (
            name
            .replace(
                ".",
                "_",
            )
            .replace(
                "-",
                "_",
            )
            .lower()
        )

        score = 0

        if "diffused" in normalized:
            score += 20

        if "diffusion" in normalized:
            score += 20

        if "semantic" in normalized:
            score += 10

        if "prob" in normalized:
            score += 5

        if "prediction" in normalized:
            score += 2

        if "anchor" in normalized:
            score -= 100

        candidates.append(
            (
                score,
                name,
                value.astype(
                    np.float32
                ),
            )
        )

    if not candidates:
        raise RuntimeError(
            "No N x 12 teacher tensor found "
            f"for target count {target_count}"
        )

    candidates.sort(
        key=lambda item: (
            item[0],
            item[1],
        ),
        reverse=True,
    )

    name, array = (
        candidates[0][1],
        candidates[0][2],
    )

    return name, array


def collect_arrays(obj, prefix=""):
    found = {}

    if isinstance(
        obj,
        dict,
    ):
        for key, value in obj.items():
            name = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            found.update(
                collect_arrays(
                    value,
                    name,
                )
            )

    elif isinstance(
        obj,
        (list, tuple),
    ):
        for index, value in enumerate(
            obj
        ):
            name = (
                f"{prefix}.{index}"
                if prefix
                else str(index)
            )

            found.update(
                collect_arrays(
                    value,
                    name,
                )
            )

    else:
        array = value_to_numpy(
            obj
        )

        if array is not None:
            found[
                prefix.lower()
            ] = array

    return found


def value_to_numpy(value):
    if torch.is_tensor(
        value
    ):
        return value.detach().cpu().numpy()

    if isinstance(
        value,
        np.ndarray,
    ):
        return value

    return None


def compute_reliability_metrics(
    scores,
    correctness,
):
    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    correctness = np.asarray(
        correctness,
        dtype=np.int64,
    )

    invalid = ~np.isfinite(
        scores
    )

    if invalid.any():
        scores = scores.copy()
        scores[
            invalid
        ] = 0.0

    positive = (
        correctness == 1
    )

    negative = (
        correctness == 0
    )

    result = {}

    result[
        "mean_score_correct"
    ] = float(
        scores[
            positive
        ].mean()
        if positive.any()
        else 0.0
    )

    result[
        "mean_score_incorrect"
    ] = float(
        scores[
            negative
        ].mean()
        if negative.any()
        else 0.0
    )

    if positive.any() and negative.any():
        result[
            "auc_correctness"
        ] = float(
            roc_auc_score(
                correctness,
                scores,
            )
        )

        result[
            "ap_correctness"
        ] = float(
            average_precision_score(
                correctness,
                scores,
            )
        )
    else:
        result[
            "auc_correctness"
        ] = None

        result[
            "ap_correctness"
        ] = None

    return result


def evaluate_confidence_bins(
    scores,
    correctness,
    bins=10,
):
    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    correctness = np.asarray(
        correctness,
        dtype=np.int64,
    )

    edges = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    rows = []

    for index in range(
        bins
    ):
        left = edges[
            index
        ]

        right = edges[
            index + 1
        ]

        if index == bins - 1:
            mask = (
                scores >= left
            ) & (
                scores <= right
            )
        else:
            mask = (
                scores >= left
            ) & (
                scores < right
            )

        if not mask.any():
            continue

        rows.append(
            {
                "lower": float(left),
                "upper": float(right),
                "count": int(
                    mask.sum()
                ),
                "accuracy": float(
                    correctness[
                        mask
                    ].mean()
                ),
            }
        )

    return rows


def top_fraction_metrics(
    score,
    correctness,
):
    order = np.argsort(
        -score,
        kind="mergesort",
    )

    fractions = [
        0.01,
        0.05,
        0.10,
        0.20,
        0.30,
        0.50,
    ]

    results = []

    for fraction in fractions:
        count = max(
            1,
            int(
                np.ceil(
                    fraction
                    * len(score)
                )
            ),
        )

        selected = order[
            :count
        ]

        results.append(
            {
                "fraction": float(
                    fraction
                ),
                "samples": int(
                    count
                ),
                "accuracy": float(
                    correctness[
                        selected
                    ].mean()
                ),
            }
        )

    return results


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
        "VISDA-2017 TARGET CLASS-CONDITIONAL SUPPORT AUDIT"
    )
    print("=" * 100)
    print(
        "device=cpu"
    )
    print(
        f"seed={args.seed}"
    )
    print(
        f"source_per_class={args.source_per_class}"
    )
    print(
        f"subprototypes={args.subprototypes}"
    )
    print(
        f"target_chunk={args.target_chunk}"
    )

    print(
        "\nLoading source cache..."
    )

    source_features, source_labels, source_files = load_cache(
        args.source_cache
    )

    if source_labels is None:
        raise RuntimeError(
            "Source labels are required for class-conditional support construction"
        )

    if len(source_features) != len(
        source_labels
    ):
        raise RuntimeError(
            "Source features and labels have different lengths"
        )

    print(
        f"source_samples={len(source_features)}"
    )

    print(
        f"source_files={len(source_files)}"
    )

    class_counts = np.bincount(
        source_labels.numpy(),
        minlength=NUM_CLASSES,
    )

    for class_id, class_name in enumerate(
        CLASSES
    ):
        print(
            f"{class_name:12s} "
            f"source_count={int(class_counts[class_id])}"
        )

    print(
        "\nLoading target cache..."
    )

    target_features, target_labels, target_files = load_cache(
        args.target_cache
    )

    print(
        f"target_samples={len(target_features)}"
    )

    print(
        f"target_files={len(target_files)}"
    )

    if target_labels is None:
        raise RuntimeError(
            "Target labels must exist in the evaluation cache"
        )

    target_count = len(
        target_features
    )

    print(
        "\nLoading frozen RPC checkpoint..."
    )

    model, state_key = load_model(
        args.rpc_checkpoint
    )

    print(
        f"rpc_state_key={state_key}"
    )

    print(
        "\nEncoding source features through frozen RPC adapter..."
    )

    adapted_source = encode_features(
        model,
        source_features,
        args.target_chunk,
    )

    print(
        f"adapted_source_shape={tuple(adapted_source.shape)}"
    )

    print(
        "\nEncoding target features through frozen RPC adapter..."
    )

    adapted_target = encode_features(
        model,
        target_features,
        args.target_chunk,
    )

    print(
        f"adapted_target_shape={tuple(adapted_target.shape)}"
    )

    del model
    del source_features
    del target_features

    print(
        "\nBuilding source class-conditional subprototypes..."
    )

    (
        prototypes,
        source_support,
    ) = build_source_subprototypes(
        adapted_source,
        source_labels,
        args.source_per_class,
        args.subprototypes,
        args.seed,
    )

    print(
        f"prototype_shape={prototypes.shape}"
    )

    for class_id, class_name in enumerate(
        CLASSES
    ):
        print(
            f"{class_name:12s} "
            f"support_used={int(source_support[class_id])}"
        )

    print(
        "\nComputing independent target class support..."
    )

    support_logits = compute_support_scores(
        adapted_target,
        prototypes,
        args.target_chunk,
    )

    if support_logits.shape != (
        target_count,
        NUM_CLASSES,
    ):
        raise RuntimeError(
            "Support score tensor has unexpected shape"
        )

    support_probabilities = softmax_support(
        support_logits,
        0.10,
    )

    support_top1 = support_probabilities.argmax(
        dim=1
    )

    support_top1_score = support_probabilities.max(
        dim=1
    ).values

    support_sorted = torch.sort(
        support_probabilities,
        dim=1,
        descending=True,
    ).values

    support_margin = (
        support_sorted[
            :,
            0,
        ]
        - support_sorted[
            :,
            1,
        ]
    )

    support_entropy = normalized_entropy(
        support_probabilities
    )

    support_reliability = (
        support_top1_score
        * (
            1.0
            - support_entropy
        )
    )

    print(
        "\nIndependent support scores are now frozen."
    )

    print(
        f"support_shape={tuple(support_probabilities.shape)}"
    )

    print(
        "\nLoading teacher predictions..."
    )

    graph_path = Path(
        "checkpoints/visda_graph_semantic_diffusion/"
        "graph_semantic_outputs_seed42.pt"
    )

    teacher_key, teacher_scores = load_teacher_predictions(
        graph_path,
        target_count,
    )

    teacher_scores_t = torch.from_numpy(
        teacher_scores
    )

    teacher_probabilities = (
        teacher_scores_t
        if (
            teacher_scores_t.min()
            >= 0.0
            and teacher_scores_t.max()
            <= 1.0
        )
        else torch.softmax(
            teacher_scores_t,
            dim=1,
        )
    )

    teacher_prediction = teacher_probabilities.argmax(
        dim=1
    )

    teacher_confidence = teacher_probabilities.max(
        dim=1
    ).values

    teacher_entropy = normalized_entropy(
        teacher_probabilities
    )

    teacher_support_agreement = (
        teacher_prediction
        == support_top1
    )

    print(
        f"teacher_tensor={teacher_key}"
    )

    print(
        f"teacher_shape={teacher_probabilities.shape}"
    )

    print(
        "\nAll label-free quantities are frozen."
    )

    print(
        "Target labels will now be used for evaluation only."
    )

    target_labels_np = target_labels.numpy().astype(
        np.int64
    )

    teacher_prediction_np = teacher_prediction.numpy().astype(
        np.int64
    )

    support_top1_np = support_top1.numpy().astype(
        np.int64
    )

    teacher_confidence_np = teacher_confidence.numpy()

    support_top1_score_np = support_top1_score.numpy()

    support_margin_np = support_margin.numpy()

    support_entropy_np = support_entropy.numpy()

    support_reliability_np = support_reliability.numpy()

    agreement_np = teacher_support_agreement.numpy().astype(
        np.int64
    )

    teacher_correct = (
        teacher_prediction_np
        == target_labels_np
    ).astype(
        np.int64
    )

    support_correct = (
        support_top1_np
        == target_labels_np
    ).astype(
        np.int64
    )

    teacher_support_correct = (
        agreement_np
        & (
            teacher_correct == 1
        )
    ).astype(
        np.int64
    )

    truck_mask = (
        target_labels_np
        == TRUCK
    )

    print(
        "\nBASELINE TEACHER"
    )

    teacher_accuracy = float(
        teacher_correct.mean()
        * 100.0
    )

    print(
        f"overall_accuracy={teacher_accuracy:.2f}%"
    )

    print(
        f"mean_confidence={teacher_confidence_np.mean():.6f}"
    )

    print(
        "\nINDEPENDENT SUPPORT PREDICTOR"
    )

    support_accuracy = float(
        support_correct.mean()
        * 100.0
    )

    print(
        f"support_top1_accuracy={support_accuracy:.2f}%"
    )

    print(
        f"support_top1_mean_probability={support_top1_score_np.mean():.6f}"
    )

    print(
        f"support_mean_margin={support_margin_np.mean():.6f}"
    )

    print(
        f"support_mean_normalized_entropy={support_entropy_np.mean():.6f}"
    )

    print(
        f"teacher_support_agreement={agreement_np.mean() * 100.0:.2f}%"
    )

    print(
        "\nRELIABILITY ANALYSIS"
    )

    teacher_confidence_metrics = compute_reliability_metrics(
        teacher_confidence_np,
        teacher_correct,
    )

    support_score_for_teacher = np.where(
        teacher_prediction_np
        == support_top1_np,
        support_reliability_np,
        -support_reliability_np,
    )

    agreement_score = (
        support_reliability_np
        * agreement_np
    )

    support_on_teacher_prediction = (
        support_probabilities[
            torch.arange(
                target_count
            ),
            teacher_prediction,
        ]
        .numpy()
    )

    support_margin_on_teacher_prediction = (
        support_margin_np
    )

    support_margin_correctness = (
        support_margin_on_teacher_prediction
    )

    support_on_teacher_correctness = compute_reliability_metrics(
        support_on_teacher_prediction,
        teacher_correct,
    )

    support_margin_metrics = compute_reliability_metrics(
        support_margin_correctness,
        teacher_correct,
    )

    agreement_metrics = compute_reliability_metrics(
        agreement_score,
        teacher_correct,
    )

    print(
        "teacher_confidence"
    )

    print(
        json.dumps(
            teacher_confidence_metrics,
            indent=2,
        )
    )

    print(
        "independent_support_on_teacher_prediction"
    )

    print(
        json.dumps(
            support_on_teacher_correctness,
            indent=2,
        )
    )

    print(
        "independent_support_margin"
    )

    print(
        json.dumps(
            support_margin_metrics,
            indent=2,
        )
    )

    print(
        "support_agreement_reliability"
    )

    print(
        json.dumps(
            agreement_metrics,
            indent=2,
        )
    )

    print(
        "\nTEACHER CONFIDENCE TOP-FRACTION ACCURACY"
    )

    teacher_top_fraction = top_fraction_metrics(
        teacher_confidence_np,
        teacher_correct,
    )

    for row in teacher_top_fraction:
        print(
            f"{row['fraction'] * 100:5.1f}% "
            f"samples={row['samples']:6d} "
            f"accuracy={row['accuracy'] * 100:6.2f}%"
        )

    print(
        "\nSUPPORT RELIABILITY TOP-FRACTION ACCURACY"
    )

    support_top_fraction = top_fraction_metrics(
        support_reliability_np,
        teacher_correct,
    )

    for row in support_top_fraction:
        print(
            f"{row['fraction'] * 100:5.1f}% "
            f"samples={row['samples']:6d} "
            f"accuracy={row['accuracy'] * 100:6.2f}%"
        )

    print(
        "\nAGREEMENT-GATED TEACHER"
    )

    agreement_thresholds = [
        0.25,
        0.50,
        0.75,
        0.90,
    ]

    agreement_gated_results = []

    for threshold in agreement_thresholds:
        support_mask = (
            support_top1_score_np
            >= threshold
        )

        accepted = (
            support_mask
            & (
                teacher_prediction_np
                == support_top1_np
            )
        )

        count = int(
            accepted.sum()
        )

        accuracy = float(
            teacher_correct[
                accepted
            ].mean()
            if count > 0
            else 0.0
        )

        coverage = (
            count
            / target_count
        )

        agreement_gated_results.append(
            {
                "support_threshold": float(
                    threshold
                ),
                "accepted": count,
                "coverage": float(
                    coverage
                ),
                "accuracy": float(
                    accuracy
                ),
            }
        )

        print(
            f"support_score>={threshold:.2f} "
            f"accepted={count:6d} "
            f"coverage={coverage * 100:6.2f}% "
            f"teacher_accuracy={accuracy * 100:6.2f}%"
        )

    print(
        "\nSTRICT SUPPORT AGREEMENT GATE"
    )

    strict_gate = (
        (
            teacher_prediction_np
            == support_top1_np
        )
        & (
            support_reliability_np
            >= np.quantile(
                support_reliability_np,
                0.50,
            )
        )
    )

    strict_count = int(
        strict_gate.sum()
    )

    strict_accuracy = float(
        teacher_correct[
            strict_gate
        ].mean()
        if strict_count > 0
        else 0.0
    )

    strict_coverage = (
        strict_count
        / target_count
    )

    print(
        f"accepted={strict_count}"
    )

    print(
        f"coverage={strict_coverage * 100:.2f}%"
    )

    print(
        f"accuracy={strict_accuracy * 100:.2f}%"
    )

    print(
        "\nTRUCK ANALYSIS"
    )

    if truck_mask.any():
        true_truck_count = int(
            truck_mask.sum()
        )

        teacher_truck_correct = (
            teacher_prediction_np[
                truck_mask
            ]
            == TRUCK
        )

        support_truck_correct = (
            support_top1_np[
                truck_mask
            ]
            == TRUCK
        )

        teacher_truck_confidence = (
            teacher_confidence_np[
                truck_mask
            ]
        )

        truck_support_probability = (
            support_probabilities[
                truck_mask,
                TRUCK,
            ]
            .numpy()
        )

        truck_support_reliability = (
            support_reliability_np[
                truck_mask
            ]
        )

        print(
            f"true_truck_samples={true_truck_count}"
        )

        print(
            f"teacher_truck_accuracy="
            f"{teacher_truck_correct.mean() * 100:.2f}%"
        )

        print(
            f"support_truck_top1_accuracy="
            f"{support_truck_correct.mean() * 100:.2f}%"
        )

        print(
            f"teacher_confidence_mean_on_truck="
            f"{teacher_truck_confidence.mean():.6f}"
        )

        print(
            f"independent_truck_support_probability_mean="
            f"{truck_support_probability.mean():.6f}"
        )

        print(
            f"truck_support_reliability_mean="
            f"{truck_support_reliability.mean():.6f}"
        )

        truck_teacher_auc = None

        if np.any(
            teacher_prediction_np
            == TRUCK
        ) and np.any(
            teacher_prediction_np
            != TRUCK
        ):
            truck_target = (
                target_labels_np
                == TRUCK
            ).astype(
                np.int64
            )

            truck_teacher_auc = float(
                roc_auc_score(
                    truck_target,
                    teacher_confidence_np,
                )
            )

        truck_support_auc = None

        if np.any(
            truck_support_probability > 0
        ):
            truck_target = (
                target_labels_np
                == TRUCK
            ).astype(
                np.int64
            )

            truck_support_auc = float(
                roc_auc_score(
                    truck_target,
                    support_probabilities[
                        :,
                        TRUCK,
                    ].numpy()
                )
            )

        print(
            f"teacher_truck_score_auc={truck_teacher_auc}"
        )

        print(
            f"independent_truck_support_auc={truck_support_auc}"
        )

    print(
        "\nPER-CLASS TEACHER VS SUPPORT"
    )

    per_class = {}

    for class_id, class_name in enumerate(
        CLASSES
    ):
        mask = (
            target_labels_np
            == class_id
        )

        teacher_class_accuracy = float(
            teacher_correct[
                mask
            ].mean()
            * 100.0
        )

        support_class_accuracy = float(
            support_correct[
                mask
            ].mean()
            * 100.0
        )

        agreement_class_accuracy = float(
            teacher_support_correct[
                mask
            ].mean()
            * 100.0
        )

        per_class[
            class_name
        ] = {
            "teacher_accuracy": teacher_class_accuracy,
            "support_accuracy": support_class_accuracy,
            "agreement_and_correct": agreement_class_accuracy,
            "samples": int(
                mask.sum()
            ),
        }

        print(
            f"{class_name:12s} "
            f"teacher={teacher_class_accuracy:6.2f}% "
            f"support={support_class_accuracy:6.2f}% "
            f"agreement_correct={agreement_class_accuracy:6.2f}%"
        )

    print(
        "\nSUPPORT-BASED PSEUDO-LABEL RELIABILITY"
    )

    support_reliability_metrics = compute_reliability_metrics(
        support_reliability_np,
        teacher_correct,
    )

    for key, value in support_reliability_metrics.items():
        print(
            f"{key}={value}"
        )

    print(
        "\nCOMPOSITE LABEL-FREE RELIABILITY SIGNAL"
    )

    composite = (
        0.50
        * teacher_confidence_np
        + 0.30
        * support_reliability_np
        + 0.20
        * agreement_np
    )

    composite_metrics = compute_reliability_metrics(
        composite,
        teacher_correct,
    )

    print(
        json.dumps(
            composite_metrics,
            indent=2,
        )
    )

    output_npz = (
        args.output_dir
        / f"target_class_conditional_support_seed{args.seed}.npz"
    )

    np.savez_compressed(
        output_npz,
        support_probabilities=support_probabilities.numpy(),
        support_logits=support_logits.numpy(),
        support_top1=support_top1_np,
        support_top1_score=support_top1_score_np,
        support_margin=support_margin_np,
        support_entropy=support_entropy_np,
        support_reliability=support_reliability_np,
        teacher_probabilities=teacher_probabilities.numpy(),
        teacher_prediction=teacher_prediction_np,
        teacher_confidence=teacher_confidence_np,
        teacher_support_agreement=agreement_np,
    )

    output_json = (
        args.output_dir
        / f"target_class_conditional_support_seed{args.seed}.json"
    )

    summary = {
        "experiment": (
            "visda_target_class_conditional_support_audit"
        ),
        "seed": args.seed,
        "device": "cpu",
        "constraints": {
            "target_labels_used_for_support_construction": False,
            "target_labels_used_for_prototype_construction": False,
            "target_labels_loaded_only_after_scores_frozen": True,
            "model_training": False,
            "rpc_model_frozen": True,
        },
        "configuration": {
            "source_per_class": args.source_per_class,
            "subprototypes": args.subprototypes,
            "target_chunk": args.target_chunk,
            "support_temperature": 0.10,
        },
        "inputs": {
            "rpc_checkpoint": str(
                args.rpc_checkpoint
            ),
            "source_cache": str(
                args.source_cache
            ),
            "target_cache": str(
                args.target_cache
            ),
            "teacher_graph_output": str(
                graph_path
            ),
            "teacher_tensor": teacher_key,
            "target_samples": int(
                target_count
            ),
        },
        "source_support": {
            class_name: int(
                source_support[class_id]
            )
            for class_id, class_name in enumerate(
                CLASSES
            )
        },
        "baseline": {
            "teacher_accuracy": teacher_accuracy,
            "mean_teacher_confidence": float(
                teacher_confidence_np.mean()
            ),
        },
        "support": {
            "top1_accuracy": support_accuracy,
            "mean_top1_probability": float(
                support_top1_score_np.mean()
            ),
            "mean_margin": float(
                support_margin_np.mean()
            ),
            "mean_normalized_entropy": float(
                support_entropy_np.mean()
            ),
            "teacher_support_agreement": float(
                agreement_np.mean()
            ),
        },
        "reliability": {
            "teacher_confidence": teacher_confidence_metrics,
            "support_on_teacher_prediction": support_on_teacher_correctness,
            "support_margin": support_margin_metrics,
            "agreement_reliability": agreement_metrics,
            "composite": composite_metrics,
            "teacher_top_fraction": teacher_top_fraction,
            "support_top_fraction": support_top_fraction,
        },
        "agreement_gated_results": agreement_gated_results,
        "strict_gate": {
            "accepted": strict_count,
            "coverage": float(
                strict_coverage
            ),
            "accuracy": float(
                strict_accuracy
            ),
        },
        "per_class": per_class,
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
        "\nTARGET CLASS-CONDITIONAL SUPPORT AUDIT COMPLETE"
    )


if __name__ == "__main__":
    main()