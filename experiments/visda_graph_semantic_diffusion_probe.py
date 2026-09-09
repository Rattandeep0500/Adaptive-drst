import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans


SEED = 42

NUM_CLASSES = 12
INPUT_DIM = 2048
HIDDEN_DIM = 512

TARGET_K = 20
ANCHOR_K = 5
ANCHORS_PER_CLASS = 2000

GRAPH_CHUNK = 256
TARGET_EVAL_BATCH = 4096

DIFFUSION_ALPHA = 0.95
DIFFUSION_ITERATIONS = 50
DIFFUSION_TOL = 1e-6

SOURCE_CACHE = Path(
    "checkpoints/visda_feature_cache/source"
)

TARGET_CACHE = Path(
    "checkpoints/visda_feature_cache/target"
)

DEFAULT_CHECKPOINT = Path(
    "checkpoints/visda_rpc_causal_experiment/"
    "rpc_seed42.pt"
)

DEFAULT_OUTPUT_DIR = Path(
    "checkpoints/visda_graph_semantic_diffusion"
)

OUTPUT_DIR = DEFAULT_OUTPUT_DIR
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TRUCK_ID = 11
CAR_ID = 3
BUS_ID = 2
TRAIN_ID = 10
SKATEBOARD_ID = 9
PERSON_ID = 7

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
            f"No cache files found in {cache_dir}"
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

        x = payload["features"].float()
        y = payload["labels"].long()

        if x.ndim != 2:
            raise RuntimeError(
                f"Invalid feature tensor in {path}"
            )

        if x.shape[1] != INPUT_DIM:
            raise RuntimeError(
                f"Expected {INPUT_DIM}-D features, "
                f"got {x.shape[1]} in {path}"
            )

        if len(x) != len(y):
            raise RuntimeError(
                f"Feature/label mismatch in {path}"
            )

        feature_parts.append(x)
        label_parts.append(y)

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
            f"Expected {INPUT_DIM}-D features, "
            f"got {features.shape[1]}"
        )

    return features, labels


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
                f"Expected {INPUT_DIM}-D input, "
                f"got {x.shape[1]}"
            )

        z = self.net(x)

        if z.shape[1] != HIDDEN_DIM:
            raise RuntimeError(
                f"Expected {HIDDEN_DIM}-D output, "
                f"got {z.shape[1]}"
            )

        return z


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


def load_model(checkpoint):
    payload = safe_load(
        checkpoint
    )

    if "student_state_dict" not in payload:
        raise RuntimeError(
            "Checkpoint missing student_state_dict"
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
    features,
    device
):
    model.eval()

    parts = []

    for start in range(
        0,
        len(features),
        TARGET_EVAL_BATCH
    ):
        end = min(
            start + TARGET_EVAL_BATCH,
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
                "Adapted feature dimension is not 512"
            )

        parts.append(
            F.normalize(
                z,
                dim=1
            ).cpu()
        )

    return torch.cat(
        parts,
        dim=0
    )


def build_source_subprototypes(
    source_features,
    source_labels
):
    prototypes = torch.zeros(
        NUM_CLASSES,
        ANCHOR_K,
        HIDDEN_DIM
    )

    counts = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long
    )

    for class_id in range(
        NUM_CLASSES
    ):
        indices = torch.nonzero(
            source_labels == class_id,
            as_tuple=False
        ).flatten()

        if len(indices) == 0:
            continue

        if len(indices) > ANCHORS_PER_CLASS:
            generator = torch.Generator(
                device="cpu"
            )

            generator.manual_seed(
                SEED + class_id
            )

            permutation = torch.randperm(
                len(indices),
                generator=generator
            )

            indices = indices[
                permutation[
                    :ANCHORS_PER_CLASS
                ]
            ]

        x = source_features[
            indices
        ].numpy()

        n_clusters = min(
            ANCHOR_K,
            len(x)
        )

        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=SEED + class_id,
            batch_size=512,
            n_init=3,
            max_iter=100
        )

        kmeans.fit(
            x
        )

        centers = torch.from_numpy(
            kmeans.cluster_centers_
        ).float()

        centers = F.normalize(
            centers,
            dim=1
        )

        prototypes[
            class_id,
            :n_clusters
        ] = centers

        if n_clusters < ANCHOR_K:
            fill = centers.mean(
                dim=0,
                keepdim=True
            )

            fill = F.normalize(
                fill,
                dim=1
            )

            prototypes[
                class_id,
                n_clusters:
            ] = fill.repeat(
                ANCHOR_K - n_clusters,
                1
            )

        counts[
            class_id
        ] = len(indices)

    return prototypes, counts


def compute_exact_target_knn(
    target_features,
    k,
    chunk_size
):
    n = len(
        target_features
    )

    if target_features.shape[1] != HIDDEN_DIM:
        raise RuntimeError(
            f"Expected {HIDDEN_DIM}-D target features"
        )

    features = F.normalize(
        target_features.float(),
        dim=1
    )

    knn_indices = torch.empty(
        n,
        k,
        dtype=torch.long
    )

    knn_similarity = torch.empty(
        n,
        k,
        dtype=torch.float32
    )

    total_chunks = (
        n + chunk_size - 1
    ) // chunk_size

    for chunk_id, start in enumerate(
        range(
            0,
            n,
            chunk_size
        ),
        start=1
    ):
        end = min(
            start + chunk_size,
            n
        )

        query = features[
            start:end
        ]

        similarity = (
            query
            @ features.T
        )

        local_rows = torch.arange(
            end - start
        )

        global_rows = (
            torch.arange(
                start,
                end
            )
        )

        similarity[
            local_rows,
            global_rows - start
        ] = -float(
            "inf"
        )

        values, indices = torch.topk(
            similarity,
            k=k,
            dim=1,
            largest=True,
            sorted=True
        )

        knn_indices[
            start:end
        ] = indices

        knn_similarity[
            start:end
        ] = values

        if (
            chunk_id == 1
            or chunk_id % 10 == 0
            or chunk_id == total_chunks
        ):
            print(
                f"  chunk "
                f"{chunk_id}/{total_chunks}"
            )

    return (
        knn_indices,
        knn_similarity
    )


def make_mutual_graph(
    knn_indices,
    knn_similarity
):
    n, k = knn_indices.shape

    adjacency = torch.zeros(
        n,
        k,
        dtype=torch.float32
    )

    neighbor_sets = [
        set(
            knn_indices[
                i
            ].tolist()
        )
        for i in range(n)
    ]

    for i in range(n):
        for j in range(k):
            neighbor = int(
                knn_indices[
                    i,
                    j
                ].item()
            )

            if i in neighbor_sets[
                neighbor
            ]:
                adjacency[
                    i,
                    j
                ] = knn_similarity[
                    i,
                    j
                ].clamp_min(
                    0.0
                )

    return adjacency


def anchor_distribution(
    target_features,
    source_prototypes
):
    n = len(
        target_features
    )

    flattened = source_prototypes.reshape(
        NUM_CLASSES * ANCHOR_K,
        HIDDEN_DIM
    )

    flattened = F.normalize(
        flattened,
        dim=1
    )

    similarities = (
        target_features
        @ flattened.T
    )

    similarities = similarities.reshape(
        n,
        NUM_CLASSES,
        ANCHOR_K
    )

    class_scores = similarities.max(
        dim=2
    ).values

    class_scores = (
        class_scores
        - class_scores.max(
            dim=1,
            keepdim=True
        ).values
    )

    anchor_probabilities = F.softmax(
        class_scores / 0.1,
        dim=1
    )

    return (
        class_scores,
        anchor_probabilities
    )


def build_target_transition(
    knn_indices,
    knn_similarity,
    n
):
    rows = []
    cols = []
    weights = []

    k = knn_indices.shape[1]

    for i in range(n):
        for j in range(k):
            neighbor = int(
                knn_indices[
                    i,
                    j
                ].item()
            )

            similarity = float(
                knn_similarity[
                    i,
                    j
                ].item()
            )

            if similarity <= 0.0:
                continue

            rows.append(i)
            cols.append(neighbor)
            weights.append(similarity)

            if i in set(
                knn_indices[
                    neighbor
                ].tolist()
            ):
                rows.append(
                    neighbor
                )
                cols.append(i)
                weights.append(
                    similarity
                )

    if not weights:
        raise RuntimeError(
            "Target graph has no positive edges"
        )

    indices = torch.tensor(
        [
            rows,
            cols
        ],
        dtype=torch.long
    )

    values = torch.tensor(
        weights,
        dtype=torch.float32
    )

    graph = torch.sparse_coo_tensor(
        indices,
        values,
        size=(
            n,
            n
        )
    ).coalesce()

    degree = torch.sparse.sum(
        graph,
        dim=1
    ).to_dense()

    inv_degree = (
        1.0
        / degree.clamp_min(
            1e-12
        )
    )

    row_indices = torch.arange(
        n
    )

    diagonal_indices = torch.stack(
        [
            row_indices,
            row_indices
        ],
        dim=0
    )

    diagonal_values = inv_degree

    degree_inv = torch.sparse_coo_tensor(
        diagonal_indices,
        diagonal_values,
        size=(
            n,
            n
        )
    ).coalesce()

    transition = torch.sparse.mm(
        degree_inv,
        graph
    ).coalesce()

    return transition


def diffuse(
    transition,
    initial_labels,
    alpha,
    iterations,
    tolerance
):
    current = initial_labels.clone()

    residuals = []

    for iteration in range(
        1,
        iterations + 1
    ):
        propagated = torch.sparse.mm(
            transition,
            current
        )

        updated = (
            alpha
            * propagated
            + (
                1.0
                - alpha
            )
            * initial_labels
        )

        residual = float(
            (
                updated
                - current
            ).abs().max().item()
        )

        residuals.append(
            residual
        )

        current = updated

        print(
            f"  diffusion iteration "
            f"{iteration:02d} | "
            f"residual={residual:.8f}"
        )

        if residual < tolerance:
            break

    current = (
        current
        / current.sum(
            dim=1,
            keepdim=True
        ).clamp_min(
            1e-12
        )
    )

    return (
        current,
        residuals
    )


def evaluate(
    probabilities,
    labels
):
    predictions = probabilities.argmax(
        dim=1
    )

    total_accuracy = (
        predictions == labels
    ).float().mean().item()

    class_accuracy = []

    for class_id in range(
        NUM_CLASSES
    ):
        mask = (
            labels == class_id
        )

        if not mask.any():
            class_accuracy.append(
                0.0
            )
        else:
            class_accuracy.append(
                100.0
                * (
                    predictions[mask]
                    == labels[mask]
                )
                .float()
                .mean()
                .item()
            )

    return {
        "overall":
            100.0 * total_accuracy,
        "mean_class":
            float(
                np.mean(
                    class_accuracy
                )
            ),
        "per_class":
            class_accuracy
    }


def evaluate_confidence(
    probabilities
):
    confidence, predictions = (
        probabilities.max(
            dim=1
        )
    )

    return {
        "mean_confidence":
            float(
                confidence.mean().item()
            ),
        "mean_entropy":
            float(
                (
                    -probabilities
                    * probabilities.clamp_min(
                        1e-12
                    ).log()
                )
                .sum(
                    dim=1
                )
                .mean()
                .item()
            ),
        "prediction_distribution":
            torch.bincount(
                predictions,
                minlength=NUM_CLASSES
            ).tolist()
    }


def print_metrics(
    title,
    metrics
):
    print()
    print(
        title
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

    for class_id, class_name in enumerate(
        CLASSES
    ):
        print(
            f"{class_name:12s}: "
            f"{metrics['per_class'][class_id]:.2f}%"
        )


def main():
    global OUTPUT_DIR

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto"
    )

    args = parser.parse_args()

    OUTPUT_DIR = args.output_dir

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    set_seed(
        SEED
    )

    if args.device == "auto":
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = torch.device(
            args.device
        )

    print("=" * 90)
    print(
        "VISDA-2017 SOURCE-ANCHORED TARGET GRAPH DIFFUSION PROBE"
    )
    print("=" * 90)

    print(
        f"device={device}"
    )

    print(
        f"seed={SEED}"
    )

    print(
        f"checkpoint={args.checkpoint}"
    )

    print(
        f"target_k={TARGET_K}"
    )

    print(
        f"anchor_k={ANCHOR_K}"
    )

    print(
        f"diffusion_alpha={DIFFUSION_ALPHA}"
    )

    print(
        f"diffusion_iterations={DIFFUSION_ITERATIONS}"
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

    model = load_model(
        args.checkpoint
    ).to(
        device
    )

    print(
        "Checkpoint loaded successfully."
    )

    base_start = evaluate(
        torch.stack(
            [
                torch.zeros(
                    NUM_CLASSES
                )
            ]
        )
        if False
        else torch.zeros(
            len(target_labels),
            NUM_CLASSES
        ),
        target_labels
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
        f"Adapted source shape: "
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
        f"Adapted target shape: "
        f"{tuple(adapted_target.shape)}"
    )

    print()

    torch.save(
        adapted_target,
        OUTPUT_DIR
        / "adapted_target_features_seed42.pt"
    )

    print(
        "Building source sub-prototypes..."
    )

    source_prototypes, source_counts = (
        build_source_subprototypes(
            adapted_source,
            source_labels
        )
    )

    print(
        f"Source sub-prototype shape: "
        f"{tuple(source_prototypes.shape)}"
    )

    print()

    for class_id in range(
        NUM_CLASSES
    ):
        print(
            f"{CLASSES[class_id]:12s} | "
            f"source support="
            f"{int(source_counts[class_id].item())}"
        )

    print()

    print(
        "Computing source-anchor semantic distribution..."
    )

    anchor_scores, anchor_probabilities = (
        anchor_distribution(
            adapted_target,
            source_prototypes
        )
    )

    anchor_metrics = evaluate(
        anchor_probabilities,
        target_labels
    )

    anchor_confidence = evaluate_confidence(
        anchor_probabilities
    )

    print_metrics(
        "SOURCE MULTI-PROTOTYPE ANCHOR",
        anchor_metrics
    )

    print()
    print(
        f"Anchor mean confidence: "
        f"{anchor_confidence['mean_confidence']:.6f}"
    )

    print()
    print(
        "Computing exact target cosine KNN graph..."
    )

    knn_start = time.perf_counter()

    knn_indices, knn_similarity = (
        compute_exact_target_knn(
            adapted_target,
            TARGET_K,
            GRAPH_CHUNK
        )
    )

    knn_seconds = (
        time.perf_counter()
        - knn_start
    )

    print(
        f"KNN time: "
        f"{knn_seconds:.2f}s"
    )

    torch.save(
        {
            "indices":
                knn_indices,
            "similarity":
                knn_similarity
        },
        OUTPUT_DIR
        / "target_knn_seed42.pt"
    )

    print()

    print(
        "Building target transition matrix..."
    )

    transition_start = time.perf_counter()

    transition = build_target_transition(
        knn_indices,
        knn_similarity,
        len(adapted_target)
    )

    transition_seconds = (
        time.perf_counter()
        - transition_start
    )

    print(
        f"Transition construction time: "
        f"{transition_seconds:.2f}s"
    )

    print()

    print(
        "Running source-anchored label diffusion..."
    )

    diffusion_start = time.perf_counter()

    (
        diffused_probabilities,
        residuals
    ) = diffuse(
        transition,
        anchor_probabilities,
        DIFFUSION_ALPHA,
        DIFFUSION_ITERATIONS,
        DIFFUSION_TOL
    )

    diffusion_seconds = (
        time.perf_counter()
        - diffusion_start
    )

    print(
        f"Diffusion time: "
        f"{diffusion_seconds:.2f}s"
    )

    diffusion_metrics = evaluate(
        diffused_probabilities,
        target_labels
    )

    diffusion_confidence = evaluate_confidence(
        diffused_probabilities
    )

    print_metrics(
        "DIFFUSED TARGET SEMANTICS",
        diffusion_metrics
    )

    print()
    print(
        f"Diffused mean confidence: "
        f"{diffusion_confidence['mean_confidence']:.6f}"
    )

    print()
    print(
        "FOCUS CLASSES"
    )

    for class_id in (
        TRUCK_ID,
        SKATEBOARD_ID,
        PERSON_ID,
        CAR_ID,
        BUS_ID,
        TRAIN_ID
    ):
        print(
            f"{CLASSES[class_id]:12s} | "
            f"anchor="
            f"{anchor_metrics['per_class'][class_id]:.2f}% | "
            f"diffused="
            f"{diffusion_metrics['per_class'][class_id]:.2f}%"
        )

    print()
    print(
        "PREDICTION DISTRIBUTION"
    )

    for class_id in range(
        NUM_CLASSES
    ):
        count = (
            diffusion_confidence[
                "prediction_distribution"
            ][class_id]
        )

        fraction = (
            100.0
            * count
            / len(target_labels)
        )

        print(
            f"{CLASSES[class_id]:12s}: "
            f"{count:6d} "
            f"({fraction:6.2f}%)"
        )

    print()
    print(
        "TARGET GRAPH STATISTICS"
    )

    positive_edges = int(
        (
            knn_similarity
            > 0
        ).sum().item()
    )

    mean_knn_similarity = float(
        knn_similarity.mean().item()
    )

    print(
        f"nodes={len(adapted_target)}"
    )

    print(
        f"k={TARGET_K}"
    )

    print(
        f"positive directed edges="
        f"{positive_edges}"
    )

    print(
        f"mean KNN similarity="
        f"{mean_knn_similarity:.6f}"
    )

    result = {
        "experiment":
            "visda_source_anchored_target_graph_diffusion",
        "seed":
            SEED,
        "checkpoint":
            str(args.checkpoint),
        "target_k":
            TARGET_K,
        "anchor_k":
            ANCHOR_K,
        "anchors_per_class":
            ANCHORS_PER_CLASS,
        "diffusion_alpha":
            DIFFUSION_ALPHA,
        "diffusion_iterations":
            DIFFUSION_ITERATIONS,
        "diffusion_tolerance":
            DIFFUSION_TOL,
        "source_samples":
            len(source_features),
        "target_samples":
            len(target_features),
        "source_feature_dimension":
            int(source_features.shape[1]),
        "adapted_feature_dimension":
            HIDDEN_DIM,
        "source_subprototype_shape":
            list(
                source_prototypes.shape
            ),
        "source_support":
            source_counts.tolist(),
        "anchor_metrics":
            anchor_metrics,
        "anchor_confidence":
            anchor_confidence,
        "diffusion_metrics":
            diffusion_metrics,
        "diffusion_confidence":
            diffusion_confidence,
        "diffusion_residuals":
            residuals,
        "knn_seconds":
            knn_seconds,
        "transition_seconds":
            transition_seconds,
        "diffusion_seconds":
            diffusion_seconds,
        "positive_directed_edges":
            positive_edges,
        "mean_knn_similarity":
            mean_knn_similarity
    }

    output_path = (
        OUTPUT_DIR
        / "graph_semantic_diffusion_seed42.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as handle:
        json.dump(
            result,
            handle,
            indent=2
        )

    torch.save(
        {
            "anchor_probabilities":
                anchor_probabilities,
            "diffused_probabilities":
                diffused_probabilities,
            "target_labels":
                target_labels,
            "source_subprototypes":
                source_prototypes
        },
        OUTPUT_DIR
        / "graph_semantic_outputs_seed42.pt"
    )

    print()
    print("=" * 90)
    print(
        "GRAPH DIFFUSION PROBE COMPLETE"
    )
    print("=" * 90)

    print(
        f"Anchor MCA: "
        f"{anchor_metrics['mean_class']:.2f}%"
    )

    print(
        f"Diffused MCA: "
        f"{diffusion_metrics['mean_class']:.2f}%"
    )

    print(
        f"Anchor overall: "
        f"{anchor_metrics['overall']:.2f}%"
    )

    print(
        f"Diffused overall: "
        f"{diffusion_metrics['overall']:.2f}%"
    )

    print(
        f"Truck anchor: "
        f"{anchor_metrics['per_class'][TRUCK_ID]:.2f}%"
    )

    print(
        f"Truck diffused: "
        f"{diffusion_metrics['per_class'][TRUCK_ID]:.2f}%"
    )

    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()