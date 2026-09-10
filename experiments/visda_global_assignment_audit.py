import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


SEED = 42
NUM_CLASSES = 12

TRUCK = 11
CAR = 3
BUS = 2
TRAIN = 10

FOCUS_CLASSES = [
    TRUCK,
    CAR,
    BUS,
    TRAIN,
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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--component-npz",
        type=Path,
        default=Path(
            "checkpoints/visda_truck_component_structure_probe/"
            "truck_component_assignments_seed42.npz"
        ),
    )

    parser.add_argument(
        "--component-json",
        type=Path,
        default=Path(
            "checkpoints/visda_truck_component_structure_probe/"
            "truck_component_structure_probe_seed42.json"
        ),
    )

    parser.add_argument(
        "--graph-outputs",
        type=Path,
        default=Path(
            "checkpoints/visda_graph_semantic_diffusion/"
            "graph_semantic_outputs_seed42.pt"
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
            "checkpoints/visda_global_assignment_audit"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    parser.add_argument(
        "--truck-components",
        type=int,
        default=13,
    )

    parser.add_argument(
        "--car-components",
        type=int,
        default=46,
    )

    parser.add_argument(
        "--bus-components",
        type=int,
        default=26,
    )

    parser.add_argument(
        "--train-components",
        type=int,
        default=26,
    )

    parser.add_argument(
        "--residual-components",
        type=int,
        default=2,
    )

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=map_location,
        )


def to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()

    if isinstance(value, np.ndarray):
        return value

    return None


def collect_arrays(obj, prefix=""):
    found = {}

    if isinstance(obj, dict):
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

    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
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
        array = to_numpy(obj)

        if array is not None:
            found[
                prefix.lower()
            ] = array

    return found


def load_components(
    npz_path,
    json_path,
):
    npz = np.load(
        npz_path,
        allow_pickle=False,
    )

    required = [
        "candidate_indices",
        "component_labels",
        "ranker_semantic_consensus",
        "ranker_graph_truck_advantage",
        "ranker_source_prototype_truck_advantage",
        "ranker_pairwise_min_margin",
        "ranker_pairwise_agreement",
    ]

    missing = [
        key
        for key in required
        if key not in npz.files
    ]

    if missing:
        raise RuntimeError(
            "Missing component arrays: "
            + ", ".join(missing)
        )

    candidate_indices = npz[
        "candidate_indices"
    ].astype(
        np.int64
    )

    component_labels = npz[
        "component_labels"
    ].astype(
        np.int64
    )

    semantic_consensus = npz[
        "ranker_semantic_consensus"
    ].astype(
        np.float64
    )

    graph_truck_advantage = npz[
        "ranker_graph_truck_advantage"
    ].astype(
        np.float64
    )

    prototype_truck_advantage = npz[
        "ranker_source_prototype_truck_advantage"
    ].astype(
        np.float64
    )

    pairwise_min_margin = npz[
        "ranker_pairwise_min_margin"
    ].astype(
        np.float64
    )

    pairwise_agreement = npz[
        "ranker_pairwise_agreement"
    ].astype(
        np.float64
    )

    n_components = (
        int(
            component_labels.max()
        )
        + 1
    )

    if len(candidate_indices) != len(
        component_labels
    ):
        raise RuntimeError(
            "candidate_indices and component_labels "
            "have different lengths"
        )

    if n_components != 113:
        raise RuntimeError(
            f"Expected 113 components, found {n_components}"
        )

    if len(candidate_indices) != 22553:
        raise RuntimeError(
            f"Expected 22553 candidate samples, "
            f"found {len(candidate_indices)}"
        )

    component_arrays = [
        (
            "semantic_consensus",
            semantic_consensus,
        ),
        (
            "graph_truck_advantage",
            graph_truck_advantage,
        ),
        (
            "prototype_truck_advantage",
            prototype_truck_advantage,
        ),
        (
            "pairwise_min_margin",
            pairwise_min_margin,
        ),
        (
            "pairwise_agreement",
            pairwise_agreement,
        ),
    ]

    for name, values in component_arrays:
        if len(values) != n_components:
            raise RuntimeError(
                f"{name} has {len(values)} values "
                f"but there are {n_components} components"
            )

    with open(
        json_path,
        "r",
        encoding="utf-8",
    ) as handle:
        metadata = json.load(
            handle
        )

    components = metadata.get(
        "components"
    )

    if components is None:
        raise RuntimeError(
            "Component JSON does not contain components"
        )

    if len(components) != n_components:
        raise RuntimeError(
            "Component JSON component count does not match NPZ"
        )

    return {
        "candidate_indices": candidate_indices,
        "component_labels": component_labels,
        "semantic_consensus": semantic_consensus,
        "graph_truck_advantage": graph_truck_advantage,
        "prototype_truck_advantage": prototype_truck_advantage,
        "pairwise_min_margin": pairwise_min_margin,
        "pairwise_agreement": pairwise_agreement,
        "components": components,
        "n_components": n_components,
    }


def find_diffusion_tensor(
    path,
    candidate_indices,
):
    payload = safe_torch_load(
        path,
        map_location="cpu",
    )

    arrays = collect_arrays(
        payload
    )

    candidate_set = np.sort(
        candidate_indices
    )

    matches = []

    for name, value in arrays.items():
        value = np.asarray(
            value
        )

        if value.ndim != 2:
            continue

        if value.shape[1] != NUM_CLASSES:
            continue

        if value.shape[0] <= int(
            candidate_indices.max()
        ):
            continue

        normalized = (
            name
            .lower()
            .replace(
                ".",
                "_",
            )
            .replace(
                "-",
                "_",
            )
        )

        if "anchor" in normalized:
            continue

        if (
            "diffusion"
            not in normalized
            and "diffused"
            not in normalized
            and "graph"
            not in normalized
        ):
            continue

        top3 = np.argpartition(
            value,
            -3,
            axis=1,
        )[
            :,
            -3:,
        ]

        detected = np.flatnonzero(
            np.any(
                top3 == TRUCK,
                axis=1,
            )
        ).astype(
            np.int64
        )

        if np.array_equal(
            np.sort(detected),
            candidate_set,
        ):
            matches.append(
                (
                    name,
                    value.astype(
                        np.float32
                    ),
                )
            )

    if not matches:
        raise RuntimeError(
            "No diffusion tensor exactly reproduces "
            "the frozen truck Top-3 candidate region"
        )

    matches.sort(
        key=lambda item: (
            "diffusion" in item[0],
            "semantic" in item[0],
            "prob" in item[0],
        ),
        reverse=True,
    )

    return matches[0]


def zscore(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    mean = float(
        np.mean(values)
    )

    std = float(
        np.std(values)
    )

    if std < 1e-12:
        return np.zeros_like(
            values
        )

    return (
        values - mean
    ) / std


def build_score_matrix(
    components,
    semantic_consensus,
    graph_advantage,
    prototype_advantage,
    pairwise_min_margin,
    pairwise_agreement,
):
    n_components = len(
        components
    )

    raw_graph = np.zeros(
        (
            n_components,
            4,
        ),
        dtype=np.float64,
    )

    raw_prototype = np.zeros_like(
        raw_graph
    )

    for cid, component in enumerate(
        components
    ):
        graph_scores = component.get(
            "graph_score_mean",
            {},
        )

        prototype_scores = component.get(
            "source_prototype_similarity",
            {},
        )

        for column, class_id in enumerate(
            FOCUS_CLASSES
        ):
            class_name = CLASSES[
                class_id
            ]

            raw_graph[
                cid,
                column,
            ] = float(
                graph_scores.get(
                    class_name,
                    0.0,
                )
            )

            raw_prototype[
                cid,
                column,
            ] = float(
                prototype_scores.get(
                    class_name,
                    0.0,
                )
            )

    graph_z = np.zeros_like(
        raw_graph
    )

    prototype_z = np.zeros_like(
        raw_prototype
    )

    for column in range(
        4
    ):
        graph_z[
            :,
            column
        ] = zscore(
            raw_graph[
                :,
                column
            ]
        )

        prototype_z[
            :,
            column
        ] = zscore(
            raw_prototype[
                :,
                column
            ]
        )

    pairwise_features = np.column_stack(
        [
            zscore(
                graph_advantage
            ),
            zscore(
                prototype_advantage
            ),
            zscore(
                pairwise_min_margin
            ),
            zscore(
                pairwise_agreement
            ),
        ]
    )

    score_matrix = np.zeros(
        (
            n_components,
            4,
        ),
        dtype=np.float64,
    )

    truck_column = 0
    car_column = 1
    bus_column = 2
    train_column = 3

    score_matrix[
        :,
        truck_column
    ] = (
        0.50
        * zscore(
            raw_graph[
                :,
                truck_column
            ]
        )
        + 0.25
        * zscore(
            raw_prototype[
                :,
                truck_column
            ]
        )
        + 0.25
        * np.mean(
            pairwise_features,
            axis=1,
        )
    )

    score_matrix[
        :,
        car_column
    ] = (
        0.60
        * graph_z[
            :,
            car_column
        ]
        + 0.40
        * prototype_z[
            :,
            car_column
        ]
    )

    score_matrix[
        :,
        bus_column
    ] = (
        0.60
        * graph_z[
            :,
            bus_column
        ]
        + 0.40
        * prototype_z[
            :,
            bus_column
        ]
    )

    score_matrix[
        :,
        train_column
    ] = (
        0.60
        * graph_z[
            :,
            train_column
        ]
        + 0.40
        * prototype_z[
            :,
            train_column
        ]
    )

    semantic_z = zscore(
        semantic_consensus
    )

    score_matrix[
        :,
        truck_column
    ] += (
        0.25
        * semantic_z
    )

    return (
        score_matrix,
        raw_graph,
        raw_prototype,
    )


def make_capacities(args):
    capacities = np.asarray(
        [
            args.truck_components,
            args.car_components,
            args.bus_components,
            args.train_components,
            args.residual_components,
        ],
        dtype=np.int64,
    )

    if int(
        capacities.sum()
    ) != 113:
        raise RuntimeError(
            "Component capacities must sum to 113"
        )

    if np.any(
        capacities < 0
    ):
        raise RuntimeError(
            "Component capacities cannot be negative"
        )

    return capacities


def greedy_global_assignment(
    score_matrix,
    capacities,
):
    n_components = score_matrix.shape[
        0
    ]

    extended = np.column_stack(
        [
            score_matrix,
            np.zeros(
                n_components,
                dtype=np.float64,
            ),
        ]
    )

    slots = []

    for class_id, count in enumerate(
        capacities
    ):
        slots.extend(
            [class_id] * int(
                count
            )
        )

    slots = np.asarray(
        slots,
        dtype=np.int64,
    )

    if len(slots) != n_components:
        raise RuntimeError(
            "Greedy slot count mismatch"
        )

    remaining = list(
        range(
            n_components
        )
    )

    assignment = np.full(
        n_components,
        -1,
        dtype=np.int64,
    )

    for _ in range(
        n_components
    ):
        best_component = None
        best_class = None
        best_value = -np.inf

        for component_id in remaining:
            available_classes = np.unique(
                slots[
                    :
                ]
            )

            for class_id in available_classes:
                candidate_slots = np.flatnonzero(
                    slots
                    == class_id
                )

                if len(candidate_slots) == 0:
                    continue

                value = extended[
                    component_id,
                    class_id,
                ]

                if value > best_value:
                    best_value = value
                    best_component = component_id
                    best_class = int(
                        class_id
                    )

        assignment[
            best_component
        ] = best_class

        remaining.remove(
            best_component
        )

        remove_position = np.flatnonzero(
            slots
            == best_class
        )[0]

        slots = np.delete(
            slots,
            remove_position,
        )

    return assignment


def exact_global_assignment(
    score_matrix,
    capacities,
):
    n_components = score_matrix.shape[
        0
    ]

    extended = np.column_stack(
        [
            score_matrix,
            np.zeros(
                n_components,
                dtype=np.float64,
            ),
        ]
    )

    slot_classes = []

    for class_id, count in enumerate(
        capacities
    ):
        slot_classes.extend(
            [class_id] * int(
                count
            )
        )

    slot_classes = np.asarray(
        slot_classes,
        dtype=np.int64,
    )

    if len(slot_classes) != n_components:
        raise RuntimeError(
            "Exact assignment slot count mismatch"
        )

    assignment_scores = (
        extended[
            :,
            slot_classes,
        ]
    )

    row_ind, col_ind = (
        linear_sum_assignment(
            -assignment_scores
        )
    )

    if len(row_ind) != n_components:
        raise RuntimeError(
            "Exact assignment did not assign every component"
        )

    assignment = np.full(
        n_components,
        -1,
        dtype=np.int64,
    )

    assignment[
        row_ind
    ] = slot_classes[
        col_ind
    ]

    counts = np.bincount(
        assignment,
        minlength=5,
    )

    if not np.array_equal(
        counts,
        capacities,
    ):
        raise RuntimeError(
            "Exact assignment violated class capacities"
        )

    return assignment


def top_truck_assignment(
    score_matrix,
    count,
):
    scores = score_matrix[
        :,
        0,
    ]

    order = np.argsort(
        -scores,
        kind="mergesort",
    )

    assignment = np.full(
        len(scores),
        -1,
        dtype=np.int64,
    )

    selected = order[
        :count
    ]

    assignment[
        selected
    ] = 0

    return assignment


def semantic_consensus_assignment(
    semantic_consensus,
    count,
):
    order = np.argsort(
        -semantic_consensus,
        kind="mergesort",
    )

    assignment = np.full(
        len(
            semantic_consensus
        ),
        -1,
        dtype=np.int64,
    )

    selected = order[
        :count
    ]

    assignment[
        selected
    ] = 0

    return assignment


def independent_argmax_assignment(
    score_matrix,
):
    assignment = np.argmax(
        score_matrix,
        axis=1,
    ).astype(
        np.int64
    )

    return assignment


def assignment_objective(
    assignment,
    score_matrix,
):
    values = np.zeros(
        len(assignment),
        dtype=np.float64,
    )

    for component_id, class_id in enumerate(
        assignment
    ):
        if class_id < 0:
            continue

        if class_id >= score_matrix.shape[1]:
            continue

        values[
            component_id
        ] = score_matrix[
            component_id,
            class_id,
        ]

    return float(
        np.sum(values)
    )


def convert_assignment_to_predictions(
    graph_top1,
    candidate_indices,
    component_labels,
    assignment,
):
    predictions = graph_top1.copy()

    truck_components = np.flatnonzero(
        assignment
        == 0
    )

    for component_id in truck_components:
        local = (
            component_labels
            == component_id
        )

        global_indices = candidate_indices[
            local
        ]

        predictions[
            global_indices
        ] = TRUCK

    return predictions


def classification_metrics(
    predictions,
    labels,
):
    overall = float(
        np.mean(
            predictions
            == labels
        )
        * 100.0
    )

    per_class = {}
    values = []

    for class_id, name in enumerate(
        CLASSES
    ):
        mask = (
            labels
            == class_id
        )

        if np.any(mask):
            accuracy = float(
                np.mean(
                    predictions[
                        mask
                    ]
                    == class_id
                )
                * 100.0
            )
        else:
            accuracy = 0.0

        per_class[
            name
        ] = accuracy

        values.append(
            accuracy
        )

    return {
        "overall": overall,
        "mean_class": float(
            np.mean(values)
        ),
        "per_class": per_class,
    }


def evaluate_assignment(
    name,
    assignment,
    score_matrix,
    candidate_indices,
    component_labels,
    graph_top1,
    target_labels,
):
    predictions = (
        convert_assignment_to_predictions(
            graph_top1,
            candidate_indices,
            component_labels,
            assignment,
        )
    )

    metrics = classification_metrics(
        predictions,
        target_labels,
    )

    truck_components = np.flatnonzero(
        assignment
        == 0
    )

    selected_mask = np.isin(
        component_labels,
        truck_components,
    )

    selected_global = candidate_indices[
        selected_mask
    ]

    selected_labels = target_labels[
        selected_global
    ]

    true_trucks = int(
        np.sum(
            selected_labels
            == TRUCK
        )
    )

    total_trucks = int(
        np.sum(
            target_labels
            == TRUCK
        )
    )

    precision = (
        true_trucks
        / max(
            len(selected_global),
            1,
        )
    )

    global_recall = (
        true_trucks
        / max(
            total_trucks,
            1,
        )
    )

    candidate_trucks = int(
        np.sum(
            target_labels[
                candidate_indices
            ]
            == TRUCK
        )
    )

    candidate_recall = (
        true_trucks
        / max(
            candidate_trucks,
            1,
        )
    )

    baseline_correct = (
        graph_top1[
            selected_global
        ]
        == target_labels[
            selected_global
        ]
    )

    overwritten_correct = int(
        np.sum(
            baseline_correct
            & (
                target_labels[
                    selected_global
                ]
                != TRUCK
            )
        )
    )

    newly_correct_truck = int(
        np.sum(
            (
                target_labels[
                    selected_global
                ]
                == TRUCK
            )
            & (
                graph_top1[
                    selected_global
                ]
                != TRUCK
            )
        )
    )

    return {
        "name": name,
        "assignment": assignment,
        "truck_component_count": int(
            len(
                truck_components
            )
        ),
        "truck_component_ids": (
            truck_components.astype(
                np.int64
            ).tolist()
        ),
        "selected_candidate_samples": int(
            len(selected_global)
        ),
        "truck_true_positives": true_trucks,
        "truck_precision": float(
            precision
        ),
        "truck_candidate_recall": float(
            candidate_recall
        ),
        "truck_global_recall": float(
            global_recall
        ),
        "newly_correct_truck_predictions": newly_correct_truck,
        "overwritten_baseline_correct_predictions": overwritten_correct,
        "objective": assignment_objective(
            assignment,
            score_matrix,
        ),
        "metrics": metrics,
    }


def print_result(result):
    metrics = result[
        "metrics"
    ]

    print(
        f"\n{result['name']}"
    )

    print(
        f"truck_components={result['truck_component_count']}"
    )

    print(
        f"selected_candidate_samples={result['selected_candidate_samples']}"
    )

    print(
        f"truck_precision={result['truck_precision'] * 100:.2f}%"
    )

    print(
        f"truck_candidate_recall={result['truck_candidate_recall'] * 100:.2f}%"
    )

    print(
        f"truck_global_recall={result['truck_global_recall'] * 100:.2f}%"
    )

    print(
        f"newly_correct_truck={result['newly_correct_truck_predictions']}"
    )

    print(
        f"overwritten_baseline_correct={result['overwritten_baseline_correct_predictions']}"
    )

    print(
        f"objective={result['objective']:.6f}"
    )

    print(
        f"OA={metrics['overall']:.2f}%"
    )

    print(
        f"MCA={metrics['mean_class']:.2f}%"
    )

    print(
        f"truck={metrics['per_class']['truck']:.2f}%"
    )

    print(
        f"car={metrics['per_class']['car']:.2f}%"
    )

    print(
        f"bus={metrics['per_class']['bus']:.2f}%"
    )

    print(
        f"train={metrics['per_class']['train']:.2f}%"
    )


def main():
    args = parse_args()

    set_seed(
        args.seed
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 96)
    print(
        "VISDA-2017 CONTROLLED GLOBAL ASSIGNMENT AUDIT"
    )
    print("=" * 96)
    print(
        "device=cpu"
    )
    print(
        f"seed={args.seed}"
    )

    print(
        "\nLoading frozen component artifacts..."
    )

    artifacts = load_components(
        args.component_npz,
        args.component_json,
    )

    candidate_indices = artifacts[
        "candidate_indices"
    ]

    component_labels = artifacts[
        "component_labels"
    ]

    semantic_consensus = artifacts[
        "semantic_consensus"
    ]

    n_components = artifacts[
        "n_components"
    ]

    print(
        f"candidate_samples={len(candidate_indices)}"
    )

    print(
        f"components={n_components}"
    )

    print(
        "\nLoading and validating diffusion outputs..."
    )

    graph_key, graph_scores = (
        find_diffusion_tensor(
            args.graph_outputs,
            candidate_indices,
        )
    )

    print(
        f"selected_graph_tensor={graph_key}"
    )

    print(
        f"graph_shape={graph_scores.shape}"
    )

    graph_top1 = graph_scores.argmax(
        axis=1
    ).astype(
        np.int64
    )

    print(
        "\nBuilding the frozen 113 x 4 semantic evidence matrix..."
    )

    (
        score_matrix,
        raw_graph,
        raw_prototype,
    ) = build_score_matrix(
        artifacts[
            "components"
        ],
        semantic_consensus,
        artifacts[
            "graph_truck_advantage"
        ],
        artifacts[
            "prototype_truck_advantage"
        ],
        artifacts[
            "pairwise_min_margin"
        ],
        artifacts[
            "pairwise_agreement"
        ],
    )

    print(
        f"score_matrix_shape={score_matrix.shape}"
    )

    print(
        "\nScore statistics..."
    )

    for column, class_id in enumerate(
        FOCUS_CLASSES
    ):
        values = score_matrix[
            :,
            column
        ]

        print(
            f"{CLASSES[class_id]:10s} "
            f"mean={np.mean(values):+.5f} "
            f"std={np.std(values):.5f} "
            f"min={np.min(values):+.5f} "
            f"max={np.max(values):+.5f}"
        )

    capacities = make_capacities(
        args
    )

    print(
        "\nFixed diagnostic capacities..."
    )

    print(
        f"truck={capacities[0]}"
    )

    print(
        f"car={capacities[1]}"
    )

    print(
        f"bus={capacities[2]}"
    )

    print(
        f"train={capacities[3]}"
    )

    print(
        f"residual={capacities[4]}"
    )

    print(
        "\nConstructing assignment variants..."
    )

    assignments = {}

    assignments[
        "semantic_consensus_top13"
    ] = semantic_consensus_assignment(
        semantic_consensus,
        args.truck_components,
    )

    assignments[
        "truck_score_top13"
    ] = top_truck_assignment(
        score_matrix,
        args.truck_components,
    )

    assignments[
        "independent_argmax"
    ] = independent_argmax_assignment(
        score_matrix
    )

    assignments[
        "greedy_global_assignment"
    ] = greedy_global_assignment(
        score_matrix,
        capacities,
    )

    assignments[
        "exact_global_assignment"
    ] = exact_global_assignment(
        score_matrix,
        capacities,
    )

    print(
        "\nAssignment construction complete."
    )

    print(
        "\nVerifying exact global assignment..."
    )

    exact_assignment = assignments[
        "exact_global_assignment"
    ]

    exact_counts = np.bincount(
        exact_assignment,
        minlength=5,
    )

    print(
        f"assignment_counts={exact_counts.tolist()}"
    )

    if not np.array_equal(
        exact_counts,
        capacities,
    ):
        raise RuntimeError(
            "Exact assignment capacity verification failed"
        )

    print(
        "exact_assignment_capacity_check=True"
    )

    print(
        "\nAll assignments are now frozen."
    )

    print(
        "Loading target labels for evaluation only..."
    )

    target_labels = np.asarray(
        [
            int(value)
            for value in load_target_labels(
                args.target_cache
            )
        ],
        dtype=np.int64,
    )

    if len(target_labels) != len(
        graph_scores
    ):
        raise RuntimeError(
            "Target label count does not match graph output count"
        )

    baseline_metrics = classification_metrics(
        graph_top1,
        target_labels,
    )

    candidate_labels = target_labels[
        candidate_indices
    ]

    total_target_trucks = int(
        np.sum(
            target_labels
            == TRUCK
        )
    )

    candidate_true_trucks = int(
        np.sum(
            candidate_labels
            == TRUCK
        )
    )

    candidate_precision = (
        candidate_true_trucks
        / len(
            candidate_indices
        )
    )

    candidate_recall = (
        candidate_true_trucks
        / max(
            total_target_trucks,
            1,
        )
    )

    print(
        "\nBASELINE"
    )

    print(
        f"OA={baseline_metrics['overall']:.2f}%"
    )

    print(
        f"MCA={baseline_metrics['mean_class']:.2f}%"
    )

    print(
        f"truck={baseline_metrics['per_class']['truck']:.2f}%"
    )

    print(
        "\nFROZEN CANDIDATE REGION"
    )

    print(
        f"precision={candidate_precision * 100:.2f}%"
    )

    print(
        f"recall={candidate_recall * 100:.2f}%"
    )

    results = []

    for name, assignment in assignments.items():
        result = evaluate_assignment(
            name,
            assignment,
            score_matrix,
            candidate_indices,
            component_labels,
            graph_top1,
            target_labels,
        )

        results.append(
            result
        )

    for result in results:
        print_result(
            result
        )

    result_map = {
        result[
            "name"
        ]: result
        for result in results
    }

    exact_result = result_map[
        "exact_global_assignment"
    ]

    greedy_result = result_map[
        "greedy_global_assignment"
    ]

    top_truck_result = result_map[
        "truck_score_top13"
    ]

    semantic_result = result_map[
        "semantic_consensus_top13"
    ]

    exact_vs_truck_precision = (
        exact_result[
            "truck_precision"
        ]
        - top_truck_result[
            "truck_precision"
        ]
    )

    exact_vs_truck_recall = (
        exact_result[
            "truck_global_recall"
        ]
        - top_truck_result[
            "truck_global_recall"
        ]
    )

    exact_vs_truck_mca = (
        exact_result[
            "metrics"
        ][
            "mean_class"
        ]
        - top_truck_result[
            "metrics"
        ][
            "mean_class"
        ]
    )

    exact_vs_greedy_mca = (
        exact_result[
            "metrics"
        ][
            "mean_class"
        ]
        - greedy_result[
            "metrics"
        ][
            "mean_class"
        ]
    )

    print(
        "\nCONTROLLED COMPARISONS"
    )

    print(
        f"exact_vs_truck_top13_precision_delta="
        f"{exact_vs_truck_precision * 100:+.2f} pp"
    )

    print(
        f"exact_vs_truck_top13_global_recall_delta="
        f"{exact_vs_truck_recall * 100:+.2f} pp"
    )

    print(
        f"exact_vs_truck_top13_MCA_delta="
        f"{exact_vs_truck_mca:+.2f} pp"
    )

    print(
        f"exact_vs_greedy_MCA_delta="
        f"{exact_vs_greedy_mca:+.2f} pp"
    )

    if (
        exact_vs_truck_mca > 0
        and exact_vs_truck_recall > 0
    ):
        global_contribution = (
            "positive_evidence_for_global_competition"
        )
    elif (
        exact_vs_truck_precision > 0
        or exact_vs_truck_recall > 0
    ):
        global_contribution = (
            "mixed_evidence_global_assignment_not_yet_superior"
        )
    else:
        global_contribution = (
            "no_evidence_global_assignment_beats_matched_truck_ranking"
        )

    if (
        exact_vs_greedy_mca > 0
    ):
        optimizer_effect = (
            "exact_optimization_improves_over_greedy"
        )
    elif (
        exact_vs_greedy_mca < 0
    ):
        optimizer_effect = (
            "exact_optimization_underperforms_greedy"
        )
    else:
        optimizer_effect = (
            "exact_and_greedy_have_equal_MCA"
        )

    print(
        f"global_contribution={global_contribution}"
    )

    print(
        f"optimizer_effect={optimizer_effect}"
    )

    print(
        "\nSCIENTIFIC DECISION"
    )

    if (
        global_contribution
        == "positive_evidence_for_global_competition"
    ):
        decision = (
            "continue_global_assignment_and_test_"
            "source_calibrated_full_12_class_soft_assignment"
        )

        print(
            "Global assignment shows evidence beyond matched "
            "truck-only ranking. Continue to score calibration "
            "and full 12-class soft assignment."
        )

    elif (
        global_contribution
        == "mixed_evidence_global_assignment_not_yet_superior"
    ):
        decision = (
            "audit_score_calibration_before_expanding_assignment"
        )

        print(
            "Global assignment shows partial signal but has not "
            "established superiority over the matched ranking. "
            "Audit cross-class score calibration next."
        )

    else:
        decision = (
            "reject_global_competition_as_primary_mechanism"
        )

        print(
            "Exact global assignment does not beat the matched "
            "truck-ranking control. Do not promote global "
            "competition as the primary mechanism."
        )

    print(
        "\nSaving outputs..."
    )

    output_npz = (
        args.output_dir
        / f"global_assignment_audit_seed{args.seed}.npz"
    )

    np.savez_compressed(
        output_npz,
        candidate_indices=candidate_indices,
        component_labels=component_labels,
        semantic_consensus=semantic_consensus,
        score_matrix=score_matrix,
        semantic_consensus_assignment=assignments[
            "semantic_consensus_top13"
        ],
        truck_score_assignment=assignments[
            "truck_score_top13"
        ],
        independent_argmax_assignment=assignments[
            "independent_argmax"
        ],
        greedy_global_assignment=assignments[
            "greedy_global_assignment"
        ],
        exact_global_assignment=assignments[
            "exact_global_assignment"
        ],
    )

    summary_results = {}

    for result in results:
        summary_results[
            result[
                "name"
            ]
        ] = {
            key: value
            for key, value in result.items()
            if key != "assignment"
        }

    summary = {
        "experiment": (
            "visda_controlled_global_assignment_audit"
        ),
        "seed": args.seed,
        "device": "cpu",
        "constraints": {
            "model_training": False,
            "candidate_region_recomputed": False,
            "component_assignments_recomputed": False,
            "target_labels_used_in_construction": False,
            "target_labels_used_in_assignment": False,
            "target_labels_loaded_only_for_evaluation": True,
        },
        "frozen_structure": {
            "candidate_count": int(
                len(candidate_indices)
            ),
            "component_count": int(
                n_components
            ),
            "candidate_rule": (
                "frozen_truck_top3_region"
            ),
        },
        "inputs": {
            "component_npz": str(
                args.component_npz
            ),
            "component_json": str(
                args.component_json
            ),
            "graph_outputs": str(
                args.graph_outputs
            ),
            "selected_graph_tensor": graph_key,
            "target_cache": str(
                args.target_cache
            ),
        },
        "capacities": capacities.tolist(),
        "baseline": baseline_metrics,
        "candidate_region": {
            "true_trucks": candidate_true_trucks,
            "total_target_trucks": total_target_trucks,
            "precision": float(
                candidate_precision
            ),
            "recall": float(
                candidate_recall
            ),
        },
        "results": summary_results,
        "controlled_comparisons": {
            "exact_vs_truck_top13_precision_delta": float(
                exact_vs_truck_precision
            ),
            "exact_vs_truck_top13_global_recall_delta": float(
                exact_vs_truck_recall
            ),
            "exact_vs_truck_top13_MCA_delta": float(
                exact_vs_truck_mca
            ),
            "exact_vs_greedy_MCA_delta": float(
                exact_vs_greedy_mca
            ),
        },
        "scientific_decision": {
            "global_contribution": (
                global_contribution
            ),
            "optimizer_effect": (
                optimizer_effect
            ),
            "decision": decision,
        },
        "outputs": {
            "npz": str(
                output_npz
            ),
        },
    }

    output_json = (
        args.output_dir
        / f"global_assignment_audit_seed{args.seed}.json"
    )

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
        "\nDONE"
    )


def load_target_labels(
    cache_dir,
):
    files = sorted(
        cache_dir.glob(
            "chunk_*.pt"
        )
    )

    if not files:
        raise RuntimeError(
            f"No target cache chunks found in {cache_dir}"
        )

    labels = []

    for path in files:
        payload = safe_torch_load(
            path,
            map_location="cpu",
        )

        if "labels" not in payload:
            raise RuntimeError(
                f"Missing labels in {path}"
            )

        labels.append(
            payload[
                "labels"
            ].long()
        )

    return torch.cat(
        labels,
        dim=0,
    ).numpy().astype(
        np.int64
    )


if __name__ == "__main__":
    main()