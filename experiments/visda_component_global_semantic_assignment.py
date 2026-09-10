import argparse
import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


SEED = 42
NUM_CLASSES = 12
TRUCK = 11
CAR = 3
BUS = 2
TRAIN = 10
FOCUS_CLASSES = [TRUCK, CAR, BUS, TRAIN]


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
            "checkpoints/visda_component_global_semantic_assignment"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--truck-component-fraction",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--car-component-fraction",
        type=float,
        default=0.35,
    )

    parser.add_argument(
        "--bus-component-fraction",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--train-component-fraction",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--residual-classes",
        type=str,
        default="aeroplane,bicycle,horse,knife,motorcycle,person,plant,skateboard",
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
            found[prefix.lower()] = array

    return found


def load_component_artifacts(
    npz_path,
    json_path,
):
    npz = np.load(
        npz_path,
        allow_pickle=False,
    )

    required_npz = [
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
        for key in required_npz
        if key not in npz.files
    ]

    if missing:
        raise RuntimeError(
            "Missing required component arrays: "
            + ", ".join(missing)
        )

    candidate_indices = npz[
        "candidate_indices"
    ].astype(np.int64)

    component_labels = npz[
        "component_labels"
    ].astype(np.int64)

    semantic_consensus = npz[
        "ranker_semantic_consensus"
    ].astype(np.float64)

    graph_truck_advantage = npz[
        "ranker_graph_truck_advantage"
    ].astype(np.float64)

    prototype_truck_advantage = npz[
        "ranker_source_prototype_truck_advantage"
    ].astype(np.float64)

    pairwise_min_margin = npz[
        "ranker_pairwise_min_margin"
    ].astype(np.float64)

    pairwise_agreement = npz[
        "ranker_pairwise_agreement"
    ].astype(np.float64)

    n_components = (
        int(
            component_labels.max()
        )
        + 1
    )

    if len(semantic_consensus) != n_components:
        raise RuntimeError(
            "semantic_consensus length does not match "
            "component count"
        )

    for name, values in [
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
    ]:
        if len(values) != n_components:
            raise RuntimeError(
                f"{name} length {len(values)} "
                f"does not match component count "
                f"{n_components}"
            )

    with open(
        json_path,
        "r",
        encoding="utf-8",
    ) as handle:
        component_json = json.load(handle)

    components = component_json.get(
        "components",
        None,
    )

    if components is None:
        raise RuntimeError(
            "Component JSON does not contain "
            "'components'"
        )

    if len(components) != n_components:
        raise RuntimeError(
            "Component JSON count does not match "
            "component-label count"
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


def extract_component_class_matrix(
    components,
    graph_scores,
):
    n_components = len(
        components
    )

    matrix = np.zeros(
        (
            n_components,
            len(FOCUS_CLASSES),
        ),
        dtype=np.float64,
    )

    for cid, component in enumerate(
        components
    ):
        graph_mean = component.get(
            "graph_score_mean",
            {},
        )

        prototype_similarity = component.get(
            "source_prototype_similarity",
            {},
        )

        for j, class_id in enumerate(
            FOCUS_CLASSES
        ):
            class_name = CLASSES[
                class_id
            ]

            values = []

            if class_name in graph_mean:
                values.append(
                    float(
                        graph_mean[
                            class_name
                        ]
                    )
                )

            if class_name in prototype_similarity:
                values.append(
                    float(
                        prototype_similarity[
                            class_name
                        ]
                    )
                )

            if values:
                matrix[
                    cid,
                    j
                ] = float(
                    np.mean(values)
                )

    return matrix


def zscore_columns(matrix):
    matrix = np.asarray(
        matrix,
        dtype=np.float64,
    )

    result = np.zeros_like(
        matrix
    )

    for column in range(
        matrix.shape[1]
    ):
        values = matrix[
            :,
            column
        ]

        mean = float(
            np.mean(values)
        )

        std = float(
            np.std(values)
        )

        if std > 1e-12:
            result[
                :,
                column
            ] = (
                values - mean
            ) / std

    return result


def build_semantic_score_matrix(
    artifacts,
):
    components = artifacts[
        "components"
    ]

    class_matrix = extract_component_class_matrix(
        components,
        None,
    )

    class_matrix_z = zscore_columns(
        class_matrix
    )

    truck_adv = zscore_columns(
        np.column_stack(
            [
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
            ]
        )
    )

    combined = np.zeros(
        (
            artifacts[
                "n_components"
            ],
            len(FOCUS_CLASSES),
        ),
        dtype=np.float64,
    )

    truck_index = FOCUS_CLASSES.index(
        TRUCK
    )

    car_index = FOCUS_CLASSES.index(
        CAR
    )

    bus_index = FOCUS_CLASSES.index(
        BUS
    )

    train_index = FOCUS_CLASSES.index(
        TRAIN
    )

    combined[
        :,
        truck_index
    ] = (
        0.50
        * class_matrix_z[
            :,
            truck_index
        ]
        + 0.50
        * np.mean(
            truck_adv,
            axis=1,
        )
    )

    combined[
        :,
        car_index
    ] = (
        class_matrix_z[
            :,
            car_index
        ]
    )

    combined[
        :,
        bus_index
    ] = (
        class_matrix_z[
            :,
            bus_index
        ]
    )

    combined[
        :,
        train_index
    ] = (
        class_matrix_z[
            :,
            train_index
        ]
    )

    return (
        combined,
        class_matrix,
    )


def normalize_fraction_vector(
    values,
    total_components,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = np.maximum(
        values,
        0.0,
    )

    if np.sum(values) <= 0:
        values = np.ones_like(
            values
        )

    values = (
        values
        / np.sum(values)
    )

    raw = (
        values
        * total_components
    )

    counts = np.floor(
        raw
    ).astype(
        np.int64
    )

    remainder = (
        total_components
        - int(
            counts.sum()
        )
    )

    if remainder > 0:
        order = np.argsort(
            -(raw - counts)
        )

        for index in order[
            :remainder
        ]:
            counts[
                index
            ] += 1

    return counts


def build_capacity_vector(
    args,
    n_components,
):
    residual_names = [
        value.strip()
        for value in args.residual_classes.split(
            ","
        )
        if value.strip()
    ]

    residual_ids = []

    for name in residual_names:
        if name not in CLASSES:
            raise RuntimeError(
                f"Unknown residual class {name}"
            )

        residual_ids.append(
            CLASSES.index(name)
        )

    if len(residual_ids) != 8:
        raise RuntimeError(
            "Exactly 8 residual classes are required"
        )

    fractions = {
        TRUCK: args.truck_component_fraction,
        CAR: args.car_component_fraction,
        BUS: args.bus_component_fraction,
        TRAIN: args.train_component_fraction,
    }

    specified = sum(
        fractions.values()
    )

    if specified >= 1.0:
        raise RuntimeError(
            "Focus-class component fractions must sum to less than 1"
        )

    residual_fraction = (
        1.0
        - specified
    )

    residual_each = (
        residual_fraction
        / len(residual_ids)
    )

    raw = []

    for class_id in FOCUS_CLASSES:
        raw.append(
            fractions[class_id]
        )

    raw.append(
        residual_each
    )

    raw = np.asarray(
        raw,
        dtype=np.float64,
    )

    counts = normalize_fraction_vector(
        raw,
        n_components,
    )

    focus_counts = counts[
        :len(FOCUS_CLASSES)
    ]

    residual_count = int(
        counts[
            -1
        ]
    )

    return (
        focus_counts,
        residual_count,
        residual_ids,
    )


def global_assignment(
    score_matrix,
    capacities,
    temperature,
):
    n_components = score_matrix.shape[0]
    n_classes = score_matrix.shape[1]

    capacities = np.asarray(
        capacities,
        dtype=np.int64,
    )

    if capacities.sum() > n_components:
        raise RuntimeError(
            "Assignment capacities exceed number of components"
        )

    if capacities.sum() < n_components:
        capacities[-1] += (
            n_components
            - capacities.sum()
        )

    expanded_classes = []

    for class_id, count in enumerate(
        capacities
    ):
        expanded_classes.extend(
            [class_id] * int(count)
        )

    if len(expanded_classes) != n_components:
        raise RuntimeError(
            "Expanded assignment dimensions are inconsistent"
        )

    expanded_classes = np.asarray(
        expanded_classes,
        dtype=np.int64,
    )

    cost = np.zeros(
        (
            n_components,
            n_components,
        ),
        dtype=np.float64,
    )

    for slot in range(
        n_components
    ):
        class_id = int(
            expanded_classes[
                slot
            ]
        )

        cost[
            :,
            slot
        ] = -score_matrix[
            :,
            class_id
        ]

    cost_scale = max(
        float(
            np.std(cost)
        ),
        1e-6,
    )

    cost = (
        cost
        / (
            temperature
            * cost_scale
        )
    )

    rows = np.arange(
        n_components
    )

    remaining_rows = list(
        rows
    )

    remaining_slots = list(
        range(
            n_components
        )
    )

    assignment = np.full(
        n_components,
        -1,
        dtype=np.int64,
    )

    while remaining_rows:
        row = remaining_rows.pop(0)

        scores = cost[
            row,
            remaining_slots,
        ]

        best_position = int(
            np.argmin(
                scores
            )
        )

        slot = remaining_slots.pop(
            best_position
        )

        assignment[
            row
        ] = expanded_classes[
            slot
        ]

    return assignment


def better_global_assignment(
    score_matrix,
    capacities,
):
    n = score_matrix.shape[0]

    expanded_classes = []

    for class_id, count in enumerate(
        capacities
    ):
        expanded_classes.extend(
            [class_id] * int(count)
        )

    expanded_classes = np.asarray(
        expanded_classes,
        dtype=np.int64,
    )

    matrix = score_matrix[
        :,
        expanded_classes,
    ]

    rows_used = np.zeros(
        n,
        dtype=bool,
    )

    columns_used = np.zeros(
        n,
        dtype=bool,
    )

    assignment = np.full(
        n,
        -1,
        dtype=np.int64,
    )

    row_order = np.argsort(
        -np.max(
            matrix,
            axis=1,
        ),
        kind="mergesort",
    )

    for row in row_order:
        available = np.flatnonzero(
            ~columns_used
        )

        values = matrix[
            row,
            available,
        ]

        best = available[
            int(
                np.argmax(values)
            )
        ]

        columns_used[
            best
        ] = True

        assignment[
            row
        ] = expanded_classes[
            best
        ]

    return assignment


def convert_component_assignment_to_samples(
    graph_top1,
    candidate_indices,
    component_labels,
    component_assignment,
):
    predictions = graph_top1.copy()

    for component_id in range(
        len(component_assignment)
    ):
        assigned_class = int(
            component_assignment[
                component_id
            ]
        )

        if assigned_class != TRUCK:
            continue

        local_mask = (
            component_labels
            == component_id
        )

        global_indices = candidate_indices[
            local_mask
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

    values = []
    per_class = {}

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


def load_diffusion_scores(
    path,
    frozen_candidate_indices,
):
    payload = safe_torch_load(
        path,
        map_location="cpu",
    )

    arrays = collect_arrays(
        payload
    )

    candidates = []

    frozen_set = np.sort(
        frozen_candidate_indices
    )

    for name, value in arrays.items():
        value = np.asarray(
            value
        )

        if (
            value.ndim != 2
            or value.shape[1]
            != NUM_CLASSES
            or value.shape[0]
            <= int(
                frozen_candidate_indices.max()
            )
        ):
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

        topk = np.argpartition(
            value,
            -3,
            axis=1,
        )[
            :,
            -3:,
        ]

        detected = np.flatnonzero(
            np.any(
                topk == TRUCK,
                axis=1,
            )
        )

        if np.array_equal(
            np.sort(detected),
            frozen_set,
        ):
            candidates.append(
                (
                    name,
                    value.astype(
                        np.float32
                    ),
                )
            )

    if len(candidates) == 0:
        raise RuntimeError(
            "No diffusion graph tensor exactly reproduces "
            "the frozen truck Top-3 candidate region."
        )

    candidates.sort(
        key=lambda item: (
            "diffusion" in item[0],
            "semantic" in item[0],
            "prob" in item[0],
        ),
        reverse=True,
    )

    return candidates[0]


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
        "VISDA-2017 GLOBAL COMPONENT-LEVEL SEMANTIC ASSIGNMENT"
    )
    print("=" * 96)
    print(
        "device=cpu"
    )
    print(
        f"seed={args.seed}"
    )
    print(
        f"temperature={args.temperature}"
    )
    print(
        f"truck_component_fraction={args.truck_component_fraction}"
    )
    print(
        f"car_component_fraction={args.car_component_fraction}"
    )
    print(
        f"bus_component_fraction={args.bus_component_fraction}"
    )
    print(
        f"train_component_fraction={args.train_component_fraction}"
    )

    print(
        "\nLoading frozen component artifacts..."
    )

    artifacts = load_component_artifacts(
        args.component_npz,
        args.component_json,
    )

    candidate_indices = artifacts[
        "candidate_indices"
    ]

    component_labels = artifacts[
        "component_labels"
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

    if len(candidate_indices) != 22553:
        raise RuntimeError(
            "Expected exactly 22,553 frozen candidate samples"
        )

    if n_components != 113:
        raise RuntimeError(
            "Expected exactly 113 frozen components"
        )

    print(
        "\nLoading and validating diffusion scores..."
    )

    graph_key, graph_scores = (
        load_diffusion_scores(
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
        "\nBuilding component semantic score matrix..."
    )

    (
        score_matrix,
        raw_class_matrix,
    ) = build_semantic_score_matrix(
        artifacts
    )

    print(
        f"score_matrix_shape={score_matrix.shape}"
    )

    if score_matrix.shape != (
        113,
        4,
    ):
        raise RuntimeError(
            "Expected a 113 x 4 focus-class score matrix"
        )

    print(
        "\nFocus classes:"
    )

    for index, class_id in enumerate(
        FOCUS_CLASSES
    ):
        print(
            f"  column={index} class={CLASSES[class_id]}"
        )

    print(
        "\nConstructing predefined component capacities..."
    )

    (
        focus_capacity,
        residual_capacity,
        residual_ids,
    ) = build_capacity_vector(
        args,
        n_components,
    )

    residual_names = [
        CLASSES[
            class_id
        ]
        for class_id in residual_ids
    ]

    print(
        "truck capacity="
        f"{int(focus_capacity[0])}"
    )

    print(
        "car capacity="
        f"{int(focus_capacity[1])}"
    )

    print(
        "bus capacity="
        f"{int(focus_capacity[2])}"
    )

    print(
        "train capacity="
        f"{int(focus_capacity[3])}"
    )

    print(
        "residual capacity="
        f"{residual_capacity}"
    )

    capacities = np.asarray(
        [
            int(
                focus_capacity[0]
            ),
            int(
                focus_capacity[1]
            ),
            int(
                focus_capacity[2]
            ),
            int(
                focus_capacity[3]
            ),
            int(
                residual_capacity
            ),
        ],
        dtype=np.int64,
    )

    extended_scores = np.zeros(
        (
            n_components,
            5,
        ),
        dtype=np.float64,
    )

    extended_scores[
        :,
        :4
    ] = score_matrix

    extended_scores[
        :,
        4
    ] = 0.0

    print(
        "\nSolving global constrained assignment..."
    )

    expanded_assignment = better_global_assignment(
        extended_scores,
        capacities,
    )

    component_assignment = expanded_assignment.copy()

    truck_component_ids = np.flatnonzero(
        component_assignment
        == 0
    )

    print(
        f"assigned_truck_components={len(truck_component_ids)}"
    )

    print(
        "assigned_truck_component_ids="
        + ",".join(
            str(
                int(value)
            )
            for value in truck_component_ids
        )
    )

    sample_scores = np.zeros(
        len(candidate_indices),
        dtype=np.float64,
    )

    for cid in range(
        n_components
    ):
        sample_scores[
            component_labels
            == cid
        ] = score_matrix[
            cid,
            0,
        ]

    assignment_predictions = (
        convert_component_assignment_to_samples(
            graph_top1,
            candidate_indices,
            component_labels,
            component_assignment,
        )
    )

    print(
        "\nAll assignment construction is now frozen."
    )

    print(
        "Loading target labels for evaluation only..."
    )

    target_labels = load_target_labels(
        args.target_cache
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

    assignment_metrics = classification_metrics(
        assignment_predictions,
        target_labels,
    )

    candidate_labels = target_labels[
        candidate_indices
    ]

    true_candidate_trucks = int(
        np.sum(
            candidate_labels
            == TRUCK
        )
    )

    total_target_trucks = int(
        np.sum(
            target_labels
            == TRUCK
        )
    )

    assigned_candidate_indices = np.flatnonzero(
        np.isin(
            component_labels,
            truck_component_ids,
        )
    )

    assigned_global_indices = (
        candidate_indices[
            assigned_candidate_indices
        ]
    )

    assigned_true_trucks = int(
        np.sum(
            target_labels[
                assigned_global_indices
            ]
            == TRUCK
        )
    )

    assigned_precision = (
        assigned_true_trucks
        / max(
            len(
                assigned_global_indices
            ),
            1,
        )
    )

    assigned_candidate_recall = (
        assigned_true_trucks
        / max(
            true_candidate_trucks,
            1,
        )
    )

    assigned_global_recall = (
        assigned_true_trucks
        / max(
            total_target_trucks,
            1,
        )
    )

    print(
        "\nBASELINE GRAPH TOP-1"
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
        "\nGLOBAL COMPONENT ASSIGNMENT"
    )

    print(
        f"assigned_truck_components={len(truck_component_ids)}"
    )

    print(
        f"selected_candidate_samples={len(assigned_global_indices)}"
    )

    print(
        f"truck_precision={assigned_precision * 100:.2f}%"
    )

    print(
        f"truck_candidate_recall={assigned_candidate_recall * 100:.2f}%"
    )

    print(
        f"truck_global_recall={assigned_global_recall * 100:.2f}%"
    )

    print(
        f"OA={assignment_metrics['overall']:.2f}%"
    )

    print(
        f"MCA={assignment_metrics['mean_class']:.2f}%"
    )

    print(
        f"truck={assignment_metrics['per_class']['truck']:.2f}%"
    )

    print(
        f"car={assignment_metrics['per_class']['car']:.2f}%"
    )

    print(
        f"bus={assignment_metrics['per_class']['bus']:.2f}%"
    )

    print(
        f"train={assignment_metrics['per_class']['train']:.2f}%"
    )

    print(
        "\nTRUCK COMPONENT DETAILS"
    )

    truck_components = []

    for cid in truck_component_ids:
        local = (
            component_labels
            == cid
        )

        size = int(
            np.sum(local)
        )

        semantic = float(
            artifacts[
                "semantic_consensus"
            ][cid]
        )

        truck_score = float(
            score_matrix[
                cid,
                0,
            ]
        )

        true_count = int(
            np.sum(
                candidate_labels[
                    local
                ]
                == TRUCK
            )
        )

        true_purity = (
            true_count
            / max(
                size,
                1,
            )
        )

        truck_components.append(
            {
                "component_id": int(
                    cid
                ),
                "size": size,
                "semantic_consensus": semantic,
                "truck_score": truck_score,
                "true_truck_count_evaluation_only": true_count,
                "true_truck_purity_evaluation_only": true_purity,
            }
        )

        print(
            f"component={cid:4d} "
            f"size={size:5d} "
            f"semantic={semantic:+.5f} "
            f"truck_score={truck_score:+.5f} "
            f"purity_eval_only={true_purity * 100:6.2f}%"
        )

    output_npz = (
        args.output_dir
        / f"global_semantic_assignment_seed{args.seed}.npz"
    )

    np.savez_compressed(
        output_npz,
        candidate_indices=candidate_indices,
        component_labels=component_labels,
        focus_score_matrix=score_matrix,
        raw_class_matrix=raw_class_matrix,
        component_assignment=component_assignment,
        selected_truck_components=truck_component_ids,
        sample_truck_scores=sample_scores,
    )

    output_json = (
        args.output_dir
        / f"global_semantic_assignment_seed{args.seed}.json"
    )

    summary = {
        "experiment": (
            "visda_global_component_semantic_assignment"
        ),
        "seed": args.seed,
        "device": "cpu",
        "constraints": {
            "model_training": False,
            "target_labels_used_in_construction": False,
            "target_labels_used_in_assignment": False,
            "target_labels_loaded_only_for_evaluation": True,
            "candidate_region_recomputed": False,
            "component_assignments_recomputed": False,
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
            "target_cache": str(
                args.target_cache
            ),
            "selected_graph_tensor": graph_key,
        },
        "frozen_structure": {
            "target_count": int(
                graph_scores.shape[0]
            ),
            "candidate_count": int(
                len(candidate_indices)
            ),
            "component_count": int(
                n_components
            ),
        },
        "assignment": {
            "focus_classes": [
                CLASSES[
                    value
                ]
                for value in FOCUS_CLASSES
            ],
            "capacities": capacities.tolist(),
            "truck_component_ids": (
                truck_component_ids.astype(
                    np.int64
                ).tolist()
            ),
            "temperature": float(
                args.temperature
            ),
            "score_matrix": score_matrix.tolist(),
        },
        "evaluation_only": {
            "candidate_true_trucks": true_candidate_trucks,
            "total_target_trucks": total_target_trucks,
            "selected_candidate_samples": int(
                len(
                    assigned_global_indices
                )
            ),
            "truck_precision": float(
                assigned_precision
            ),
            "truck_candidate_recall": float(
                assigned_candidate_recall
            ),
            "truck_global_recall": float(
                assigned_global_recall
            ),
            "baseline_metrics": baseline_metrics,
            "assignment_metrics": assignment_metrics,
            "truck_components": truck_components,
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
        "\nDONE"
    )


if __name__ == "__main__":
    main()