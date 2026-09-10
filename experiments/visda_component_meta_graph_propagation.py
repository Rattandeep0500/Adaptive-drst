import json
import math
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
TOP_K = 3
EXPECTED_TARGET_COUNT = 55388
EXPECTED_CANDIDATE_COUNT = 22553
EXPECTED_COMPONENT_COUNT = 113
SEED_FRACTION = 0.01
META_K = 15
PROPAGATION_ALPHA = 0.85
PROPAGATION_ITERATIONS = 100
ENCODE_BATCH_SIZE = 4096
CANDIDATE_MASSES = (0.10, 0.20, 0.30, 0.40, 0.50)
EPS = 1e-12

COMPONENT_ASSIGNMENTS = Path(
    "checkpoints/visda_truck_component_structure_probe/truck_component_assignments_seed42.npz"
)

GRAPH_OUTPUTS = Path(
    "checkpoints/visda_graph_semantic_diffusion/graph_semantic_outputs_seed42.pt"
)

RPC_CHECKPOINT = Path(
    "checkpoints/visda_rpc_causal_experiment/rpc_seed42.pt"
)

TARGET_CACHE = Path(
    "checkpoints/visda_feature_cache/target"
)

OUTPUT_DIR = Path(
    "checkpoints/visda_component_meta_graph_propagation"
)

OUTPUT_NPZ = (
    OUTPUT_DIR
    / "component_meta_graph_scores_seed42.npz"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "component_meta_graph_propagation_seed42.json"
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
                HIDDEN_DIM,
            ),
            nn.BatchNorm1d(
                HIDDEN_DIM,
            ),
            nn.ReLU(
                inplace=True,
            ),
        )

    def forward(self, x):
        if x.ndim != 2:
            raise RuntimeError(
                f"Adapter expected a 2-D tensor, got {tuple(x.shape)}"
            )

        if x.shape[1] != INPUT_DIM:
            raise RuntimeError(
                f"Adapter expected N x {INPUT_DIM}, got {tuple(x.shape)}"
            )

        z = self.net(x)

        if z.ndim != 2:
            raise RuntimeError(
                f"Adapter produced a non-2-D tensor: {tuple(z.shape)}"
            )

        if z.shape[1] != HIDDEN_DIM:
            raise RuntimeError(
                f"Adapter produced invalid feature dimension {z.shape[1]}"
            )

        return z


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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def safe_torch_load(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing required artifact: {path}"
        )

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


def to_numpy(value):
    if torch.is_tensor(value):
        return (
            value
            .detach()
            .cpu()
            .numpy()
        )

    if isinstance(
        value,
        np.ndarray,
    ):
        return value

    return None


def collect_arrays(
    obj,
    prefix="",
):
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


def load_frozen_components(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing frozen component artifact: {path}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as payload:
        available_keys = list(
            payload.files
        )

        required_keys = [
            "candidate_indices",
            "component_labels",
            "ranker_semantic_consensus",
        ]

        missing_keys = [
            key
            for key in required_keys
            if key not in payload
        ]

        if missing_keys:
            raise RuntimeError(
                f"Frozen component artifact is missing keys "
                f"{missing_keys}. Available keys: {available_keys}"
            )

        candidate_indices = np.asarray(
            payload[
                "candidate_indices"
            ],
            dtype=np.int64,
        ).reshape(-1)

        component_labels = np.asarray(
            payload[
                "component_labels"
            ],
            dtype=np.int64,
        ).reshape(-1)

        semantic_consensus = np.asarray(
            payload[
                "ranker_semantic_consensus"
            ],
            dtype=np.float64,
        ).reshape(-1)

    if (
        candidate_indices.size
        != EXPECTED_CANDIDATE_COUNT
    ):
        raise RuntimeError(
            f"candidate_indices length must be "
            f"{EXPECTED_CANDIDATE_COUNT}, "
            f"got {candidate_indices.size}"
        )

    if (
        component_labels.size
        != EXPECTED_CANDIDATE_COUNT
    ):
        raise RuntimeError(
            f"component_labels length must be "
            f"{EXPECTED_CANDIDATE_COUNT}, "
            f"got {component_labels.size}"
        )

    if (
        candidate_indices.size
        != component_labels.size
    ):
        raise RuntimeError(
            "candidate_indices and component_labels "
            "have different sample-level lengths"
        )

    if (
        np.unique(
            candidate_indices
        ).size
        != candidate_indices.size
    ):
        raise RuntimeError(
            "candidate_indices contains duplicate target sample IDs"
        )

    if candidate_indices.size == 0:
        raise RuntimeError(
            "Frozen candidate region is empty"
        )

    minimum_candidate = int(
        candidate_indices.min()
    )

    maximum_candidate = int(
        candidate_indices.max()
    )

    if minimum_candidate < 0:
        raise RuntimeError(
            f"candidate_indices contains negative index "
            f"{minimum_candidate}"
        )

    if (
        maximum_candidate
        >= EXPECTED_TARGET_COUNT
    ):
        raise RuntimeError(
            f"candidate_indices must lie in "
            f"[0, {EXPECTED_TARGET_COUNT}), "
            f"observed max={maximum_candidate}"
        )

    if component_labels.size == 0:
        raise RuntimeError(
            "component_labels is empty"
        )

    minimum_component = int(
        component_labels.min()
    )

    if minimum_component < 0:
        raise RuntimeError(
            f"component_labels contains negative "
            f"component ID {minimum_component}"
        )

    component_count = (
        int(
            component_labels.max()
        )
        + 1
    )

    if (
        component_count
        != EXPECTED_COMPONENT_COUNT
    ):
        raise RuntimeError(
            f"max(component_labels) + 1 must be "
            f"{EXPECTED_COMPONENT_COUNT}, "
            f"got {component_count}"
        )

    unique_components = np.unique(
        component_labels
    )

    expected_components = np.arange(
        EXPECTED_COMPONENT_COUNT,
        dtype=np.int64,
    )

    if not np.array_equal(
        unique_components,
        expected_components,
    ):
        missing_components = np.setdiff1d(
            expected_components,
            unique_components,
        ).tolist()

        extra_components = np.setdiff1d(
            unique_components,
            expected_components,
        ).tolist()

        raise RuntimeError(
            f"Frozen component IDs are not exactly "
            f"0..{EXPECTED_COMPONENT_COUNT - 1}. "
            f"Missing={missing_components} "
            f"extra={extra_components}"
        )

    if (
        semantic_consensus.size
        != EXPECTED_COMPONENT_COUNT
    ):
        raise RuntimeError(
            f"ranker_semantic_consensus length must be "
            f"{EXPECTED_COMPONENT_COUNT}, "
            f"got {semantic_consensus.size}"
        )

    if not np.all(
        np.isfinite(
            semantic_consensus
        )
    ):
        bad = np.flatnonzero(
            ~np.isfinite(
                semantic_consensus
            )
        ).tolist()

        raise RuntimeError(
            f"ranker_semantic_consensus contains "
            f"non-finite values at components {bad}"
        )

    candidate_indices.setflags(
        write=False
    )

    component_labels.setflags(
        write=False
    )

    semantic_consensus.setflags(
        write=False
    )

    return (
        candidate_indices,
        component_labels,
        semantic_consensus,
        available_keys,
    )


def truck_topk_candidate_indices(
    scores,
):
    array = np.asarray(
        scores
    )

    if array.ndim != 2:
        raise RuntimeError(
            f"Graph score tensor must be 2-D, "
            f"got shape {array.shape}"
        )

    if (
        array.shape[1]
        != NUM_CLASSES
    ):
        raise RuntimeError(
            f"Graph score tensor must have "
            f"{NUM_CLASSES} classes, "
            f"got shape {array.shape}"
        )

    tensor = torch.as_tensor(
        array
    )

    topk_indices = torch.topk(
        tensor,
        k=TOP_K,
        dim=1,
        largest=True,
        sorted=True,
    ).indices

    candidate_mask = (
        topk_indices
        == TRUCK_ID
    ).any(
        dim=1
    )

    candidate_indices = torch.nonzero(
        candidate_mask,
        as_tuple=False,
    ).flatten()

    return (
        candidate_indices
        .cpu()
        .numpy()
        .astype(
            np.int64
        )
    )


def graph_tensor_diagnostic(
    name,
    array,
    frozen_candidate_indices,
):
    shape = tuple(
        int(value)
        for value in array.shape
    )

    result = {
        "key": name,
        "shape": list(shape),
        "candidate_count": None,
        "overlap_count": None,
        "jaccard": None,
        "exact_frozen_candidate_match": False,
    }

    if array.ndim != 2:
        return result

    if (
        array.shape[1]
        != NUM_CLASSES
    ):
        return result

    candidate_indices = (
        truck_topk_candidate_indices(
            array
        )
    )

    result[
        "candidate_count"
    ] = int(
        candidate_indices.size
    )

    if (
        array.shape[0]
        != EXPECTED_TARGET_COUNT
    ):
        return result

    intersection_count = int(
        np.intersect1d(
            candidate_indices,
            frozen_candidate_indices,
            assume_unique=True,
        ).size
    )

    union_count = int(
        np.union1d(
            candidate_indices,
            frozen_candidate_indices,
        ).size
    )

    if union_count == 0:
        jaccard = 1.0
    else:
        jaccard = (
            intersection_count
            / union_count
        )

    exact = (
        candidate_indices.size
        == frozen_candidate_indices.size
        and np.array_equal(
            np.sort(
                candidate_indices
            ),
            np.sort(
                frozen_candidate_indices
            ),
        )
    )

    result[
        "overlap_count"
    ] = intersection_count

    result[
        "jaccard"
    ] = float(
        jaccard
    )

    result[
        "exact_frozen_candidate_match"
    ] = bool(
        exact
    )

    return result


def format_graph_diagnostics(
    diagnostics,
):
    rows = []

    for item in diagnostics:
        if (
            item[
                "overlap_count"
            ]
            is None
        ):
            overlap_text = "n/a"
        else:
            overlap_text = str(
                item[
                    "overlap_count"
                ]
            )

        if (
            item[
                "jaccard"
            ]
            is None
        ):
            jaccard_text = "n/a"
        else:
            jaccard_text = (
                f"{item['jaccard']:.8f}"
            )

        rows.append(
            f"key={item['key']} "
            f"shape={tuple(item['shape'])} "
            f"truck_top{TOP_K}_count="
            f"{item['candidate_count']} "
            f"overlap={overlap_text} "
            f"jaccard={jaccard_text} "
            f"exact="
            f"{item['exact_frozen_candidate_match']}"
        )

    return "\n".join(
        rows
    )


def identify_diffusion_graph_tensor(
    path,
    frozen_candidate_indices,
):
    graph_payload = safe_torch_load(
        path
    )

    arrays = collect_arrays(
        graph_payload
    )

    candidate_arrays = {}

    for name, array in arrays.items():
        array = np.asarray(
            array
        )

        if array.ndim != 2:
            continue

        if (
            array.shape[1]
            != NUM_CLASSES
        ):
            continue

        candidate_arrays[
            name
        ] = array

    del arrays
    del graph_payload

    if not candidate_arrays:
        raise RuntimeError(
            f"No 2-D N x {NUM_CLASSES} tensors "
            f"were found in graph output {path}"
        )

    diagnostics = []

    for name in sorted(
        candidate_arrays
    ):
        diagnostics.append(
            graph_tensor_diagnostic(
                name,
                candidate_arrays[
                    name
                ],
                frozen_candidate_indices,
            )
        )

    exact_names = [
        item[
            "key"
        ]
        for item in diagnostics
        if item[
            "exact_frozen_candidate_match"
        ]
    ]

    if not exact_names:
        raise RuntimeError(
            "No graph tensor exactly reproduces "
            "the frozen truck Top-3 candidate region.\n"
            f"Frozen candidate count="
            f"{frozen_candidate_indices.size}\n"
            f"Expected target count="
            f"{EXPECTED_TARGET_COUNT}\n"
            "Candidate graph tensors:\n"
            + format_graph_diagnostics(
                diagnostics
            )
        )

    normalized_names = {
        name:
            name
            .lower()
            .replace(
                "-",
                "_",
            )
            .replace(
                " ",
                "_",
            )
        for name in exact_names
    }

    diffusion_names = []

    for name in exact_names:
        normalized = (
            normalized_names[
                name
            ]
        )

        diffusion_like = any(
            term in normalized
            for term in (
                "diff",
                "diffus",
                "propagat",
                "graph_semantic",
            )
        )

        anchor_like = (
            "anchor"
            in normalized
        )

        if (
            diffusion_like
            and not anchor_like
        ):
            diffusion_names.append(
                name
            )

    if (
        len(
            diffusion_names
        )
        == 1
    ):
        selected_name = (
            diffusion_names[
                0
            ]
        )

    elif (
        len(
            diffusion_names
        )
        > 1
    ):
        reference = (
            candidate_arrays[
                diffusion_names[
                    0
                ]
            ]
        )

        identical = all(
            np.array_equal(
                reference,
                candidate_arrays[
                    name
                ],
            )
            for name
            in diffusion_names[
                1:
            ]
        )

        if identical:
            selected_name = sorted(
                diffusion_names,
                key=lambda name: (
                    len(name),
                    name,
                ),
            )[0]

        else:
            raise RuntimeError(
                "Multiple distinct diffusion-like graph "
                "tensors exactly reproduce the frozen "
                "candidate region, so the graph output "
                "is ambiguous.\n"
                + format_graph_diagnostics(
                    diagnostics
                )
            )

    else:
        non_anchor_names = [
            name
            for name in exact_names
            if (
                "anchor"
                not in normalized_names[
                    name
                ]
            )
        ]

        if (
            len(
                non_anchor_names
            )
            == 1
        ):
            selected_name = (
                non_anchor_names[
                    0
                ]
            )

        else:
            raise RuntimeError(
                "The frozen candidate region is reproduced, "
                "but the matching tensor cannot be identified "
                "unambiguously as a diffusion or graph-semantic "
                "output. Refusing to select an anchor tensor.\n"
                + format_graph_diagnostics(
                    diagnostics
                )
            )

    if (
        "anchor"
        in normalized_names[
            selected_name
        ]
    ):
        raise RuntimeError(
            f"Refusing to use anchor-like graph tensor "
            f"'{selected_name}'"
        )

    selected = np.asarray(
        candidate_arrays[
            selected_name
        ],
        dtype=np.float32,
    )

    expected_shape = (
        EXPECTED_TARGET_COUNT,
        NUM_CLASSES,
    )

    if (
        selected.shape
        != expected_shape
    ):
        raise RuntimeError(
            f"Selected graph tensor '{selected_name}' "
            f"must have shape {expected_shape}, "
            f"got {selected.shape}"
        )

    if not np.all(
        np.isfinite(
            selected
        )
    ):
        raise RuntimeError(
            f"Selected graph tensor '{selected_name}' "
            f"contains non-finite values"
        )

    reproduced_indices = (
        truck_topk_candidate_indices(
            selected
        )
    )

    if (
        reproduced_indices.size
        != EXPECTED_CANDIDATE_COUNT
    ):
        raise RuntimeError(
            f"Selected graph tensor '{selected_name}' "
            f"produces {reproduced_indices.size} "
            f"truck Top-{TOP_K} candidates, expected "
            f"{EXPECTED_CANDIDATE_COUNT}"
        )

    if not np.array_equal(
        np.sort(
            reproduced_indices
        ),
        np.sort(
            frozen_candidate_indices
        ),
    ):
        raise RuntimeError(
            f"Selected graph tensor '{selected_name}' "
            f"does not exactly reproduce the frozen "
            f"candidate sample IDs"
        )

    selected.setflags(
        write=False
    )

    return (
        selected_name,
        selected,
        diagnostics,
    )


def load_rpc_model(
    path,
    device,
):
    checkpoint = safe_torch_load(
        path
    )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise RuntimeError(
            f"RPC checkpoint {path} "
            f"is not a dictionary"
        )

    if (
        "student_state_dict"
        in checkpoint
    ):
        state_dict = checkpoint[
            "student_state_dict"
        ]

        state_key = (
            "student_state_dict"
        )

    elif (
        "state_dict"
        in checkpoint
    ):
        state_dict = checkpoint[
            "state_dict"
        ]

        state_key = (
            "state_dict"
        )

    else:
        tensor_keys = [
            key
            for key, value
            in checkpoint.items()
            if torch.is_tensor(
                value
            )
        ]

        root_state_compatible = (
            bool(
                tensor_keys
            )
            and any(
                str(key).startswith(
                    "adapter."
                )
                for key
                in tensor_keys
            )
        )

        if root_state_compatible:
            state_dict = checkpoint
            state_key = "<root>"

        else:
            checkpoint_keys = sorted(
                str(key)
                for key
                in checkpoint.keys()
            )

            raise RuntimeError(
                f"Could not locate the RPC student "
                f"state dict in {path}. "
                f"Checkpoint keys: "
                f"{checkpoint_keys}"
            )

    if not isinstance(
        state_dict,
        dict,
    ):
        raise RuntimeError(
            f"RPC state under '{state_key}' "
            f"is not a state-dict mapping"
        )

    model = MCDModel().to(
        device
    )

    expected_keys = set(
        model
        .state_dict()
        .keys()
    )

    actual_keys = set(
        state_dict.keys()
    )

    missing_keys = sorted(
        expected_keys
        - actual_keys
    )

    unexpected_keys = sorted(
        actual_keys
        - expected_keys
    )

    if (
        missing_keys
        or unexpected_keys
    ):
        raise RuntimeError(
            f"RPC state dict is incompatible "
            f"with the verified MCDModel. "
            f"Missing keys={missing_keys} "
            f"unexpected keys={unexpected_keys}"
        )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(
            False
        )

    return (
        model,
        state_key,
    )


def cache_chunk_files(
    cache_dir,
):
    files = sorted(
        Path(
            cache_dir
        ).glob(
            "chunk_*.pt"
        )
    )

    if not files:
        raise RuntimeError(
            f"No cache chunks found "
            f"in {cache_dir}"
        )

    return files


def load_candidate_raw_features(
    cache_dir,
    candidate_indices,
):
    files = cache_chunk_files(
        cache_dir
    )

    output = torch.empty(
        (
            candidate_indices.size,
            INPUT_DIM,
        ),
        dtype=torch.float16,
        device="cpu",
    )

    fill_count = np.zeros(
        candidate_indices.size,
        dtype=np.int8,
    )

    expected_start = 0
    total_count = 0

    for path in files:
        payload = safe_torch_load(
            path
        )

        required_keys = [
            "features",
            "start_index",
            "end_index",
        ]

        missing_keys = [
            key
            for key
            in required_keys
            if key not in payload
        ]

        if missing_keys:
            raise RuntimeError(
                f"Target cache chunk {path} "
                f"is missing keys {missing_keys}"
            )

        features = payload[
            "features"
        ]

        if not torch.is_tensor(
            features
        ):
            raise RuntimeError(
                f"features in {path} "
                f"is not a tensor"
            )

        if features.ndim != 2:
            raise RuntimeError(
                f"features in {path} "
                f"must be 2-D, "
                f"got {tuple(features.shape)}"
            )

        if (
            features.shape[1]
            != INPUT_DIM
        ):
            raise RuntimeError(
                f"Expected N x {INPUT_DIM} "
                f"features in {path}, "
                f"got {tuple(features.shape)}"
            )

        start_index = int(
            payload[
                "start_index"
            ]
        )

        end_index = int(
            payload[
                "end_index"
            ]
        )

        if (
            start_index
            != expected_start
        ):
            raise RuntimeError(
                f"Target cache has a gap, "
                f"overlap, or reordering at "
                f"{path}: expected "
                f"start_index={expected_start}, "
                f"got {start_index}"
            )

        if (
            end_index
            <= start_index
        ):
            raise RuntimeError(
                f"Invalid target cache range "
                f"in {path}: "
                f"start={start_index} "
                f"end={end_index}"
            )

        expected_chunk_count = (
            end_index
            - start_index
        )

        if (
            features.shape[0]
            != expected_chunk_count
        ):
            raise RuntimeError(
                f"Feature count/range mismatch "
                f"in {path}: "
                f"features={features.shape[0]} "
                f"range={expected_chunk_count}"
            )

        local_candidate_positions = (
            np.flatnonzero(
                (
                    candidate_indices
                    >= start_index
                )
                & (
                    candidate_indices
                    < end_index
                )
            )
        )

        if (
            local_candidate_positions.size
            > 0
        ):
            global_ids = (
                candidate_indices[
                    local_candidate_positions
                ]
            )

            local_ids_numpy = (
                global_ids
                - start_index
            ).astype(
                np.int64,
                copy=False,
            )

            local_ids = torch.from_numpy(
                local_ids_numpy
            )

            selected_features = (
                features
                .index_select(
                    0,
                    local_ids,
                )
                .to(
                    dtype=torch.float16,
                    device="cpu",
                )
            )

            destination_positions = (
                torch.from_numpy(
                    local_candidate_positions.astype(
                        np.int64,
                        copy=False,
                    )
                )
            )

            output.index_copy_(
                0,
                destination_positions,
                selected_features,
            )

            fill_count[
                local_candidate_positions
            ] += 1

        total_count += (
            expected_chunk_count
        )

        expected_start = (
            end_index
        )

        del payload
        del features

    if (
        total_count
        != EXPECTED_TARGET_COUNT
    ):
        raise RuntimeError(
            f"Target feature cache count "
            f"must be "
            f"{EXPECTED_TARGET_COUNT}, "
            f"got {total_count}"
        )

    if (
        expected_start
        != EXPECTED_TARGET_COUNT
    ):
        raise RuntimeError(
            f"Target feature cache final "
            f"end_index must be "
            f"{EXPECTED_TARGET_COUNT}, "
            f"got {expected_start}"
        )

    missing_positions = (
        np.flatnonzero(
            fill_count
            == 0
        )
    )

    duplicate_positions = (
        np.flatnonzero(
            fill_count
            > 1
        )
    )

    if (
        missing_positions.size
        or duplicate_positions.size
    ):
        raise RuntimeError(
            "Candidate feature extraction "
            "did not map every frozen "
            "candidate ID exactly once. "
            f"Missing candidate positions="
            f"{missing_positions[:20].tolist()} "
            f"duplicate candidate positions="
            f"{duplicate_positions[:20].tolist()}"
        )

    expected_shape = (
        EXPECTED_CANDIDATE_COUNT,
        INPUT_DIM,
    )

    if (
        tuple(
            output.shape
        )
        != expected_shape
    ):
        raise RuntimeError(
            f"Candidate raw feature tensor "
            f"must have shape "
            f"{expected_shape}, "
            f"got {tuple(output.shape)}"
        )

    return (
        output,
        total_count,
        len(files),
    )


@torch.no_grad()
def encode_candidate_features(
    model,
    raw_candidate_features,
    device,
):
    expected_shape = (
        EXPECTED_CANDIDATE_COUNT,
        INPUT_DIM,
    )

    if (
        tuple(
            raw_candidate_features.shape
        )
        != expected_shape
    ):
        raise RuntimeError(
            f"Candidate raw features "
            f"must have shape "
            f"{expected_shape}, "
            f"got "
            f"{tuple(raw_candidate_features.shape)}"
        )

    output_parts = []

    model.eval()

    for start_index in range(
        0,
        raw_candidate_features.shape[
            0
        ],
        ENCODE_BATCH_SIZE,
    ):
        end_index = min(
            start_index
            + ENCODE_BATCH_SIZE,
            raw_candidate_features.shape[
                0
            ],
        )

        batch = (
            raw_candidate_features[
                start_index:
                end_index
            ]
            .to(
                device=device,
                dtype=torch.float32,
            )
        )

        adapted = model.encode(
            batch
        )

        if adapted.ndim != 2:
            raise RuntimeError(
                f"RPC adapter produced "
                f"a non-2-D tensor: "
                f"{tuple(adapted.shape)}"
            )

        if (
            adapted.shape[1]
            != HIDDEN_DIM
        ):
            raise RuntimeError(
                f"RPC adapter produced "
                f"invalid dimension "
                f"{adapted.shape[1]}"
            )

        output_parts.append(
            adapted.cpu()
        )

    adapted_features = torch.cat(
        output_parts,
        dim=0,
    )

    expected_adapted_shape = (
        EXPECTED_CANDIDATE_COUNT,
        HIDDEN_DIM,
    )

    if (
        tuple(
            adapted_features.shape
        )
        != expected_adapted_shape
    ):
        raise RuntimeError(
            f"Adapted candidate features "
            f"must have shape "
            f"{expected_adapted_shape}, "
            f"got "
            f"{tuple(adapted_features.shape)}"
        )

    if not torch.isfinite(
        adapted_features
    ).all():
        raise RuntimeError(
            "Adapted candidate features "
            "contain non-finite values"
        )

    return adapted_features


def compute_component_centroids(
    adapted_candidate_features,
    component_labels,
):
    expected_shape = (
        EXPECTED_CANDIDATE_COUNT,
        HIDDEN_DIM,
    )

    if (
        tuple(
            adapted_candidate_features.shape
        )
        != expected_shape
    ):
        raise RuntimeError(
            f"Expected adapted candidate "
            f"features {expected_shape}, "
            f"got "
            f"{tuple(adapted_candidate_features.shape)}"
        )

    component_sizes = np.bincount(
        component_labels,
        minlength=EXPECTED_COMPONENT_COUNT,
    ).astype(
        np.int64
    )

    if (
        component_sizes.shape[
            0
        ]
        != EXPECTED_COMPONENT_COUNT
    ):
        raise RuntimeError(
            f"Component size vector "
            f"must have length "
            f"{EXPECTED_COMPONENT_COUNT}, "
            f"got "
            f"{component_sizes.shape[0]}"
        )

    if (
        int(
            component_sizes.sum()
        )
        != EXPECTED_CANDIDATE_COUNT
    ):
        raise RuntimeError(
            f"Component sizes sum to "
            f"{int(component_sizes.sum())}, "
            f"expected "
            f"{EXPECTED_CANDIDATE_COUNT}"
        )

    empty_components = np.flatnonzero(
        component_sizes
        == 0
    )

    if (
        empty_components.size
        > 0
    ):
        raise RuntimeError(
            f"Cannot construct centroids "
            f"because components are empty: "
            f"{empty_components.tolist()}"
        )

    label_tensor = torch.from_numpy(
        component_labels.copy()
    ).long()

    centroid_sums = torch.zeros(
        (
            EXPECTED_COMPONENT_COUNT,
            HIDDEN_DIM,
        ),
        dtype=torch.float32,
        device="cpu",
    )

    centroid_sums.index_add_(
        0,
        label_tensor,
        adapted_candidate_features.float(),
    )

    size_tensor = (
        torch
        .from_numpy(
            component_sizes
        )
        .float()
        .unsqueeze(
            1
        )
    )

    centroids = (
        centroid_sums
        / size_tensor
    )

    centroid_norms = (
        torch.linalg.vector_norm(
            centroids,
            dim=1,
        )
    )

    invalid_centroids = (
        torch.nonzero(
            (
                ~torch.isfinite(
                    centroid_norms
                )
            )
            | (
                centroid_norms
                <= EPS
            ),
            as_tuple=False,
        )
        .flatten()
    )

    if (
        invalid_centroids.numel()
        > 0
    ):
        raise RuntimeError(
            f"Invalid or zero component "
            f"centroids at IDs "
            f"{invalid_centroids.tolist()}"
        )

    centroids = F.normalize(
        centroids,
        p=2,
        dim=1,
    )

    expected_centroid_shape = (
        EXPECTED_COMPONENT_COUNT,
        HIDDEN_DIM,
    )

    if (
        tuple(
            centroids.shape
        )
        != expected_centroid_shape
    ):
        raise RuntimeError(
            f"Component centroid shape "
            f"must be "
            f"{expected_centroid_shape}, "
            f"got "
            f"{tuple(centroids.shape)}"
        )

    if not torch.isfinite(
        centroids
    ).all():
        raise RuntimeError(
            "Normalized component "
            "centroids contain "
            "non-finite values"
        )

    return (
        centroids
        .cpu()
        .numpy()
        .astype(
            np.float32
        ),
        component_sizes,
    )


def similarity_statistics(
    similarity,
):
    node_count = (
        similarity.shape[
            0
        ]
    )

    off_diagonal_mask = (
        ~np.eye(
            node_count,
            dtype=bool,
        )
    )

    values = similarity[
        off_diagonal_mask
    ]

    if values.size == 0:
        raise RuntimeError(
            "No off-diagonal component "
            "similarities were produced"
        )

    if not np.all(
        np.isfinite(
            values
        )
    ):
        raise RuntimeError(
            "Component similarity matrix "
            "contains invalid "
            "off-diagonal values"
        )

    return {
        "off_diagonal_min":
            float(
                np.min(
                    values
                )
            ),
        "off_diagonal_max":
            float(
                np.max(
                    values
                )
            ),
        "off_diagonal_mean":
            float(
                np.mean(
                    values
                )
            ),
        "off_diagonal_std":
            float(
                np.std(
                    values
                )
            ),
        "off_diagonal_median":
            float(
                np.median(
                    values
                )
            ),
        "off_diagonal_nonnegative_fraction":
            float(
                np.mean(
                    values
                    >= 0.0
                )
            ),
        "off_diagonal_positive_fraction":
            float(
                np.mean(
                    values
                    > 0.0
                )
            ),
    }


def build_component_meta_graph(
    component_centroids,
):
    expected_shape = (
        EXPECTED_COMPONENT_COUNT,
        HIDDEN_DIM,
    )

    if (
        component_centroids.shape
        != expected_shape
    ):
        raise RuntimeError(
            f"Component centroids must "
            f"have shape {expected_shape}, "
            f"got "
            f"{component_centroids.shape}"
        )

    if META_K <= 0:
        raise RuntimeError(
            f"META_K must be positive, "
            f"got {META_K}"
        )

    if (
        META_K
        >= EXPECTED_COMPONENT_COUNT
    ):
        raise RuntimeError(
            f"META_K must be less than "
            f"{EXPECTED_COMPONENT_COUNT}, "
            f"got {META_K}"
        )

    similarity = (
        component_centroids
        @ component_centroids.T
    )

    similarity = np.asarray(
        similarity,
        dtype=np.float32,
    )

    expected_similarity_shape = (
        EXPECTED_COMPONENT_COUNT,
        EXPECTED_COMPONENT_COUNT,
    )

    if (
        similarity.shape
        != expected_similarity_shape
    ):
        raise RuntimeError(
            f"Component similarity "
            f"matrix must have shape "
            f"{expected_similarity_shape}, "
            f"got {similarity.shape}"
        )

    if not np.all(
        np.isfinite(
            similarity
        )
    ):
        raise RuntimeError(
            "Component similarity matrix "
            "contains non-finite values"
        )

    statistics = (
        similarity_statistics(
            similarity
        )
    )

    neighbor_search_similarity = (
        similarity.copy()
    )

    np.fill_diagonal(
        neighbor_search_similarity,
        -np.inf,
    )

    directed_adjacency = np.zeros(
        expected_similarity_shape,
        dtype=np.float32,
    )

    for component_id in range(
        EXPECTED_COMPONENT_COUNT
    ):
        neighbor_order = np.argsort(
            -neighbor_search_similarity[
                component_id
            ],
            kind="mergesort",
        )

        neighbor_ids = (
            neighbor_order[
                :META_K
            ]
        )

        neighbor_weights = (
            np.maximum(
                similarity[
                    component_id,
                    neighbor_ids,
                ],
                0.0,
            )
            .astype(
                np.float32
            )
        )

        directed_adjacency[
            component_id,
            neighbor_ids,
        ] = neighbor_weights

    adjacency = np.maximum(
        directed_adjacency,
        directed_adjacency.T,
    )

    np.fill_diagonal(
        adjacency,
        0.0,
    )

    if np.any(
        adjacency
        < 0.0
    ):
        raise RuntimeError(
            "Component adjacency "
            "contains negative weights"
        )

    if not np.all(
        np.isfinite(
            adjacency
        )
    ):
        raise RuntimeError(
            "Component adjacency "
            "contains non-finite values"
        )

    row_sums = adjacency.sum(
        axis=1,
        dtype=np.float64,
    )

    invalid_rows = np.flatnonzero(
        (
            ~np.isfinite(
                row_sums
            )
        )
        | (
            row_sums
            <= EPS
        )
    )

    if (
        invalid_rows.size
        > 0
    ):
        raise RuntimeError(
            f"Component graph contains "
            f"invalid zero-weight rows: "
            f"{invalid_rows.tolist()}"
        )

    transition = (
        adjacency.astype(
            np.float64
        )
        / row_sums[
            :,
            None,
        ]
    )

    if not np.all(
        np.isfinite(
            transition
        )
    ):
        raise RuntimeError(
            "Component transition matrix "
            "contains non-finite values"
        )

    if np.any(
        transition
        < -EPS
    ):
        raise RuntimeError(
            "Component transition matrix "
            "contains negative values"
        )

    transition_row_sums = (
        transition.sum(
            axis=1
        )
    )

    valid_rows = np.isclose(
        transition_row_sums,
        1.0,
        rtol=1e-7,
        atol=1e-9,
    )

    if not np.all(
        valid_rows
    ):
        bad_rows = np.flatnonzero(
            ~valid_rows
        )

        raise RuntimeError(
            f"Component transition matrix "
            f"has invalid row sums at IDs "
            f"{bad_rows.tolist()}"
        )

    edge_mask = np.triu(
        adjacency
        > 0.0,
        k=1,
    )

    edge_weights = (
        adjacency[
            edge_mask
        ]
    )

    edge_count = int(
        edge_weights.size
    )

    if edge_count == 0:
        raise RuntimeError(
            "Component meta-graph "
            "has no positive-weight edges"
        )

    degrees = np.sum(
        adjacency
        > 0.0,
        axis=1,
    )

    statistics.update(
        {
            "knn_k":
                int(
                    META_K
                ),
            "undirected_positive_edge_count":
                edge_count,
            "edge_weight_min":
                float(
                    np.min(
                        edge_weights
                    )
                ),
            "edge_weight_max":
                float(
                    np.max(
                        edge_weights
                    )
                ),
            "edge_weight_mean":
                float(
                    np.mean(
                        edge_weights
                    )
                ),
            "edge_weight_std":
                float(
                    np.std(
                        edge_weights
                    )
                ),
            "degree_min":
                int(
                    np.min(
                        degrees
                    )
                ),
            "degree_max":
                int(
                    np.max(
                        degrees
                    )
                ),
            "degree_mean":
                float(
                    np.mean(
                        degrees
                    )
                ),
            "transition_row_sum_min":
                float(
                    np.min(
                        transition_row_sums
                    )
                ),
            "transition_row_sum_max":
                float(
                    np.max(
                        transition_row_sums
                    )
                ),
        }
    )

    return (
        adjacency.astype(
            np.float32
        ),
        transition.astype(
            np.float64
        ),
        statistics,
        edge_count,
    )


def choose_seed_components(
    semantic_consensus,
):
    expected_shape = (
        EXPECTED_COMPONENT_COUNT,
    )

    if (
        semantic_consensus.shape
        != expected_shape
    ):
        raise RuntimeError(
            f"semantic_consensus "
            f"must have shape "
            f"{expected_shape}, "
            f"got "
            f"{semantic_consensus.shape}"
        )

    if not np.all(
        np.isfinite(
            semantic_consensus
        )
    ):
        raise RuntimeError(
            "semantic_consensus "
            "contains non-finite values"
        )

    seed_count = max(
        1,
        int(
            math.ceil(
                SEED_FRACTION
                * EXPECTED_COMPONENT_COUNT
            )
        ),
    )

    component_ids = np.arange(
        EXPECTED_COMPONENT_COUNT,
        dtype=np.int64,
    )

    ranking_order = np.lexsort(
        (
            component_ids,
            -semantic_consensus,
        )
    )

    seed_components = (
        ranking_order[
            :seed_count
        ]
        .astype(
            np.int64
        )
    )

    initial_signal = np.zeros(
        EXPECTED_COMPONENT_COUNT,
        dtype=np.float64,
    )

    initial_signal[
        seed_components
    ] = 1.0

    if (
        int(
            np.sum(
                initial_signal
                > 0.0
            )
        )
        != seed_count
    ):
        raise RuntimeError(
            "Initial component signal "
            "contains an unexpected "
            "number of seeds"
        )

    return (
        seed_components,
        initial_signal,
    )


def propagate_component_signal(
    transition,
    initial_signal,
):
    expected_transition_shape = (
        EXPECTED_COMPONENT_COUNT,
        EXPECTED_COMPONENT_COUNT,
    )

    if (
        transition.shape
        != expected_transition_shape
    ):
        raise RuntimeError(
            f"Transition matrix has "
            f"invalid shape "
            f"{transition.shape}"
        )

    if (
        initial_signal.shape
        != (
            EXPECTED_COMPONENT_COUNT,
        )
    ):
        raise RuntimeError(
            f"Initial component signal "
            f"has invalid shape "
            f"{initial_signal.shape}"
        )

    if not (
        0.0
        < PROPAGATION_ALPHA
        < 1.0
    ):
        raise RuntimeError(
            f"PROPAGATION_ALPHA "
            f"must lie in (0, 1), "
            f"got "
            f"{PROPAGATION_ALPHA}"
        )

    if (
        PROPAGATION_ITERATIONS
        <= 0
    ):
        raise RuntimeError(
            f"PROPAGATION_ITERATIONS "
            f"must be positive, "
            f"got "
            f"{PROPAGATION_ITERATIONS}"
        )

    current = initial_signal.astype(
        np.float64,
        copy=True,
    )

    residuals = []

    for _ in range(
        PROPAGATION_ITERATIONS
    ):
        propagated = (
            transition
            @ current
        )

        updated = (
            PROPAGATION_ALPHA
            * propagated
            + (
                1.0
                - PROPAGATION_ALPHA
            )
            * initial_signal
        )

        if not np.all(
            np.isfinite(
                updated
            )
        ):
            raise RuntimeError(
                "Propagation produced "
                "non-finite component scores"
            )

        residual = float(
            np.max(
                np.abs(
                    updated
                    - current
                )
            )
        )

        residuals.append(
            residual
        )

        current = updated

    minimum_score = float(
        np.min(
            current
        )
    )

    maximum_score = float(
        np.max(
            current
        )
    )

    if not np.isfinite(
        minimum_score
    ):
        raise RuntimeError(
            "Final propagation minimum "
            "is non-finite"
        )

    if not np.isfinite(
        maximum_score
    ):
        raise RuntimeError(
            "Final propagation maximum "
            "is non-finite"
        )

    score_range = (
        maximum_score
        - minimum_score
    )

    if (
        score_range
        <= EPS
    ):
        raise RuntimeError(
            "Final propagation scores "
            "are constant and cannot "
            "form a ranking"
        )

    normalized_scores = (
        current
        - minimum_score
    ) / score_range

    if not np.all(
        np.isfinite(
            normalized_scores
        )
    ):
        raise RuntimeError(
            "Normalized propagation "
            "scores are non-finite"
        )

    information = {
        "raw_score_min":
            minimum_score,
        "raw_score_max":
            maximum_score,
        "raw_score_mean":
            float(
                np.mean(
                    current
                )
            ),
        "normalized_score_min":
            float(
                np.min(
                    normalized_scores
                )
            ),
        "normalized_score_max":
            float(
                np.max(
                    normalized_scores
                )
            ),
        "normalized_score_mean":
            float(
                np.mean(
                    normalized_scores
                )
            ),
        "final_residual":
            float(
                residuals[
                    -1
                ]
            ),
        "max_residual":
            float(
                np.max(
                    residuals
                )
            ),
    }

    return (
        normalized_scores.astype(
            np.float64
        ),
        information,
    )


def freeze_score_artifacts(
    component_scores,
    component_labels,
):
    expected_component_shape = (
        EXPECTED_COMPONENT_COUNT,
    )

    if (
        component_scores.shape
        != expected_component_shape
    ):
        raise RuntimeError(
            f"Component score shape "
            f"must be "
            f"{expected_component_shape}, "
            f"got "
            f"{component_scores.shape}"
        )

    sample_scores = (
        component_scores[
            component_labels
        ]
    )

    expected_sample_shape = (
        EXPECTED_CANDIDATE_COUNT,
    )

    if (
        sample_scores.shape
        != expected_sample_shape
    ):
        raise RuntimeError(
            f"Sample score shape "
            f"must be "
            f"{expected_sample_shape}, "
            f"got "
            f"{sample_scores.shape}"
        )

    frozen_component_scores = (
        np.array(
            component_scores,
            dtype=np.float64,
            copy=True,
        )
    )

    frozen_sample_scores = (
        np.array(
            sample_scores,
            dtype=np.float64,
            copy=True,
        )
    )

    frozen_component_scores.setflags(
        write=False
    )

    frozen_sample_scores.setflags(
        write=False
    )

    return (
        frozen_component_scores,
        frozen_sample_scores,
    )


def load_target_labels(
    cache_dir,
):
    files = cache_chunk_files(
        cache_dir
    )

    label_parts = []

    expected_start = 0
    total_count = 0

    for path in files:
        payload = safe_torch_load(
            path
        )

        required_keys = [
            "labels",
            "start_index",
            "end_index",
        ]

        missing_keys = [
            key
            for key
            in required_keys
            if key not in payload
        ]

        if missing_keys:
            raise RuntimeError(
                f"Target cache chunk {path} "
                f"is missing keys "
                f"{missing_keys}"
            )

        labels = payload[
            "labels"
        ]

        if not torch.is_tensor(
            labels
        ):
            raise RuntimeError(
                f"labels in {path} "
                f"is not a tensor"
            )

        labels = (
            labels
            .long()
            .cpu()
            .reshape(
                -1
            )
        )

        start_index = int(
            payload[
                "start_index"
            ]
        )

        end_index = int(
            payload[
                "end_index"
            ]
        )

        if (
            start_index
            != expected_start
        ):
            raise RuntimeError(
                f"Target label cache has "
                f"a gap, overlap, or "
                f"reordering at {path}: "
                f"expected "
                f"start_index="
                f"{expected_start}, "
                f"got "
                f"{start_index}"
            )

        if (
            end_index
            <= start_index
        ):
            raise RuntimeError(
                f"Invalid target label "
                f"cache range in {path}: "
                f"start={start_index} "
                f"end={end_index}"
            )

        expected_chunk_count = (
            end_index
            - start_index
        )

        if (
            labels.numel()
            != expected_chunk_count
        ):
            raise RuntimeError(
                f"Label count/range "
                f"mismatch in {path}: "
                f"labels={labels.numel()} "
                f"range="
                f"{expected_chunk_count}"
            )

        label_parts.append(
            labels
        )

        total_count += (
            expected_chunk_count
        )

        expected_start = (
            end_index
        )

        del payload

    target_labels = (
        torch.cat(
            label_parts,
            dim=0,
        )
        .numpy()
        .astype(
            np.int64
        )
    )

    if (
        total_count
        != EXPECTED_TARGET_COUNT
    ):
        raise RuntimeError(
            f"Target labels count "
            f"must be "
            f"{EXPECTED_TARGET_COUNT}, "
            f"range total="
            f"{total_count}"
        )

    if (
        target_labels.size
        != EXPECTED_TARGET_COUNT
    ):
        raise RuntimeError(
            f"Concatenated target "
            f"labels must contain "
            f"{EXPECTED_TARGET_COUNT} "
            f"samples, got "
            f"{target_labels.size}"
        )

    if (
        expected_start
        != EXPECTED_TARGET_COUNT
    ):
        raise RuntimeError(
            f"Target label cache "
            f"final end_index must be "
            f"{EXPECTED_TARGET_COUNT}, "
            f"got {expected_start}"
        )

    minimum_label = int(
        target_labels.min()
    )

    maximum_label = int(
        target_labels.max()
    )

    if minimum_label < 0:
        raise RuntimeError(
            f"Target labels contain "
            f"negative class ID "
            f"{minimum_label}"
        )

    if (
        maximum_label
        >= NUM_CLASSES
    ):
        raise RuntimeError(
            f"Target labels must lie "
            f"in [0, {NUM_CLASSES}), "
            f"observed max="
            f"{maximum_label}"
        )

    target_labels.setflags(
        write=False
    )

    return target_labels


def binary_average_precision(
    scores,
    binary_labels,
):
    scores = np.asarray(
        scores,
        dtype=np.float64,
    ).reshape(-1)

    binary_labels = np.asarray(
        binary_labels,
        dtype=np.int64,
    ).reshape(-1)

    if (
        scores.size
        != binary_labels.size
    ):
        raise RuntimeError(
            f"AP score/label "
            f"length mismatch: "
            f"{scores.size} vs "
            f"{binary_labels.size}"
        )

    if scores.size == 0:
        raise RuntimeError(
            "Cannot compute AP "
            "for an empty ranking"
        )

    if not np.all(
        np.isfinite(
            scores
        )
    ):
        raise RuntimeError(
            "AP scores contain "
            "non-finite values"
        )

    valid_binary_labels = (
        (
            binary_labels
            == 0
        )
        | (
            binary_labels
            == 1
        )
    )

    if not np.all(
        valid_binary_labels
    ):
        raise RuntimeError(
            "AP labels must be binary"
        )

    positive_count = int(
        binary_labels.sum()
    )

    if positive_count == 0:
        return 0.0

    ranking_order = np.argsort(
        -scores,
        kind="mergesort",
    )

    sorted_scores = scores[
        ranking_order
    ]

    sorted_labels = binary_labels[
        ranking_order
    ]

    cumulative_true_positives = (
        np.cumsum(
            sorted_labels,
            dtype=np.int64,
        )
    )

    distinct_score_ends = (
        np.flatnonzero(
            np.diff(
                sorted_scores
            )
            != 0.0
        )
    )

    group_ends = np.concatenate(
        [
            distinct_score_ends,
            np.asarray(
                [
                    scores.size
                    - 1
                ],
                dtype=np.int64,
            ),
        ]
    )

    true_positives = (
        cumulative_true_positives[
            group_ends
        ]
        .astype(
            np.float64
        )
    )

    selected_counts = (
        group_ends
        + 1
    ).astype(
        np.float64
    )

    precision = (
        true_positives
        / selected_counts
    )

    recall = (
        true_positives
        / positive_count
    )

    recall_deltas = np.diff(
        np.concatenate(
            [
                np.asarray(
                    [0.0],
                    dtype=np.float64,
                ),
                recall,
            ]
        )
    )

    average_precision = float(
        np.sum(
            precision
            * recall_deltas
        )
    )

    if (
        average_precision
        < -1e-10
        or average_precision
        > 1.0 + 1e-10
    ):
        raise RuntimeError(
            f"Computed AP lies "
            f"outside [0, 1]: "
            f"{average_precision}"
        )

    return float(
        np.clip(
            average_precision,
            0.0,
            1.0,
        )
    )


def classification_metrics(
    predictions,
    labels,
):
    predictions = np.asarray(
        predictions,
        dtype=np.int64,
    ).reshape(-1)

    labels = np.asarray(
        labels,
        dtype=np.int64,
    ).reshape(-1)

    if (
        predictions.size
        != labels.size
    ):
        raise RuntimeError(
            f"Prediction/label "
            f"length mismatch: "
            f"{predictions.size} "
            f"vs {labels.size}"
        )

    if (
        labels.size
        != EXPECTED_TARGET_COUNT
    ):
        raise RuntimeError(
            f"Classification "
            f"evaluation expected "
            f"{EXPECTED_TARGET_COUNT} "
            f"samples, got "
            f"{labels.size}"
        )

    overall_accuracy = float(
        np.mean(
            predictions
            == labels
        )
        * 100.0
    )

    per_class_accuracy = {}

    class_accuracy_values = []

    for class_id, class_name in enumerate(
        CLASSES
    ):
        class_mask = (
            labels
            == class_id
        )

        if not np.any(
            class_mask
        ):
            raise RuntimeError(
                f"Target evaluation "
                f"contains no samples "
                f"for class "
                f"'{class_name}'"
            )

        class_accuracy = float(
            np.mean(
                predictions[
                    class_mask
                ]
                == class_id
            )
            * 100.0
        )

        per_class_accuracy[
            class_name
        ] = class_accuracy

        class_accuracy_values.append(
            class_accuracy
        )

    mean_class_accuracy = float(
        np.mean(
            class_accuracy_values
        )
    )

    return {
        "overall_accuracy":
            overall_accuracy,
        "mean_class_accuracy":
            mean_class_accuracy,
        "per_class_accuracy":
            per_class_accuracy,
    }


def component_ranking_order(
    component_scores,
):
    component_ids = np.arange(
        EXPECTED_COMPONENT_COUNT,
        dtype=np.int64,
    )

    return (
        np.lexsort(
            (
                component_ids,
                -np.asarray(
                    component_scores,
                    dtype=np.float64,
                ),
            )
        )
        .astype(
            np.int64
        )
    )


def select_for_candidate_mass(
    fraction,
    component_scores,
    component_labels,
    component_sizes,
):
    if not (
        0.0
        < fraction
        <= 1.0
    ):
        raise RuntimeError(
            f"Candidate mass "
            f"must lie in (0, 1], "
            f"got {fraction}"
        )

    requested_sample_count = int(
        math.ceil(
            fraction
            * EXPECTED_CANDIDATE_COUNT
        )
    )

    ranking_order = (
        component_ranking_order(
            component_scores
        )
    )

    selected_components = []

    accumulated_samples = 0

    for component_id in ranking_order:
        selected_components.append(
            int(
                component_id
            )
        )

        accumulated_samples += int(
            component_sizes[
                component_id
            ]
        )

        if (
            accumulated_samples
            >= requested_sample_count
        ):
            break

    selected_components_array = (
        np.asarray(
            selected_components,
            dtype=np.int64,
        )
    )

    local_mask = np.isin(
        component_labels,
        selected_components_array,
    )

    local_positions = (
        np.flatnonzero(
            local_mask
        )
        .astype(
            np.int64
        )
    )

    if (
        local_positions.size
        != accumulated_samples
    ):
        raise RuntimeError(
            f"Selected sample "
            f"count mismatch at "
            f"candidate mass {fraction}: "
            f"mask="
            f"{local_positions.size} "
            f"accumulated="
            f"{accumulated_samples}"
        )

    if (
        local_positions.size
        < requested_sample_count
    ):
        raise RuntimeError(
            f"Selection failed to "
            f"reach requested "
            f"candidate mass {fraction}: "
            f"selected="
            f"{local_positions.size} "
            f"target="
            f"{requested_sample_count}"
        )

    return (
        selected_components_array,
        local_positions,
        requested_sample_count,
    )


def evaluate_candidate_ranking(
    candidate_indices,
    component_labels,
    component_sizes,
    sample_scores,
    component_scores,
    graph_predictions,
    target_labels,
):
    candidate_labels = (
        target_labels[
            candidate_indices
        ]
    )

    candidate_binary_labels = (
        candidate_labels
        == TRUCK_ID
    ).astype(
        np.int64
    )

    total_target_trucks = int(
        np.sum(
            target_labels
            == TRUCK_ID
        )
    )

    candidate_true_trucks = int(
        np.sum(
            candidate_binary_labels
        )
    )

    if (
        total_target_trucks
        <= 0
    ):
        raise RuntimeError(
            "Target set contains "
            "no truck samples"
        )

    if (
        candidate_true_trucks
        <= 0
    ):
        raise RuntimeError(
            "Frozen candidate region "
            "contains no true "
            "truck samples"
        )

    candidate_precision = float(
        candidate_true_trucks
        / candidate_indices.size
    )

    candidate_recall = float(
        candidate_true_trucks
        / total_target_trucks
    )

    sample_average_precision = (
        binary_average_precision(
            sample_scores,
            candidate_binary_labels,
        )
    )

    mass_evaluations = []

    for fraction in CANDIDATE_MASSES:
        (
            selected_components,
            local_positions,
            requested_sample_count,
        ) = select_for_candidate_mass(
            fraction,
            component_scores,
            component_labels,
            component_sizes,
        )

        selected_global_indices = (
            candidate_indices[
                local_positions
            ]
        )

        selected_labels = (
            target_labels[
                selected_global_indices
            ]
        )

        truck_true_positives = int(
            np.sum(
                selected_labels
                == TRUCK_ID
            )
        )

        selected_sample_count = int(
            selected_global_indices.size
        )

        truck_precision = float(
            truck_true_positives
            / max(
                selected_sample_count,
                1,
            )
        )

        truck_recall_within_candidate = float(
            truck_true_positives
            / candidate_true_trucks
        )

        truck_recall_global = float(
            truck_true_positives
            / total_target_trucks
        )

        converted_predictions = np.array(
            graph_predictions,
            dtype=np.int64,
            copy=True,
        )

        converted_predictions[
            selected_global_indices
        ] = TRUCK_ID

        converted_metrics = (
            classification_metrics(
                converted_predictions,
                target_labels,
            )
        )

        mass_evaluations.append(
            {
                "requested_candidate_mass":
                    float(
                        fraction
                    ),
                "requested_sample_count":
                    int(
                        requested_sample_count
                    ),
                "actual_candidate_mass":
                    float(
                        selected_sample_count
                        / candidate_indices.size
                    ),
                "selected_sample_count":
                    selected_sample_count,
                "selected_component_count":
                    int(
                        selected_components.size
                    ),
                "selected_component_ids":
                    [
                        int(
                            component_id
                        )
                        for component_id
                        in selected_components.tolist()
                    ],
                "truck_true_positives":
                    truck_true_positives,
                "truck_precision":
                    truck_precision,
                "truck_recall_within_candidate_region":
                    truck_recall_within_candidate,
                "truck_recall_over_all_target_trucks":
                    truck_recall_global,
                "converted_metrics":
                    converted_metrics,
            }
        )

    return {
        "candidate_true_trucks":
            candidate_true_trucks,
        "total_target_trucks":
            total_target_trucks,
        "candidate_precision":
            candidate_precision,
        "candidate_recall":
            candidate_recall,
        "sample_average_precision":
            sample_average_precision,
        "candidate_mass_evaluations":
            mass_evaluations,
    }


def scientific_interpretation(
    evaluation,
):
    sample_ap = evaluation[
        "sample_average_precision"
    ]

    if sample_ap > 0.50:
        sample_ap_state = "strong"

    elif sample_ap > 0.40:
        sample_ap_state = "meaningful"

    else:
        sample_ap_state = "weak"

    key_operating_points = []

    for row in evaluation[
        "candidate_mass_evaluations"
    ]:
        requested_mass = row[
            "requested_candidate_mass"
        ]

        is_twenty_percent = (
            math.isclose(
                requested_mass,
                0.20,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )

        is_thirty_percent = (
            math.isclose(
                requested_mass,
                0.30,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )

        if (
            is_twenty_percent
            or is_thirty_percent
        ):
            key_operating_points.append(
                row
            )

    qualifying_points = [
        row
        for row
        in key_operating_points
        if (
            row[
                "truck_precision"
            ]
            > 0.60
            and row[
                "truck_recall_over_all_target_trucks"
            ]
            > 0.40
        )
    ]

    continue_branch = (
        len(
            qualifying_points
        )
        > 0
    )

    if continue_branch:
        conclusion = (
            "Continue the component meta-graph propagation branch: "
            "at least one predefined 20-30% candidate-mass operating "
            "point exceeds both the 60% truck-precision and 40% "
            "global-truck-recall thresholds."
        )

    elif sample_ap > 0.40:
        conclusion = (
            "The propagated ranking contains useful signal, but it "
            "does not satisfy the predefined 20-30% candidate-mass "
            "continuation criterion. Do not perform broad alpha, k, "
            "or seed-fraction tuning; the next branch should test a "
            "more principled component-level semantic or global "
            "assignment method."
        )

    else:
        conclusion = (
            "Seeded component meta-graph propagation is not "
            "sufficiently informative under the predefined diagnostic "
            "thresholds. Do not endlessly tune alpha, k, or seed "
            "fraction; move to a more principled component-level "
            "semantic or global assignment method."
        )

    return {
        "sample_ap_state":
            sample_ap_state,
        "sample_ap_meaningful_threshold":
            0.40,
        "sample_ap_strong_threshold":
            0.50,
        "continuation_precision_threshold":
            0.60,
        "continuation_global_truck_recall_threshold":
            0.40,
        "continuation_candidate_masses":
            [
                0.20,
                0.30,
            ],
        "continue_component_propagation_branch":
            continue_branch,
        "qualifying_candidate_masses":
            [
                float(
                    row[
                        "requested_candidate_mass"
                    ]
                )
                for row
                in qualifying_points
            ],
        "conclusion":
            conclusion,
    }


def print_classification_metrics(
    title,
    metrics,
):
    print(
        title
    )

    print(
        f"OA="
        f"{metrics['overall_accuracy']:.2f}% "
        f"MCA="
        f"{metrics['mean_class_accuracy']:.2f}%"
    )

    for class_name in CLASSES:
        class_accuracy = (
            metrics[
                "per_class_accuracy"
            ][
                class_name
            ]
        )

        print(
            f"{class_name:12s} "
            f"{class_accuracy:.2f}%"
        )


def print_candidate_evaluation(
    evaluation,
):
    print()

    print(
        "FROZEN CANDIDATE REGION"
    )

    print(
        f"candidate_precision="
        f"{100.0 * evaluation['candidate_precision']:.2f}%"
    )

    print(
        f"candidate_recall="
        f"{100.0 * evaluation['candidate_recall']:.2f}%"
    )

    print(
        f"sample_average_precision="
        f"{evaluation['sample_average_precision']:.6f}"
    )

    print()

    print(
        "META-GRAPH RANKING AND CONVERSION"
    )

    for row in evaluation[
        "candidate_mass_evaluations"
    ]:
        metrics = row[
            "converted_metrics"
        ]

        per_class = metrics[
            "per_class_accuracy"
        ]

        print(
            f"requested_mass="
            f"{100.0 * row['requested_candidate_mass']:5.1f}% "
            f"actual_mass="
            f"{100.0 * row['actual_candidate_mass']:5.1f}% "
            f"samples="
            f"{row['selected_sample_count']:5d} "
            f"precision="
            f"{100.0 * row['truck_precision']:6.2f}% "
            f"candidate_recall="
            f"{100.0 * row['truck_recall_within_candidate_region']:6.2f}% "
            f"global_recall="
            f"{100.0 * row['truck_recall_over_all_target_trucks']:6.2f}%"
        )

        print(
            f"  OA="
            f"{metrics['overall_accuracy']:.2f}% "
            f"MCA="
            f"{metrics['mean_class_accuracy']:.2f}% "
            f"truck="
            f"{per_class['truck']:.2f}% "
            f"car="
            f"{per_class['car']:.2f}% "
            f"bus="
            f"{per_class['bus']:.2f}% "
            f"train="
            f"{per_class['train']:.2f}%"
        )


def save_outputs(
    candidate_indices,
    component_labels,
    semantic_consensus,
    initial_component_signal,
    propagated_component_scores,
    propagated_sample_scores,
    component_centroids,
    component_sizes,
    component_adjacency,
    component_transition,
    result,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        OUTPUT_NPZ,
        candidate_indices=np.asarray(
            candidate_indices,
            dtype=np.int64,
        ),
        component_labels=np.asarray(
            component_labels,
            dtype=np.int64,
        ),
        semantic_consensus=np.asarray(
            semantic_consensus,
            dtype=np.float64,
        ),
        initial_component_signal=np.asarray(
            initial_component_signal,
            dtype=np.float64,
        ),
        propagated_component_scores=np.asarray(
            propagated_component_scores,
            dtype=np.float64,
        ),
        propagated_sample_scores=np.asarray(
            propagated_sample_scores,
            dtype=np.float64,
        ),
        component_centroids=np.asarray(
            component_centroids,
            dtype=np.float32,
        ),
        component_sizes=np.asarray(
            component_sizes,
            dtype=np.int64,
        ),
        component_adjacency=np.asarray(
            component_adjacency,
            dtype=np.float32,
        ),
        component_transition=np.asarray(
            component_transition,
            dtype=np.float64,
        ),
    )

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            result,
            handle,
            indent=2,
        )


def main():
    set_seed(
        SEED
    )

    device = torch.device(
        "cpu"
    )

    print(
        "="
        * 96
    )

    print(
        "VISDA-2017 SEEDED COMPONENT META-GRAPH PROPAGATION"
    )

    print(
        "="
        * 96
    )

    print(
        f"device={device}"
    )

    print(
        f"seed={SEED}"
    )

    print(
        f"seed_fraction="
        f"{SEED_FRACTION}"
    )

    print(
        f"meta_k="
        f"{META_K}"
    )

    print(
        f"alpha="
        f"{PROPAGATION_ALPHA}"
    )

    print(
        f"iterations="
        f"{PROPAGATION_ITERATIONS}"
    )

    print()

    print(
        "Loading frozen component assignments "
        "and semantic-consensus ranker..."
    )

    (
        candidate_indices,
        component_labels,
        semantic_consensus,
        component_npz_keys,
    ) = load_frozen_components(
        COMPONENT_ASSIGNMENTS
    )

    print(
        f"candidate_samples="
        f"{candidate_indices.size}"
    )

    print(
        f"components="
        f"{int(component_labels.max()) + 1}"
    )

    print(
        f"semantic_consensus_scores="
        f"{semantic_consensus.size}"
    )

    print()

    print(
        "Inspecting graph outputs and verifying "
        "the frozen truck Top-3 candidate region..."
    )

    (
        graph_tensor_key,
        graph_scores,
        graph_tensor_diagnostics,
    ) = identify_diffusion_graph_tensor(
        GRAPH_OUTPUTS,
        candidate_indices,
    )

    if (
        graph_scores.shape[
            0
        ]
        != EXPECTED_TARGET_COUNT
    ):
        raise RuntimeError(
            f"Graph target count must be "
            f"{EXPECTED_TARGET_COUNT}, "
            f"got "
            f"{graph_scores.shape[0]}"
        )

    graph_predictions = (
        np.argmax(
            graph_scores,
            axis=1,
        )
        .astype(
            np.int64
        )
    )

    if (
        graph_predictions.size
        != EXPECTED_TARGET_COUNT
    ):
        raise RuntimeError(
            f"Graph prediction count "
            f"must be "
            f"{EXPECTED_TARGET_COUNT}, "
            f"got "
            f"{graph_predictions.size}"
        )

    graph_predictions.setflags(
        write=False
    )

    print(
        f"selected_graph_tensor="
        f"{graph_tensor_key}"
    )

    print(
        f"graph_shape="
        f"{graph_scores.shape}"
    )

    print()

    print(
        "Selecting label-free seed components "
        "from frozen semantic consensus..."
    )

    (
        seed_components,
        initial_component_signal,
    ) = choose_seed_components(
        semantic_consensus
    )

    print(
        f"seed_components="
        f"{seed_components.tolist()}"
    )

    print()

    print(
        "Loading frozen RPC model on CPU..."
    )

    (
        model,
        rpc_state_key,
    ) = load_rpc_model(
        RPC_CHECKPOINT,
        device,
    )

    print(
        f"rpc_state_key="
        f"{rpc_state_key}"
    )

    print()

    print(
        "Loading only frozen target features "
        "required by the candidate region..."
    )

    (
        raw_candidate_features,
        target_feature_count,
        target_cache_chunk_count,
    ) = load_candidate_raw_features(
        TARGET_CACHE,
        candidate_indices,
    )

    if (
        target_feature_count
        != EXPECTED_TARGET_COUNT
    ):
        raise RuntimeError(
            f"Target feature cache "
            f"count must be "
            f"{EXPECTED_TARGET_COUNT}, "
            f"got "
            f"{target_feature_count}"
        )

    print(
        f"target_feature_cache_count="
        f"{target_feature_count} "
        f"chunks="
        f"{target_cache_chunk_count}"
    )

    print(
        f"candidate_raw_feature_shape="
        f"{tuple(raw_candidate_features.shape)}"
    )

    print()

    print(
        "Encoding frozen candidate features "
        "through the RPC adapter..."
    )

    adapted_candidate_features = (
        encode_candidate_features(
            model,
            raw_candidate_features,
            device,
        )
    )

    del raw_candidate_features

    print(
        f"adapted_candidate_feature_shape="
        f"{tuple(adapted_candidate_features.shape)}"
    )

    print()

    print(
        "Computing centroids for all "
        "frozen components..."
    )

    (
        component_centroids,
        component_sizes,
    ) = compute_component_centroids(
        adapted_candidate_features,
        component_labels,
    )

    del adapted_candidate_features

    if (
        component_centroids.shape[
            0
        ]
        != EXPECTED_COMPONENT_COUNT
    ):
        raise RuntimeError(
            f"Component graph must "
            f"have "
            f"{EXPECTED_COMPONENT_COUNT} "
            f"nodes, got "
            f"{component_centroids.shape[0]}"
        )

    print(
        f"component_centroid_shape="
        f"{component_centroids.shape}"
    )

    print(
        f"component_size_min="
        f"{int(component_sizes.min())} "
        f"component_size_max="
        f"{int(component_sizes.max())} "
        f"component_size_mean="
        f"{float(component_sizes.mean()):.2f}"
    )

    print()

    print(
        "Constructing the 113-node centroid "
        "cosine KNN meta-graph..."
    )

    (
        component_adjacency,
        component_transition,
        graph_statistics,
        meta_graph_edge_count,
    ) = build_component_meta_graph(
        component_centroids
    )

    expected_graph_shape = (
        EXPECTED_COMPONENT_COUNT,
        EXPECTED_COMPONENT_COUNT,
    )

    if (
        component_adjacency.shape
        != expected_graph_shape
    ):
        raise RuntimeError(
            f"Component adjacency "
            f"must have shape "
            f"{expected_graph_shape}, "
            f"got "
            f"{component_adjacency.shape}"
        )

    if (
        component_transition.shape
        != expected_graph_shape
    ):
        raise RuntimeError(
            f"Component transition "
            f"must have shape "
            f"{expected_graph_shape}, "
            f"got "
            f"{component_transition.shape}"
        )

    print(
        f"meta_graph_edges="
        f"{meta_graph_edge_count}"
    )

    print(
        f"centroid_similarity_mean="
        f"{graph_statistics['off_diagonal_mean']:.6f} "
        f"min="
        f"{graph_statistics['off_diagonal_min']:.6f} "
        f"max="
        f"{graph_statistics['off_diagonal_max']:.6f}"
    )

    print(
        f"meta_graph_degree_min="
        f"{graph_statistics['degree_min']} "
        f"mean="
        f"{graph_statistics['degree_mean']:.2f} "
        f"max="
        f"{graph_statistics['degree_max']}"
    )

    print()

    print(
        "Running personalized component propagation..."
    )

    (
        propagated_component_scores,
        propagation_statistics,
    ) = propagate_component_signal(
        component_transition,
        initial_component_signal,
    )

    (
        frozen_component_scores,
        frozen_sample_scores,
    ) = freeze_score_artifacts(
        propagated_component_scores,
        component_labels,
    )

    if (
        frozen_component_scores.size
        != EXPECTED_COMPONENT_COUNT
    ):
        raise RuntimeError(
            f"Frozen component "
            f"score count must be "
            f"{EXPECTED_COMPONENT_COUNT}, "
            f"got "
            f"{frozen_component_scores.size}"
        )

    if (
        frozen_sample_scores.size
        != EXPECTED_CANDIDATE_COUNT
    ):
        raise RuntimeError(
            f"Frozen sample score "
            f"count must be "
            f"{EXPECTED_CANDIDATE_COUNT}, "
            f"got "
            f"{frozen_sample_scores.size}"
        )

    component_centroids.setflags(
        write=False
    )

    component_sizes.setflags(
        write=False
    )

    component_adjacency.setflags(
        write=False
    )

    component_transition.setflags(
        write=False
    )

    initial_component_signal.setflags(
        write=False
    )

    print(
        f"propagated_component_scores="
        f"{frozen_component_scores.size} "
        f"propagated_sample_scores="
        f"{frozen_sample_scores.size}"
    )

    print(
        f"final_propagation_residual="
        f"{propagation_statistics['final_residual']:.8e}"
    )

    print()

    print(
        "All candidate construction, component assignments, "
        "seed selection, representation construction, "
        "meta-graph construction, propagation, and ranking "
        "scores are frozen."
    )

    print(
        "Loading target labels now for evaluation only..."
    )

    target_labels = load_target_labels(
        TARGET_CACHE
    )

    if (
        target_labels.size
        != EXPECTED_TARGET_COUNT
    ):
        raise RuntimeError(
            f"Target labels count "
            f"must be "
            f"{EXPECTED_TARGET_COUNT}, "
            f"got "
            f"{target_labels.size}"
        )

    print(
        f"target_label_count="
        f"{target_labels.size}"
    )

    print()

    baseline_metrics = (
        classification_metrics(
            graph_predictions,
            target_labels,
        )
    )

    print_classification_metrics(
        "BASELINE GRAPH TOP-1",
        baseline_metrics,
    )

    evaluation = (
        evaluate_candidate_ranking(
            candidate_indices,
            component_labels,
            component_sizes,
            frozen_sample_scores,
            frozen_component_scores,
            graph_predictions,
            target_labels,
        )
    )

    print_candidate_evaluation(
        evaluation
    )

    interpretation = (
        scientific_interpretation(
            evaluation
        )
    )

    print()

    print(
        "SCIENTIFIC INTERPRETATION"
    )

    print(
        f"sample_ap_state="
        f"{interpretation['sample_ap_state']} "
        f"continue_branch="
        f"{interpretation['continue_component_propagation_branch']}"
    )

    print(
        interpretation[
            "conclusion"
        ]
    )

    result = {
        "experiment":
            "visda_component_meta_graph_propagation",
        "seed":
            SEED,
        "device":
            str(
                device
            ),
        "hyperparameters": {
            "num_classes":
                NUM_CLASSES,
            "input_dim":
                INPUT_DIM,
            "adapted_dim":
                HIDDEN_DIM,
            "truck_class_id":
                TRUCK_ID,
            "candidate_top_k":
                TOP_K,
            "seed_fraction":
                SEED_FRACTION,
            "seed_component_count":
                int(
                    seed_components.size
                ),
            "meta_graph_k":
                META_K,
            "propagation_alpha":
                PROPAGATION_ALPHA,
            "propagation_iterations":
                PROPAGATION_ITERATIONS,
            "encode_batch_size":
                ENCODE_BATCH_SIZE,
            "candidate_masses":
                [
                    float(
                        fraction
                    )
                    for fraction
                    in CANDIDATE_MASSES
                ],
            "candidate_mass_selection":
                "whole_components_in_descending_propagated_score_order_until_requested_candidate_mass_is_reached",
            "expected_target_count":
                EXPECTED_TARGET_COUNT,
            "expected_candidate_count":
                EXPECTED_CANDIDATE_COUNT,
            "expected_component_count":
                EXPECTED_COMPONENT_COUNT,
        },
        "input_paths": {
            "component_assignments":
                str(
                    COMPONENT_ASSIGNMENTS
                ),
            "graph_outputs":
                str(
                    GRAPH_OUTPUTS
                ),
            "rpc_checkpoint":
                str(
                    RPC_CHECKPOINT
                ),
            "target_cache":
                str(
                    TARGET_CACHE
                ),
        },
        "component_npz_keys":
            component_npz_keys,
        "actual_graph_tensor_key_selected":
            graph_tensor_key,
        "graph_tensor_diagnostics":
            graph_tensor_diagnostics,
        "rpc_state_key":
            rpc_state_key,
        "candidate_count":
            int(
                candidate_indices.size
            ),
        "component_count":
            int(
                EXPECTED_COMPONENT_COUNT
            ),
        "target_count":
            int(
                target_labels.size
            ),
        "target_feature_cache_count":
            int(
                target_feature_count
            ),
        "target_cache_chunk_count":
            int(
                target_cache_chunk_count
            ),
        "seed_component_ids":
            [
                int(
                    component_id
                )
                for component_id
                in seed_components.tolist()
            ],
        "graph_construction_method":
            "centroid_knn",
        "number_of_meta_graph_edges":
            int(
                meta_graph_edge_count
            ),
        "centroid_similarity_statistics":
            graph_statistics,
        "propagation_statistics":
            propagation_statistics,
        "baseline_graph_metrics":
            baseline_metrics,
        "candidate_precision":
            evaluation[
                "candidate_precision"
            ],
        "candidate_recall":
            evaluation[
                "candidate_recall"
            ],
        "candidate_true_trucks":
            evaluation[
                "candidate_true_trucks"
            ],
        "total_target_trucks":
            evaluation[
                "total_target_trucks"
            ],
        "sample_average_precision":
            evaluation[
                "sample_average_precision"
            ],
        "precision_recall_at_candidate_masses":
            evaluation[
                "candidate_mass_evaluations"
            ],
        "converted_metrics_by_candidate_mass":
            [
                {
                    "requested_candidate_mass":
                        row[
                            "requested_candidate_mass"
                        ],
                    "actual_candidate_mass":
                        row[
                            "actual_candidate_mass"
                        ],
                    "selected_sample_count":
                        row[
                            "selected_sample_count"
                        ],
                    "overall_accuracy":
                        row[
                            "converted_metrics"
                        ][
                            "overall_accuracy"
                        ],
                    "mean_class_accuracy":
                        row[
                            "converted_metrics"
                        ][
                            "mean_class_accuracy"
                        ],
                    "per_class_accuracy":
                        row[
                            "converted_metrics"
                        ][
                            "per_class_accuracy"
                        ],
                }
                for row
                in evaluation[
                    "candidate_mass_evaluations"
                ]
            ],
        "scientific_interpretation":
            interpretation,
        "leakage_constraints": {
            "target_labels_used_for_component_construction":
                False,
            "target_labels_used_for_candidate_definition":
                False,
            "target_labels_used_for_seed_selection":
                False,
            "target_labels_used_for_centroids":
                False,
            "target_labels_used_for_graph_edges":
                False,
            "target_labels_used_for_propagation":
                False,
            "target_labels_used_for_ranking":
                False,
            "target_label_values_accessed_from_target_cache_only_after_scores_frozen":
                True,
            "frozen_candidate_indices_recomputed":
                False,
            "frozen_component_labels_recomputed":
                False,
            "source_cache_used_by_meta_graph":
                False,
        },
        "output_paths": {
            "npz":
                str(
                    OUTPUT_NPZ
                ),
            "json":
                str(
                    OUTPUT_JSON
                ),
        },
    }

    save_outputs(
        candidate_indices,
        component_labels,
        semantic_consensus,
        initial_component_signal,
        frozen_component_scores,
        frozen_sample_scores,
        component_centroids,
        component_sizes,
        component_adjacency,
        component_transition,
        result,
    )

    print()

    print(
        "OUTPUTS"
    )

    print(
        f"npz="
        f"{OUTPUT_NPZ}"
    )

    print(
        f"json="
        f"{OUTPUT_JSON}"
    )


if __name__ == "__main__":
    main()