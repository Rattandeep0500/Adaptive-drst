import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
NUM_CLASSES = 12
INPUT_DIM = 2048
HIDDEN_DIM = 512
TRUCK = 11
CAR = 3
BUS = 2
TRAIN = 10
FOCUS = [TRUCK, CAR, BUS, TRAIN]
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
    parser.add_argument("--rpc-checkpoint", type=Path, default=Path("checkpoints/visda_rpc_causal_experiment/rpc_seed42.pt"))
    parser.add_argument("--graph-outputs", type=Path, default=Path("checkpoints/visda_graph_semantic_diffusion/graph_semantic_outputs_seed42.pt"))
    parser.add_argument("--pairwise-outputs", type=Path, default=None)
    parser.add_argument("--source-cache", type=Path, default=Path("checkpoints/visda_feature_cache/source"))
    parser.add_argument("--target-cache", type=Path, default=Path("checkpoints/visda_feature_cache/target"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/visda_truck_component_structure_probe"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--local-k", type=int, default=20)
    parser.add_argument("--edge-quantile", type=float, default=0.75)
    parser.add_argument("--knn-chunk", type=int, default=384)
    parser.add_argument("--eval-batch", type=int, default=4096)
    parser.add_argument("--source-support", type=int, default=2000)
    parser.add_argument("--anchor-k", type=int, default=5)
    parser.add_argument("--prototype-iters", type=int, default=20)
    parser.add_argument("--fallback-target-size", type=int, default=200)
    parser.add_argument("--fallback-iters", type=int, default=20)
    parser.add_argument("--max-cc-fraction", type=float, default=0.25)
    parser.add_argument("--max-singleton-fraction", type=float, default=0.30)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--threads", type=int, default=0)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


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
            name = f"{prefix}.{key}" if prefix else str(key)
            found.update(collect_arrays(value, name))
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            name = f"{prefix}.{i}" if prefix else str(i)
            found.update(collect_arrays(value, name))
    else:
        array = to_numpy(obj)
        if array is not None:
            found[prefix.lower()] = array
    return found


def choose_array(arrays, required_terms, preferred_terms=(), ndim=2, last_dim=None):
    candidates = []
    for name, array in arrays.items():
        normalized = name.replace("-", "_")
        if not all(term in normalized for term in required_terms):
            continue
        if ndim is not None and array.ndim != ndim:
            continue
        if last_dim is not None and (array.ndim == 0 or array.shape[-1] != last_dim):
            continue
        bonus = sum(term in normalized for term in preferred_terms)
        candidates.append((bonus, -len(name), name, array))
    if not candidates:
        return None, None
    candidates.sort(reverse=True)
    return candidates[0][2], candidates[0][3]


def find_graph_scores(payload, n_target_hint=None):
    arrays = collect_arrays(payload)
    anchor_name, anchor = choose_array(arrays, ["anchor"], ["score", "semantic", "prob"], ndim=2, last_dim=NUM_CLASSES)
    diffusion_name, diffusion = choose_array(arrays, ["diff"], ["score", "semantic", "prob", "output"], ndim=2, last_dim=NUM_CLASSES)
    if anchor is None:
        anchor_name, anchor = choose_array(arrays, ["source"], ["anchor", "score", "semantic"], ndim=2, last_dim=NUM_CLASSES)
    if diffusion is None:
        diffusion_name, diffusion = choose_array(arrays, ["target"], ["diff", "semantic", "score"], ndim=2, last_dim=NUM_CLASSES)
    if anchor is None or diffusion is None:
        keys = sorted(arrays.keys())
        raise RuntimeError(f"Could not locate anchor/diffusion score tensors in graph output. Available tensor keys include: {keys[:80]}")
    if anchor.shape != diffusion.shape:
        raise RuntimeError(f"Anchor/diffusion shape mismatch: {anchor.shape} vs {diffusion.shape}")
    if n_target_hint is not None and anchor.shape[0] != n_target_hint:
        raise RuntimeError(f"Graph score count {anchor.shape[0]} does not match target count {n_target_hint}")
    return anchor_name, anchor.astype(np.float32), diffusion_name, diffusion.astype(np.float32), arrays


def find_adapted_target(arrays, n_target):
    candidates = []
    for name, array in arrays.items():
        if array.ndim != 2 or array.shape != (n_target, HIDDEN_DIM):
            continue
        score = 0
        for term in ["target", "adapt", "feature", "embedding", "z"]:
            if term in name:
                score += 1
        candidates.append((score, -len(name), name, array))
    if not candidates:
        return None, None
    candidates.sort(reverse=True)
    return candidates[0][2], candidates[0][3].astype(np.float32)


def load_cache_features(cache_dir):
    files = sorted(cache_dir.glob("chunk_*.pt"))
    if not files:
        raise RuntimeError(f"No cache chunks found in {cache_dir}")
    features = []
    for path in files:
        payload = safe_torch_load(path, map_location="cpu")
        if "features" not in payload:
            raise RuntimeError(f"Missing features in {path}")
        features.append(payload["features"].cpu())
    result = torch.cat(features, dim=0)
    if result.ndim != 2 or result.shape[1] != INPUT_DIM:
        raise RuntimeError(f"Expected N x {INPUT_DIM} cache features, got {tuple(result.shape)}")
    return result


def load_cache_labels(cache_dir):
    files = sorted(cache_dir.glob("chunk_*.pt"))
    if not files:
        raise RuntimeError(f"No cache chunks found in {cache_dir}")
    labels = []
    for path in files:
        payload = safe_torch_load(path, map_location="cpu")
        if "labels" not in payload:
            raise RuntimeError(f"Missing labels in {path}")
        labels.append(payload["labels"].long().cpu())
    return torch.cat(labels, dim=0).numpy().astype(np.int64)


def load_source_focus(cache_dir, support, seed):
    files = sorted(cache_dir.glob("chunk_*.pt"))
    if not files:
        raise RuntimeError(f"No source cache chunks found in {cache_dir}")
    by_class = {c: [] for c in FOCUS}
    for path in files:
        payload = safe_torch_load(path, map_location="cpu")
        x = payload["features"].cpu()
        y = payload["labels"].long().cpu()
        for c in FOCUS:
            mask = y == c
            if mask.any():
                by_class[c].append(x[mask])
    rng = np.random.default_rng(seed)
    selected = {}
    for c in FOCUS:
        if not by_class[c]:
            raise RuntimeError(f"No source samples for {CLASSES[c]}")
        x = torch.cat(by_class[c], dim=0)
        if len(x) < support:
            raise RuntimeError(f"Need {support} source samples for {CLASSES[c]}, found {len(x)}")
        idx = rng.choice(len(x), size=support, replace=False)
        selected[c] = x.index_select(0, torch.from_numpy(idx).long())
    return selected


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class MCDModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = Adapter()
        self.classifier1 = nn.Linear(HIDDEN_DIM, NUM_CLASSES)
        self.classifier2 = nn.Linear(HIDDEN_DIM, NUM_CLASSES)

    def encode(self, x):
        return self.adapter(x)

    def forward(self, x):
        z = self.encode(x)
        return self.classifier1(z), self.classifier2(z)


def load_rpc_model(path, device):
    payload = safe_torch_load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError("RPC checkpoint is not a dictionary")
    if "student_state_dict" in payload:
        state = payload["student_state_dict"]
    elif "state_dict" in payload:
        state = payload["state_dict"]
    else:
        tensor_keys = [k for k, v in payload.items() if torch.is_tensor(v)]
        if tensor_keys and any(str(k).startswith("adapter.") for k in tensor_keys):
            state = payload
        else:
            raise RuntimeError("Could not locate student_state_dict in RPC checkpoint")
    model = MCDModel().to(device)
    model.load_state_dict(copy.deepcopy(state), strict=True)
    model.eval()
    return model, payload


def encode_raw(model, raw_features, device, batch_size):
    outputs = []
    with torch.no_grad():
        for start in range(0, len(raw_features), batch_size):
            batch = raw_features[start:start + batch_size].to(device=device, dtype=torch.float32)
            z = model.encode(batch)
            outputs.append(z.cpu())
    return torch.cat(outputs, dim=0).numpy().astype(np.float32)


def spherical_kmeans(features, k, seed, iterations):
    x = torch.as_tensor(features, dtype=torch.float32)
    x = F.normalize(x, dim=1)
    n = len(x)
    if k <= 1:
        return np.zeros(n, dtype=np.int64), x.mean(dim=0, keepdim=True).numpy()
    if k > n:
        k = n
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    first = int(torch.randint(0, n, (1,), generator=generator).item())
    centers = [x[first]]
    min_dist = 1.0 - x @ centers[0]
    for _ in range(1, k):
        weights = min_dist.clamp_min(1e-8)
        idx = int(torch.multinomial(weights, 1, generator=generator).item())
        centers.append(x[idx])
        dist = 1.0 - x @ centers[-1]
        min_dist = torch.minimum(min_dist, dist)
    centers = torch.stack(centers, dim=0)
    labels = torch.zeros(n, dtype=torch.long)
    for _ in range(iterations):
        scores = x @ centers.T
        new_labels = scores.argmax(dim=1)
        if torch.equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        new_centers = []
        for j in range(k):
            mask = labels == j
            if mask.any():
                center = F.normalize(x[mask].mean(dim=0, keepdim=True), dim=1)[0]
            else:
                center = x[int(torch.randint(0, n, (1,), generator=generator).item())]
            new_centers.append(center)
        centers = torch.stack(new_centers, dim=0)
    return labels.numpy().astype(np.int64), centers.numpy().astype(np.float32)


def build_source_prototypes(model, cache_dir, support, k, seed, iterations, device, batch_size):
    selected = load_source_focus(cache_dir, support, seed)
    prototypes = {}
    for c in FOCUS:
        adapted = encode_raw(model, selected[c], device, batch_size)
        _, centers = spherical_kmeans(adapted, k, seed + c * 1009, iterations)
        centers = centers / np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
        prototypes[c] = centers
    return prototypes


def discover_pairwise_file(root):
    patterns = [
        "**/*truck*boundary*outputs*.pt",
        "**/*truck*pairwise*outputs*.pt",
        "**/*truck*local*probe*.pt",
        "**/*truck*boundary*.pt",
        "**/*pairwise*.pt",
        "**/*truck*boundary*.npz",
        "**/*pairwise*.npz",
    ]
    found = []
    for pattern in patterns:
        found.extend(root.glob(pattern))
    unique = []
    seen = set()
    for path in found:
        resolved = str(path.resolve())
        if resolved not in seen and "feature_cache" not in str(path):
            seen.add(resolved)
            unique.append(path)
    return unique


def load_generic_array_file(path):
    if path.suffix.lower() == ".npz":
        payload = np.load(path, allow_pickle=True)
        return {k: payload[k] for k in payload.files}
    return safe_torch_load(path, map_location="cpu")


def score_name_match(name, competitor):
    compact = name.lower().replace("-", "_").replace(" ", "_")
    return "truck" in compact and competitor in compact and any(term in compact for term in ["score", "margin", "prob", "logit", "output", "pred"])


def find_pairwise_score(arrays, competitor, n_target, n_candidate, candidate_indices):
    candidates = []
    for name, array in arrays.items():
        if not score_name_match(name, competitor):
            continue
        a = np.asarray(array)
        if a.ndim == 2 and 1 in a.shape:
            a = a.reshape(-1)
        if a.ndim != 1:
            continue
        if len(a) not in [n_target, n_candidate]:
            continue
        bonus = int("margin" in name) + int("score" in name) + int("prob" in name)
        candidates.append((bonus, -len(name), name, a.astype(np.float32)))
    if not candidates:
        return None, None
    candidates.sort(reverse=True)
    name, a = candidates[0][2], candidates[0][3]
    if len(a) == n_target:
        a = a[candidate_indices]
    return name, a


def canonical_margin(values):
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        raise RuntimeError("Pairwise score array contains no finite values")
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if lo >= -1e-6 and hi <= 1.0 + 1e-6:
        return values - 0.5, "probability_minus_0.5"
    return values, "raw_margin"


def load_pairwise_scores(path, n_target, candidate_indices):
    payload = load_generic_array_file(path)
    arrays = collect_arrays(payload)
    n_candidate = len(candidate_indices)
    result = {}
    modes = {}
    names = {}
    for competitor in ["car", "bus", "train"]:
        name, score = find_pairwise_score(arrays, competitor, n_target, n_candidate, candidate_indices)
        if score is None:
            raise RuntimeError(f"Could not find truck-vs-{competitor} pointwise score in {path}. Tensor keys include: {sorted(arrays.keys())[:100]}")
        margin, mode = canonical_margin(score)
        result[competitor] = margin.astype(np.float32)
        modes[competitor] = mode
        names[competitor] = name
    return result, modes, names


def exact_knn_cosine(features, k, chunk, device):
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    x = F.normalize(x, dim=1)
    n = len(x)
    if n <= 1:
        raise RuntimeError("Candidate region contains fewer than two samples")
    k = min(k, n - 1)
    all_indices = []
    all_values = []
    total_chunks = math.ceil(n / chunk)
    for ci, start in enumerate(range(0, n, chunk), 1):
        end = min(start + chunk, n)
        sims = x[start:end] @ x.T
        row = torch.arange(end - start, device=device)
        col = torch.arange(start, end, device=device)
        sims[row, col] = -float("inf")
        values, indices = torch.topk(sims, k=k, dim=1, largest=True, sorted=True)
        all_indices.append(indices.cpu())
        all_values.append(values.cpu())
        if ci == 1 or ci % 10 == 0 or ci == total_chunks:
            print(f"  local KNN chunk {ci}/{total_chunks}")
    return torch.cat(all_indices, dim=0).numpy().astype(np.int32), torch.cat(all_values, dim=0).numpy().astype(np.float32)


def mutual_edges(knn_indices, knn_values):
    n, k = knn_indices.shape
    src = np.repeat(np.arange(n, dtype=np.int64), k)
    dst = knn_indices.reshape(-1).astype(np.int64)
    sim = knn_values.reshape(-1).astype(np.float32)
    codes = src * n + dst
    order = np.argsort(codes)
    sorted_codes = codes[order]
    reverse = dst * n + src
    pos = np.searchsorted(sorted_codes, reverse)
    valid = pos < len(sorted_codes)
    mutual = np.zeros(len(codes), dtype=bool)
    valid_idx = np.flatnonzero(valid)
    mutual[valid_idx] = sorted_codes[pos[valid_idx]] == reverse[valid_idx]
    keep = mutual & (src < dst)
    return src[keep], dst[keep], sim[keep]


class UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != x:
            nxt = self.parent[x]
            self.parent[x] = root
            x = nxt
        return root

    def union(self, a, b):
        ra = self.find(int(a))
        rb = self.find(int(b))
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def connected_component_labels(n, src, dst, sim, threshold):
    uf = UnionFind(n)
    mask = sim >= threshold
    for a, b in zip(src[mask], dst[mask]):
        uf.union(a, b)
    roots = np.array([uf.find(i) for i in range(n)], dtype=np.int64)
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int64), int(mask.sum())


def relabel_by_size(labels):
    unique, counts = np.unique(labels, return_counts=True)
    order = unique[np.argsort(-counts)]
    mapping = {int(old): new for new, old in enumerate(order)}
    return np.array([mapping[int(x)] for x in labels], dtype=np.int64)


def partition_stats(labels):
    counts = np.bincount(labels)
    n = int(counts.sum())
    return {
        "num_components": int(len(counts)),
        "largest_component_size": int(counts.max()),
        "largest_component_fraction": float(counts.max() / max(n, 1)),
        "singleton_component_count": int(np.sum(counts == 1)),
        "singleton_sample_fraction": float(np.sum(counts == 1) / max(n, 1)),
        "median_component_size": float(np.median(counts)),
        "mean_component_size": float(np.mean(counts)),
    }


def choose_partition(features, cc_labels, args):
    stats = partition_stats(cc_labels)
    degenerate = stats["largest_component_fraction"] > args.max_cc_fraction or stats["singleton_sample_fraction"] > args.max_singleton_fraction or stats["num_components"] < 8
    if not degenerate:
        return relabel_by_size(cc_labels), "thresholded_mutual_knn_components", stats, None
    k = int(np.clip(math.ceil(len(features) / args.fallback_target_size), 8, 256))
    labels, _ = spherical_kmeans(features, k, args.seed + 4049, args.fallback_iters)
    labels = relabel_by_size(labels)
    return labels, "spherical_kmeans_fallback", stats, partition_stats(labels)


def mean_pairwise_similarity(normalized_features):
    n = len(normalized_features)
    if n < 2:
        return 1.0
    summed = normalized_features.sum(axis=0, dtype=np.float64)
    numerator = float(np.dot(summed, summed) - n)
    denominator = float(n * (n - 1))
    return numerator / max(denominator, 1.0)


def zscore(values):
    x = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(x))
    std = float(np.std(x))
    if std < 1e-12:
        return np.zeros_like(x)
    return (x - mean) / std


def compute_component_metrics(labels, candidate_features, candidate_indices, anchor_scores, diffusion_scores, graph_top1, pairwise, prototypes, mutual_src, mutual_dst, mutual_sim, local_k):
    x = candidate_features / np.maximum(np.linalg.norm(candidate_features, axis=1, keepdims=True), 1e-12)
    n_components = int(labels.max()) + 1
    edge_component = labels[mutual_src] == labels[mutual_dst]
    edge_lists = [[] for _ in range(n_components)]
    for a, sim, internal in zip(mutual_src, mutual_sim, edge_component):
        if internal:
            edge_lists[int(labels[a])].append(float(sim))
    components = []
    arrays = {
        "size": np.zeros(n_components, dtype=np.float64),
        "mean_pairwise_similarity": np.zeros(n_components, dtype=np.float64),
        "mean_internal_edge_similarity": np.zeros(n_components, dtype=np.float64),
        "neighbor_retention": np.zeros(n_components, dtype=np.float64),
        "prototype_advantage": np.zeros(n_components, dtype=np.float64),
        "graph_advantage": np.zeros(n_components, dtype=np.float64),
        "pairwise_min_margin": np.zeros(n_components, dtype=np.float64),
        "pairwise_mean_margin": np.zeros(n_components, dtype=np.float64),
        "pairwise_agreement": np.zeros(n_components, dtype=np.float64),
    }
    for cid in range(n_components):
        local = np.flatnonzero(labels == cid)
        global_idx = candidate_indices[local]
        z = x[local]
        centroid = z.mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
        proto_sim = {}
        for c in FOCUS:
            proto_sim[CLASSES[c]] = float(np.max(prototypes[c] @ centroid))
        anchor_mean = {CLASSES[c]: float(np.mean(anchor_scores[global_idx, c])) for c in FOCUS}
        graph_mean = {CLASSES[c]: float(np.mean(diffusion_scores[global_idx, c])) for c in FOCUS}
        pairwise_mean = {name: float(np.mean(pairwise[name][local])) for name in ["car", "bus", "train"]}
        agreement = float(np.mean((pairwise["car"][local] > 0) & (pairwise["bus"][local] > 0) & (pairwise["train"][local] > 0)))
        graph_counts = np.bincount(graph_top1[global_idx], minlength=NUM_CLASSES)
        graph_composition = {CLASSES[c]: int(graph_counts[c]) for c in range(NUM_CLASSES) if graph_counts[c] > 0}
        mp = mean_pairwise_similarity(z)
        edge_values = edge_lists[cid]
        mean_edge = float(np.mean(edge_values)) if edge_values else 0.0
        retention = float(min(1.0, 2.0 * len(edge_values) / max(len(local) * float(local_k), 1.0)))
        proto_adv = proto_sim["truck"] - max(proto_sim["car"], proto_sim["bus"], proto_sim["train"])
        graph_adv = graph_mean["truck"] - max(graph_mean["car"], graph_mean["bus"], graph_mean["train"])
        pair_min = min(pairwise_mean.values())
        pair_mean = float(np.mean(list(pairwise_mean.values())))
        arrays["size"][cid] = len(local)
        arrays["mean_pairwise_similarity"][cid] = mp
        arrays["mean_internal_edge_similarity"][cid] = mean_edge
        arrays["neighbor_retention"][cid] = retention
        arrays["prototype_advantage"][cid] = proto_adv
        arrays["graph_advantage"][cid] = graph_adv
        arrays["pairwise_min_margin"][cid] = pair_min
        arrays["pairwise_mean_margin"][cid] = pair_mean
        arrays["pairwise_agreement"][cid] = agreement
        components.append({
            "component_id": cid,
            "size": int(len(local)),
            "mean_pairwise_similarity": float(mp),
            "mean_internal_mutual_edge_similarity": float(mean_edge),
            "mutual_neighbor_retention": float(retention),
            "graph_label_composition": graph_composition,
            "source_prototype_similarity": proto_sim,
            "source_anchor_score_mean": anchor_mean,
            "graph_score_mean": graph_mean,
            "pairwise_margin_mean": pairwise_mean,
            "pairwise_all_three_agreement": agreement,
            "prototype_truck_advantage": float(proto_adv),
            "graph_truck_advantage": float(graph_adv),
            "pairwise_min_margin": float(pair_min),
            "pairwise_mean_margin": float(pair_mean),
        })
    consensus = (
        zscore(arrays["prototype_advantage"]) +
        zscore(arrays["graph_advantage"]) +
        zscore(arrays["pairwise_min_margin"]) +
        zscore(arrays["pairwise_agreement"])
    ) / 4.0
    consensus_density = (consensus + zscore(arrays["mean_pairwise_similarity"])) / 2.0
    rankers = {
        "internal_similarity": arrays["mean_pairwise_similarity"],
        "local_density": arrays["mean_internal_edge_similarity"],
        "source_prototype_truck_advantage": arrays["prototype_advantage"],
        "graph_truck_advantage": arrays["graph_advantage"],
        "pairwise_min_margin": arrays["pairwise_min_margin"],
        "pairwise_agreement": arrays["pairwise_agreement"],
        "semantic_consensus": consensus,
        "semantic_density_consensus": consensus_density,
    }
    for name, values in rankers.items():
        for cid, value in enumerate(values):
            components[cid].setdefault("label_free_scores", {})[name] = float(value)
    for cid, value in enumerate(consensus_density):
        components[cid]["component_level_truck_score"] = float(value)
    return components, rankers


def classification_metrics(pred, labels):
    overall = float(np.mean(pred == labels) * 100.0)
    per_class = []
    for c in range(NUM_CLASSES):
        mask = labels == c
        per_class.append(float(np.mean(pred[mask] == c) * 100.0) if np.any(mask) else 0.0)
    return {
        "overall": overall,
        "mean_class": float(np.mean(per_class)),
        "per_class": {CLASSES[c]: per_class[c] for c in range(NUM_CLASSES)},
    }


def selection_evaluation(component_labels, rank_values, target_labels, candidate_indices, graph_top1, fractions):
    n_components = int(component_labels.max()) + 1
    order = np.argsort(-np.asarray(rank_values))
    total_true_truck = int(np.sum(target_labels == TRUCK))
    results = []
    for frac in fractions:
        count = max(1, int(math.ceil(frac * n_components)))
        selected_components = order[:count]
        selected_local = np.isin(component_labels, selected_components)
        selected_global = candidate_indices[selected_local]
        true_selected = target_labels[selected_global]
        tp = int(np.sum(true_selected == TRUCK))
        precision = float(tp / max(len(selected_global), 1))
        recall = float(tp / max(total_true_truck, 1))
        candidate_trucks = int(np.sum(target_labels[candidate_indices] == TRUCK))
        candidate_conditional_recall = float(tp / max(candidate_trucks, 1))
        contamination = {}
        for c in [CAR, BUS, TRAIN]:
            contamination[CLASSES[c]] = int(np.sum(true_selected == c))
        contamination["other"] = int(len(true_selected) - tp - sum(contamination.values()))
        pred = graph_top1.copy()
        pred[selected_global] = TRUCK
        metrics = classification_metrics(pred, target_labels)
        results.append({
            "top_component_fraction": float(frac),
            "selected_component_count": int(count),
            "selected_sample_count": int(len(selected_global)),
            "selected_candidate_sample_fraction": float(len(selected_global) / max(len(candidate_indices), 1)),
            "truck_true_positives": tp,
            "truck_precision": precision,
            "truck_recall_all_target": recall,
            "truck_recall_within_candidate_region": candidate_conditional_recall,
            "contamination_counts": contamination,
            "converted_prediction_metrics": metrics,
        })
    return results


def oracle_component_evaluation(component_labels, target_labels, candidate_indices, fractions):
    n_components = int(component_labels.max()) + 1
    candidate_labels = target_labels[candidate_indices]
    purities = np.zeros(n_components, dtype=np.float64)
    sizes = np.zeros(n_components, dtype=np.int64)
    truck_counts = np.zeros(n_components, dtype=np.int64)
    for cid in range(n_components):
        local = component_labels == cid
        sizes[cid] = int(np.sum(local))
        truck_counts[cid] = int(np.sum(candidate_labels[local] == TRUCK))
        purities[cid] = truck_counts[cid] / max(sizes[cid], 1)
    order = np.lexsort((-sizes, -purities))
    total_candidate_trucks = int(np.sum(candidate_labels == TRUCK))
    total_target_trucks = int(np.sum(target_labels == TRUCK))
    concentration = []
    for frac in fractions:
        count = max(1, int(math.ceil(frac * n_components)))
        chosen = order[:count]
        tp = int(truck_counts[chosen].sum())
        selected = int(sizes[chosen].sum())
        concentration.append({
            "top_component_fraction": float(frac),
            "selected_component_count": int(count),
            "selected_sample_count": selected,
            "truck_precision": float(tp / max(selected, 1)),
            "truck_recall_all_target": float(tp / max(total_target_trucks, 1)),
            "truck_recall_within_candidate_region": float(tp / max(total_candidate_trucks, 1)),
        })
    weighted_purity = float(np.sum(truck_counts * purities) / max(total_candidate_trucks, 1))
    high_purity_mask = (purities >= 0.5) & (sizes >= 5)
    high_purity_capture = float(truck_counts[high_purity_mask].sum() / max(total_candidate_trucks, 1))
    hhi = float(np.sum((truck_counts / max(total_candidate_trucks, 1)) ** 2))
    return purities, sizes, truck_counts, concentration, weighted_purity, high_purity_capture, hhi


def average_precision_components(rank_values, truck_counts, sizes):
    order = np.argsort(-np.asarray(rank_values))
    total_trucks = int(truck_counts.sum())
    tp = 0
    fp = 0
    prev_recall = 0.0
    ap = 0.0
    for cid in order:
        tp += int(truck_counts[cid])
        fp += int(sizes[cid] - truck_counts[cid])
        recall = tp / max(total_trucks, 1)
        precision = tp / max(tp + fp, 1)
        ap += precision * max(0.0, recall - prev_recall)
        prev_recall = recall
    return float(ap)


def enrich_components_with_labels(components, purities, sizes, truck_counts, target_labels, candidate_indices, component_labels):
    candidate_labels = target_labels[candidate_indices]
    for cid, component in enumerate(components):
        local = component_labels == cid
        labels = candidate_labels[local]
        counts = np.bincount(labels, minlength=NUM_CLASSES)
        component["evaluation_only"] = {
            "true_truck_count": int(truck_counts[cid]),
            "true_truck_purity": float(purities[cid]),
            "true_label_composition": {CLASSES[c]: int(counts[c]) for c in range(NUM_CLASSES) if counts[c] > 0},
        }


def plot_size_distribution(sizes, path):
    plt.figure(figsize=(8, 5))
    plt.hist(sizes, bins=min(50, max(10, int(math.sqrt(len(sizes))))))
    plt.xlabel("Component size")
    plt.ylabel("Number of components")
    plt.title("Truck-candidate component size distribution")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_largest_composition(components, path, top_n=20):
    largest = sorted(components, key=lambda x: x["size"], reverse=True)[:top_n]
    ids = [str(x["component_id"]) for x in largest]
    truck = []
    car = []
    bus = []
    train = []
    other = []
    for comp in largest:
        counts = comp["evaluation_only"]["true_label_composition"]
        n = max(comp["size"], 1)
        truck.append(counts.get("truck", 0) / n)
        car.append(counts.get("car", 0) / n)
        bus.append(counts.get("bus", 0) / n)
        train.append(counts.get("train", 0) / n)
        other.append(1.0 - truck[-1] - car[-1] - bus[-1] - train[-1])
    x = np.arange(len(ids))
    plt.figure(figsize=(12, 6))
    bottom = np.zeros(len(ids))
    for values, label in [(truck, "truck"), (car, "car"), (bus, "bus"), (train, "train"), (other, "other")]:
        plt.bar(x, values, bottom=bottom, label=label)
        bottom += np.asarray(values)
    plt.xticks(x, ids, rotation=45)
    plt.xlabel("Component ID")
    plt.ylabel("True-label fraction, evaluation only")
    plt.title("Largest candidate components")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_score_vs_purity(rank_values, purities, sizes, ranker_name, path):
    plt.figure(figsize=(7, 6))
    marker_sizes = 10.0 + 90.0 * np.sqrt(sizes / max(float(np.max(sizes)), 1.0))
    plt.scatter(rank_values, purities, s=marker_sizes, alpha=0.65)
    plt.xlabel(ranker_name)
    plt.ylabel("True truck purity, evaluation only")
    plt.title("Label-free component score vs truck purity")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_precision_recall(rankers, component_labels, target_labels, candidate_indices, path):
    candidate_labels = target_labels[candidate_indices]
    total_target_trucks = max(int(np.sum(target_labels == TRUCK)), 1)
    plt.figure(figsize=(8, 6))
    for name, values in rankers.items():
        order = np.argsort(-np.asarray(values))
        precisions = []
        recalls = []
        selected = np.zeros(len(component_labels), dtype=bool)
        for cid in order:
            selected |= component_labels == cid
            labels = candidate_labels[selected]
            tp = int(np.sum(labels == TRUCK))
            precisions.append(tp / max(len(labels), 1))
            recalls.append(tp / total_target_trucks)
        plt.plot(recalls, precisions, label=name)
    plt.xlabel("Truck recall over all target trucks")
    plt.ylabel("Truck precision")
    plt.title("Component-ranking precision-recall")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def serialize_number(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)}")


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("VISDA-2017 TRUCK COMPONENT-LEVEL TARGET SEMANTIC STRUCTURE PROBE")
    print("=" * 96)
    print(f"device={device}")
    print(f"seed={args.seed}")
    print(f"rpc_checkpoint={args.rpc_checkpoint}")
    print(f"graph_outputs={args.graph_outputs}")
    print(f"local_k={args.local_k}")
    print(f"edge_quantile={args.edge_quantile}")
    print("Loading graph semantic outputs...")
    graph_payload = safe_torch_load(args.graph_outputs, map_location="cpu")
    anchor_name, anchor_scores, diffusion_name, diffusion_scores, graph_arrays = find_graph_scores(graph_payload)
    n_target = diffusion_scores.shape[0]
    print(f"anchor_scores={anchor_name} {anchor_scores.shape}")
    print(f"diffusion_scores={diffusion_name} {diffusion_scores.shape}")
    graph_top1 = diffusion_scores.argmax(axis=1).astype(np.int64)
    graph_top3 = np.argsort(-diffusion_scores, axis=1)[:, :3]
    candidate_mask = np.any(graph_top3 == TRUCK, axis=1)
    candidate_indices = np.flatnonzero(candidate_mask).astype(np.int64)
    print(f"candidate_samples={len(candidate_indices)}")
    if len(candidate_indices) < 10:
        raise RuntimeError("Truck Top-3 candidate region is unexpectedly small")
    print("Loading RPC model...")
    model, rpc_payload = load_rpc_model(args.rpc_checkpoint, device)
    adapted_name, adapted_target = find_adapted_target(graph_arrays, n_target)
    if adapted_target is None:
        print("Adapted target features were not found in graph outputs; reconstructing 512-D RPC-adapted target features...")
        raw_target = load_cache_features(args.target_cache)
        if len(raw_target) != n_target:
            raise RuntimeError(f"Target cache count {len(raw_target)} does not match graph output count {n_target}")
        adapted_target = encode_raw(model, raw_target, device, args.eval_batch)
        del raw_target
    else:
        print(f"Using adapted target features from {adapted_name}")
    candidate_features = adapted_target[candidate_indices]
    print(f"candidate_feature_shape={candidate_features.shape}")
    print("Building source truck/car/bus/train sub-prototypes...")
    prototypes = build_source_prototypes(model, args.source_cache, args.source_support, args.anchor_k, args.seed, args.prototype_iters, device, args.eval_batch)
    print("Loading existing pointwise pairwise scores...")
    pairwise_path = args.pairwise_outputs
    if pairwise_path is None:
        discovered = discover_pairwise_file(Path("checkpoints"))
        selected = None
        errors = []
        for path in discovered:
            try:
                pairwise, pairwise_modes, pairwise_names = load_pairwise_scores(path, n_target, candidate_indices)
                selected = path
                break
            except Exception as exc:
                errors.append(f"{path}: {exc}")
        if selected is None:
            preview = "\n".join(errors[:8])
            raise RuntimeError("Could not auto-discover a saved pointwise truck-vs-car/bus/train score file. Pass --pairwise-outputs explicitly. Discovery errors:\n" + preview)
        pairwise_path = selected
    pairwise, pairwise_modes, pairwise_names = load_pairwise_scores(pairwise_path, n_target, candidate_indices)
    print(f"pairwise_outputs={pairwise_path}")
    for key in ["car", "bus", "train"]:
        print(f"  truck_vs_{key}: key={pairwise_names[key]} mode={pairwise_modes[key]} n={len(pairwise[key])}")
    print("Computing exact cosine KNN inside truck Top-3 candidate region...")
    t0 = time.time()
    knn_indices, knn_values = exact_knn_cosine(candidate_features, args.local_k, args.knn_chunk, device)
    print(f"local_knn_time={time.time() - t0:.2f}s")
    src, dst, sim = mutual_edges(knn_indices, knn_values)
    if len(sim) == 0:
        raise RuntimeError("No mutual KNN edges found in candidate region")
    threshold = float(np.quantile(sim, args.edge_quantile))
    print(f"mutual_edges={len(sim)}")
    print(f"mutual_similarity_mean={float(np.mean(sim)):.6f}")
    print(f"component_similarity_threshold={threshold:.6f}")
    cc_labels, retained_edge_count = connected_component_labels(len(candidate_indices), src, dst, sim, threshold)
    cc_labels = relabel_by_size(cc_labels)
    component_labels, construction_method, cc_stats, fallback_stats = choose_partition(candidate_features, cc_labels, args)
    print(f"construction_method={construction_method}")
    print(f"cc_components={cc_stats['num_components']} cc_largest_fraction={cc_stats['largest_component_fraction']:.4f} cc_singleton_fraction={cc_stats['singleton_sample_fraction']:.4f}")
    if fallback_stats is not None:
        print(f"fallback_components={fallback_stats['num_components']} fallback_largest_fraction={fallback_stats['largest_component_fraction']:.4f}")
    print("Computing label-free component descriptors and rankers...")
    components, rankers = compute_component_metrics(
        component_labels,
        candidate_features,
        candidate_indices,
        anchor_scores,
        diffusion_scores,
        graph_top1,
        pairwise,
        prototypes,
        src,
        dst,
        sim,
        args.local_k,
    )
    frozen_rankers = {name: values.copy() for name, values in rankers.items()}
    frozen_component_labels = component_labels.copy()
    print("Component construction and all label-free scores are frozen. Loading target labels for evaluation only...")
    target_labels = load_cache_labels(args.target_cache)
    if len(target_labels) != n_target:
        raise RuntimeError(f"Target label count {len(target_labels)} does not match graph output count {n_target}")
    candidate_true = target_labels[candidate_indices]
    true_trucks = int(np.sum(target_labels == TRUCK))
    candidate_true_trucks = int(np.sum(candidate_true == TRUCK))
    candidate_recall = float(candidate_true_trucks / max(true_trucks, 1))
    candidate_precision = float(candidate_true_trucks / max(len(candidate_indices), 1))
    print(f"candidate_truck_recall={candidate_recall * 100:.2f}%")
    print(f"candidate_truck_precision={candidate_precision * 100:.2f}%")
    fractions = [0.01, 0.05, 0.10, 0.20]
    purities, sizes, truck_counts, oracle_concentration, weighted_purity, high_purity_capture, hhi = oracle_component_evaluation(
        frozen_component_labels,
        target_labels,
        candidate_indices,
        fractions,
    )
    enrich_components_with_labels(components, purities, sizes, truck_counts, target_labels, candidate_indices, frozen_component_labels)
    ranker_evaluations = {}
    ranker_ap = {}
    for name, values in frozen_rankers.items():
        ranker_evaluations[name] = selection_evaluation(frozen_component_labels, values, target_labels, candidate_indices, graph_top1, fractions)
        ranker_ap[name] = average_precision_components(values, truck_counts, sizes)
    best_ranker = max(ranker_ap, key=ranker_ap.get)
    best_eval = ranker_evaluations[best_ranker]
    baseline_metrics = classification_metrics(graph_top1, target_labels)
    best_mca_entry = None
    for name, entries in ranker_evaluations.items():
        for entry in entries:
            candidate = {
                "ranker": name,
                "top_component_fraction": entry["top_component_fraction"],
                "mean_class": entry["converted_prediction_metrics"]["mean_class"],
                "overall": entry["converted_prediction_metrics"]["overall"],
                "truck_precision": entry["truck_precision"],
                "truck_recall": entry["truck_recall_all_target"],
                "entry": entry,
            }
            if best_mca_entry is None or candidate["mean_class"] > best_mca_entry["mean_class"]:
                best_mca_entry = candidate
    pure_components = sorted(components, key=lambda c: (c["evaluation_only"]["true_truck_purity"], c["size"]), reverse=True)
    largest_components = sorted(components, key=lambda c: c["size"], reverse=True)
    top_pure = [c for c in pure_components if c["size"] >= 5][:20]
    top_largest = largest_components[:20]
    if weighted_purity >= 0.50 and high_purity_capture >= 0.50:
        structure_state = "coherent_truck_components_exist"
    elif weighted_purity < 0.35 and high_purity_capture < 0.30:
        structure_state = "truck_is_dispersed_and_mixed"
    else:
        structure_state = "mixed_intermediate_structure"
    best_at_20 = None
    for name, entries in ranker_evaluations.items():
        entry = next(x for x in entries if abs(x["top_component_fraction"] - 0.20) < 1e-9)
        if best_at_20 is None or entry["truck_recall_all_target"] > best_at_20["truck_recall_all_target"]:
            best_at_20 = {"ranker": name, **entry}
    if structure_state == "coherent_truck_components_exist" and best_at_20["truck_precision"] >= 0.50 and best_at_20["truck_recall_all_target"] >= 0.20:
        decision = "proceed_to_component_level_semantic_assignment"
    elif structure_state == "coherent_truck_components_exist":
        decision = "components_are_coherent_but_current_label_free_scores_do_not_identify_them_reliably"
    else:
        decision = "abandon_component_rescue_and_move_to_semantic_transport"
    print("\nCOMPONENT STRUCTURE SUMMARY")
    print("=" * 96)
    print(f"components={len(components)}")
    print(f"component_size_min={int(np.min(sizes))} median={float(np.median(sizes)):.1f} mean={float(np.mean(sizes)):.1f} max={int(np.max(sizes))}")
    print(f"truck_weighted_component_purity={weighted_purity * 100:.2f}%")
    print(f"truck_capture_in_components_purity_ge_50pct_size_ge_5={high_purity_capture * 100:.2f}%")
    print(f"truck_component_hhi={hhi:.6f}")
    print(f"best_label_free_ranker_by_component_AP={best_ranker} AP={ranker_ap[best_ranker]:.6f}")
    print(f"best_component_conversion_MCA={best_mca_entry['mean_class']:.2f}% OA={best_mca_entry['overall']:.2f}% ranker={best_mca_entry['ranker']} top_components={best_mca_entry['top_component_fraction'] * 100:.0f}%")
    print("\nLARGEST COMPONENTS")
    for comp in top_largest[:10]:
        ev = comp["evaluation_only"]
        print(f"component={comp['component_id']:4d} size={comp['size']:5d} purity={ev['true_truck_purity'] * 100:6.2f}% truck={ev['true_truck_count']:4d} score={comp['component_level_truck_score']:+.4f}")
    print("\nHIGHEST-PURITY COMPONENTS, TARGET LABELS FOR EVALUATION ONLY")
    for comp in top_pure[:10]:
        ev = comp["evaluation_only"]
        print(f"component={comp['component_id']:4d} size={comp['size']:5d} purity={ev['true_truck_purity'] * 100:6.2f}% truck={ev['true_truck_count']:4d} score={comp['component_level_truck_score']:+.4f}")
    print(f"structure_state={structure_state}")
    print(f"scientific_decision={decision}")
    print("\nTRUCK RECALL AT TOP COMPONENT FRACTIONS")
    for entry in best_eval:
        contamination = entry["contamination_counts"]
        print(f"{best_ranker:34s} top {entry['top_component_fraction'] * 100:>4.0f}% | precision={entry['truck_precision'] * 100:6.2f}% | recall={entry['truck_recall_all_target'] * 100:6.2f}% | candidate-recall={entry['truck_recall_within_candidate_region'] * 100:6.2f}% | samples={entry['selected_sample_count']} | car={contamination['car']} bus={contamination['bus']} train={contamination['train']} other={contamination['other']}")
    print("\nORACLE PURITY CONCENTRATION, EVALUATION ONLY")
    for entry in oracle_concentration:
        print(f"top {entry['top_component_fraction'] * 100:>4.0f}% purity components | precision={entry['truck_precision'] * 100:6.2f}% | recall={entry['truck_recall_all_target'] * 100:6.2f}% | candidate-recall={entry['truck_recall_within_candidate_region'] * 100:6.2f}%")
    size_png = args.output_dir / f"component_size_distribution_seed{args.seed}.png"
    largest_png = args.output_dir / f"largest_component_composition_seed{args.seed}.png"
    score_png = args.output_dir / f"best_label_free_score_vs_purity_seed{args.seed}.png"
    pr_png = args.output_dir / f"component_ranking_precision_recall_seed{args.seed}.png"
    plot_size_distribution(sizes, size_png)
    plot_largest_composition(components, largest_png)
    plot_score_vs_purity(frozen_rankers[best_ranker], purities, sizes, best_ranker, score_png)
    plot_precision_recall(frozen_rankers, frozen_component_labels, target_labels, candidate_indices, pr_png)
    size_summary = {
        "min": int(np.min(sizes)),
        "q01": float(np.quantile(sizes, 0.01)),
        "q05": float(np.quantile(sizes, 0.05)),
        "q25": float(np.quantile(sizes, 0.25)),
        "median": float(np.median(sizes)),
        "mean": float(np.mean(sizes)),
        "q75": float(np.quantile(sizes, 0.75)),
        "q95": float(np.quantile(sizes, 0.95)),
        "q99": float(np.quantile(sizes, 0.99)),
        "max": int(np.max(sizes)),
    }
    summary = {
        "experiment": "visda_truck_component_structure_probe",
        "seed": args.seed,
        "constraints": {
            "model_training": False,
            "target_labels_used_in_construction": False,
            "target_labels_used_in_scoring": False,
            "target_labels_loaded_only_after_components_and_rankers_frozen": True,
            "rpc_modified": False,
            "graph_diffusion_retrained": False,
            "pseudo_label_loss": False,
        },
        "inputs": {
            "rpc_checkpoint": str(args.rpc_checkpoint),
            "graph_outputs": str(args.graph_outputs),
            "pairwise_outputs": str(pairwise_path),
            "anchor_tensor_key": anchor_name,
            "diffusion_tensor_key": diffusion_name,
            "pairwise_tensor_keys": pairwise_names,
            "pairwise_score_modes": pairwise_modes,
            "target_count": int(n_target),
        },
        "construction": {
            "candidate_rule": "truck_in_graph_top3",
            "candidate_sample_count": int(len(candidate_indices)),
            "local_k": args.local_k,
            "mutual_edge_count": int(len(sim)),
            "mutual_edge_similarity_mean": float(np.mean(sim)),
            "edge_quantile": args.edge_quantile,
            "edge_similarity_threshold": threshold,
            "thresholded_edge_count": int(retained_edge_count),
            "construction_method": construction_method,
            "connected_component_partition": cc_stats,
            "fallback_partition": fallback_stats,
            "final_component_count": int(len(components)),
            "component_size_distribution": size_summary,
        },
        "baseline": {
            "graph_metrics": baseline_metrics,
            "candidate_truck_recall": candidate_recall,
            "candidate_truck_precision": candidate_precision,
            "true_trucks_total": true_trucks,
            "true_trucks_in_candidate_region": candidate_true_trucks,
        },
        "evaluation_only_component_structure": {
            "truck_weighted_component_purity": weighted_purity,
            "truck_capture_in_components_purity_ge_0_5_size_ge_5": high_purity_capture,
            "truck_component_hhi": hhi,
            "oracle_purity_concentration": oracle_concentration,
            "highest_purity_components_size_ge_5": top_pure,
            "largest_components": top_largest,
        },
        "label_free_rankers": {
            "component_average_precision_evaluation_only": ranker_ap,
            "best_ranker_evaluation_only": best_ranker,
            "selection_results": ranker_evaluations,
            "best_mca_conversion_evaluation_only": best_mca_entry,
            "best_top20_recall_evaluation_only": best_at_20,
        },
        "scientific_decision": {
            "structure_state": structure_state,
            "decision": decision,
            "criteria": {
                "coherent": "truck_weighted_component_purity >= 0.50 and >=50% of candidate true trucks lie in components with purity >=0.50 and size >=5",
                "dispersed": "truck_weighted_component_purity < 0.35 and <30% of candidate true trucks lie in components with purity >=0.50 and size >=5",
                "component_assignment_go": "coherent plus a predefined label-free ranking at top 20% components achieves >=50% truck precision and >=20% recall over all target trucks",
            },
        },
        "components": components,
        "diagnostic_pngs": [str(size_png), str(largest_png), str(score_png), str(pr_png)],
    }
    json_path = args.output_dir / f"truck_component_structure_probe_seed{args.seed}.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=serialize_number)
    compact_path = args.output_dir / f"truck_component_assignments_seed{args.seed}.npz"
    np.savez_compressed(
        compact_path,
        candidate_indices=candidate_indices,
        component_labels=frozen_component_labels,
        component_purity_evaluation_only=purities,
        component_sizes=sizes,
        **{f"ranker_{name}": values for name, values in frozen_rankers.items()},
    )
    print("\nOUTPUTS")
    print("=" * 96)
    print(f"json={json_path}")
    print(f"assignments={compact_path}")
    print(f"png={size_png}")
    print(f"png={largest_png}")
    print(f"png={score_png}")
    print(f"png={pr_png}")


if __name__ == "__main__":
    main()
