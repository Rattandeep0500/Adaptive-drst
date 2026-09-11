import argparse
import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
NUM_CLASSES = 12
INPUT_DIM = 2048
HIDDEN_DIM = 512
TARGET_COUNT = 55388
BATCH_SIZE = 512
EVAL_BATCH_SIZE = 4096
EPOCHS = 2
LR_ADAPTER = 0.001
LR_CLASSIFIER = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
KAPPA = 2.0
EPS = 1e-12
ECE_BINS = 15
CATASTROPHIC_CLASS_DROP_PP = 10.0
STRONG_CLASS_ACCURACY_THRESHOLD = 80.0

CACHE_ROOT = Path("checkpoints/visda_feature_cache")
SOURCE_CACHE = CACHE_ROOT / "source"
TARGET_CACHE = CACHE_ROOT / "target"
DEFAULT_CHECKPOINT = Path("checkpoints/visda_rpc_causal_experiment/rpc_seed42.pt")
DEFAULT_SUPPORT_ARTIFACT = Path(
    "checkpoints/visda_target_class_conditional_support/target_class_conditional_support_seed42.npz"
)
DEFAULT_OUTPUT_DIR = Path("checkpoints/visda_support_weighted_target_adaptation")

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

ARMS = [
    "vanilla",
    "norm_matched_global_attenuation",
    "confidence_weighted",
    "confidence_support_weighted",
]

FROZEN_PROMOTION_MASSES = torch.tensor(
    [
        0.0233335,
        0.0298017,
        0.0450411,
        0.1275856,
        0.0338252,
        0.0379790,
        0.0512989,
        0.0666487,
        0.0500399,
        0.0291361,
        0.0396248,
        0.0055053,
    ],
    dtype=torch.float64,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def sha256_array(array):
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def clone_state_dict(state_dict):
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def load_source_cache(cache_dir):
    files = sorted(Path(cache_dir).glob("chunk_*.pt"))
    if not files:
        raise RuntimeError(f"No source cache chunks found in {cache_dir}")
    features = []
    labels = []
    for path in files:
        payload = safe_torch_load(path)
        if "features" not in payload or "labels" not in payload:
            raise RuntimeError(f"Source cache chunk {path} must contain features and labels")
        x = payload["features"]
        y = payload["labels"].long()
        if x.ndim != 2 or x.shape[1] != INPUT_DIM:
            raise RuntimeError(f"Expected N x {INPUT_DIM} source features in {path}, got {tuple(x.shape)}")
        if len(x) != len(y):
            raise RuntimeError(f"Source feature/label mismatch in {path}")
        features.append(x.cpu())
        labels.append(y.cpu())
    source_features = torch.cat(features, dim=0)
    source_labels = torch.cat(labels, dim=0)
    if len(source_features) != len(source_labels):
        raise RuntimeError("Final source feature/label count mismatch")
    if source_labels.min().item() < 0 or source_labels.max().item() >= NUM_CLASSES:
        raise RuntimeError("Source labels lie outside the expected VisDA class range")
    return source_features, source_labels


def load_target_features_only(cache_dir):
    files = sorted(Path(cache_dir).glob("chunk_*.pt"))
    if not files:
        raise RuntimeError(f"No target cache chunks found in {cache_dir}")
    features = []
    for path in files:
        payload = safe_torch_load(path)
        if "features" not in payload:
            raise RuntimeError(f"Target cache chunk {path} does not contain features")
        x = payload["features"]
        if x.ndim != 2 or x.shape[1] != INPUT_DIM:
            raise RuntimeError(f"Expected N x {INPUT_DIM} target features in {path}, got {tuple(x.shape)}")
        features.append(x.cpu())
        del payload
    target_features = torch.cat(features, dim=0)
    if len(target_features) != TARGET_COUNT:
        raise RuntimeError(f"Target feature cache must contain {TARGET_COUNT} samples, got {len(target_features)}")
    return target_features


def load_target_labels_for_evaluation(cache_dir):
    files = sorted(Path(cache_dir).glob("chunk_*.pt"))
    if not files:
        raise RuntimeError(f"No target cache chunks found in {cache_dir}")
    labels = []
    for path in files:
        payload = safe_torch_load(path)
        if "labels" not in payload:
            raise RuntimeError(f"Target cache chunk {path} does not contain labels")
        y = payload["labels"].long().cpu()
        labels.append(y)
    target_labels = torch.cat(labels, dim=0)
    if len(target_labels) != TARGET_COUNT:
        raise RuntimeError(f"Target label cache must contain {TARGET_COUNT} samples, got {len(target_labels)}")
    if target_labels.min().item() < 0 or target_labels.max().item() >= NUM_CLASSES:
        raise RuntimeError("Target labels lie outside the expected VisDA class range")
    return target_labels


def load_support_artifact(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing support artifact: {path}")
    with np.load(path, allow_pickle=False) as payload:
        keys = list(payload.files)
        required = [
            "support_probabilities",
            "support_top1",
            "support_top1_score",
            "support_margin",
            "support_entropy",
            "support_reliability",
            "teacher_probabilities",
            "teacher_prediction",
            "teacher_confidence",
            "teacher_support_agreement",
        ]
        missing = [key for key in required if key not in payload]
        if missing:
            raise RuntimeError(f"Support artifact is missing keys {missing}; available keys are {keys}")
        support_probabilities = np.asarray(payload["support_probabilities"], dtype=np.float32)
        stored_support_margin = np.asarray(payload["support_margin"], dtype=np.float32).reshape(-1)
        support_reliability = np.asarray(payload["support_reliability"], dtype=np.float32).reshape(-1)
    if support_probabilities.shape != (TARGET_COUNT, NUM_CLASSES):
        raise RuntimeError(
            f"support_probabilities must have shape {(TARGET_COUNT, NUM_CLASSES)}, got {support_probabilities.shape}"
        )
    if stored_support_margin.shape != (TARGET_COUNT,):
        raise RuntimeError(f"support_margin must have shape {(TARGET_COUNT,)}, got {stored_support_margin.shape}")
    if support_reliability.shape != (TARGET_COUNT,):
        raise RuntimeError(f"support_reliability must have shape {(TARGET_COUNT,)}, got {support_reliability.shape}")
    if not np.all(np.isfinite(support_probabilities)):
        raise RuntimeError("support_probabilities contains non-finite values")
    if np.any(support_probabilities < -1e-7):
        raise RuntimeError("support_probabilities contains negative values")
    row_sums = support_probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-4, atol=1e-5):
        raise RuntimeError(
            f"support_probabilities rows do not sum to 1 within tolerance: min={row_sums.min():.8f} max={row_sums.max():.8f}"
        )
    return {
        "keys": keys,
        "support_probabilities": support_probabilities,
        "stored_support_margin": stored_support_margin,
        "support_reliability": support_reliability,
    }


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != INPUT_DIM:
            raise RuntimeError(f"Adapter expected N x {INPUT_DIM}, got {tuple(x.shape)}")
        z = self.net(x)
        if z.ndim != 2 or z.shape[1] != HIDDEN_DIM:
            raise RuntimeError(f"Adapter produced invalid shape {tuple(z.shape)}")
        return z


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


def model_from_state(state_dict, device):
    model = MCDModel().to(device)
    model.load_state_dict(copy.deepcopy(state_dict), strict=True)
    return model


def load_initial_state(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    checkpoint = safe_torch_load(path)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("RPC checkpoint is not a dictionary")
    if "student_state_dict" not in checkpoint:
        raise RuntimeError("RPC checkpoint does not contain student_state_dict")
    state = checkpoint["student_state_dict"]
    verifier = MCDModel()
    verifier.load_state_dict(state, strict=True)
    return clone_state_dict(state), sorted(checkpoint.keys())


def make_optimizer_pair(model):
    optimizer_adapter = torch.optim.SGD(
        model.adapter.parameters(),
        lr=LR_ADAPTER,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    optimizer_classifier = torch.optim.SGD(
        list(model.classifier1.parameters()) + list(model.classifier2.parameters()),
        lr=LR_CLASSIFIER,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    return optimizer_adapter, optimizer_classifier


def make_epoch_batches(num_examples, batch_size, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(num_examples, generator=generator)
    usable = (num_examples // batch_size) * batch_size
    if usable == 0:
        raise RuntimeError("Dataset is smaller than one full batch")
    return permutation[:usable].view(-1, batch_size)


def build_schedules(n_source, n_target):
    schedules = []
    for epoch in range(1, EPOCHS + 1):
        source_batches = make_epoch_batches(n_source, BATCH_SIZE, SEED + 1000 * epoch + 11)
        target_batches = make_epoch_batches(n_target, BATCH_SIZE, SEED + 1000 * epoch + 29)
        steps = int(source_batches.shape[0])
        exposure = torch.cat(
            [target_batches[step % int(target_batches.shape[0])] for step in range(steps)],
            dim=0,
        )
        selected_indices = torch.unique(target_batches.reshape(-1), sorted=True)
        schedules.append(
            {
                "epoch": epoch,
                "source_batches": source_batches,
                "target_batches": target_batches,
                "steps": steps,
                "target_exposure_indices": exposure,
                "selected_target_indices": selected_indices,
                "source_batch_hash": sha256_array(source_batches.numpy().astype(np.int64, copy=False)),
                "target_batch_hash": sha256_array(target_batches.numpy().astype(np.int64, copy=False)),
                "target_exposure_hash": sha256_array(exposure.numpy().astype(np.int64, copy=False)),
                "selected_index_hash": sha256_array(selected_indices.numpy().astype(np.int64, copy=False)),
            }
        )
    return schedules


def fetch_features(features, indices, device):
    return features.index_select(0, indices).to(device=device, dtype=torch.float32)


def fetch_labels(labels, indices, device):
    return labels.index_select(0, indices).to(device=device, dtype=torch.long)


def parameter_tuple(module):
    return tuple(parameter for parameter in module.parameters() if parameter.requires_grad)


def all_pseudo_label_parameters(model):
    return parameter_tuple(model.adapter) + parameter_tuple(model.classifier1) + parameter_tuple(model.classifier2)


def assign_parameter_grads(parameters, gradients):
    if len(parameters) != len(gradients):
        raise RuntimeError("Parameter/gradient length mismatch")
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = gradient.detach().clone()


def grad_l2_norm(gradients):
    total = torch.tensor(0.0, dtype=torch.float64)
    for gradient in gradients:
        total = total + gradient.detach().double().pow(2).sum().cpu()
    return float(torch.sqrt(total).item())


def split_grad_norms(model, gradients):
    adapter_count = len(parameter_tuple(model.adapter))
    adapter_gradients = gradients[:adapter_count]
    classifier_gradients = gradients[adapter_count:]
    return grad_l2_norm(adapter_gradients), grad_l2_norm(classifier_gradients), grad_l2_norm(gradients)


def scale_gradients(gradients, scale):
    return tuple(gradient * float(scale) for gradient in gradients)


def discrepancy(logits1, logits2):
    p1 = F.softmax(logits1, dim=1)
    p2 = F.softmax(logits2, dim=1)
    return (p1 - p2).abs().mean()


def frozen_rpc_alphas(device, dtype=torch.float32):
    values = FROZEN_PROMOTION_MASSES.clone()
    sorted_values, _ = torch.sort(values)
    lower_median = sorted_values[(len(sorted_values) - 1) // 2]
    cap = KAPPA * lower_median
    alpha = torch.minimum(torch.ones_like(values), cap / values.clamp_min(EPS))
    return alpha.to(device=device, dtype=dtype), float(lower_median.item()), float(cap.item())


def source_adapter_step(model, optimizer_adapter, source_x, source_y):
    optimizer_adapter.zero_grad(set_to_none=True)
    logits1, logits2 = model(source_x)
    loss = F.cross_entropy(logits1, source_y) + F.cross_entropy(logits2, source_y)
    parameters = parameter_tuple(model.adapter)
    gradients = torch.autograd.grad(loss, parameters, retain_graph=False, create_graph=False)
    assign_parameter_grads(parameters, gradients)
    optimizer_adapter.step()
    return float(loss.detach().item())


def classifier_adversarial_step(model, optimizer_classifier, source_x, source_y, target_x):
    optimizer_classifier.zero_grad(set_to_none=True)
    with torch.no_grad():
        source_z = model.encode(source_x)
        target_z = model.encode(target_x)
    source_logits1 = model.classifier1(source_z)
    source_logits2 = model.classifier2(source_z)
    target_logits1 = model.classifier1(target_z)
    target_logits2 = model.classifier2(target_z)
    source_loss = F.cross_entropy(source_logits1, source_y) + F.cross_entropy(source_logits2, source_y)
    target_discrepancy = discrepancy(target_logits1, target_logits2)
    objective = source_loss - target_discrepancy
    parameters = parameter_tuple(model.classifier1) + parameter_tuple(model.classifier2)
    gradients = torch.autograd.grad(objective, parameters, retain_graph=False, create_graph=False)
    assign_parameter_grads(parameters, gradients)
    optimizer_classifier.step()
    return float(source_loss.detach().item()), float(target_discrepancy.detach().item())


def logit_gradient_components(logit_gradient):
    promotion = (-logit_gradient).clamp_min(0.0)
    suppression = logit_gradient.clamp_min(0.0)
    return promotion, suppression


def rpc_transform_single_head(logit_gradient, alpha):
    promotion, suppression = logit_gradient_components(logit_gradient)
    clipped_promotion = promotion * alpha.view(1, -1)
    clipped_promotion_total = clipped_promotion.sum(dim=1, keepdim=True)
    suppression_total = suppression.sum(dim=1, keepdim=True)
    suppression_scale = torch.where(
        suppression_total > EPS,
        clipped_promotion_total / suppression_total.clamp_min(EPS),
        torch.zeros_like(suppression_total),
    )
    suppression_scale = suppression_scale.clamp(min=0.0, max=1.0)
    transformed = suppression * suppression_scale - clipped_promotion
    max_amplification = float((transformed.abs() - logit_gradient.abs()).max().detach().item())
    if max_amplification > 1e-10:
        raise RuntimeError(f"RPC amplified a logit-gradient component by {max_amplification:.12e}")
    return transformed


def target_rpc_adapter_step(model, optimizer_adapter, target_x, alpha):
    optimizer_adapter.zero_grad(set_to_none=True)
    target_z = model.encode(target_x)
    target_logits1 = model.classifier1(target_z)
    target_logits2 = model.classifier2(target_z)
    target_discrepancy = discrepancy(target_logits1, target_logits2)
    grad1, grad2 = torch.autograd.grad(
        target_discrepancy,
        (target_logits1, target_logits2),
        retain_graph=True,
        create_graph=False,
    )
    transformed1 = rpc_transform_single_head(grad1.detach(), alpha)
    transformed2 = rpc_transform_single_head(grad2.detach(), alpha)
    adapter_parameters = parameter_tuple(model.adapter)
    gradients = torch.autograd.grad(
        outputs=(target_logits1, target_logits2),
        inputs=adapter_parameters,
        grad_outputs=(transformed1, transformed2),
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )
    assign_parameter_grads(adapter_parameters, gradients)
    optimizer_adapter.step()
    return float(target_discrepancy.detach().item()), grad_l2_norm(gradients)


@torch.no_grad()
def infer_probabilities(model, features, device):
    model.eval()
    outputs = []
    for start in range(0, len(features), EVAL_BATCH_SIZE):
        end = min(start + EVAL_BATCH_SIZE, len(features))
        x = features[start:end].to(device=device, dtype=torch.float32)
        logits1, logits2 = model(x)
        probabilities = (F.softmax(logits1, dim=1) + F.softmax(logits2, dim=1)) / 2.0
        outputs.append(probabilities.cpu())
    probabilities = torch.cat(outputs, dim=0)
    if probabilities.shape != (len(features), NUM_CLASSES):
        raise RuntimeError(f"Probability tensor has invalid shape {tuple(probabilities.shape)}")
    return probabilities


def current_class_support_margin(support_probabilities, predictions):
    rows = np.arange(len(predictions), dtype=np.int64)
    current_support = support_probabilities[rows, predictions]
    other_scores = support_probabilities.copy()
    other_scores[rows, predictions] = -np.inf
    best_other = other_scores.max(axis=1)
    margin = current_support - best_other
    return current_support.astype(np.float32), margin.astype(np.float32)


def build_epoch_pseudo_label_plan(model, target_features, support_probabilities, schedule, device):
    probabilities = infer_probabilities(model, target_features, device)
    confidence_values, prediction_values = probabilities.max(dim=1)
    predictions = prediction_values.numpy().astype(np.int64)
    confidence = confidence_values.numpy().astype(np.float32)
    support_current, support_signed_margin = current_class_support_margin(support_probabilities, predictions)
    support_top1 = support_probabilities.argmax(axis=1).astype(np.int64)
    support_agreement = support_top1 == predictions

    confidence_raw = confidence.astype(np.float64)
    confidence_support_raw = confidence.astype(np.float64) * support_current.astype(np.float64)
    exposure_indices = schedule["target_exposure_indices"].numpy().astype(np.int64, copy=False)
    confidence_mean_exposure = float(confidence_raw[exposure_indices].mean())
    confidence_support_mean_exposure = float(confidence_support_raw[exposure_indices].mean())
    if confidence_mean_exposure <= EPS:
        raise RuntimeError("Confidence weighting has zero exposure mean")
    if confidence_support_mean_exposure <= EPS:
        raise RuntimeError("Confidence-support weighting has zero exposure mean")
    confidence_weight = (confidence_raw / confidence_mean_exposure).astype(np.float32)
    confidence_support_weight = (confidence_support_raw / confidence_support_mean_exposure).astype(np.float32)

    selected_indices = schedule["selected_target_indices"].numpy().astype(np.int64, copy=False)
    if not np.array_equal(selected_indices, np.unique(exposure_indices)):
        raise RuntimeError("Selected target index set does not equal the unique exposure set")

    return {
        "predictions": predictions,
        "confidence": confidence,
        "support_current_class": support_current,
        "support_signed_margin": support_signed_margin,
        "support_agreement": support_agreement,
        "confidence_weight": confidence_weight,
        "confidence_support_weight": confidence_support_weight,
        "selected_indices": selected_indices,
        "pseudo_label_prediction_hash": sha256_array(predictions[selected_indices]),
        "confidence_weight_mean_exposure": float(confidence_weight[exposure_indices].mean()),
        "confidence_support_weight_mean_exposure": float(confidence_support_weight[exposure_indices].mean()),
    }


def pseudo_label_step(
    model,
    optimizer_adapter,
    optimizer_classifier,
    target_x,
    pseudo_y,
    applied_weight,
    counterfactual_support_weight,
    arm,
):
    optimizer_adapter.zero_grad(set_to_none=True)
    optimizer_classifier.zero_grad(set_to_none=True)
    logits1, logits2 = model(target_x)
    ce1 = F.cross_entropy(logits1, pseudo_y, reduction="none")
    ce2 = F.cross_entropy(logits2, pseudo_y, reduction="none")
    per_sample_ce = ce1 + ce2
    vanilla_loss = per_sample_ce.mean()
    weighted_loss = (applied_weight.detach() * per_sample_ce).mean()
    parameters = all_pseudo_label_parameters(model)

    counterfactual_norm = None
    vanilla_norm = None
    global_scale = 1.0
    norm_match_error = 0.0

    if arm == "norm_matched_global_attenuation":
        counterfactual_loss = (counterfactual_support_weight.detach() * per_sample_ce).mean()
        vanilla_gradients = torch.autograd.grad(
            vanilla_loss,
            parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )
        counterfactual_gradients = torch.autograd.grad(
            counterfactual_loss,
            parameters,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )
        vanilla_norm = grad_l2_norm(vanilla_gradients)
        counterfactual_norm = grad_l2_norm(counterfactual_gradients)
        if vanilla_norm <= EPS:
            global_scale = 0.0 if counterfactual_norm <= EPS else 1.0
        else:
            global_scale = counterfactual_norm / vanilla_norm
        gradients = scale_gradients(vanilla_gradients, global_scale)
        applied_loss_value = float((global_scale * vanilla_loss.detach()).item())
        applied_norm = grad_l2_norm(gradients)
        norm_match_error = abs(applied_norm - counterfactual_norm)
        tolerance = 1e-8 + 1e-6 * max(counterfactual_norm, 1.0)
        if norm_match_error > tolerance:
            raise RuntimeError(
                f"Pseudo-label norm matching failed: applied={applied_norm:.12e} "
                f"counterfactual={counterfactual_norm:.12e} error={norm_match_error:.12e}"
            )
    else:
        gradients = torch.autograd.grad(
            weighted_loss,
            parameters,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )
        applied_loss_value = float(weighted_loss.detach().item())

    adapter_norm, classifier_norm, total_norm = split_grad_norms(model, gradients)
    assign_parameter_grads(parameters, gradients)
    optimizer_adapter.step()
    optimizer_classifier.step()

    return {
        "unweighted_pseudo_label_ce": float(vanilla_loss.detach().item()),
        "applied_pseudo_label_loss": applied_loss_value,
        "adapter_gradient_norm": adapter_norm,
        "classifier_gradient_norm": classifier_norm,
        "total_pseudo_label_gradient_norm": total_norm,
        "vanilla_counterfactual_gradient_norm": vanilla_norm,
        "support_weighted_counterfactual_gradient_norm": counterfactual_norm,
        "global_norm_match_scale": float(global_scale),
        "norm_match_error": float(norm_match_error),
    }


def summarize_array(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def run_arm(
    arm,
    initial_state,
    source_features,
    source_labels,
    target_features,
    support_probabilities,
    schedules,
    alpha,
    device,
):
    set_seed(SEED)
    model = model_from_state(initial_state, device)
    optimizer_adapter, optimizer_classifier = make_optimizer_pair(model)
    history = []
    snapshots = []
    plans = []
    arm_start = time.perf_counter()

    for schedule in schedules:
        epoch = schedule["epoch"]
        epoch_start = time.perf_counter()
        plan = build_epoch_pseudo_label_plan(
            model,
            target_features,
            support_probabilities,
            schedule,
            device,
        )
        plans.append(plan)
        model.train()

        source_adapter_loss_sum = 0.0
        source_classifier_loss_sum = 0.0
        classifier_discrepancy_sum = 0.0
        rpc_target_discrepancy_sum = 0.0
        rpc_adapter_gradient_norm_sum = 0.0
        pseudo_unweighted_ce_sum = 0.0
        pseudo_applied_loss_sum = 0.0
        pseudo_adapter_grad_norm_sum = 0.0
        pseudo_classifier_grad_norm_sum = 0.0
        pseudo_total_grad_norm_sum = 0.0
        norm_scale_values = []
        norm_match_error_max = 0.0

        source_batches = schedule["source_batches"]
        target_batches = schedule["target_batches"]
        steps = schedule["steps"]

        for step in range(steps):
            source_indices = source_batches[step]
            target_indices = target_batches[step % int(target_batches.shape[0])]
            source_x = fetch_features(source_features, source_indices, device)
            source_y = fetch_labels(source_labels, source_indices, device)
            target_x = fetch_features(target_features, target_indices, device)

            source_adapter_loss_sum += source_adapter_step(
                model,
                optimizer_adapter,
                source_x,
                source_y,
            )
            classifier_source_loss, classifier_discrepancy = classifier_adversarial_step(
                model,
                optimizer_classifier,
                source_x,
                source_y,
                target_x,
            )
            source_classifier_loss_sum += classifier_source_loss
            classifier_discrepancy_sum += classifier_discrepancy
            target_discrepancy, rpc_adapter_grad_norm = target_rpc_adapter_step(
                model,
                optimizer_adapter,
                target_x,
                alpha,
            )
            rpc_target_discrepancy_sum += target_discrepancy
            rpc_adapter_gradient_norm_sum += rpc_adapter_grad_norm

            target_index_numpy = target_indices.numpy().astype(np.int64, copy=False)
            pseudo_y = torch.from_numpy(plan["predictions"][target_index_numpy]).to(device=device, dtype=torch.long)
            confidence_weight = torch.from_numpy(plan["confidence_weight"][target_index_numpy]).to(
                device=device, dtype=torch.float32
            )
            confidence_support_weight = torch.from_numpy(
                plan["confidence_support_weight"][target_index_numpy]
            ).to(device=device, dtype=torch.float32)

            if arm == "vanilla":
                applied_weight = torch.ones_like(confidence_weight)
            elif arm == "confidence_weighted":
                applied_weight = confidence_weight
            elif arm == "confidence_support_weighted":
                applied_weight = confidence_support_weight
            elif arm == "norm_matched_global_attenuation":
                applied_weight = torch.ones_like(confidence_weight)
            else:
                raise ValueError(f"Unknown arm {arm}")

            pseudo_stats = pseudo_label_step(
                model=model,
                optimizer_adapter=optimizer_adapter,
                optimizer_classifier=optimizer_classifier,
                target_x=target_x,
                pseudo_y=pseudo_y,
                applied_weight=applied_weight,
                counterfactual_support_weight=confidence_support_weight,
                arm=arm,
            )
            pseudo_unweighted_ce_sum += pseudo_stats["unweighted_pseudo_label_ce"]
            pseudo_applied_loss_sum += pseudo_stats["applied_pseudo_label_loss"]
            pseudo_adapter_grad_norm_sum += pseudo_stats["adapter_gradient_norm"]
            pseudo_classifier_grad_norm_sum += pseudo_stats["classifier_gradient_norm"]
            pseudo_total_grad_norm_sum += pseudo_stats["total_pseudo_label_gradient_norm"]
            norm_scale_values.append(pseudo_stats["global_norm_match_scale"])
            norm_match_error_max = max(norm_match_error_max, pseudo_stats["norm_match_error"])

        selected_indices = plan["selected_indices"]
        if arm == "vanilla" or arm == "norm_matched_global_attenuation":
            selected_weights = np.ones(len(selected_indices), dtype=np.float32)
            weighting_formula = "1"
        elif arm == "confidence_weighted":
            selected_weights = plan["confidence_weight"][selected_indices]
            weighting_formula = "c / E_exposure[c]"
        else:
            selected_weights = plan["confidence_support_weight"][selected_indices]
            weighting_formula = "(c * s_current_class) / E_exposure[c * s_current_class]"

        record = {
            "epoch": int(epoch),
            "arm": arm,
            "steps": int(steps),
            "pseudo_label_count_unique": int(len(selected_indices)),
            "pseudo_label_fraction_unique": float(len(selected_indices) / len(target_features)),
            "pseudo_label_exposure_count": int(len(schedule["target_exposure_indices"])),
            "selected_index_hash": schedule["selected_index_hash"],
            "pseudo_label_prediction_hash": plan["pseudo_label_prediction_hash"],
            "source_batch_hash": schedule["source_batch_hash"],
            "target_batch_hash": schedule["target_batch_hash"],
            "target_exposure_hash": schedule["target_exposure_hash"],
            "weighting_formula": weighting_formula,
            "weights_detached": True,
            "pseudo_label_confidence": summarize_array(plan["confidence"][selected_indices]),
            "pseudo_label_support": summarize_array(plan["support_current_class"][selected_indices]),
            "pseudo_label_signed_support_margin": summarize_array(plan["support_signed_margin"][selected_indices]),
            "teacher_support_agreement_rate": float(plan["support_agreement"][selected_indices].mean()),
            "pseudo_label_weight": summarize_array(selected_weights),
            "confidence_weight_mean_exposure": plan["confidence_weight_mean_exposure"],
            "confidence_support_weight_mean_exposure": plan["confidence_support_weight_mean_exposure"],
            "source_adapter_loss_mean": source_adapter_loss_sum / max(steps, 1),
            "source_classifier_loss_mean": source_classifier_loss_sum / max(steps, 1),
            "classifier_target_discrepancy_mean": classifier_discrepancy_sum / max(steps, 1),
            "total_target_discrepancy": rpc_target_discrepancy_sum / max(steps, 1),
            "rpc_adapter_gradient_norm_mean": rpc_adapter_gradient_norm_sum / max(steps, 1),
            "pseudo_label_ce": pseudo_unweighted_ce_sum / max(steps, 1),
            "applied_pseudo_label_loss": pseudo_applied_loss_sum / max(steps, 1),
            "adapter_gradient_norm": pseudo_adapter_grad_norm_sum / max(steps, 1),
            "classifier_gradient_norm": pseudo_classifier_grad_norm_sum / max(steps, 1),
            "total_pseudo_label_gradient_norm": pseudo_total_grad_norm_sum / max(steps, 1),
            "global_norm_match_scale_mean": float(np.mean(norm_scale_values)),
            "global_norm_match_scale_min": float(np.min(norm_scale_values)),
            "global_norm_match_scale_max": float(np.max(norm_scale_values)),
            "norm_match_error_max": float(norm_match_error_max),
            "epoch_seconds": float(time.perf_counter() - epoch_start),
        }
        history.append(record)
        snapshots.append(clone_state_dict(model.state_dict()))
        print(
            f"{arm:34s} epoch {epoch}/{EPOCHS} | "
            f"PL-CE {record['pseudo_label_ce']:.6f} | "
            f"PL-grad {record['total_pseudo_label_gradient_norm']:.6e} | "
            f"conf {record['pseudo_label_confidence']['mean']:.6f} | "
            f"support {record['pseudo_label_support']['mean']:.6f}"
        )

    return {
        "arm": arm,
        "history": history,
        "snapshots": snapshots,
        "plans": plans,
        "final_state": clone_state_dict(model.state_dict()),
        "training_seconds": float(time.perf_counter() - arm_start),
    }


def expected_calibration_error(probabilities, labels, bins=ECE_BINS):
    confidence, predictions = probabilities.max(dim=1)
    correct = predictions.eq(labels).float()
    ece = 0.0
    bin_edges = torch.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        lower = float(bin_edges[index].item())
        upper = float(bin_edges[index + 1].item())
        if index == 0:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence > lower) & (confidence <= upper)
        count = int(mask.sum().item())
        if count == 0:
            continue
        bin_accuracy = float(correct[mask].mean().item())
        bin_confidence = float(confidence[mask].mean().item())
        ece += (count / len(labels)) * abs(bin_accuracy - bin_confidence)
    return float(ece)


@torch.no_grad()
def evaluate_state(state_dict, target_features, target_labels, device):
    model = model_from_state(state_dict, device)
    model.eval()
    probabilities_parts = []
    discrepancy_sum = 0.0
    discrepancy_count = 0
    for start in range(0, len(target_features), EVAL_BATCH_SIZE):
        end = min(start + EVAL_BATCH_SIZE, len(target_features))
        x = target_features[start:end].to(device=device, dtype=torch.float32)
        logits1, logits2 = model(x)
        p1 = F.softmax(logits1, dim=1)
        p2 = F.softmax(logits2, dim=1)
        probabilities_parts.append(((p1 + p2) / 2.0).cpu())
        batch_discrepancy = (p1 - p2).abs().mean(dim=1)
        discrepancy_sum += float(batch_discrepancy.sum().item())
        discrepancy_count += int(len(batch_discrepancy))
    probabilities = torch.cat(probabilities_parts, dim=0)
    labels = target_labels.long().cpu()
    if probabilities.shape != (TARGET_COUNT, NUM_CLASSES):
        raise RuntimeError(f"Evaluation probability shape mismatch: {tuple(probabilities.shape)}")
    predictions = probabilities.argmax(dim=1)
    confidence = probabilities.max(dim=1).values
    correct = predictions.eq(labels)
    overall_accuracy_fraction = float(correct.float().mean().item())
    per_class_accuracy = []
    for class_id in range(NUM_CLASSES):
        mask = labels == class_id
        if not bool(mask.any()):
            raise RuntimeError(f"No evaluation samples for class {CLASSES[class_id]}")
        per_class_accuracy.append(float(predictions[mask].eq(labels[mask]).float().mean().item() * 100.0))
    mean_class_accuracy = float(np.mean(per_class_accuracy))
    one_hot = F.one_hot(labels, num_classes=NUM_CLASSES).float()
    brier = float(((probabilities - one_hot) ** 2).sum(dim=1).mean().item())
    true_probabilities = probabilities.gather(1, labels.view(-1, 1)).squeeze(1).clamp_min(EPS)
    nll = float((-true_probabilities.log()).mean().item())
    ece = expected_calibration_error(probabilities, labels, ECE_BINS)
    mean_confidence = float(confidence.mean().item())
    mean_entropy = float((-(probabilities.clamp_min(EPS) * probabilities.clamp_min(EPS).log()).sum(dim=1)).mean().item())
    return {
        "overall_accuracy": overall_accuracy_fraction * 100.0,
        "accuracy_fraction": overall_accuracy_fraction,
        "mean_class_accuracy": mean_class_accuracy,
        "per_class_accuracy": {name: per_class_accuracy[index] for index, name in enumerate(CLASSES)},
        "mean_confidence": mean_confidence,
        "confidence_correctness_gap": mean_confidence - overall_accuracy_fraction,
        "ece": ece,
        "brier_score": brier,
        "nll": nll,
        "mean_entropy": mean_entropy,
        "total_target_discrepancy_eval": discrepancy_sum / max(discrepancy_count, 1),
    }


def attach_evaluation(results, initial_state, target_features, target_labels, device):
    initial_metrics = evaluate_state(initial_state, target_features, target_labels, device)
    for result in results:
        if len(result["snapshots"]) != EPOCHS or len(result["history"]) != EPOCHS:
            raise RuntimeError(f"Arm {result['arm']} has an unexpected number of snapshots/history rows")
        for epoch_index in range(EPOCHS):
            metrics = evaluate_state(result["snapshots"][epoch_index], target_features, target_labels, device)
            result["history"][epoch_index]["evaluation"] = metrics
        result["final_metrics"] = result["history"][-1]["evaluation"]
    return initial_metrics


def verify_arm_identity_invariants(results, schedules):
    identity = {
        "same_selected_target_positions_all_arms": True,
        "same_source_batch_order_all_arms": True,
        "same_target_batch_order_all_arms": True,
        "same_target_exposure_order_all_arms": True,
        "epoch1_same_pseudo_label_predictions_all_arms": True,
        "later_epoch_predictions_allowed_to_diverge_as_causal_consequence": True,
        "details": [],
    }
    for epoch_index in range(EPOCHS):
        selected_hashes = {result["history"][epoch_index]["selected_index_hash"] for result in results}
        source_hashes = {result["history"][epoch_index]["source_batch_hash"] for result in results}
        target_hashes = {result["history"][epoch_index]["target_batch_hash"] for result in results}
        exposure_hashes = {result["history"][epoch_index]["target_exposure_hash"] for result in results}
        prediction_hashes = {result["history"][epoch_index]["pseudo_label_prediction_hash"] for result in results}
        schedule = schedules[epoch_index]
        if selected_hashes != {schedule["selected_index_hash"]}:
            identity["same_selected_target_positions_all_arms"] = False
        if source_hashes != {schedule["source_batch_hash"]}:
            identity["same_source_batch_order_all_arms"] = False
        if target_hashes != {schedule["target_batch_hash"]}:
            identity["same_target_batch_order_all_arms"] = False
        if exposure_hashes != {schedule["target_exposure_hash"]}:
            identity["same_target_exposure_order_all_arms"] = False
        if epoch_index == 0 and len(prediction_hashes) != 1:
            identity["epoch1_same_pseudo_label_predictions_all_arms"] = False
        identity["details"].append(
            {
                "epoch": epoch_index + 1,
                "selected_index_hash": schedule["selected_index_hash"],
                "source_batch_hash": schedule["source_batch_hash"],
                "target_batch_hash": schedule["target_batch_hash"],
                "target_exposure_hash": schedule["target_exposure_hash"],
                "pseudo_label_prediction_hashes": {
                    result["arm"]: result["history"][epoch_index]["pseudo_label_prediction_hash"]
                    for result in results
                },
            }
        )
    required = [
        identity["same_selected_target_positions_all_arms"],
        identity["same_source_batch_order_all_arms"],
        identity["same_target_batch_order_all_arms"],
        identity["same_target_exposure_order_all_arms"],
        identity["epoch1_same_pseudo_label_predictions_all_arms"],
    ]
    if not all(required):
        raise RuntimeError(f"Arm identity invariant failure: {identity}")
    return identity


def scientific_decision(results):
    by_arm = {result["arm"]: result["final_metrics"] for result in results}
    vanilla = by_arm["vanilla"]
    norm_control = by_arm["norm_matched_global_attenuation"]
    confidence = by_arm["confidence_weighted"]
    support = by_arm["confidence_support_weighted"]

    strong_classes = [
        name
        for name in CLASSES
        if vanilla["per_class_accuracy"][name] >= STRONG_CLASS_ACCURACY_THRESHOLD
    ]
    strong_class_deltas = {
        name: support["per_class_accuracy"][name] - vanilla["per_class_accuracy"][name]
        for name in strong_classes
    }
    catastrophic_classes = [
        name for name, delta in strong_class_deltas.items() if delta <= -CATASTROPHIC_CLASS_DROP_PP
    ]

    hierarchy = (
        support["mean_class_accuracy"] > confidence["mean_class_accuracy"]
        and confidence["mean_class_accuracy"] > vanilla["mean_class_accuracy"]
    )
    beats_norm = support["mean_class_accuracy"] > norm_control["mean_class_accuracy"]
    calibration_improvement = (
        support["ece"] < vanilla["ece"]
        or support["brier_score"] < vanilla["brier_score"]
        or support["nll"] < vanilla["nll"]
    )
    no_catastrophic_degradation = len(catastrophic_classes) == 0
    strong_evidence = hierarchy and beats_norm and no_catastrophic_degradation

    if strong_evidence:
        conclusion = (
            "Confidence-support weighting passes the preregistered causal comparison at the fixed final epoch: "
            "it exceeds confidence-only weighting, vanilla pseudo-labeling, and the norm-matched global attenuation control "
            "without catastrophic degradation of vanilla-strong classes. Replicate across seeds before making a method claim."
        )
    else:
        conclusion = (
            "Independent support predicts pseudo-label correctness but does not provide sufficient causal guidance through "
            "simple scalar loss weighting under this preregistered experiment. Do not tune the weighting function repeatedly; "
            "use this result to decide whether a different risk-control mechanism is needed."
        )

    return {
        "primary_metric": "fixed_final_epoch_mean_class_accuracy",
        "target_accuracy_not_used_for_checkpoint_selection": True,
        "required_ordering": "confidence_support_weighted > confidence_weighted > vanilla",
        "support_must_also_beat": "norm_matched_global_attenuation",
        "observed_hierarchy": bool(hierarchy),
        "support_beats_norm_matched_control": bool(beats_norm),
        "calibration_improves_on_at_least_one_of_ece_brier_nll": bool(calibration_improvement),
        "strong_class_definition": f"vanilla final per-class accuracy >= {STRONG_CLASS_ACCURACY_THRESHOLD:.1f}%",
        "catastrophic_drop_definition": f"support-weighted drop <= -{CATASTROPHIC_CLASS_DROP_PP:.1f} percentage points versus vanilla",
        "strong_class_deltas_pp": strong_class_deltas,
        "catastrophically_degraded_classes": catastrophic_classes,
        "no_catastrophic_degradation": bool(no_catastrophic_degradation),
        "strong_evidence": bool(strong_evidence),
        "conclusion": conclusion,
    }


def build_final_comparison(results):
    vanilla = next(result for result in results if result["arm"] == "vanilla")["final_metrics"]
    comparison = {}
    for result in results:
        metrics = result["final_metrics"]
        comparison[result["arm"]] = {
            "final_metrics": metrics,
            "delta_vs_vanilla": {
                "mean_class_accuracy_pp": metrics["mean_class_accuracy"] - vanilla["mean_class_accuracy"],
                "overall_accuracy_pp": metrics["overall_accuracy"] - vanilla["overall_accuracy"],
                "ece": metrics["ece"] - vanilla["ece"],
                "brier_score": metrics["brier_score"] - vanilla["brier_score"],
                "nll": metrics["nll"] - vanilla["nll"],
                "mean_confidence": metrics["mean_confidence"] - vanilla["mean_confidence"],
                "confidence_correctness_gap": metrics["confidence_correctness_gap"]
                - vanilla["confidence_correctness_gap"],
            },
        }
    return comparison


def save_npz(path, results, schedules):
    arrays = {
        "classes": np.asarray(CLASSES),
        "arms": np.asarray(ARMS),
        "seed": np.asarray([SEED], dtype=np.int64),
    }
    for epoch_index, schedule in enumerate(schedules, 1):
        arrays[f"epoch{epoch_index}_selected_target_indices"] = schedule["selected_target_indices"].numpy().astype(np.int64)
        arrays[f"epoch{epoch_index}_target_exposure_indices"] = schedule["target_exposure_indices"].numpy().astype(np.int64)
    for result in results:
        arm = result["arm"]
        for epoch_index, plan in enumerate(result["plans"], 1):
            prefix = f"{arm}_epoch{epoch_index}"
            selected = plan["selected_indices"]
            arrays[f"{prefix}_pseudo_label_prediction"] = plan["predictions"][selected].astype(np.int16)
            arrays[f"{prefix}_confidence"] = plan["confidence"][selected].astype(np.float32)
            arrays[f"{prefix}_support_current_class"] = plan["support_current_class"][selected].astype(np.float32)
            arrays[f"{prefix}_support_signed_margin"] = plan["support_signed_margin"][selected].astype(np.float32)
            arrays[f"{prefix}_support_agreement"] = plan["support_agreement"][selected].astype(np.uint8)
            arrays[f"{prefix}_confidence_weight"] = plan["confidence_weight"][selected].astype(np.float32)
            arrays[f"{prefix}_confidence_support_weight"] = plan["confidence_support_weight"][selected].astype(np.float32)
    np.savez_compressed(path, **arrays)


def save_checkpoints(output_dir, results):
    output_paths = {}
    for result in results:
        arm = result["arm"]
        arm_paths = []
        for epoch_index, state in enumerate(result["snapshots"], 1):
            path = output_dir / f"{arm}_epoch{epoch_index}_seed{SEED}.pt"
            torch.save(
                {
                    "student_state_dict": state,
                    "arm": arm,
                    "epoch": epoch_index,
                    "seed": SEED,
                    "classes": CLASSES,
                    "input_dim": INPUT_DIM,
                    "hidden_dim": HIDDEN_DIM,
                },
                path,
            )
            arm_paths.append(str(path))
        output_paths[arm] = arm_paths
    return output_paths


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--support-artifact", type=Path, default=DEFAULT_SUPPORT_ARTIFACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(SEED)
    device = torch.device("cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 104)
    print("VISDA-2017 CONFIDENCE-SUPPORT DECOUPLED PSEUDO-LABEL WEIGHTING")
    print("=" * 104)
    print(f"device={device}")
    print(f"seed={SEED}")
    print(f"checkpoint={args.checkpoint}")
    print(f"support_artifact={args.support_artifact}")
    print(f"epochs={EPOCHS} batch_size={BATCH_SIZE}")

    print("Loading source cache...")
    source_features, source_labels = load_source_cache(SOURCE_CACHE)
    print(f"source_features={tuple(source_features.shape)}")

    print("Loading target features without accessing the target label field...")
    target_features = load_target_features_only(TARGET_CACHE)
    print(f"target_features={tuple(target_features.shape)}")

    print("Loading frozen independent-support artifact...")
    support = load_support_artifact(args.support_artifact)
    support_probabilities = support["support_probabilities"]
    print(f"support_probabilities={support_probabilities.shape}")

    print("Loading exact frozen RPC student checkpoint...")
    initial_state, checkpoint_keys = load_initial_state(args.checkpoint)
    alpha, lower_median, cap = frozen_rpc_alphas(device)

    schedules = build_schedules(len(source_features), len(target_features))
    for schedule in schedules:
        print(
            f"epoch={schedule['epoch']} source_steps={schedule['steps']} "
            f"target_batches={len(schedule['target_batches'])} "
            f"unique_target_PL={len(schedule['selected_target_indices'])}"
        )

    results = []
    for arm in ARMS:
        print()
        print("=" * 104)
        print(f"ARM {arm}")
        print("=" * 104)
        result = run_arm(
            arm=arm,
            initial_state=initial_state,
            source_features=source_features,
            source_labels=source_labels,
            target_features=target_features,
            support_probabilities=support_probabilities,
            schedules=schedules,
            alpha=alpha,
            device=device,
        )
        results.append(result)

    identity_checks = verify_arm_identity_invariants(results, schedules)

    checkpoint_output_paths = save_checkpoints(args.output_dir, results)

    print()
    print("All training arms and epoch snapshots are frozen. Target labels are now loaded for evaluation only.")
    target_labels = load_target_labels_for_evaluation(TARGET_CACHE)
    initial_metrics = attach_evaluation(results, initial_state, target_features, target_labels, device)

    final_comparison = build_final_comparison(results)
    decision = scientific_decision(results)

    json_path = args.output_dir / f"support_weighted_target_adaptation_seed{SEED}.json"
    npz_path = args.output_dir / f"support_weighted_target_adaptation_seed{SEED}.npz"
    save_npz(npz_path, results, schedules)

    summary = {
        "experiment": "visda_support_weighted_target_adaptation",
        "seed": SEED,
        "device": str(device),
        "cpu_only": True,
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_keys": checkpoint_keys,
        "support_artifact_path": str(args.support_artifact),
        "support_artifact_keys": support["keys"],
        "source_cache": str(SOURCE_CACHE),
        "target_cache": str(TARGET_CACHE),
        "target_count": int(len(target_features)),
        "source_count": int(len(source_features)),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "pseudo_label_fraction_policy": "all deterministically scheduled target positions; no confidence/support filtering",
        "effective_unique_pseudo_label_fraction_by_epoch": [
            float(len(schedule["selected_target_indices"]) / len(target_features)) for schedule in schedules
        ],
        "weighting_formulas": {
            "vanilla": "w_i = 1",
            "norm_matched_global_attenuation": (
                "w_i = global scalar on vanilla CE gradient chosen per batch so ||g_vanilla_scaled||_2 "
                "equals ||g_confidence_support||_2 at the same model state and pseudo-label batch"
            ),
            "confidence_weighted": "w_i = c_i / E_exposure[c]",
            "confidence_support_weighted": (
                "w_i = (c_i * s_i[current_model_pseudo_label_class]) / "
                "E_exposure[c * s_current_class]"
            ),
        },
        "weights_detached": True,
        "support_disagreement_policy": (
            "never replace or reject the current model pseudo-label; use support probability assigned to the current pseudo-label "
            "class so disagreement only reduces continuous influence"
        ),
        "support_margin_policy": (
            "diagnostic only in this first causal test; signed current-class margin is computed as "
            "support(current pseudo-label) - max support(other class) and is not used for weighting"
        ),
        "optimizer": {
            "adapter": {
                "type": "SGD",
                "lr": LR_ADAPTER,
                "momentum": MOMENTUM,
                "weight_decay": WEIGHT_DECAY,
            },
            "classifiers": {
                "type": "SGD",
                "lr": LR_CLASSIFIER,
                "momentum": MOMENTUM,
                "weight_decay": WEIGHT_DECAY,
            },
        },
        "rpc_base_update": {
            "kappa": KAPPA,
            "frozen_promotion_masses": {name: float(FROZEN_PROMOTION_MASSES[i]) for i, name in enumerate(CLASSES)},
            "lower_median": lower_median,
            "cap": cap,
            "alpha": {name: float(alpha[i].item()) for i, name in enumerate(CLASSES)},
        },
        "pseudo_label_set_identity_checks": identity_checks,
        "leakage_safeguards": {
            "target_label_field_accessed_for_training": False,
            "target_labels_used_for_training": False,
            "target_labels_used_for_pseudo_label_generation": False,
            "target_labels_used_for_weights": False,
            "target_labels_used_for_thresholds": False,
            "target_labels_used_for_checkpoint_selection": False,
            "target_labels_concatenated_and_retained_only_in_explicit_evaluation_phase": True,
            "target_cache_serialization_note": (
                "feature and label tensors share each torch cache chunk, so torch.load deserializes the chunk object; "
                "the pre-evaluation loader never indexes, concatenates, retains, inspects, or passes the label field to training"
            ),
            "support_artifact_teacher_prediction_compared_to_current_rpc_prediction": False,
            "support_artifact_regenerated": False,
            "truck_specific_training_logic": False,
            "current_model_predictions_are_pseudo_label_source": True,
            "support_probabilities_are_auxiliary_weights_only": True,
        },
        "initial_evaluation": initial_metrics,
        "arms": [
            {
                "arm": result["arm"],
                "training_seconds": result["training_seconds"],
                "history": result["history"],
                "epoch_checkpoints": checkpoint_output_paths[result["arm"]],
            }
            for result in results
        ],
        "final_comparison": final_comparison,
        "scientific_decision": decision,
        "output_paths": {
            "json": str(json_path),
            "npz": str(npz_path),
            "epoch_checkpoints": checkpoint_output_paths,
        },
    }

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print()
    print("=" * 104)
    print("FINAL FIXED-EPOCH COMPARISON")
    print("=" * 104)
    for arm in ARMS:
        metrics = final_comparison[arm]["final_metrics"]
        delta = final_comparison[arm]["delta_vs_vanilla"]
        print(
            f"{arm:34s} MCA={metrics['mean_class_accuracy']:.2f}% "
            f"OA={metrics['overall_accuracy']:.2f}% "
            f"ECE={metrics['ece']:.6f} "
            f"Brier={metrics['brier_score']:.6f} "
            f"NLL={metrics['nll']:.6f} "
            f"conf={metrics['mean_confidence']:.6f} "
            f"gap={metrics['confidence_correctness_gap']:.6f} "
            f"dMCA={delta['mean_class_accuracy_pp']:+.2f}pp"
        )
    print()
    print(decision["conclusion"])
    print()
    print(f"json={json_path}")
    print(f"npz={npz_path}")


if __name__ == "__main__":
    main()
