import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
NUM_CLASSES = 12
INPUT_DIM = 2048
HIDDEN_DIM = 512
BATCH_SIZE = 512
EVAL_BATCH_SIZE = 4096
EPOCHS = 2
LR_ADAPTER = 0.001
LR_CLASSIFIER = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
KAPPA = 2.0
EPS = 1e-12
EXPECTED_START_MCA = 73.89
EXPECTED_START_OVERALL = 73.95
START_METRIC_TOLERANCE = 0.20
CACHE_ROOT = Path("checkpoints/visda_feature_cache")
SOURCE_CACHE = CACHE_ROOT / "source"
TARGET_CACHE = CACHE_ROOT / "target"
DEFAULT_CHECKPOINT = Path("checkpoints/visda_geometry_gated_mcd/geometry_gated_mcd_seed42.pt")
DEFAULT_OUTPUT_DIR = Path("checkpoints/visda_rpc_causal_experiment")
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_cache(cache_dir):
    files = sorted(cache_dir.glob("chunk_*.pt"))
    if not files:
        raise RuntimeError(f"No cache chunks found in {cache_dir}")
    features = []
    labels = []
    for path in files:
        payload = safe_torch_load(path, map_location="cpu")
        if "features" not in payload or "labels" not in payload:
            raise RuntimeError(f"Invalid cache payload in {path}")
        batch_features = payload["features"]
        batch_labels = payload["labels"].long()
        if batch_features.ndim != 2 or batch_features.shape[1] != INPUT_DIM:
            raise RuntimeError(f"Expected N x {INPUT_DIM} features in {path}, got {tuple(batch_features.shape)}")
        if len(batch_features) != len(batch_labels):
            raise RuntimeError(f"Feature/label mismatch in {path}")
        features.append(batch_features.cpu())
        labels.append(batch_labels.cpu())
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    if features.ndim != 2 or features.shape[1] != INPUT_DIM:
        raise RuntimeError(f"Invalid concatenated feature tensor {tuple(features.shape)}")
    if len(features) != len(labels):
        raise RuntimeError("Final feature/label count mismatch")
    return features, labels


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


def model_from_state(state_dict, device):
    model = MCDModel().to(device)
    model.load_state_dict(copy.deepcopy(state_dict), strict=True)
    return model


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
        raise RuntimeError("Dataset is smaller than one training batch")
    return permutation[:usable].view(-1, batch_size)


def fetch_features(features, indices, device):
    return features.index_select(0, indices).to(device=device, dtype=torch.float32, non_blocking=False)


def fetch_labels(labels, indices, device):
    return labels.index_select(0, indices).to(device=device, dtype=torch.long, non_blocking=False)


def parameter_tuple(module):
    return tuple(parameter for parameter in module.parameters() if parameter.requires_grad)


def assign_parameter_grads(parameters, gradients):
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = gradient.detach().clone()


def grad_l2_norm(gradients):
    total = None
    for gradient in gradients:
        value = gradient.detach().double().pow(2).sum()
        total = value if total is None else total + value
    if total is None:
        return 0.0
    return float(torch.sqrt(total).item())


def scale_gradients(gradients, scale):
    return tuple(gradient * scale for gradient in gradients)


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
    amplification = transformed.abs() - logit_gradient.abs()
    max_amplification = float(amplification.max().detach().item())
    zero_sum_error = float(transformed.sum(dim=1).abs().max().detach().item())
    if max_amplification > 1e-10:
        raise RuntimeError(f"RPC amplified a logit-gradient component by {max_amplification:.12e}")
    return transformed, clipped_promotion, suppression_scale, zero_sum_error


def native_logit_gradients(target_logits1, target_logits2):
    target_discrepancy = discrepancy(target_logits1, target_logits2)
    grad1, grad2 = torch.autograd.grad(
        target_discrepancy,
        (target_logits1, target_logits2),
        retain_graph=True,
        create_graph=False,
    )
    return target_discrepancy, grad1.detach(), grad2.detach()


def adapter_vjp(target_logits1, target_logits2, grad1, grad2, adapter_parameters, retain_graph):
    return torch.autograd.grad(
        outputs=(target_logits1, target_logits2),
        inputs=adapter_parameters,
        grad_outputs=(grad1, grad2),
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=False,
    )


def target_adapter_step(model, optimizer_adapter, target_x, arm, alpha):
    optimizer_adapter.zero_grad(set_to_none=True)
    target_z = model.encode(target_x)
    target_logits1 = model.classifier1(target_z)
    target_logits2 = model.classifier2(target_z)
    target_discrepancy, native_grad1, native_grad2 = native_logit_gradients(target_logits1, target_logits2)
    native_promotion1, _ = logit_gradient_components(native_grad1)
    native_promotion2, _ = logit_gradient_components(native_grad2)
    native_promotion_mass = native_promotion1.sum(dim=0) + native_promotion2.sum(dim=0)
    adapter_parameters = parameter_tuple(model.adapter)
    rpc_grad1, rpc_promotion1, rpc_suppression_scale1, rpc_zero_sum1 = rpc_transform_single_head(native_grad1, alpha)
    rpc_grad2, rpc_promotion2, rpc_suppression_scale2, rpc_zero_sum2 = rpc_transform_single_head(native_grad2, alpha)
    rpc_promotion_mass = rpc_promotion1.sum(dim=0) + rpc_promotion2.sum(dim=0)
    global_scale = 1.0
    counterfactual_rpc_norm = None
    native_parameter_norm = None
    if arm == "vanilla_mcd":
        applied_grad1 = native_grad1
        applied_grad2 = native_grad2
        parameter_gradients = adapter_vjp(
            target_logits1,
            target_logits2,
            applied_grad1,
            applied_grad2,
            adapter_parameters,
            retain_graph=False,
        )
        applied_promotion_mass = native_promotion_mass
    elif arm == "rpc":
        applied_grad1 = rpc_grad1
        applied_grad2 = rpc_grad2
        parameter_gradients = adapter_vjp(
            target_logits1,
            target_logits2,
            applied_grad1,
            applied_grad2,
            adapter_parameters,
            retain_graph=False,
        )
        applied_promotion_mass = rpc_promotion_mass
    elif arm == "norm_matched_global":
        native_parameter_gradients = adapter_vjp(
            target_logits1,
            target_logits2,
            native_grad1,
            native_grad2,
            adapter_parameters,
            retain_graph=True,
        )
        rpc_parameter_gradients = adapter_vjp(
            target_logits1,
            target_logits2,
            rpc_grad1,
            rpc_grad2,
            adapter_parameters,
            retain_graph=False,
        )
        native_parameter_norm = grad_l2_norm(native_parameter_gradients)
        counterfactual_rpc_norm = grad_l2_norm(rpc_parameter_gradients)
        if native_parameter_norm <= EPS:
            global_scale = 0.0 if counterfactual_rpc_norm <= EPS else 1.0
        else:
            global_scale = counterfactual_rpc_norm / native_parameter_norm
        parameter_gradients = scale_gradients(native_parameter_gradients, global_scale)
        applied_grad1 = native_grad1 * global_scale
        applied_grad2 = native_grad2 * global_scale
        applied_promotion_mass = native_promotion_mass * global_scale
    else:
        raise ValueError(f"Unknown arm {arm}")
    applied_parameter_norm = grad_l2_norm(parameter_gradients)
    if arm == "norm_matched_global" and counterfactual_rpc_norm is not None:
        match_error = abs(applied_parameter_norm - counterfactual_rpc_norm)
        tolerance = 1e-8 + 1e-6 * max(counterfactual_rpc_norm, 1.0)
        if match_error > tolerance:
            raise RuntimeError(
                f"Norm-matched control failed: applied={applied_parameter_norm:.12e}, rpc={counterfactual_rpc_norm:.12e}"
            )
    else:
        match_error = 0.0
    assign_parameter_grads(adapter_parameters, parameter_gradients)
    optimizer_adapter.step()
    applied_zero_sum1 = float(applied_grad1.sum(dim=1).abs().max().detach().item())
    applied_zero_sum2 = float(applied_grad2.sum(dim=1).abs().max().detach().item())
    return {
        "target_discrepancy": float(target_discrepancy.detach().item()),
        "native_promotion_mass": native_promotion_mass.detach().double().cpu(),
        "rpc_promotion_mass": rpc_promotion_mass.detach().double().cpu(),
        "applied_promotion_mass": applied_promotion_mass.detach().double().cpu(),
        "target_gradient_norm": applied_parameter_norm,
        "native_parameter_gradient_norm": native_parameter_norm,
        "counterfactual_rpc_gradient_norm": counterfactual_rpc_norm,
        "global_scale": float(global_scale),
        "norm_match_error": float(match_error),
        "rpc_zero_sum_error": max(rpc_zero_sum1, rpc_zero_sum2),
        "applied_zero_sum_error": max(applied_zero_sum1, applied_zero_sum2),
        "rpc_suppression_scale_min": float(
            min(
                rpc_suppression_scale1.min().detach().item(),
                rpc_suppression_scale2.min().detach().item(),
            )
        ),
        "rpc_suppression_scale_mean": float(
            0.5
            * (
                rpc_suppression_scale1.mean().detach().item()
                + rpc_suppression_scale2.mean().detach().item()
            )
        ),
    }


@torch.no_grad()
def evaluate(model, features, labels, device):
    model.eval()
    loader = DataLoader(
        TensorDataset(features, labels),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    total_correct = 0
    total_examples = 0
    class_correct = torch.zeros(NUM_CLASSES, dtype=torch.long)
    class_total = torch.zeros(NUM_CLASSES, dtype=torch.long)
    for x, y in loader:
        x = x.to(device=device, dtype=torch.float32)
        y_device = y.to(device=device, dtype=torch.long)
        logits1, logits2 = model(x)
        probabilities = (F.softmax(logits1, dim=1) + F.softmax(logits2, dim=1)) / 2.0
        predictions = probabilities.argmax(dim=1)
        total_correct += int((predictions == y_device).sum().item())
        total_examples += int(y.numel())
        for class_id in range(NUM_CLASSES):
            mask = y_device == class_id
            count = int(mask.sum().item())
            if count > 0:
                class_total[class_id] += count
                class_correct[class_id] += int((predictions[mask] == y_device[mask]).sum().item())
    per_class = 100.0 * class_correct.float() / class_total.clamp_min(1)
    return {
        "overall_accuracy": 100.0 * total_correct / max(total_examples, 1),
        "mean_class_accuracy": float(per_class.mean().item()),
        "per_class_accuracy": [float(value) for value in per_class.tolist()],
    }


def assert_start_checkpoint(metrics):
    mca_error = abs(metrics["mean_class_accuracy"] - EXPECTED_START_MCA)
    overall_error = abs(metrics["overall_accuracy"] - EXPECTED_START_OVERALL)
    if mca_error > START_METRIC_TOLERANCE or overall_error > START_METRIC_TOLERANCE:
        raise RuntimeError(
            "Starting checkpoint does not match the expected 73.89/73.95 checkpoint: "
            f"got MCA={metrics['mean_class_accuracy']:.4f}, overall={metrics['overall_accuracy']:.4f}"
        )


def tensor_dict(values):
    return {name: float(values[index]) for index, name in enumerate(CLASSES)}


def share_dict(values):
    total = float(values.sum().item())
    if total <= EPS:
        return {name: 0.0 for name in CLASSES}
    return {name: float(values[index].item() / total) for index, name in enumerate(CLASSES)}


def run_arm(
    arm,
    initial_state,
    source_features,
    source_labels,
    target_features,
    target_labels,
    device,
    output_dir,
    alpha,
    lower_median,
    cap,
):
    set_seed(SEED)
    model = model_from_state(initial_state, device)
    optimizer_adapter, optimizer_classifier = make_optimizer_pair(model)
    start_metrics = evaluate(model, target_features, target_labels, device)
    assert_start_checkpoint(start_metrics)
    history = []
    arm_start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_start = time.perf_counter()
        source_batches = make_epoch_batches(len(source_features), BATCH_SIZE, SEED + 1000 * epoch + 11)
        target_batches = make_epoch_batches(len(target_features), BATCH_SIZE, SEED + 1000 * epoch + 29)
        native_promotion_total = torch.zeros(NUM_CLASSES, dtype=torch.float64)
        rpc_promotion_total = torch.zeros(NUM_CLASSES, dtype=torch.float64)
        applied_promotion_total = torch.zeros(NUM_CLASSES, dtype=torch.float64)
        target_gradient_norm_sum = 0.0
        target_gradient_norm_sq_sum = 0.0
        target_gradient_norm_max = 0.0
        source_adapter_loss_sum = 0.0
        source_classifier_loss_sum = 0.0
        classifier_discrepancy_sum = 0.0
        generator_discrepancy_sum = 0.0
        global_scales = []
        norm_match_error_max = 0.0
        rpc_zero_sum_error_max = 0.0
        applied_zero_sum_error_max = 0.0
        rpc_suppression_scale_min = 1.0
        rpc_suppression_scale_sum = 0.0
        steps = int(source_batches.shape[0])
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
            target_stats = target_adapter_step(
                model,
                optimizer_adapter,
                target_x,
                arm,
                alpha,
            )
            generator_discrepancy_sum += target_stats["target_discrepancy"]
            native_promotion_total += target_stats["native_promotion_mass"]
            rpc_promotion_total += target_stats["rpc_promotion_mass"]
            applied_promotion_total += target_stats["applied_promotion_mass"]
            target_gradient_norm = target_stats["target_gradient_norm"]
            target_gradient_norm_sum += target_gradient_norm
            target_gradient_norm_sq_sum += target_gradient_norm * target_gradient_norm
            target_gradient_norm_max = max(target_gradient_norm_max, target_gradient_norm)
            global_scales.append(target_stats["global_scale"])
            norm_match_error_max = max(norm_match_error_max, target_stats["norm_match_error"])
            rpc_zero_sum_error_max = max(rpc_zero_sum_error_max, target_stats["rpc_zero_sum_error"])
            applied_zero_sum_error_max = max(applied_zero_sum_error_max, target_stats["applied_zero_sum_error"])
            rpc_suppression_scale_min = min(
                rpc_suppression_scale_min,
                target_stats["rpc_suppression_scale_min"],
            )
            rpc_suppression_scale_sum += target_stats["rpc_suppression_scale_mean"]
        metrics = evaluate(model, target_features, target_labels, device)
        per_class = metrics["per_class_accuracy"]
        epoch_seconds = time.perf_counter() - epoch_start
        global_scale_array = np.asarray(global_scales, dtype=np.float64)
        row = {
            "epoch": epoch,
            "arm": arm,
            "mean_class_accuracy": metrics["mean_class_accuracy"],
            "overall_accuracy": metrics["overall_accuracy"],
            "truck_accuracy": per_class[CLASSES.index("truck")],
            "car_accuracy": per_class[CLASSES.index("car")],
            "bus_accuracy": per_class[CLASSES.index("bus")],
            "train_accuracy": per_class[CLASSES.index("train")],
            "per_class_accuracy": tensor_dict(torch.tensor(per_class, dtype=torch.float64)),
            "native_target_promotion_mass": tensor_dict(native_promotion_total),
            "native_target_promotion_share": share_dict(native_promotion_total),
            "rpc_counterfactual_promotion_mass": tensor_dict(rpc_promotion_total),
            "rpc_counterfactual_promotion_share": share_dict(rpc_promotion_total),
            "applied_target_promotion_mass": tensor_dict(applied_promotion_total),
            "applied_target_promotion_share": share_dict(applied_promotion_total),
            "target_gradient_norm_sum": target_gradient_norm_sum,
            "target_gradient_norm_mean": target_gradient_norm_sum / max(steps, 1),
            "target_gradient_norm_rms": math.sqrt(target_gradient_norm_sq_sum / max(steps, 1)),
            "target_gradient_norm_max": target_gradient_norm_max,
            "frozen_rpc_alpha": tensor_dict(alpha.detach().double().cpu()),
            "rpc_lower_median_mass": lower_median,
            "rpc_cap": cap,
            "global_scale_mean": float(global_scale_array.mean()),
            "global_scale_min": float(global_scale_array.min()),
            "global_scale_max": float(global_scale_array.max()),
            "global_scale_above_one_steps": int((global_scale_array > 1.0 + 1e-10).sum()),
            "norm_match_error_max": norm_match_error_max,
            "rpc_zero_sum_error_max": rpc_zero_sum_error_max,
            "applied_zero_sum_error_max": applied_zero_sum_error_max,
            "rpc_suppression_scale_min": rpc_suppression_scale_min,
            "rpc_suppression_scale_mean": rpc_suppression_scale_sum / max(steps, 1),
            "source_adapter_loss_mean": source_adapter_loss_sum / max(steps, 1),
            "source_classifier_loss_mean": source_classifier_loss_sum / max(steps, 1),
            "classifier_discrepancy_mean": classifier_discrepancy_sum / max(steps, 1),
            "generator_discrepancy_mean": generator_discrepancy_sum / max(steps, 1),
            "steps": steps,
            "epoch_seconds": epoch_seconds,
        }
        history.append(row)
        print(
            f"{arm:20s} epoch {epoch}/{EPOCHS} | "
            f"MCA {metrics['mean_class_accuracy']:.2f}% | "
            f"OA {metrics['overall_accuracy']:.2f}% | "
            f"truck {row['truck_accuracy']:.2f}% | "
            f"car {row['car_accuracy']:.2f}% | "
            f"bus {row['bus_accuracy']:.2f}% | "
            f"train {row['train_accuracy']:.2f}% | "
            f"grad-norm {row['target_gradient_norm_mean']:.6e}"
        )
        print(
            "promotion-share "
            + " ".join(
                f"{name}={100.0 * row['applied_target_promotion_share'][name]:.2f}%"
                for name in CLASSES
            )
        )
        if arm == "norm_matched_global":
            print(
                f"global-scale mean={row['global_scale_mean']:.6f} "
                f"min={row['global_scale_min']:.6f} "
                f"max={row['global_scale_max']:.6f} "
                f"above1={row['global_scale_above_one_steps']} "
                f"match-error-max={row['norm_match_error_max']:.3e}"
            )
    total_seconds = time.perf_counter() - arm_start
    final_metrics = evaluate(model, target_features, target_labels, device)
    payload = {
        "arm": arm,
        "student_state_dict": model.state_dict(),
        "seed": SEED,
        "epochs": EPOCHS,
        "classes": CLASSES,
        "input_dim": INPUT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "kappa": KAPPA,
        "frozen_promotion_masses": tensor_dict(FROZEN_PROMOTION_MASSES),
        "frozen_rpc_alpha": tensor_dict(alpha.detach().double().cpu()),
        "rpc_lower_median_mass": lower_median,
        "rpc_cap": cap,
        "start_metrics": start_metrics,
        "final_metrics": final_metrics,
        "history": history,
        "total_seconds": total_seconds,
    }
    checkpoint_path = output_dir / f"{arm}_seed{SEED}.pt"
    json_path = output_dir / f"{arm}_seed{SEED}.json"
    torch.save(payload, checkpoint_path)
    serializable_payload = dict(payload)
    serializable_payload.pop("student_state_dict")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(serializable_payload, handle, indent=2)
    return {
        "arm": arm,
        "checkpoint": str(checkpoint_path),
        "history_json": str(json_path),
        "start_metrics": start_metrics,
        "final_metrics": final_metrics,
        "history": history,
        "total_seconds": total_seconds,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(SEED)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 100)
    print("VISDA-2017 RECIPIENT PROMOTION CAPPING CAUSAL EXPERIMENT")
    print("=" * 100)
    print(f"device={device}")
    print(f"checkpoint={args.checkpoint}")
    print(f"epochs={EPOCHS}")
    print(f"batch_size={BATCH_SIZE}")
    print(f"adapter_path={INPUT_DIM}->{HIDDEN_DIM}")
    print("Loading cached features")
    source_features, source_labels = load_cache(SOURCE_CACHE)
    target_features, target_labels = load_cache(TARGET_CACHE)
    print(f"source={tuple(source_features.shape)} target={tuple(target_features.shape)}")
    checkpoint = safe_torch_load(args.checkpoint, map_location="cpu")
    if "student_state_dict" not in checkpoint:
        raise RuntimeError("Checkpoint does not contain student_state_dict")
    initial_state = copy.deepcopy(checkpoint["student_state_dict"])
    verifier = model_from_state(initial_state, device)
    start_metrics = evaluate(verifier, target_features, target_labels, device)
    assert_start_checkpoint(start_metrics)
    del verifier
    alpha, lower_median, cap = frozen_rpc_alphas(device)
    print(f"start_mca={start_metrics['mean_class_accuracy']:.4f}")
    print(f"start_overall={start_metrics['overall_accuracy']:.4f}")
    print(f"rpc_lower_median_mass={lower_median:.7f}")
    print(f"rpc_cap={cap:.7f}")
    print("frozen_rpc_alpha")
    for name, value in zip(CLASSES, alpha.detach().cpu().tolist()):
        print(f"{name:12s} {value:.7f}")
    nontrivial = [(name, value) for name, value in zip(CLASSES, alpha.detach().cpu().tolist()) if value < 1.0 - 1e-10]
    if len(nontrivial) != 1 or nontrivial[0][0] != "car":
        raise RuntimeError(f"Expected only car to be capped, got {nontrivial}")
    results = []
    for arm in ["vanilla_mcd", "norm_matched_global", "rpc"]:
        print()
        print("=" * 100)
        print(f"ARM {arm}")
        print("=" * 100)
        result = run_arm(
            arm=arm,
            initial_state=initial_state,
            source_features=source_features,
            source_labels=source_labels,
            target_features=target_features,
            target_labels=target_labels,
            device=device,
            output_dir=args.output_dir,
            alpha=alpha,
            lower_median=lower_median,
            cap=cap,
        )
        results.append(result)
    summary = {
        "experiment": "visda_rpc_causal_allocation_vs_norm_control",
        "seed": SEED,
        "device": str(device),
        "source_cache": str(SOURCE_CACHE),
        "target_cache": str(TARGET_CACHE),
        "starting_checkpoint": str(args.checkpoint),
        "expected_start_mean_class_accuracy": EXPECTED_START_MCA,
        "expected_start_overall_accuracy": EXPECTED_START_OVERALL,
        "observed_start_metrics": start_metrics,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "input_dim": INPUT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "lr_adapter": LR_ADAPTER,
        "lr_classifier": LR_CLASSIFIER,
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "kappa": KAPPA,
        "frozen_promotion_masses": tensor_dict(FROZEN_PROMOTION_MASSES),
        "rpc_lower_median_mass": lower_median,
        "rpc_cap": cap,
        "frozen_rpc_alpha": tensor_dict(alpha.detach().double().cpu()),
        "arms": [
            {
                "arm": result["arm"],
                "checkpoint": result["checkpoint"],
                "history_json": result["history_json"],
                "start_metrics": result["start_metrics"],
                "final_metrics": result["final_metrics"],
                "total_seconds": result["total_seconds"],
            }
            for result in results
        ],
    }
    summary_path = args.output_dir / f"summary_seed{SEED}.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print()
    print("=" * 100)
    print("FINAL COMPARISON")
    print("=" * 100)
    for result in results:
        final_metrics = result["final_metrics"]
        per_class = final_metrics["per_class_accuracy"]
        print(
            f"{result['arm']:20s} "
            f"MCA={final_metrics['mean_class_accuracy']:.2f}% "
            f"OA={final_metrics['overall_accuracy']:.2f}% "
            f"truck={per_class[11]:.2f}% "
            f"car={per_class[3]:.2f}% "
            f"bus={per_class[2]:.2f}% "
            f"train={per_class[10]:.2f}%"
        )
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
