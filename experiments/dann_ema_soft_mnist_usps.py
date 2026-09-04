from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, Sampler
from torchvision import datasets, transforms

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.digits import get_datasets


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class GradientReversal(nn.Module):
    def forward(self, x, lambd):
        return GradientReversalFunction.apply(x, lambd)


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.network(x)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.network(x)


class DomainDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.network(x)


class IndexedTargetDataset(Dataset):
    def __init__(self, dataset, indices, labels, weights):
        self.dataset = dataset
        self.indices = indices
        self.labels = labels
        self.weights = weights

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        dataset_index = int(self.indices[position])
        x, _ = self.dataset[dataset_index]
        y = int(self.labels[position])
        w = float(self.weights[position])
        return x, y, w, dataset_index


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transform():
    return transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ]
    )


def build_datasets(root):
    mnist, usps_train = get_datasets(root=root)
    usps_test = datasets.USPS(
        root=root,
        train=False,
        download=True,
        transform=get_transform(),
    )
    return mnist, usps_train, usps_test


def make_subset(dataset, count):
    if len(dataset) < count:
        raise ValueError(
            f"Requested {count} samples but dataset contains {len(dataset)} samples"
        )
    return Subset(dataset, list(range(count)))


def evaluate(feature, classifier, loader, device):
    feature.eval()
    classifier.eval()

    total = 0
    correct = 0
    loss_total = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = classifier(feature(x))
            loss = F.cross_entropy(logits, y)
            loss_total += loss.item() * y.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)

    return 100.0 * correct / total, loss_total / total


def evaluate_domain(domain, feature, source_loader, target_loader, device):
    feature.eval()
    domain.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, _ in source_loader:
            x = x.to(device)
            logits = domain(feature(x))
            correct += (logits.argmax(dim=1) == 0).sum().item()
            total += x.size(0)

        for x, _ in target_loader:
            x = x.to(device)
            logits = domain(feature(x))
            correct += (logits.argmax(dim=1) == 1).sum().item()
            total += x.size(0)

    return 100.0 * correct / total


def save_dann_checkpoint(
    path,
    feature,
    classifier,
    domain,
    optimizer,
    epoch,
    seed,
):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    torch.save(
        {
            "feature": feature.state_dict(),
            "classifier": classifier.state_dict(),
            "domain": domain.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "seed": seed,
        },
        path,
    )


def load_dann_checkpoint(path, feature, classifier, domain, device):
    checkpoint = torch.load(path, map_location=device)

    feature_key = (
        "feature"
        if "feature" in checkpoint
        else "feature_extractor"
    )

    classifier_key = "classifier"

    domain_key = (
        "domain"
        if "domain" in checkpoint
        else "domain_discriminator"
    )

    feature.load_state_dict(
        checkpoint[feature_key]
    )

    classifier.load_state_dict(
        checkpoint[classifier_key]
    )

    domain.load_state_dict(
        checkpoint[domain_key]
    )

    return checkpoint


def train_dann(
    feature,
    classifier,
    domain,
    source_loader,
    target_loader,
    source_eval_loader,
    target_adaptation_eval_loader,
    target_test_loader,
    device,
    epochs,
    batch_size,
    lr,
    weight_decay,
    domain_lambda,
):
    optimizer = torch.optim.Adam(
        list(feature.parameters())
        + list(classifier.parameters())
        + list(domain.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    grl = GradientReversal()
    history = []

    for epoch in range(epochs):
        feature.train()
        classifier.train()
        domain.train()

        source_iterator = iter(source_loader)
        target_iterator = iter(target_loader)

        steps = min(len(source_loader), len(target_loader))

        source_loss_total = 0.0
        domain_loss_total = 0.0

        for _ in range(steps):
            source_x, source_y = next(source_iterator)
            target_x, _ = next(target_iterator)

            source_x = source_x.to(device)
            source_y = source_y.to(device)
            target_x = target_x.to(device)

            source_features = feature(source_x)
            target_features = feature(target_x)

            source_logits = classifier(source_features)

            combined = torch.cat(
                [source_features, target_features],
                dim=0,
            )

            reversed_features = grl(
                combined,
                domain_lambda,
            )

            domain_logits = domain(reversed_features)

            domain_labels = torch.cat(
                [
                    torch.zeros(
                        source_features.size(0),
                        dtype=torch.long,
                        device=device,
                    ),
                    torch.ones(
                        target_features.size(0),
                        dtype=torch.long,
                        device=device,
                    ),
                ]
            )

            source_loss = F.cross_entropy(
                source_logits,
                source_y,
            )

            domain_loss = F.cross_entropy(
                domain_logits,
                domain_labels,
            )

            loss = source_loss + domain_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            source_loss_total += source_loss.item()
            domain_loss_total += domain_loss.item()

        source_accuracy, _ = evaluate(
            feature,
            classifier,
            source_eval_loader,
            device,
        )

        target_adaptation_accuracy, _ = evaluate(
            feature,
            classifier,
            target_adaptation_eval_loader,
            device,
        )

        target_test_accuracy, _ = evaluate(
            feature,
            classifier,
            target_test_loader,
            device,
        )

        current_domain_accuracy = evaluate_domain(
            domain,
            feature,
            source_loader,
            target_adaptation_eval_loader,
            device,
        )

        row = {
            "epoch": epoch + 1,
            "source_loss": source_loss_total / steps,
            "domain_loss": domain_loss_total / steps,
            "source_accuracy": source_accuracy,
            "target_adaptation_accuracy": target_adaptation_accuracy,
            "target_test_accuracy": target_test_accuracy,
            "domain_accuracy": current_domain_accuracy,
        }

        history.append(row)

        print(
            f"DANN Epoch {epoch + 1:02d}/{epochs} "
            f"| Source Acc {source_accuracy:.2f}% "
            f"| USPS Acc {target_test_accuracy:.2f}% "
            f"| Domain Acc {current_domain_accuracy:.2f}%"
        )

    return optimizer, history


def clone_model(feature, classifier, device):
    teacher_feature = FeatureExtractor().to(device)
    teacher_classifier = Classifier().to(device)

    teacher_feature.load_state_dict(feature.state_dict())
    teacher_classifier.load_state_dict(classifier.state_dict())

    teacher_feature.eval()
    teacher_classifier.eval()

    for parameter in teacher_feature.parameters():
        parameter.requires_grad_(False)

    for parameter in teacher_classifier.parameters():
        parameter.requires_grad_(False)

    return teacher_feature, teacher_classifier


@torch.no_grad()
def update_ema(
    student_feature,
    student_classifier,
    teacher_feature,
    teacher_classifier,
    decay,
):
    student_feature_state = student_feature.state_dict()
    teacher_feature_state = teacher_feature.state_dict()

    for name in teacher_feature_state:
        student_value = student_feature_state[name]
        teacher_value = teacher_feature_state[name]

        if torch.is_floating_point(teacher_value):
            teacher_value.mul_(decay).add_(
                student_value,
                alpha=1.0 - decay,
            )
        else:
            teacher_value.copy_(student_value)

    student_classifier_state = student_classifier.state_dict()
    teacher_classifier_state = teacher_classifier.state_dict()

    for name in teacher_classifier_state:
        student_value = student_classifier_state[name]
        teacher_value = teacher_classifier_state[name]

        if torch.is_floating_point(teacher_value):
            teacher_value.mul_(decay).add_(
                student_value,
                alpha=1.0 - decay,
            )
        else:
            teacher_value.copy_(student_value)


@torch.no_grad()
def collect_teacher_predictions(
    teacher_feature,
    teacher_classifier,
    target_dataset,
    device,
    batch_size,
):
    loader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    teacher_feature.eval()
    teacher_classifier.eval()

    all_confidences = []
    all_labels = []
    all_indices = []

    for batch_indices, (x, _) in enumerate(loader):
        x = x.to(device)

        logits = teacher_classifier(
            teacher_feature(x)
        )

        probabilities = F.softmax(
            logits,
            dim=1,
        )

        confidence, labels = probabilities.max(
            dim=1
        )

        start = batch_indices * batch_size
        positions = torch.arange(
            start,
            start + x.size(0),
            device=device,
        )

        all_confidences.append(
            confidence.detach().cpu()
        )

        all_labels.append(
            labels.detach().cpu()
        )

        all_indices.append(
            positions.detach().cpu()
        )

    confidences = torch.cat(all_confidences)
    labels = torch.cat(all_labels)
    indices = torch.cat(all_indices)

    return indices, labels, confidences


def build_selection(
    target_dataset,
    target_indices,
    target_labels,
    target_confidences,
    confidence_floor,
):
    keep = target_confidences >= confidence_floor

    selected_indices = target_indices[keep]
    selected_labels = target_labels[keep]
    selected_confidences = target_confidences[keep]

    if len(selected_indices) == 0:
        return None

    class_counts = torch.bincount(
        selected_labels,
        minlength=10,
    )

    total_selected = len(selected_labels)

    class_shares = (
        class_counts.float()
        / max(float(total_selected), 1.0)
    )

    weights = selected_confidences.clone()

    rare_classes = class_shares < 0.05

    if rare_classes.any():
        for class_id in range(10):
            if not rare_classes[class_id]:
                continue

            share = float(class_shares[class_id].item())

            if share <= 0.0:
                continue

            factor = min(
                2.0,
                0.05 / share,
            )

            class_mask = selected_labels == class_id

            weights[class_mask] *= factor

    normalized_weights = weights / weights.mean().clamp_min(1e-8)

    return {
        "dataset": target_dataset,
        "indices": selected_indices,
        "labels": selected_labels,
        "confidences": selected_confidences,
        "weights": normalized_weights,
        "class_counts": class_counts,
        "class_shares": class_shares,
    }


class BalancedClassSampler(Sampler):
    def __init__(self, labels, num_samples):
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.num_samples = int(num_samples)

        counts = torch.bincount(
            self.labels,
            minlength=10,
        ).float()

        present = counts > 0
        weights = torch.ones_like(counts)

        if present.any():
            mean_count = counts[present].mean()
            imbalance_ratio = (
                counts[present].max()
                / counts[present].min().clamp_min(1.0)
            )

            if imbalance_ratio > 2.0:
                weights[present] = (
                    mean_count
                    / counts[present]
                ).sqrt()

                weights[present] = weights[present].clamp(
                    min=0.5,
                    max=2.0,
                )

        sample_weights = weights[self.labels]
        self.sample_weights = sample_weights / sample_weights.sum()

    def __iter__(self):
        sampled = torch.multinomial(
            self.sample_weights,
            self.num_samples,
            replacement=True,
        )

        return iter(sampled.tolist())

    def __len__(self):
        return self.num_samples


def make_target_loader(selection, batch_size):
    indices = selection["indices"].tolist()
    labels = selection["labels"].tolist()
    weights = selection["weights"].tolist()

    dataset = IndexedTargetDataset(
        selection["dataset"],
        indices,
        labels,
        weights,
    )

    class_counts = torch.bincount(
        selection["labels"],
        minlength=10,
    )

    present_counts = class_counts[
        class_counts > 0
    ]

    imbalance_ratio = 1.0

    if len(present_counts) > 0:
        imbalance_ratio = float(
            present_counts.max().item()
            / present_counts.min().clamp_min(1).item()
        )

    use_balanced_sampler = imbalance_ratio > 2.0

    if use_balanced_sampler:
        sampler = BalancedClassSampler(
            labels,
            len(dataset),
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=False,
            num_workers=0,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )

    return loader, use_balanced_sampler, imbalance_ratio


def format_class_counts(class_counts):
    values = [
        int(class_counts[i].item())
        for i in range(10)
    ]
    return ",".join(
        f"{i}:{values[i]}"
        for i in range(10)
    )


def pseudo_train_epoch(
    feature,
    classifier,
    teacher_feature,
    teacher_classifier,
    source_loader,
    target_loader,
    optimizer,
    device,
    pseudo_loss_weight,
    ema_decay,
):
    feature.train()
    classifier.train()
    teacher_feature.eval()
    teacher_classifier.eval()

    source_iterator = iter(source_loader)
    target_iterator = iter(target_loader)

    steps = max(len(source_loader), len(target_loader))

    source_loss_total = 0.0
    pseudo_loss_total = 0.0
    source_steps = 0
    target_steps = 0

    for step in range(steps):
        has_source = step < len(source_loader)
        has_target = step < len(target_loader)

        source_loss = torch.zeros((), device=device)
        pseudo_loss = torch.zeros((), device=device)

        if has_source:
            source_x, source_y = next(source_iterator)

            source_x = source_x.to(device)
            source_y = source_y.to(device)

            source_logits = classifier(
                feature(source_x)
            )

            source_loss = F.cross_entropy(
                source_logits,
                source_y,
            )

        if has_target:
            target_x, pseudo_y, sample_weights, _ = next(
                target_iterator
            )

            target_x = target_x.to(device)
            pseudo_y = pseudo_y.to(device)
            sample_weights = sample_weights.to(device)

            target_logits = classifier(
                feature(target_x)
            )

            per_sample_loss = F.cross_entropy(
                target_logits,
                pseudo_y,
                reduction="none",
            )

            pseudo_loss = (
                per_sample_loss * sample_weights
            ).mean()

        losses = []

        if has_source:
            losses.append(source_loss)

        if has_target:
            losses.append(
                pseudo_loss_weight * pseudo_loss
            )

        if not losses:
            continue

        combined_loss = sum(losses)

        optimizer.zero_grad(set_to_none=True)
        combined_loss.backward()
        optimizer.step()

        update_ema(
            feature,
            classifier,
            teacher_feature,
            teacher_classifier,
            ema_decay,
        )

        if has_source:
            source_loss_total += source_loss.item()
            source_steps += 1

        if has_target:
            pseudo_loss_total += pseudo_loss.item()
            target_steps += 1

    source_loss_value = (
        source_loss_total / max(source_steps, 1)
    )

    pseudo_loss_value = (
        pseudo_loss_total / max(target_steps, 1)
    )

    return source_loss_value, pseudo_loss_value

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        default="data/digits",
    )
    parser.add_argument(
        "--dann-checkpoint",
        required=True,
    )
    parser.add_argument(
        "--output-checkpoint",
        default="checkpoints/dann_ema_soft_seed42.pt",
    )
    parser.add_argument(
        "--history",
        default="checkpoints/dann_ema_soft_seed42.json",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--dann-epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--dann-lr",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--dann-weight-decay",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--dann-domain-lambda",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--pseudo-epochs",
        type=int,
        default=9,
    )
    parser.add_argument(
        "--pseudo-lr",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--pseudo-weight-decay",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.99,
    )
    parser.add_argument(
        "--confidence-floor",
        type=float,
        default=0.85,
    )
    parser.add_argument(
        "--refresh-every",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--pseudo-loss-weight",
        type=float,
        default=1.0,
    )

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device)

    print("========================================")
    print("DANN + EMA + SOFT PSEUDO-LABELING")
    print("========================================")
    print(f"device={device}")
    print(f"seed={args.seed}")
    print(f"dann_epochs={args.dann_epochs}")
    print(f"pseudo_epochs={args.pseudo_epochs}")
    print(f"ema_decay={args.ema_decay}")
    print(f"confidence_floor={args.confidence_floor}")
    print(f"refresh_every={args.refresh_every}")
    print()

    mnist, usps_train, usps_test = build_datasets(
        args.data_root
    )

    source_dataset = make_subset(
        mnist,
        2000,
    )

    target_dataset = make_subset(
        usps_train,
        1800,
    )

    source_loader = DataLoader(
        source_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    target_unlabeled_loader = DataLoader(
        target_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    source_eval_loader = DataLoader(
        source_dataset,
        batch_size=256,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    target_adaptation_eval_loader = DataLoader(
        target_dataset,
        batch_size=256,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    target_test_loader = DataLoader(
        usps_test,
        batch_size=256,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    feature = FeatureExtractor().to(device)
    classifier = Classifier().to(device)
    domain = DomainDiscriminator().to(device)

    if not os.path.exists(args.dann_checkpoint):
        raise FileNotFoundError(
            f"DANN checkpoint not found: {args.dann_checkpoint}"
        )

    print(
        f"Loading DANN checkpoint: "
        f"{args.dann_checkpoint}"
    )

    load_dann_checkpoint(
        args.dann_checkpoint,
        feature,
        classifier,
        domain,
        device,
    )

    baseline_test_accuracy = evaluate(
        feature,
        classifier,
        target_test_loader,
        device,
    )[0]

    print(
        f"Loaded DANN USPS Accuracy: "
        f"{baseline_test_accuracy:.2f}%"
    )

    teacher_feature, teacher_classifier = clone_model(
        feature,
        classifier,
        device,
    )

    teacher_feature.eval()
    teacher_classifier.eval()

    optimizer = torch.optim.Adam(
        list(feature.parameters())
        + list(classifier.parameters()),
        lr=args.pseudo_lr,
        weight_decay=args.pseudo_weight_decay,
    )

    history = []

    selection = None

    for epoch in range(args.pseudo_epochs):
        refresh = (
            epoch == 0
            or epoch % args.refresh_every == 0
        )

        if refresh:
            indices, labels, confidences = (
                collect_teacher_predictions(
                    teacher_feature,
                    teacher_classifier,
                    target_dataset,
                    device,
                    args.batch_size,
                )
            )

            selection = build_selection(
                target_dataset,
                indices,
                labels,
                confidences,
                args.confidence_floor,
            )

            if selection is None:
                raise RuntimeError(
                    "No target samples passed the confidence floor."
                )

            selected_count = len(
                selection["indices"]
            )

            coverage = (
                100.0
                * selected_count
                / len(target_dataset)
            )

            mean_confidence = float(
                selection["confidences"].mean().item()
            )

            print(
                f"REFRESH Epoch {epoch + 1:02d} "
                f"| Selected {selected_count}/{len(target_dataset)} "
                f"| Coverage {coverage:.2f}% "
                f"| Mean Confidence {mean_confidence:.4f} "
                f"| Classes "
                f"{format_class_counts(selection['class_counts'])}"
            )

        (
            target_loader,
            balanced_sampler,
            imbalance_ratio,
        ) = make_target_loader(
            selection,
            args.batch_size,
        )

        source_loss, pseudo_loss = pseudo_train_epoch(
            feature,
            classifier,
            teacher_feature,
            teacher_classifier,
            source_loader,
            target_loader,
            optimizer,
            device,
            args.pseudo_loss_weight,
            args.ema_decay,
        )

        source_accuracy, _ = evaluate(
            feature,
            classifier,
            source_eval_loader,
            device,
        )

        target_adaptation_accuracy, _ = evaluate(
            feature,
            classifier,
            target_adaptation_eval_loader,
            device,
        )

        target_test_accuracy, _ = evaluate(
            feature,
            classifier,
            target_test_loader,
            device,
        )

        selected_count = len(
            selection["indices"]
        )

        coverage = (
            100.0
            * selected_count
            / len(target_dataset)
        )

        mean_confidence = float(
            selection["confidences"].mean().item()
        )

        row = {
            "epoch": epoch + 1,
            "source_loss": source_loss,
            "pseudo_loss": pseudo_loss,
            "source_accuracy": source_accuracy,
            "target_adaptation_accuracy": target_adaptation_accuracy,
            "target_test_accuracy": target_test_accuracy,
            "selected_count": selected_count,
            "coverage": coverage,
            "mean_confidence": mean_confidence,
            "balanced_sampler": balanced_sampler,
            "imbalance_ratio": imbalance_ratio,
            "class_counts": [
                int(x.item())
                for x in selection["class_counts"]
            ],
            "class_shares": [
                float(x.item())
                for x in selection["class_shares"]
            ],
        }

        history.append(row)

        print(
            f"Pseudo Epoch {epoch + 1:02d}/{args.pseudo_epochs} "
            f"| Source Loss {source_loss:.4f} "
            f"| Pseudo Loss {pseudo_loss:.4f} "
            f"| Source Acc {source_accuracy:.2f}% "
            f"| USPS Acc {target_test_accuracy:.2f}% "
            f"| Selected {selected_count} "
            f"| Coverage {coverage:.2f}% "
            f"| Mean Conf {mean_confidence:.4f} "
            f"| Balanced {balanced_sampler} "
            f"| Imbalance {imbalance_ratio:.2f}x"
        )

    directory = os.path.dirname(args.output_checkpoint)

    if directory:
        os.makedirs(directory, exist_ok=True)

    torch.save(
        {
            "feature": feature.state_dict(),
            "classifier": classifier.state_dict(),
            "teacher_feature": teacher_feature.state_dict(),
            "teacher_classifier": teacher_classifier.state_dict(),
            "seed": args.seed,
            "baseline_dann_accuracy": baseline_test_accuracy,
            "history": history,
        },
        args.output_checkpoint,
    )

    history_directory = os.path.dirname(args.history)

    if history_directory:
        os.makedirs(history_directory, exist_ok=True)

    with open(
        args.history,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "baseline_dann_accuracy": baseline_test_accuracy,
                "history": history,
            },
            handle,
            indent=2,
        )

    best_accuracy = max(
        row["target_test_accuracy"]
        for row in history
    )

    final_accuracy = history[-1][
        "target_test_accuracy"
    ]

    print()
    print("========================================")
    print("FINAL RESULT")
    print("========================================")
    print(
        f"baseline_dann_usps={baseline_test_accuracy:.2f}%"
    )
    print(
        f"best_ema_soft_usps={best_accuracy:.2f}%"
    )
    print(
        f"final_ema_soft_usps={final_accuracy:.2f}%"
    )
    print(
        f"improvement_over_dann="
        f"{best_accuracy - baseline_test_accuracy:+.2f} pp"
    )
    print(
        f"improvement_over_85.25="
        f"{best_accuracy - 85.25:+.2f} pp"
    )
    print(
        f"checkpoint={args.output_checkpoint}"
    )
    print(
        f"history={args.history}"
    )


if __name__ == "__main__":
    main()
