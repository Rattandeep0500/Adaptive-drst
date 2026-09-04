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
from torch.utils.data import DataLoader, Subset
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
    def __init__(self, feature_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(64 * 7 * 7, feature_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = x.flatten(1)
        x = self.fc(x)
        x = F.relu(x)

        return x


class Classifier(nn.Module):
    def __init__(self, feature_dim=256, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class DomainDiscriminator(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.net(x)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transform():
    return transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])


def build_datasets(root):
    mnist, usps = get_datasets(root=root)

    transform = get_transform()

    usps_test = datasets.USPS(
        root=root,
        train=False,
        download=True,
        transform=transform,
    )

    return mnist, usps, usps_test


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

            features = feature(x)
            logits = classifier(features)
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

            features = feature(x)
            logits = domain(features)

            correct += (logits.argmax(dim=1) == 0).sum().item()
            total += x.size(0)

        for x, _ in target_loader:
            x = x.to(device)

            features = feature(x)
            logits = domain(features)

            correct += (logits.argmax(dim=1) == 1).sum().item()
            total += x.size(0)

    return 100.0 * correct / total


def linear_lambda(epoch, total_epochs, max_lambda):
    if total_epochs <= 1:
        return max_lambda

    progress = float(epoch) / float(total_epochs - 1)

    return max_lambda * progress


def save_checkpoint(
    path,
    feature,
    classifier,
    domain,
    optimizer,
    epoch,
    args,
    history,
):
    directory = os.path.dirname(path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    checkpoint = {
        "feature": feature.state_dict(),
        "classifier": classifier.state_dict(),
        "domain": domain.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_domain_lambda": args.max_domain_lambda,
        "history": history,
    }

    torch.save(checkpoint, path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        default="data/digits",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--max-domain-lambda",
        type=float,
        default=0.10,
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
        "--checkpoint",
        default="checkpoints/dann_long_ramp_seed42.pt",
    )
    parser.add_argument(
        "--history",
        default="checkpoints/dann_long_ramp_seed42.json",
    )

    args = parser.parse_args()

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    if args.lr <= 0:
        raise ValueError("--lr must be positive")

    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be non-negative")

    if args.max_domain_lambda < 0:
        raise ValueError("--max-domain-lambda must be non-negative")

    set_seed(args.seed)

    device = torch.device(args.device)

    print("========================================")
    print("DANN LONG-RAMP MNIST -> USPS")
    print("========================================")
    print(f"device={device}")
    print(f"seed={args.seed}")
    print(f"epochs={args.epochs}")
    print(f"batch_size={args.batch_size}")
    print(f"lr={args.lr}")
    print(f"weight_decay={args.weight_decay}")
    print(f"max_domain_lambda={args.max_domain_lambda}")
    print("domain_lambda_schedule=linear")
    print()

    mnist, usps, usps_test = build_datasets(args.data_root)

    source_dataset = make_subset(mnist, 2000)
    target_dataset = make_subset(usps, 1800)

    source_loader = DataLoader(
        source_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    target_loader = DataLoader(
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

    print(f"source_samples={len(source_dataset)}")
    print(f"target_adaptation_samples={len(target_dataset)}")
    print(f"target_test_samples={len(usps_test)}")
    print()

    feature = FeatureExtractor(feature_dim=256).to(device)
    classifier = Classifier(feature_dim=256, num_classes=10).to(device)
    domain = DomainDiscriminator(feature_dim=256).to(device)

    grl = GradientReversal()

    optimizer = torch.optim.Adam(
        list(feature.parameters())
        + list(classifier.parameters())
        + list(domain.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = min(
        len(source_loader),
        len(target_loader),
    )

    history = []

    best_test_accuracy = -1.0
    best_test_epoch = -1

    for epoch in range(args.epochs):
        feature.train()
        classifier.train()
        domain.train()

        current_lambda = linear_lambda(
            epoch,
            args.epochs,
            args.max_domain_lambda,
        )

        source_iterator = iter(source_loader)
        target_iterator = iter(target_loader)

        total_classification_loss = 0.0
        total_domain_loss = 0.0
        total_loss = 0.0

        for _ in range(steps_per_epoch):
            try:
                source_x, source_y = next(source_iterator)
            except StopIteration:
                source_iterator = iter(source_loader)
                source_x, source_y = next(source_iterator)

            try:
                target_x, _ = next(target_iterator)
            except StopIteration:
                target_iterator = iter(target_loader)
                target_x, _ = next(target_iterator)

            source_x = source_x.to(device)
            source_y = source_y.to(device)
            target_x = target_x.to(device)

            source_features = feature(source_x)
            target_features = feature(target_x)

            source_logits = classifier(source_features)

            combined_features = torch.cat(
                [
                    source_features,
                    target_features,
                ],
                dim=0,
            )

            reversed_features = grl(
                combined_features,
                current_lambda,
            )

            domain_logits = domain(reversed_features)

            source_domain_labels = torch.zeros(
                source_features.size(0),
                dtype=torch.long,
                device=device,
            )

            target_domain_labels = torch.ones(
                target_features.size(0),
                dtype=torch.long,
                device=device,
            )

            domain_labels = torch.cat(
                [
                    source_domain_labels,
                    target_domain_labels,
                ],
                dim=0,
            )

            classification_loss = F.cross_entropy(
                source_logits,
                source_y,
            )

            domain_loss = F.cross_entropy(
                domain_logits,
                domain_labels,
            )

            loss = classification_loss + domain_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_classification_loss += classification_loss.item()
            total_domain_loss += domain_loss.item()
            total_loss += loss.item()

        classification_loss_value = (
            total_classification_loss / steps_per_epoch
        )

        domain_loss_value = (
            total_domain_loss / steps_per_epoch
        )

        total_loss_value = (
            total_loss / steps_per_epoch
        )

        source_accuracy, source_loss = evaluate(
            feature,
            classifier,
            source_eval_loader,
            device,
        )

        target_adaptation_accuracy, target_adaptation_loss = evaluate(
            feature,
            classifier,
            target_adaptation_eval_loader,
            device,
        )

        target_test_accuracy, target_test_loss = evaluate(
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
            "lambda": current_lambda,
            "classification_loss": classification_loss_value,
            "domain_loss": domain_loss_value,
            "total_loss": total_loss_value,
            "source_accuracy": source_accuracy,
            "source_loss": source_loss,
            "target_adaptation_accuracy": target_adaptation_accuracy,
            "target_adaptation_loss": target_adaptation_loss,
            "target_test_accuracy": target_test_accuracy,
            "target_test_loss": target_test_loss,
            "domain_accuracy": current_domain_accuracy,
        }

        history.append(row)

        if target_test_accuracy > best_test_accuracy:
            best_test_accuracy = target_test_accuracy
            best_test_epoch = epoch + 1

        print(
            f"epoch {epoch + 1:02d}/{args.epochs} "
            f"lambda={current_lambda:.4f} "
            f"cls={classification_loss_value:.4f} "
            f"dom={domain_loss_value:.4f} "
            f"src={source_accuracy:.2f}% "
            f"target_train={target_adaptation_accuracy:.2f}% "
            f"usps={target_test_accuracy:.2f}% "
            f"domain_acc={current_domain_accuracy:.2f}%"
        )

    save_checkpoint(
        path=args.checkpoint,
        feature=feature,
        classifier=classifier,
        domain=domain,
        optimizer=optimizer,
        epoch=args.epochs,
        args=args,
        history=history,
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
            history,
            handle,
            indent=2,
        )

    final_row = history[-1]

    print()
    print("========================================")
    print("FINAL RESULT")
    print("========================================")
    print(
        f"final_source_accuracy="
        f"{final_row['source_accuracy']:.2f}%"
    )
    print(
        f"final_target_adaptation_accuracy="
        f"{final_row['target_adaptation_accuracy']:.2f}%"
    )
    print(
        f"final_usps_accuracy="
        f"{final_row['target_test_accuracy']:.2f}%"
    )
    print(
        f"best_usps_accuracy="
        f"{best_test_accuracy:.2f}%"
    )
    print(
        f"best_usps_epoch="
        f"{best_test_epoch}"
    )
    print(
        f"final_domain_accuracy="
        f"{final_row['domain_accuracy']:.2f}%"
    )
    print(
        f"checkpoint={args.checkpoint}"
    )
    print(
        f"history={args.history}"
    )


if __name__ == "__main__":
    main()