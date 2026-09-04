import copy
import random
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


SEED = 42

SOURCE_SAMPLES = 2000
TARGET_SAMPLES = 1800

BATCH_SIZE = 128
EPOCHS = 10

DATA_ROOT = "data/digits"

EPS = 1e-8


class DigitDRLModel(nn.Module):
    def __init__(
        self,
        feature_dim=128,
        hidden_dim=64,
        num_classes=10,
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(
            feature_dim,
            num_classes,
        )

        self.domain = nn.Sequential(
            nn.Linear(
                feature_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def extract_features(self, x):
        return self.features(x)

    def classification_logits(self, x):
        features = self.extract_features(x)
        return self.classifier(features)

    def domain_logits(self, features):
        return self.domain(features)

    def domain_probabilities(self, features):
        return torch.softmax(
            self.domain_logits(features),
            dim=1,
        )


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def load_datasets():
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

    mnist = datasets.MNIST(
        root=DATA_ROOT,
        train=True,
        download=True,
        transform=transform,
    )

    usps_train = datasets.USPS(
        root=DATA_ROOT,
        train=True,
        download=True,
        transform=transform,
    )

    usps_test = datasets.USPS(
        root=DATA_ROOT,
        train=False,
        download=True,
        transform=transform,
    )

    return (
        mnist,
        usps_train,
        usps_test,
    )


def seeded_subset(
    dataset,
    count,
    seed,
):
    generator = torch.Generator()
    generator.manual_seed(seed)

    indices = torch.randperm(
        len(dataset),
        generator=generator,
    )[:count]

    return Subset(
        dataset,
        indices.tolist(),
    )


def make_loaders(
    source_dataset,
    target_dataset,
):
    source_loader = DataLoader(
        source_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    target_loader = DataLoader(
        target_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    return (
        source_loader,
        target_loader,
    )


def next_batch(
    iterator,
    loader,
):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def density_ratio_from_probs(
    domain_probs,
):
    source_prob = domain_probs[:, 0].clamp_min(EPS)
    target_prob = domain_probs[:, 1].clamp_min(EPS)

    return source_prob / target_prob


def expected_score(
    classifier_logits,
    density_ratio,
):
    drl_logits = (
        classifier_logits
        * density_ratio.unsqueeze(1)
    )

    probabilities = torch.softmax(
        drl_logits,
        dim=1,
    )

    return (
        probabilities
        * classifier_logits
    ).sum(dim=1).mean()


def train_step(
    model,
    source_x,
    source_y,
    target_x,
    classifier_optimizer,
    domain_optimizer,
):
    # --------------------------------------------------
    # DOMAIN PHASE
    #
    # 1. Compute Eq. 9 task gradient.
    # 2. Compute domain classification gradient.
    # 3. One domain optimizer step.
    # --------------------------------------------------

    for parameter in model.classifier.parameters():
        parameter.requires_grad_(False)

    for parameter in model.features.parameters():
        parameter.requires_grad_(False)

    for parameter in model.domain.parameters():
        parameter.requires_grad_(True)

    domain_optimizer.zero_grad(
        set_to_none=True
    )

    with torch.no_grad():
        source_features = model.extract_features(
            source_x
        )

        target_features = model.extract_features(
            target_x
        )

    # Eq. 9 target-domain task gradient.
    target_domain_probs = model.domain_probabilities(
        target_features
    )

    ds = target_domain_probs[:, 0].clamp_min(EPS)
    dt = target_domain_probs[:, 1].clamp_min(EPS)

    with torch.no_grad():
        target_logits = model.classifier(
            target_features
        )

        ratio = ds / dt

        score = expected_score(
            target_logits,
            ratio,
        )

    grad_ds = score / dt

    grad_dt = (
        -(
            ds
            / (dt ** 2)
        )
        * score
    )

    torch.autograd.backward(
        tensors=[ds, dt],
        grad_tensors=[
            grad_ds,
            grad_dt,
        ],
    )

    # Domain classification gradient.
    domain_features = torch.cat(
        [
            source_features,
            target_features,
        ],
        dim=0,
    )

    domain_labels = torch.cat(
        [
            torch.zeros(
                len(source_x),
                dtype=torch.long,
                device=source_x.device,
            ),
            torch.ones(
                len(target_x),
                dtype=torch.long,
                device=target_x.device,
            ),
        ],
        dim=0,
    )

    domain_logits = model.domain_logits(
        domain_features
    )

    domain_loss = F.cross_entropy(
        domain_logits,
        domain_labels,
    )

    domain_loss.backward()

    # ONE domain update.
    domain_optimizer.step()

    # --------------------------------------------------
    # CLASSIFIER PHASE
    #
    # Recompute the ratio AFTER the domain update.
    # --------------------------------------------------

    for parameter in model.features.parameters():
        parameter.requires_grad_(True)

    for parameter in model.classifier.parameters():
        parameter.requires_grad_(True)

    for parameter in model.domain.parameters():
        parameter.requires_grad_(False)

    classifier_optimizer.zero_grad(
        set_to_none=True
    )

    source_features = model.extract_features(
        source_x
    )

    source_logits = model.classifier(
        source_features
    )

    with torch.no_grad():
        updated_domain_probs = (
            model.domain_probabilities(
                source_features
            )
        )

        density_ratio = (
            density_ratio_from_probs(
                updated_domain_probs
            )
        )

    drl_logits = (
        source_logits
        * density_ratio.unsqueeze(1)
    )

    classification_loss = F.cross_entropy(
        drl_logits,
        source_y,
    )

    classification_loss.backward()

    classifier_optimizer.step()

    return {
        "domain_loss": float(
            domain_loss.detach()
        ),
        "classification_loss": float(
            classification_loss.detach()
        ),
        "target_score": float(
            score.detach()
        ),
    }


def train_epoch(
    model,
    source_loader,
    target_loader,
    classifier_optimizer,
    domain_optimizer,
):
    model.train()

    target_iterator = iter(
        target_loader
    )

    total_domain_loss = 0.0
    total_classification_loss = 0.0
    total_target_score = 0.0

    ratio_values = []

    steps = 0

    for source_x, source_y in source_loader:
        (target_x, _), target_iterator = next_batch(
            target_iterator,
            target_loader,
        )

        metrics = train_step(
            model=model,
            source_x=source_x,
            source_y=source_y,
            target_x=target_x,
            classifier_optimizer=classifier_optimizer,
            domain_optimizer=domain_optimizer,
        )

        total_domain_loss += metrics[
            "domain_loss"
        ]

        total_classification_loss += metrics[
            "classification_loss"
        ]

        total_target_score += metrics[
            "target_score"
        ]

        with torch.no_grad():
            target_features = model.extract_features(
                target_x
            )

            target_probs = (
                model.domain_probabilities(
                    target_features
                )
            )

            ratios = density_ratio_from_probs(
                target_probs
            )

            ratio_values.append(
                ratios.detach().cpu()
            )

        steps += 1

    ratios = torch.cat(
        ratio_values
    )

    return {
        "domain_loss":
            total_domain_loss / max(steps, 1),

        "classification_loss":
            total_classification_loss
            / max(steps, 1),

        "target_score":
            total_target_score
            / max(steps, 1),

        "ratio_min":
            float(ratios.min()),

        "ratio_median":
            float(ratios.median()),

        "ratio_max":
            float(ratios.max()),

        "ratio_mean":
            float(ratios.mean()),
    }


@torch.no_grad()
def evaluate_source(
    model,
    loader,
):
    model.eval()

    correct = 0
    total = 0

    for images, labels in loader:
        logits = model.classification_logits(
            images
        )

        predictions = logits.argmax(
            dim=1
        )

        correct += int(
            (predictions == labels).sum()
        )

        total += labels.size(0)

    return correct / total


@torch.no_grad()
def evaluate_drl(
    model,
    loader,
):
    model.eval()

    correct = 0
    total = 0

    for images, labels in loader:
        features = model.extract_features(
            images
        )

        logits = model.classifier(
            features
        )

        domain_probs = (
            model.domain_probabilities(
                features
            )
        )

        density_ratio = (
            density_ratio_from_probs(
                domain_probs
            )
        )

        drl_logits = (
            logits
            * density_ratio.unsqueeze(1)
        )

        predictions = drl_logits.argmax(
            dim=1
        )

        correct += int(
            (predictions == labels).sum()
        )

        total += labels.size(0)

    return correct / total


def main():
    seed_everything(SEED)

    print(
        "=== LOCKED MNIST -> USPS DRL ==="
    )

    (
        mnist,
        usps_train,
        usps_test,
    ) = load_datasets()

    source_dataset = seeded_subset(
        mnist,
        SOURCE_SAMPLES,
        SEED,
    )

    target_dataset = seeded_subset(
        usps_train,
        TARGET_SAMPLES,
        SEED,
    )

    source_eval_loader = DataLoader(
        source_dataset,
        batch_size=256,
        shuffle=False,
    )

    target_eval_loader = DataLoader(
        usps_test,
        batch_size=256,
        shuffle=False,
    )

    source_loader, target_loader = (
        make_loaders(
            source_dataset,
            target_dataset,
        )
    )

    print(
        f"Source training samples: "
        f"{len(source_dataset)}"
    )

    print(
        f"Target adaptation samples: "
        f"{len(target_dataset)}"
    )

    print(
        f"Target evaluation samples: "
        f"{len(usps_test)}"
    )

    model = DigitDRLModel()

    classifier_optimizer = torch.optim.Adam(
        list(model.features.parameters())
        + list(model.classifier.parameters()),
        lr=1e-3,
    )

    domain_optimizer = torch.optim.Adam(
        model.domain.parameters(),
        lr=1e-4,
    )

    print(
        "\n=== TRAINING DRL ==="
    )

    for epoch in range(EPOCHS):
        metrics = train_epoch(
            model=model,
            source_loader=source_loader,
            target_loader=target_loader,
            classifier_optimizer=classifier_optimizer,
            domain_optimizer=domain_optimizer,
        )

        print(
            f"epoch={epoch + 1}/{EPOCHS} "
            f"domain="
            f"{metrics['domain_loss']:.4f} "
            f"cls="
            f"{metrics['classification_loss']:.4f} "
            f"score="
            f"{metrics['target_score']:.4f} "
            f"ratio_min="
            f"{metrics['ratio_min']:.4f} "
            f"ratio_med="
            f"{metrics['ratio_median']:.4f} "
            f"ratio_max="
            f"{metrics['ratio_max']:.4f}"
        )

    source_accuracy = evaluate_source(
        model,
        source_eval_loader,
    )

    target_source_only_model = copy.deepcopy(
        model
    )

    # Evaluate ordinary classifier logits
    # on the USPS test set.
    target_source_accuracy = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        target_source_only_model.eval()

        for images, labels in target_eval_loader:
            logits = target_source_only_model.classification_logits(
                images
            )

            predictions = logits.argmax(
                dim=1
            )

            correct += int(
                (predictions == labels).sum()
            )

            total += labels.size(0)

    target_source_accuracy = (
        correct / total
    )

    target_drl_accuracy = evaluate_drl(
        model,
        target_eval_loader,
    )

    print(
        "\n=== FINAL ==="
    )

    print(
        f"Source accuracy: "
        f"{source_accuracy * 100:.2f}%"
    )

    print(
        f"USPS source-only accuracy: "
        f"{target_source_accuracy * 100:.2f}%"
    )

    print(
        f"USPS DRL accuracy: "
        f"{target_drl_accuracy * 100:.2f}%"
    )


if __name__ == "__main__":
    main()