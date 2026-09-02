import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.digits import get_datasets
from src.evaluation.metrics import accuracy
from src.drl.predictor import drl_probabilities


class DigitDRLModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(128, 10)
        self.domain = nn.Linear(128, 2)

    def extract_features(self, x):
        return self.features(x)

    def classification_logits(self, x):
        return self.classifier(
            self.extract_features(x)
        )


def train_epoch(
    model,
    source_loader,
    target_loader,
    classifier_optimizer,
    domain_optimizer,
):
    model.train()

    total_classification_loss = 0.0
    total_domain_loss = 0.0
    total_target_score = 0.0
    steps = 0

    for (source_x, source_y), (target_x, _) in zip(
        source_loader,
        target_loader,
    ):
        domain_optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            source_features = model.extract_features(source_x)
            target_features = model.extract_features(target_x)

        target_domain_probs = torch.softmax(
            model.domain(target_features),
            dim=1,
        )

        ds = target_domain_probs[:, 0].clamp_min(1e-8)
        dt = target_domain_probs[:, 1].clamp_min(1e-8)

        with torch.no_grad():
            target_logits = model.classifier(
                target_features
            )

            ratio = ds / dt

            drl_logits = target_logits * ratio.unsqueeze(1)

            target_probabilities = torch.softmax(
                drl_logits,
                dim=1,
            )

            expected_score = (
                target_probabilities * target_logits
            ).sum(dim=1)

        grad_ds = expected_score / dt
        grad_dt = (
            -(ds / (dt ** 2))
            * expected_score
        )

        torch.autograd.backward(
            [ds, dt],
            [grad_ds, grad_dt],
        )

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
                ),
                torch.ones(
                    len(target_x),
                    dtype=torch.long,
                ),
            ]
        )

        domain_logits = model.domain(
            domain_features
        )

        domain_loss = nn.functional.cross_entropy(
            domain_logits,
            domain_labels,
        )

        domain_loss.backward()
        domain_optimizer.step()

        classifier_optimizer.zero_grad(set_to_none=True)

        source_features = model.extract_features(source_x)
        source_logits = model.classifier(
            source_features
        )

        with torch.no_grad():
            domain_probs = torch.softmax(
                model.domain(
                    source_features.detach()
                ),
                dim=1,
            )

            source_prob = domain_probs[:, 0].clamp_min(
                1e-8
            )

            target_prob = domain_probs[:, 1].clamp_min(
                1e-8
            )

            ratio = source_prob / target_prob

        drl_logits = source_logits * ratio.unsqueeze(1)

        classification_loss = nn.functional.cross_entropy(
            drl_logits,
            source_y,
        )

        classification_loss.backward()
        classifier_optimizer.step()

        total_classification_loss += float(
            classification_loss.detach()
        )

        total_domain_loss += float(
            domain_loss.detach()
        )

        total_target_score += float(
            expected_score.mean().detach()
        )

        steps += 1

    return {
        "classification_loss":
            total_classification_loss / max(steps, 1),
        "domain_loss":
            total_domain_loss / max(steps, 1),
        "target_score":
            total_target_score / max(steps, 1),
    }


def evaluate_source(
    model,
    loader,
):
    model.eval()

    probabilities = []
    labels = []

    with torch.no_grad():
        for images, y in loader:
            logits = model.classification_logits(images)

            probabilities.append(
                torch.softmax(logits, dim=1)
            )

            labels.append(y)

    return accuracy(
        torch.cat(probabilities),
        torch.cat(labels),
    )


def evaluate_drl(
    model,
    loader,
):
    model.eval()

    probabilities = []
    labels = []

    with torch.no_grad():
        for images, y in loader:
            features = model.extract_features(images)
            logits = model.classifier(features)

            domain_probs = torch.softmax(
                model.domain(features),
                dim=1,
            )

            ds = domain_probs[:, 0].clamp_min(1e-8)
            dt = domain_probs[:, 1].clamp_min(1e-8)

            ratio = ds / dt

            probabilities.append(
                drl_probabilities(
                    logits,
                    ratio,
                )
            )

            labels.append(y)

    return accuracy(
        torch.cat(probabilities),
        torch.cat(labels),
    )


def main():
    torch.manual_seed(42)

    mnist, usps = get_datasets()

    source_loader = DataLoader(
        mnist,
        batch_size=128,
        shuffle=True,
    )

    target_loader = DataLoader(
        usps,
        batch_size=128,
        shuffle=True,
    )

    target_eval_loader = DataLoader(
        usps,
        batch_size=256,
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

    for epoch in range(10):
        metrics = train_epoch(
            model,
            source_loader,
            target_loader,
            classifier_optimizer,
            domain_optimizer,
        )

        print(
            f"Epoch {epoch + 1}/10 "
            f"classification_loss="
            f"{metrics['classification_loss']:.4f} "
            f"domain_loss="
            f"{metrics['domain_loss']:.4f} "
            f"target_score="
            f"{metrics['target_score']:.4f}"
        )

    source_accuracy = evaluate_source(
        model,
        source_loader,
    )

    target_source_accuracy = evaluate_source(
        model,
        target_eval_loader,
    )

    target_drl_accuracy = evaluate_drl(
        model,
        target_eval_loader,
    )

    print("Source accuracy:", source_accuracy)
    print(
        "Target source-only accuracy:",
        target_source_accuracy,
    )
    print(
        "Target DRL accuracy:",
        target_drl_accuracy,
    )


if __name__ == "__main__":
    main()
