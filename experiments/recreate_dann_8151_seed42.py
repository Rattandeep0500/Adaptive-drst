import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torchvision import datasets, transforms


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


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coefficient):
        ctx.coefficient = coefficient
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.coefficient * grad_output, None


def gradient_reverse(x, coefficient):
    return GradientReversalFunction.apply(
        x,
        coefficient,
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data():
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

    mnist = datasets.MNIST(
        root="data/digits",
        train=True,
        download=True,
        transform=transform,
    )

    usps_train = datasets.USPS(
        root="data/digits",
        train=True,
        download=True,
        transform=transform,
    )

    usps_test = datasets.USPS(
        root="data/digits",
        train=False,
        download=True,
        transform=transform,
    )

    source = Subset(
        mnist,
        list(range(2000)),
    )

    target = Subset(
        usps_train,
        list(range(1800)),
    )

    return source, target, usps_test


def collect_batch(dataset, indices, device):
    xs = []
    ys = []

    for index in indices:
        x, y = dataset[int(index)]
        xs.append(x)
        ys.append(y)

    x = torch.stack(
        xs,
        dim=0,
    ).to(device)

    y = torch.tensor(
        ys,
        dtype=torch.long,
        device=device,
    )

    return x, y


def collect_target_batch(
    dataset,
    indices,
    device,
):
    xs = []

    for index in indices:
        x, _ = dataset[int(index)]
        xs.append(x)

    return torch.stack(
        xs,
        dim=0,
    ).to(device)


def evaluate(
    feature_extractor,
    classifier,
    dataset,
    device,
):
    feature_extractor.eval()
    classifier.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for start in range(
            0,
            len(dataset),
            128,
        ):
            end = min(
                start + 128,
                len(dataset),
            )

            xs = []
            ys = []

            for i in range(
                start,
                end,
            ):
                x, y = dataset[i]
                xs.append(x)
                ys.append(y)

            x = torch.stack(
                xs,
                dim=0,
            ).to(device)

            y = torch.tensor(
                ys,
                dtype=torch.long,
                device=device,
            )

            features = feature_extractor(x)
            logits = classifier(features)

            correct += (
                logits.argmax(dim=1) == y
            ).sum().item()

            total += y.size(0)

    return 100.0 * correct / total


def evaluate_domain(
    feature_extractor,
    domain_discriminator,
    source,
    target,
    device,
):
    feature_extractor.eval()
    domain_discriminator.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for dataset, label in [
            (source, 0),
            (target, 1),
        ]:
            for start in range(
                0,
                len(dataset),
                128,
            ):
                end = min(
                    start + 128,
                    len(dataset),
                )

                xs = []

                for i in range(
                    start,
                    end,
                ):
                    x, _ = dataset[i]
                    xs.append(x)

                x = torch.stack(
                    xs,
                    dim=0,
                ).to(device)

                features = feature_extractor(x)
                logits = domain_discriminator(
                    features
                )

                predictions = logits.argmax(
                    dim=1
                )

                labels = torch.full(
                    (x.size(0),),
                    label,
                    dtype=torch.long,
                    device=device,
                )

                correct += (
                    predictions == labels
                ).sum().item()

                total += x.size(0)

    return 100.0 * correct / total


def predict_target(
    feature_extractor,
    classifier,
    target,
    device,
):
    feature_extractor.eval()
    classifier.eval()

    probabilities = []
    indices = []

    all_indices = list(
        range(len(target))
    )

    with torch.no_grad():
        for start in range(
            0,
            len(all_indices),
            128,
        ):
            batch_indices = all_indices[
                start:start + 128
            ]

            x = collect_target_batch(
                target,
                batch_indices,
                device,
            )

            features = feature_extractor(x)
            logits = classifier(features)
            probs = F.softmax(
                logits,
                dim=1,
            )

            probabilities.append(
                probs.cpu()
            )

            indices.extend(
                batch_indices
            )

    return (
        torch.cat(
            probabilities,
            dim=0,
        ),
        indices,
    )


def select_pseudo_labels(
    probabilities,
    indices,
    threshold,
    fraction,
):
    confidence, labels = probabilities.max(
        dim=1
    )

    valid = torch.where(
        confidence >= threshold
    )[0]

    if valid.numel() == 0:
        return [], [], []

    count = max(
        1,
        int(
            len(indices)
            * fraction
        ),
    )

    count = min(
        count,
        valid.numel(),
    )

    ranking = valid[
        torch.argsort(
            confidence[valid],
            descending=True,
        )
    ]

    selected_positions = ranking[
        :count
    ]

    selected_indices = [
        indices[int(i)]
        for i in selected_positions
    ]

    selected_labels = [
        int(labels[int(i)])
        for i in selected_positions
    ]

    selected_confidence = [
        float(confidence[int(i)])
        for i in selected_positions
    ]

    return (
        selected_indices,
        selected_labels,
        selected_confidence,
    )


def train_dann_epoch(
    feature_extractor,
    classifier,
    domain_discriminator,
    optimizer,
    source_dataset,
    target_dataset,
    batches,
    device,
    lambda_domain,
):
    feature_extractor.train()
    classifier.train()
    domain_discriminator.train()

    class_losses = []
    domain_losses = []

    for batch in batches:
        source_x, source_y = collect_batch(
            source_dataset,
            batch["source"],
            device,
        )

        target_x, _ = collect_batch(
            target_dataset,
            batch["target"],
            device,
        )

        source_features = feature_extractor(
            source_x
        )

        target_features = feature_extractor(
            target_x
        )

        source_logits = classifier(
            source_features
        )

        class_loss = F.cross_entropy(
            source_logits,
            source_y,
        )

        features = torch.cat(
            [
                source_features,
                target_features,
            ],
            dim=0,
        )

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
            ],
            dim=0,
        )

        reversed_features = gradient_reverse(
            features,
            lambda_domain,
        )

        domain_logits = domain_discriminator(
            reversed_features
        )

        domain_loss = F.cross_entropy(
            domain_logits,
            domain_labels,
        )

        loss = (
            class_loss
            + domain_loss
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        optimizer.step()

        class_losses.append(
            float(class_loss.detach())
        )

        domain_losses.append(
            float(domain_loss.detach())
        )

    return (
        float(np.mean(class_losses)),
        float(np.mean(domain_losses)),
    )


def train_pseudo_epoch(
    feature_extractor,
    classifier,
    domain_discriminator,
    optimizer,
    source_dataset,
    target_dataset,
    selected_indices,
    pseudo_labels,
    locked_source_batches,
    device,
    lambda_domain,
):
    feature_extractor.train()
    classifier.train()
    domain_discriminator.train()

    source_class_losses = []
    target_class_losses = []
    domain_losses = []

    target_order = list(
        range(
            len(selected_indices)
        )
    )

    random.shuffle(target_order)

    for batch in locked_source_batches:
        source_x, source_y = collect_batch(
            source_dataset,
            batch["source"],
            device,
        )

        source_features = feature_extractor(
            source_x
        )

        source_logits = classifier(
            source_features
        )

        source_loss = F.cross_entropy(
            source_logits,
            source_y,
        )

        target_positions = target_order[
            :len(batch["source"])
        ]

        if len(target_positions) > 0:
            target_batch_indices = [
                selected_indices[i]
                for i in target_positions
            ]

            target_batch_labels = torch.tensor(
                [
                    pseudo_labels[i]
                    for i in target_positions
                ],
                dtype=torch.long,
                device=device,
            )

            target_x = collect_target_batch(
                target_dataset,
                target_batch_indices,
                device,
            )

            target_features = feature_extractor(
                target_x
            )

            target_logits = classifier(
                target_features
            )

            target_loss = F.cross_entropy(
                target_logits,
                target_batch_labels,
            )
        else:
            target_features = None
            target_loss = torch.tensor(
                0.0,
                device=device,
            )

        target_domain_x = collect_target_batch(
            target_dataset,
            batch["target"],
            device,
        )

        target_domain_features = (
            feature_extractor(
                target_domain_x
            )
        )

        domain_features = torch.cat(
            [
                source_features,
                target_domain_features,
            ],
            dim=0,
        )

        domain_labels = torch.cat(
            [
                torch.zeros(
                    source_features.size(0),
                    dtype=torch.long,
                    device=device,
                ),
                torch.ones(
                    target_domain_features.size(0),
                    dtype=torch.long,
                    device=device,
                ),
            ],
            dim=0,
        )

        reversed_features = gradient_reverse(
            domain_features,
            lambda_domain,
        )

        domain_logits = domain_discriminator(
            reversed_features
        )

        domain_loss = F.cross_entropy(
            domain_logits,
            domain_labels,
        )

        loss = (
            source_loss
            + target_loss
            + domain_loss
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        optimizer.step()

        source_class_losses.append(
            float(source_loss.detach())
        )

        target_class_losses.append(
            float(target_loss.detach())
        )

        domain_losses.append(
            float(domain_loss.detach())
        )

        target_order = target_order[
            len(target_positions):
        ]

        if len(target_order) == 0:
            target_order = list(
                range(
                    len(selected_indices)
                )
            )

            random.shuffle(
                target_order
            )

    return (
        float(np.mean(source_class_losses)),
        float(np.mean(target_class_losses)),
        float(np.mean(domain_losses)),
    )



import json
from pathlib import Path


def main():
    seed = 42
    set_seed(seed)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    source_dataset, target_dataset, target_test = load_data()

    locked_batches = torch.load(
        "data/locked_digits_batches.pt",
        map_location="cpu",
    )

    feature_extractor = FeatureExtractor().to(device)
    classifier = Classifier().to(device)
    domain_discriminator = DomainDiscriminator().to(device)

    optimizer = torch.optim.Adam(
        list(feature_extractor.parameters())
        + list(classifier.parameters())
        + list(domain_discriminator.parameters()),
        lr=1e-3,
        weight_decay=1e-4,
    )

    lambda_domain = 0.10
    history = []

    for epoch in range(10):
        batches = [
            batch
            for batch in locked_batches
            if int(batch["epoch"]) == epoch % 5
        ]

        class_loss, domain_loss = train_dann_epoch(
            feature_extractor,
            classifier,
            domain_discriminator,
            optimizer,
            source_dataset,
            target_dataset,
            batches,
            device,
            lambda_domain,
        )

        source_accuracy = evaluate(
            feature_extractor,
            classifier,
            source_dataset,
            device,
        )

        target_accuracy = evaluate(
            feature_extractor,
            classifier,
            target_test,
            device,
        )

        domain_accuracy = evaluate_domain(
            feature_extractor,
            domain_discriminator,
            source_dataset,
            target_dataset,
            device,
        )

        row = {
            "epoch": epoch + 1,
            "class_loss": class_loss,
            "domain_loss": domain_loss,
            "source_accuracy": source_accuracy,
            "usps_accuracy": target_accuracy,
            "domain_accuracy": domain_accuracy,
        }

        history.append(row)

        print(
            f"DANN Epoch {epoch + 1}/10 | "
            f"Class Loss {class_loss:.4f} | "
            f"Domain Loss {domain_loss:.4f} | "
            f"Source Acc {source_accuracy:.2f}% | "
            f"USPS Acc {target_accuracy:.2f}% | "
            f"Domain Acc {domain_accuracy:.2f}%"
        )

    dann_target_accuracy = evaluate(
        feature_extractor,
        classifier,
        target_test,
        device,
    )

    torch.save(
        {
            "feature_extractor": feature_extractor.state_dict(),
            "classifier": classifier.state_dict(),
            "domain_discriminator": domain_discriminator.state_dict(),
            "seed": seed,
            "epochs": 10,
            "batch_size": 16,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "lambda_domain": 0.10,
            "protocol": {
                "source_samples": 2000,
                "target_adaptation_samples": 1800,
                "target_test_samples": len(target_test),
                "transform": "Resize(28,28)+ToTensor",
            },
            "usps_accuracy": dann_target_accuracy,
            "history": history,
        },
        "checkpoints/dann_81_51_seed42.pt",
    )

    Path("checkpoints").mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        "checkpoints/dann_81_51_seed42.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "usps_accuracy": dann_target_accuracy,
                "history": history,
            },
            handle,
            indent=2,
        )

    print()
    print("DANN Baseline USPS Accuracy:")
    print(f"{dann_target_accuracy:.2f}%")
    print("Saved checkpoint: checkpoints/dann_81_51_seed42.pt")


if __name__ == "__main__":
    main()
