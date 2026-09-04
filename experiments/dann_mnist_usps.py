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

    datasets_to_check = [
        (source, 0),
        (target, 1),
    ]

    with torch.no_grad():
        for dataset, domain_label in datasets_to_check:
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

                predicted = logits.argmax(
                    dim=1
                )

                labels = torch.full(
                    (x.size(0),),
                    domain_label,
                    dtype=torch.long,
                    device=device,
                )

                correct += (
                    predicted == labels
                ).sum().item()

                total += x.size(0)

    return 100.0 * correct / total


def main():
    seed = 42
    set_seed(seed)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    source_dataset, target_dataset, target_test = (
        load_data()
    )

    locked_batches = torch.load(
        "data/locked_digits_batches.pt",
        map_location="cpu",
    )

    feature_extractor = FeatureExtractor().to(
        device
    )

    classifier = Classifier().to(
        device
    )

    domain_discriminator = DomainDiscriminator().to(
        device
    )

    optimizer = torch.optim.Adam(
        list(feature_extractor.parameters())
        + list(classifier.parameters())
        + list(domain_discriminator.parameters()),
        lr=1e-3,
        weight_decay=1e-4,
    )

    classification_loss = nn.CrossEntropyLoss()
    domain_loss = nn.CrossEntropyLoss()

    epochs = 10
    lambda_domain = 0.10

    for epoch in range(epochs):
        feature_extractor.train()
        classifier.train()
        domain_discriminator.train()

        epoch_classification_loss = []
        epoch_domain_loss = []

        epoch_batches = [
            batch
            for batch in locked_batches
            if int(batch["epoch"]) < 5
            and int(batch["epoch"]) == epoch % 5
        ]

        for batch in epoch_batches:
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

            class_loss = classification_loss(
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

            d_loss = domain_loss(
                domain_logits,
                domain_labels,
            )

            loss = (
                class_loss
                + d_loss
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            optimizer.step()

            epoch_classification_loss.append(
                class_loss.detach().item()
            )

            epoch_domain_loss.append(
                d_loss.detach().item()
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

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Class Loss "
            f"{np.mean(epoch_classification_loss):.4f} | "
            f"Domain Loss "
            f"{np.mean(epoch_domain_loss):.4f} | "
            f"Source Acc "
            f"{source_accuracy:.2f}% | "
            f"USPS Acc "
            f"{target_accuracy:.2f}% | "
            f"Domain Acc "
            f"{domain_accuracy:.2f}%"
        )

    final_source = evaluate(
        feature_extractor,
        classifier,
        source_dataset,
        device,
    )

    final_target = evaluate(
        feature_extractor,
        classifier,
        target_test,
        device,
    )

    final_domain = evaluate_domain(
        feature_extractor,
        domain_discriminator,
        source_dataset,
        target_dataset,
        device,
    )

    print()
    print(
        f"Final Source Accuracy: "
        f"{final_source:.2f}%"
    )

    print(
        f"Final USPS Accuracy: "
        f"{final_target:.2f}%"
    )

    print(
        f"Final Domain Accuracy: "
        f"{final_domain:.2f}%"
    )


if __name__ == "__main__":
    main()