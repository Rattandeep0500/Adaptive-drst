import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


SEED = 42
SOURCE_SAMPLES = 2000
TARGET_SAMPLES = 1800
BATCH_SIZE = 128
EPOCHS = 5


class DigitModel(nn.Module):
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

    def forward(self, x):
        return self.classifier(
            self.features(x)
        )


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

    return mnist, usps_train, usps_test


def select_subset(dataset, count, seed):
    generator = random.Random(seed)

    indices = list(range(len(dataset)))
    generator.shuffle(indices)

    return Subset(
        dataset,
        indices[:count],
    )


def train(model, loader):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0.0
        steps = 0

        for images, labels in loader:
            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(images)

            loss = nn.functional.cross_entropy(
                logits,
                labels,
            )

            loss.backward()
            optimizer.step()

            total_loss += float(
                loss.detach()
            )
            steps += 1

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"loss={total_loss / max(steps, 1):.4f}"
        )


def evaluate(model, loader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images).argmax(
                dim=1
            )

            correct += int(
                (predictions == labels).sum()
            )

            total += labels.size(0)

    return correct / total


def main():
    torch.manual_seed(SEED)

    print("Loading MNIST -> USPS...")

    mnist, usps_train, usps_test = load_data()

    source_subset = select_subset(
        mnist,
        SOURCE_SAMPLES,
        SEED,
    )

    target_subset = select_subset(
        usps_train,
        TARGET_SAMPLES,
        SEED,
    )

    source_loader = DataLoader(
        source_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    target_adaptation_loader = DataLoader(
        target_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    target_test_loader = DataLoader(
        usps_test,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print(
        f"Source training samples: "
        f"{len(source_subset)}"
    )

    print(
        f"Target adaptation samples: "
        f"{len(target_subset)}"
    )

    print(
        f"Target test samples: "
        f"{len(usps_test)}"
    )

    model = DigitModel()

    print("\n=== SOURCE-ONLY ===")

    train(
        model,
        source_loader,
    )

    source_accuracy = evaluate(
        model,
        source_loader,
    )

    target_accuracy = evaluate(
        model,
        target_test_loader,
    )

    print(
        f"Source accuracy: "
        f"{source_accuracy * 100:.2f}%"
    )

    print(
        f"USPS target accuracy: "
        f"{target_accuracy * 100:.2f}%"
    )

    print(
        "\nMNIST -> USPS protocol benchmark complete."
    )


if __name__ == "__main__":
    main()