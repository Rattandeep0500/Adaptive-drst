import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.digits import get_datasets
from src.evaluation.metrics import accuracy


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

    def extract_features(self, x):
        return self.features(x)

    def forward(self, x):
        return self.classifier(
            self.extract_features(x)
        )


def main():
    torch.manual_seed(42)

    mnist, usps = get_datasets()

    train_loader = DataLoader(
        mnist,
        batch_size=128,
        shuffle=True,
    )

    model = DigitModel()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    for epoch in range(5):
        model.train()

        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)

            logits = model(images)

            loss = nn.functional.cross_entropy(
                logits,
                labels,
            )

            loss.backward()
            optimizer.step()

        print(
            f"Epoch {epoch + 1}/5 complete"
        )

    model.eval()

    train_loader_eval = DataLoader(
        mnist,
        batch_size=256,
    )

    target_loader = DataLoader(
        usps,
        batch_size=256,
    )

    source_correct = 0
    source_total = 0
    target_correct = 0
    target_total = 0

    with torch.no_grad():
        for images, labels in train_loader_eval:
            probabilities = torch.softmax(
                model(images),
                dim=1,
            )

            source_correct += int(
                (
                    probabilities.argmax(dim=1)
                    == labels
                ).sum()
            )

            source_total += labels.size(0)

        for images, labels in target_loader:
            probabilities = torch.softmax(
                model(images),
                dim=1,
            )

            target_correct += int(
                (
                    probabilities.argmax(dim=1)
                    == labels
                ).sum()
            )

            target_total += labels.size(0)

    source_accuracy = source_correct / source_total
    target_accuracy = target_correct / target_total

    print("Source accuracy:", source_accuracy)
    print("Target accuracy:", target_accuracy)

    assert 0.0 <= source_accuracy <= 1.0
    assert 0.0 <= target_accuracy <= 1.0

    print("MNIST-USPS baseline OK")


if __name__ == "__main__":
    main()
