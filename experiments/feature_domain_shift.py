import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from src.evaluation.metrics import accuracy


def make_data(n, shift):
    x = torch.randn(n, 2048) + shift
    y = torch.randint(0, 12, (n,))
    return x, y


def main():
    torch.manual_seed(42)

    source_x, source_y = make_data(1000, 0.0)
    target_x, target_y = make_data(500, 0.75)

    model = nn.Linear(2048, 12)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)

        logits = model(source_x)

        loss = nn.functional.cross_entropy(
            logits,
            source_y,
        )

        loss.backward()
        optimizer.step()

    model.eval()

    with torch.no_grad():
        source_probabilities = torch.softmax(
            model(source_x),
            dim=1,
        )

        target_probabilities = torch.softmax(
            model(target_x),
            dim=1,
        )

    source_accuracy = accuracy(
        source_probabilities,
        source_y,
    )

    target_accuracy = accuracy(
        target_probabilities,
        target_y,
    )

    print("Source accuracy:", source_accuracy)
    print("Target accuracy:", target_accuracy)
    print("Domain-shift baseline OK")


if __name__ == "__main__":
    main()
