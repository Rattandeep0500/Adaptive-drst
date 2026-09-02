import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import accuracy


def main():
    torch.manual_seed(42)

    source_x = torch.randn(200, 2048)
    source_y = torch.randint(0, 12, (200,))

    loader = DataLoader(
        TensorDataset(source_x, source_y),
        batch_size=32,
        shuffle=True,
    )

    model = nn.Linear(2048, 12)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    for _ in range(10):
        model.train()

        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)

            logits = model(x)

            loss = nn.functional.cross_entropy(
                logits,
                y,
            )

            loss.backward()
            optimizer.step()

    model.eval()

    with torch.no_grad():
        probabilities = torch.softmax(
            model(source_x),
            dim=1,
        )

    acc = accuracy(
        probabilities,
        source_y,
    )

    print("Source feature accuracy:", acc)

    assert 0.0 <= acc <= 1.0

    print("Feature baseline OK")


if __name__ == "__main__":
    main()
