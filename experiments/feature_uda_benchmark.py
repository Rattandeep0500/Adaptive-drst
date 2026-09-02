import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from src.evaluation.metrics import accuracy


def make_data(
    prototypes,
    samples_per_class,
    shifts,
    noise,
):
    xs = []
    ys = []

    for class_id, prototype in enumerate(prototypes):
        x = (
            prototype
            + shifts[class_id]
            + noise * torch.randn(
                samples_per_class,
                prototype.numel(),
            )
        )

        y = torch.full(
            (samples_per_class,),
            class_id,
            dtype=torch.long,
        )

        xs.append(x)
        ys.append(y)

    return torch.cat(xs), torch.cat(ys)


def main():
    torch.manual_seed(42)

    num_classes = 12
    feature_dim = 64

    prototypes = torch.randn(
        num_classes,
        feature_dim,
    ) * 2.0

    source_shifts = torch.zeros(
        num_classes,
        feature_dim,
    )

    target_shifts = torch.randn(
        num_classes,
        feature_dim,
    ) * 1.5

    source_x, source_y = make_data(
        prototypes,
        100,
        source_shifts,
        1.0,
    )

    target_x, target_y = make_data(
        prototypes,
        100,
        target_shifts,
        1.5,
    )

    model = nn.Linear(
        feature_dim,
        num_classes,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-2,
    )

    for _ in range(100):
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
    print("Controlled domain-shift benchmark OK")


if __name__ == "__main__":
    main()
