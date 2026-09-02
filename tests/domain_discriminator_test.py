import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from src.data.digits import get_datasets
from experiments.digits_drl import DigitDRLModel


def main():
    torch.manual_seed(42)

    mnist, usps = get_datasets()

    source_loader = DataLoader(
        mnist,
        batch_size=256,
        shuffle=True,
    )

    target_loader = DataLoader(
        usps,
        batch_size=256,
        shuffle=True,
    )

    model = DigitDRLModel()

    optimizer = torch.optim.Adam(
        model.domain.parameters(),
        lr=1e-3,
    )

    for _ in range(20):
        source_x, _ = next(iter(source_loader))
        target_x, _ = next(iter(target_loader))

        source_features = model.extract_features(source_x).detach()
        target_features = model.extract_features(target_x).detach()

        features = torch.cat(
            [source_features, target_features],
            dim=0,
        )

        labels = torch.cat(
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

        optimizer.zero_grad(set_to_none=True)

        logits = model.domain(features)
        loss = torch.nn.functional.cross_entropy(
            logits,
            labels,
        )

        loss.backward()
        optimizer.step()

    model.eval()

    source_x, _ = next(iter(source_loader))
    target_x, _ = next(iter(target_loader))

    with torch.no_grad():
        source_pred = model.domain(
            model.extract_features(source_x)
        ).argmax(dim=1)

        target_pred = model.domain(
            model.extract_features(target_x)
        ).argmax(dim=1)

    source_accuracy = (
        source_pred == 0
    ).float().mean()

    target_accuracy = (
        target_pred == 1
    ).float().mean()

    print(
        "Source domain accuracy:",
        float(source_accuracy),
    )

    print(
        "Target domain accuracy:",
        float(target_accuracy),
    )

    print("Domain discriminator test OK")


if __name__ == "__main__":
    main()
