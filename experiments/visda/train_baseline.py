import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.models.domain_adapter import ImageDRLModel
from src.drl.image_trainer import ImageDRLTrainer


def train_one_epoch(
    model,
    trainer,
    source_loader,
    target_loader,
    device,
):
    model.train()

    total_domain = 0.0
    total_classification = 0.0
    steps = 0

    for (source_x, source_y), (target_x, _) in zip(
        source_loader,
        target_loader,
    ):
        source_x = source_x.to(device)
        source_y = source_y.to(device)
        target_x = target_x.to(device)

        metrics = trainer.train_step(
            source_x,
            source_y,
            target_x,
        )

        total_domain += metrics["domain_loss"]
        total_classification += metrics["classification_loss"]
        steps += 1

    return {
        "domain_loss": total_domain / max(steps, 1),
        "classification_loss": total_classification / max(steps, 1),
    }


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = ImageDRLModel(
        num_classes=12,
    ).to(device)

    classifier_optimizer = torch.optim.Adam(
        model.classifier.parameters(),
        lr=1e-4,
    )

    domain_optimizer = torch.optim.Adam(
        model.domain_network.parameters(),
        lr=1e-4,
    )

    trainer = ImageDRLTrainer(
        model,
        classifier_optimizer,
        domain_optimizer,
    )

    print("Device:", device)
    print("VisDA baseline trainer ready")


if __name__ == "__main__":
    main()
