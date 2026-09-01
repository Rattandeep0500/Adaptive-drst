import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.drl.density_ratio import DensityRatioNetwork
from src.drl.trainer import DRLTrainer
from src.models.classifier import Classifier


def parameter_snapshot(model):
    return [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]


def parameters_changed(before, model) -> bool:
    for old, new in zip(before, model.parameters()):
        if not torch.allclose(old, new.detach()):
            return True

    return False


def main():
    torch.manual_seed(42)

    source_x = torch.randn(8, 4)
    target_x = torch.randn(8, 4) + 0.5
    source_y = torch.randint(0, 3, (8,))

    classifier = Classifier(
        input_dim=4,
        feature_dim=8,
        num_classes=3,
    )

    domain_network = DensityRatioNetwork(
        input_dim=4,
        hidden_dim=8,
    )

    classifier_optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=1e-3,
    )

    domain_optimizer = torch.optim.Adam(
        domain_network.parameters(),
        lr=1e-3,
    )

    trainer = DRLTrainer(
        classifier=classifier,
        domain_network=domain_network,
        classifier_optimizer=classifier_optimizer,
        domain_optimizer=domain_optimizer,
    )

    classifier_before = parameter_snapshot(classifier)
    domain_before = parameter_snapshot(domain_network)

    metrics = trainer.train_step(
        source_x=source_x,
        source_y=source_y,
        target_x=target_x,
    )

    classifier_changed = parameters_changed(
        classifier_before,
        classifier,
    )

    domain_changed = parameters_changed(
        domain_before,
        domain_network,
    )

    assert classifier_changed
    assert domain_changed

    print("Domain loss:", metrics["domain_loss"])
    print(
        "Classification loss:",
        metrics["classification_loss"],
    )
    print(
        "Target score:",
        metrics["target_score"],
    )
    print("Classifier parameters updated: OK")
    print("Domain parameters updated: OK")
    print("DRL training step OK")


if __name__ == "__main__":
    main()
