import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.drl.density_ratio import DensityRatioNetwork
from src.drl.trainer import DRLTrainer
from src.drst.loop import DRSTLoop
from src.models.classifier import Classifier


def main():
    torch.manual_seed(42)

    source_x = torch.randn(16, 4)
    source_y = torch.randint(0, 3, (16,))
    target_x = torch.randn(20, 4)

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

    loop = DRSTLoop(
        classifier=classifier,
        domain_network=domain_network,
        trainer=trainer,
        pseudo_label_portion=0.2,
    )

    result = loop.run_iteration(
        source_x=source_x,
        source_y=source_y,
        target_x=target_x,
    )

    print(
        "Original source samples:",
        len(source_x),
    )

    print(
        "Selected pseudo-labeled samples:",
        len(result["pseudo_x"]),
    )

    print(
        "Updated training samples:",
        len(result["source_x"]),
    )

    print(
        "Remaining unlabeled target samples:",
        len(result["remaining_target_x"]),
    )

    print(
        "Training metrics:",
        result["training_metrics"],
    )

    assert len(result["pseudo_x"]) == 4
    assert len(result["source_x"]) == 20
    assert len(result["remaining_target_x"]) == 16
    assert result["training_metrics"] is not None

    print("Complete DRST iteration OK")


if __name__ == "__main__":
    main()
