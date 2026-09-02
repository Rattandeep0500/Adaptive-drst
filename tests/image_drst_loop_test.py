import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.drl.image_trainer import ImageDRLTrainer
from src.drst.image_loop import ImageDRSTLoop
from src.models.domain_adapter import ImageDRLModel


def main():
    torch.manual_seed(42)

    source_x = torch.randn(4, 3, 224, 224)
    source_y = torch.randint(0, 12, (4,))
    target_x = torch.randn(6, 3, 224, 224)

    model = ImageDRLModel()

    classifier_optimizer = torch.optim.Adam(
        model.classifier.parameters(),
        lr=1e-4,
    )

    domain_optimizer = torch.optim.Adam(
        model.domain_network.parameters(),
        lr=1e-4,
    )

    trainer = ImageDRLTrainer(
        model=model,
        classifier_optimizer=classifier_optimizer,
        domain_optimizer=domain_optimizer,
    )

    loop = ImageDRSTLoop(
        model=model,
        trainer=trainer,
        pseudo_label_portion=0.5,
        iterations=2,
    )

    result = loop.run(
        source_x=source_x,
        source_y=source_y,
        target_x=target_x,
    )

    for item in result["history"]:
        print(
            f"Iteration {item['iteration']}: "
            f"selected={item['selected']}, "
            f"training={item['training_size']}, "
            f"remaining={item['remaining_target']}, "
            f"confidence={item['mean_confidence']:.4f}"
        )

    print(
        "Final training samples:",
        len(result["source_x"]),
    )

    print(
        "Remaining target samples:",
        len(result["remaining_target_x"]),
    )

    assert len(result["history"]) == 2
    assert len(result["source_x"]) > len(source_x)
    assert len(result["remaining_target_x"]) < len(target_x)

    print("Image DRST loop OK")


if __name__ == "__main__":
    main()
