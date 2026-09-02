import torch

from src.drl.image_predictor import image_drl_probabilities
from src.drst.self_training import select_pseudo_labels


class ImageDRSTLoop:
    def __init__(
        self,
        model,
        trainer,
        pseudo_label_portion=0.2,
        iterations=3,
    ):
        if not 0.0 < pseudo_label_portion <= 1.0:
            raise ValueError("pseudo_label_portion must be in (0, 1]")

        if iterations < 1:
            raise ValueError("iterations must be >= 1")

        self.model = model
        self.trainer = trainer
        self.pseudo_label_portion = pseudo_label_portion
        self.iterations = iterations

    def run(
        self,
        source_x,
        source_y,
        target_x,
    ):
        labeled_x = source_x
        labeled_y = source_y
        unlabeled_x = target_x
        history = []

        for iteration in range(1, self.iterations + 1):
            if len(unlabeled_x) == 0:
                break

            probabilities = image_drl_probabilities(
                self.model,
                unlabeled_x,
            )

            indices, pseudo_labels, confidences = (
                select_pseudo_labels(
                    probabilities,
                    self.pseudo_label_portion,
                )
            )

            pseudo_x = unlabeled_x[indices]

            mask = torch.ones(
                len(unlabeled_x),
                dtype=torch.bool,
            )

            mask[indices] = False

            remaining_x = unlabeled_x[mask]

            labeled_x = torch.cat(
                [labeled_x, pseudo_x],
                dim=0,
            )

            labeled_y = torch.cat(
                [labeled_y, pseudo_labels],
                dim=0,
            )

            metrics = None

            if len(remaining_x) > 0:
                metrics = self.trainer.train_step(
                    source_x=labeled_x,
                    source_y=labeled_y,
                    target_x=remaining_x,
                )

            history.append(
                {
                    "iteration": iteration,
                    "selected": len(pseudo_x),
                    "training_size": len(labeled_x),
                    "remaining_target": len(remaining_x),
                    "mean_confidence": float(
                        confidences.mean().detach()
                    ),
                    "metrics": metrics,
                }
            )

            unlabeled_x = remaining_x

        return {
            "source_x": labeled_x,
            "source_y": labeled_y,
            "remaining_target_x": unlabeled_x,
            "history": history,
        }
