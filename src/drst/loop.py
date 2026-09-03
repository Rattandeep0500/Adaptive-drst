import torch

from src.drst.drst import generate_pseudo_labels


class DRSTLoop:
    """
    Iterative DRST self-training loop.

    Each iteration:
        1. Select confident target samples.
        2. Assign pseudo-labels.
        3. Add them to the labeled training pool.
        4. Train DRL on the expanded pool.
        5. Remove selected samples from the unlabeled pool.
        6. Repeat.
    """

    def __init__(
        self,
        classifier,
        domain_network,
        trainer,
        pseudo_label_portion: float = 0.2,
        iterations: int = 5,
    ) -> None:

        if not 0.0 < pseudo_label_portion <= 1.0:
            raise ValueError(
                "pseudo_label_portion must be in (0, 1]"
            )

        if iterations < 1:
            raise ValueError(
                "iterations must be >= 1"
            )

        self.classifier = classifier
        self.domain_network = domain_network
        self.trainer = trainer
        self.pseudo_label_portion = pseudo_label_portion
        self.iterations = iterations

    def run(
        self,
        source_x: torch.Tensor,
        source_y: torch.Tensor,
        target_x: torch.Tensor,
    ) -> dict:

        labeled_x = source_x.clone()
        labeled_y = source_y.clone()
        unlabeled_x = target_x.clone()

        history = []

        for iteration in range(1, self.iterations + 1):

            if len(unlabeled_x) == 0:
                break

            (
                selected_indices,
                pseudo_labels,
                confidences,
                probabilities,
            ) = generate_pseudo_labels(
                classifier=self.classifier,
                domain_network=self.domain_network,
                target_x=unlabeled_x,
                portion=self.pseudo_label_portion,
            )

            pseudo_x = unlabeled_x[selected_indices]

            mask = torch.ones(
                len(unlabeled_x),
                dtype=torch.bool,
                device=unlabeled_x.device,
            )

            mask[selected_indices] = False

            remaining_x = unlabeled_x[mask]

            labeled_x = torch.cat(
                [labeled_x, pseudo_x],
                dim=0,
            )

            labeled_y = torch.cat(
                [labeled_y, pseudo_labels],
                dim=0,
            )

            training_metrics = None

            if len(remaining_x) > 0:
                training_metrics = self.trainer.train_step(
                    source_x=labeled_x,
                    source_y=labeled_y,
                    target_x=remaining_x,
                )

            iteration_result = {
                "iteration": iteration,
                "selected_count": len(pseudo_x),
                "training_count": len(labeled_x),
                "remaining_target_count": len(remaining_x),
                "mean_confidence": float(
                    confidences.mean().detach()
                ),
                "training_metrics": training_metrics,
            }

            history.append(iteration_result)

            unlabeled_x = remaining_x

        return {
            "source_x": labeled_x,
            "source_y": labeled_y,
            "remaining_target_x": unlabeled_x,
            "history": history,
        }
    def run_iteration(
        self,
        source_x: torch.Tensor,
        source_y: torch.Tensor,
        target_x: torch.Tensor,
    ) -> dict:
        result = self.run(
            source_x=source_x,
            source_y=source_y,
            target_x=target_x,
        )

        first = result["history"][0]

        selected_count = first["selected_count"]

        return {
            "pseudo_x": result["source_x"][-selected_count:],
            "pseudo_y": result["source_y"][-selected_count:],
            "source_x": result["source_x"],
            "source_y": result["source_y"],
            "remaining_target_x": result["remaining_target_x"],
            "training_metrics": first["training_metrics"],
        }
    def run_iteration(
        self,
        source_x: torch.Tensor,
        source_y: torch.Tensor,
        target_x: torch.Tensor,
    ) -> dict:
        (
            selected_indices,
            pseudo_labels,
            confidences,
            probabilities,
        ) = generate_pseudo_labels(
            classifier=self.classifier,
            domain_network=self.domain_network,
            target_x=target_x,
            portion=self.pseudo_label_portion,
        )

        pseudo_x = target_x[selected_indices]

        mask = torch.ones(
            len(target_x),
            dtype=torch.bool,
            device=target_x.device,
        )
        mask[selected_indices] = False

        remaining_x = target_x[mask]

        updated_source_x = torch.cat(
            [source_x, pseudo_x],
            dim=0,
        )
        updated_source_y = torch.cat(
            [source_y, pseudo_labels],
            dim=0,
        )

        training_metrics = None

        if len(remaining_x) > 0:
            training_metrics = self.trainer.train_step(
                source_x=updated_source_x,
                source_y=updated_source_y,
                target_x=remaining_x,
            )

        return {
            "pseudo_x": pseudo_x,
            "pseudo_y": pseudo_labels,
            "source_x": updated_source_x,
            "source_y": updated_source_y,
            "remaining_target_x": remaining_x,
            "training_metrics": training_metrics,
        }
