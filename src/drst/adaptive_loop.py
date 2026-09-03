import torch

from src.drl.predictor import drl_probabilities
from src.drst.adaptive_selector import adaptive_select_pseudo_labels


def generate_adaptive_pseudo_labels(
    classifier,
    domain_network,
    target_x,
    min_threshold=0.70,
    max_threshold=0.95,
    max_per_class=None,
):
    classifier.eval()
    domain_network.eval()

    with torch.no_grad():
        logits = classifier(target_x)
        domain_probs = domain_network.domain_probabilities(target_x)

        source_prob = domain_probs[:, 0].clamp_min(1e-8)
        target_prob = domain_probs[:, 1].clamp_min(1e-8)
        density_ratio = source_prob / target_prob

        probabilities = drl_probabilities(
            logits,
            density_ratio,
        )

    return adaptive_select_pseudo_labels(
        probabilities,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
        max_per_class=max_per_class,
    )


class AdaptiveDRSTLoop:
    def __init__(
        self,
        classifier,
        domain_network,
        trainer,
        min_threshold=0.70,
        max_threshold=0.95,
        max_per_class=None,
        iterations=5,
    ):
        self.classifier = classifier
        self.domain_network = domain_network
        self.trainer = trainer
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.max_per_class = max_per_class
        self.iterations = iterations

    def run_iteration(self, source_x, source_y, target_x):
        (
            selected_indices,
            pseudo_labels,
            confidences,
            thresholds,
        ) = generate_adaptive_pseudo_labels(
            self.classifier,
            self.domain_network,
            target_x,
            self.min_threshold,
            self.max_threshold,
            self.max_per_class,
        )

        pseudo_x = target_x[selected_indices]

        mask = torch.ones(
            len(target_x),
            dtype=torch.bool,
            device=target_x.device,
        )
        mask[selected_indices] = False

        remaining_target_x = target_x[mask]

        updated_source_x = torch.cat(
            [source_x, pseudo_x],
            dim=0,
        )
        updated_source_y = torch.cat(
            [source_y, pseudo_labels],
            dim=0,
        )

        training_metrics = None

        if len(remaining_target_x) > 0:
            training_metrics = self.trainer.train_step(
                source_x=updated_source_x,
                source_y=updated_source_y,
                target_x=remaining_target_x,
            )

        class_histogram = torch.bincount(
            pseudo_labels,
            minlength=thresholds.shape[0],
        )

        return {
            "pseudo_x": pseudo_x,
            "pseudo_y": pseudo_labels,
            "source_x": updated_source_x,
            "source_y": updated_source_y,
            "remaining_target_x": remaining_target_x,
            "selected_indices": selected_indices,
            "confidences": confidences,
            "thresholds": thresholds,
            "class_histogram": class_histogram,
            "training_metrics": training_metrics,
        }

    def run(self, source_x, source_y, target_x):
        labeled_x = source_x.clone()
        labeled_y = source_y.clone()
        unlabeled_x = target_x.clone()
        history = []

        for iteration in range(1, self.iterations + 1):
            if len(unlabeled_x) == 0:
                break

            result = self.run_iteration(
                labeled_x,
                labeled_y,
                unlabeled_x,
            )

            selected_count = len(result["pseudo_x"])

            history.append({
                "iteration": iteration,
                "selected_count": selected_count,
                "training_count": len(result["source_x"]),
                "remaining_target_count": len(result["remaining_target_x"]),
                "mean_confidence": (
                    float(result["confidences"].mean())
                    if selected_count > 0 else 0.0
                ),
                "thresholds": result["thresholds"].detach().cpu(),
                "class_histogram": result["class_histogram"].detach().cpu(),
                "training_metrics": result["training_metrics"],
            })

            labeled_x = result["source_x"]
            labeled_y = result["source_y"]
            unlabeled_x = result["remaining_target_x"]

            if selected_count == 0:
                break

        return {
            "source_x": labeled_x,
            "source_y": labeled_y,
            "remaining_target_x": unlabeled_x,
            "history": history,
        }
