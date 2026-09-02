import torch
import torch.nn.functional as F


class ImageDRLTrainer:
    def __init__(
        self,
        model,
        classifier_optimizer,
        domain_optimizer,
    ):
        self.model = model
        self.classifier_optimizer = classifier_optimizer
        self.domain_optimizer = domain_optimizer

    def train_step(
        self,
        source_x,
        source_y,
        target_x,
    ):
        self.model.train()

        source_features = self.model.extract_features(source_x)
        target_features = self.model.extract_features(target_x)

        domain_input = torch.cat(
            [
                source_features.detach(),
                target_features.detach(),
            ],
            dim=0,
        )

        domain_logits = self.model.domain_network(domain_input)

        domain_labels = torch.cat(
            [
                torch.zeros(
                    source_x.size(0),
                    dtype=torch.long,
                    device=source_x.device,
                ),
                torch.ones(
                    target_x.size(0),
                    dtype=torch.long,
                    device=target_x.device,
                ),
            ]
        )

        domain_loss = F.cross_entropy(
            domain_logits,
            domain_labels,
        )

        self.domain_optimizer.zero_grad(set_to_none=True)
        domain_loss.backward()
        self.domain_optimizer.step()

        self.classifier_optimizer.zero_grad(set_to_none=True)

        source_features = self.model.extract_features(source_x)

        source_logits = self.model.classifier.backbone.fc(
            source_features
        )

        with torch.no_grad():
            source_domain_probs = (
                self.model.domain_network.domain_probabilities(
                    source_features.detach()
                )
            )

            source_prob = source_domain_probs[:, 0].clamp_min(1e-8)
            target_prob = source_domain_probs[:, 1].clamp_min(1e-8)

            ratio = source_prob / target_prob

        drl_logits = source_logits * ratio.unsqueeze(1)

        classification_loss = F.cross_entropy(
            drl_logits,
            source_y,
        )

        classification_loss.backward()
        self.classifier_optimizer.step()

        return {
            "domain_loss": float(domain_loss.detach()),
            "classification_loss": float(
                classification_loss.detach()
            ),
        }
