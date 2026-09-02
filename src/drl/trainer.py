"""
Algorithm 1 – End-to-end training for DRL.

Faithful implementation of the joint optimisation described in the paper.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .gradients import (
    classifier_loss_and_grads,
    total_domain_loss,
)
from .density_ratio import domain_logits_to_probs, density_ratio
from .predictor import drl_predict, standard_predict


class DRLTrainer:
    """
    Implements Algorithm 1 of the paper.

    Two distinct parameter groups / optimisers:
      - opt_domain  updates the domain head (and optionally the backbone)
      - opt_cls     updates the classification head (and the backbone)
    """

    def __init__(
        self,
        model: nn.Module,
        lr_domain: float = 1e-3,
        lr_cls: float = 1e-3,
        weight_decay: float = 1e-4,
        lambda_drl: float = 1.0,
        r: float = 0.0,
        device: str | torch.device = "cpu",
        share_backbone: bool = True,
    ):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.lambda_drl = lambda_drl
        self.r = r
        self.share_backbone = share_backbone

        # Parameter groups matching Algorithm 1
        domain_params = list(model.domain_head.parameters())
        cls_params = list(model.classifier.parameters())
        backbone_params = list(model.backbone.parameters())

        if share_backbone:
            # Backbone is updated by both steps (paper allows this)
            domain_params = domain_params + backbone_params
            cls_params = cls_params + backbone_params

        self.opt_domain = torch.optim.SGD(
            domain_params, lr=lr_domain, momentum=0.9, weight_decay=weight_decay
        )
        self.opt_cls = torch.optim.SGD(
            cls_params, lr=lr_cls, momentum=0.9, weight_decay=weight_decay
        )

    def _move(self, batch):
        return tuple(t.to(self.device) for t in batch)

    def train_step(
        self,
        src_x: torch.Tensor,
        src_y: torch.Tensor,
        tgt_x: torch.Tensor,
    ) -> Dict[str, float]:
        """
        One iteration of Algorithm 1 (lines 5–7).

        Returns a dict of scalar losses for logging.
        """
        self.model.train()
        B = src_x.size(0)

        # ------------------------------------------------------------------
        # 1. Forward pass on the concatenated batch
        # ------------------------------------------------------------------
        x_all = torch.cat([src_x, tgt_x], dim=0)
        domain_labels = torch.cat(
            [
                torch.zeros(B, dtype=torch.long, device=self.device),
                torch.ones(B, dtype=torch.long, device=self.device),
            ],
            dim=0,
        )

        feat_all = self.model.extract(x_all)
        domain_logits = self.model.domain(feat_all)
        feat_src = feat_all[:B]
        domain_logits_src = domain_logits[:B]

        # ------------------------------------------------------------------
        # 2. Update domain network (Algorithm 1 step 5)
        #    gradients from BOTH loss terms of Eq. (7)
        # ------------------------------------------------------------------
        self.opt_domain.zero_grad()
        loss_domain = total_domain_loss(
            domain_logits=domain_logits,
            domain_labels=domain_labels,
            features_src=feat_src,
            labels_src=src_y,
            classifier=self.model.classifier,
            lambda_drl=self.lambda_drl,
            r=self.r,
        )
        loss_domain.backward()
        self.opt_domain.step()

        # ------------------------------------------------------------------
        # 3. Recompute features / domain probs AFTER the domain update
        #    (the paper computes f with the freshly updated τ)
        # ------------------------------------------------------------------
        with torch.no_grad():
            feat_src = self.model.extract(src_x)
            domain_logits_src = self.model.domain(feat_src)
            tau_s, tau_t = domain_logits_to_probs(domain_logits_src)

        # ------------------------------------------------------------------
        # 4. Update classifier + backbone (Algorithm 1 step 7)
        #    using the derived source-side gradients
        # ------------------------------------------------------------------
        self.opt_cls.zero_grad()
        # We need a fresh forward that is connected to the graph
        feat_src = self.model.extract(src_x)
        loss_cls = classifier_loss_and_grads(
            features=feat_src,
            labels=src_y,
            classifier=self.model.classifier,
            tau_s=tau_s,
            tau_t=tau_t,
            r=self.r,
        )
        loss_cls.backward()
        self.opt_cls.step()

        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------
        with torch.no_grad():
            ratio = density_ratio(tau_s, tau_t)
            ratio_stats = {
                "ratio_mean": ratio.mean().item(),
                "ratio_std": ratio.std().item(),
                "ratio_min": ratio.min().item(),
                "ratio_max": ratio.max().item(),
            }

        return {
            "loss_domain": loss_domain.item(),
            "loss_cls": loss_cls.item(),
            **ratio_stats,
        }

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        use_drl: bool = True,
    ) -> Dict[str, float]:
        """
        Compute accuracy, mean confidence and Brier score on a labelled loader.
        Target labels are used **only** for evaluation, never for training.
        """
        self.model.eval()
        correct = 0
        total = 0
        conf_sum = 0.0
        brier_sum = 0.0

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            feat = self.model.extract(x)
            if use_drl:
                domain_logits = self.model.domain(feat)
                probs = drl_predict(feat, self.model.classifier, domain_logits, r=self.r)
            else:
                probs = standard_predict(feat, self.model.classifier)

            pred = probs.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            conf = probs.max(dim=1).values
            conf_sum += conf.sum().item()

            # Brier score (multi-class)
            onehot = F.one_hot(y, num_classes=probs.size(1)).float()
            brier_sum += ((probs - onehot) ** 2).sum(dim=1).sum().item()

        acc = 100.0 * correct / max(total, 1)
        mean_conf = conf_sum / max(total, 1)
        brier = brier_sum / max(total, 1)
        return {
            "accuracy": acc,
            "mean_confidence": mean_conf,
            "brier": brier,
            "n": total,
        }


# Convenience import for F.one_hot
import torch.nn.functional as F
