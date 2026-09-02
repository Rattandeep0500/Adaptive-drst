from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .density_ratio import density_ratio, domain_logits_to_probs


def classifier_loss_and_grads(
    features,
    labels,
    classifier,
    tau_s,
    tau_t,
    r=0.0,
):
    """
    Compatibility API.

    The ratio is detached for the classifier update.
    """
    ratio = density_ratio(
        tau_s.detach().clamp_min(1e-8),
        tau_t.detach().clamp_min(1e-8),
    )

    z = classifier(features)

    # Scalar training loss retained for the existing test/API.
    return F.cross_entropy(
        ratio.unsqueeze(1) * z,
        labels,
    )


def domain_classification_loss(
    domain_logits,
    domain_labels,
):
    return F.cross_entropy(
        domain_logits,
        domain_labels,
    )


def domain_drl_regulariser(
    features,
    labels,
    classifier,
    domain_logits,
    r=0.0,
):
    """
    Backward-compatible scalar API used by the existing tests.

    This is kept only as a compatibility helper. The actual Algorithm-1
    trainer uses target_eq9_gradients() below for the Eq. (9) signal.
    """
    tau_s, tau_t = domain_logits_to_probs(
        domain_logits
    )

    ratio = density_ratio(
        tau_s.clamp_min(1e-8),
        tau_t.clamp_min(1e-8),
    ).unsqueeze(1)

    raw_scores = classifier(
        features
    ).detach()

    logits = ratio * raw_scores

    return F.cross_entropy(
        logits,
        labels,
    )


def target_eq9_gradients(
    target_features,
    classifier,
    target_domain_logits,
):
    """
    Equation (9) signal for the DOMAIN network.

    Uses unlabeled target inputs.
    """
    ds, dt = domain_logits_to_probs(
        target_domain_logits
    )

    ds = ds.clamp_min(1e-8)
    dt = dt.clamp_min(1e-8)

    with torch.no_grad():
        z = classifier(
            target_features.detach()
        )

        ratio = density_ratio(
            ds.detach(),
            dt.detach(),
        )

        p_hat = torch.softmax(
            ratio.unsqueeze(1) * z,
            dim=1,
        )

        expected_score = (
            p_hat * z
        ).sum(dim=1)

        g_ds = expected_score / dt
        g_dt = (
            -ds * expected_score
            / (dt * dt)
        )

        B = target_features.size(0)

        g_ds = g_ds / B
        g_dt = g_dt / B

    return ds, dt, g_ds, g_dt


def total_domain_loss(
    domain_logits,
    domain_labels,
    features_src=None,
    labels_src=None,
    classifier=None,
    lambda_drl=1.0,
    r=0.0,
):
    """
    Standard domain classification loss.

    Eq. (9) is intentionally handled separately by the trainer because
    it must be evaluated on unlabeled TARGET inputs.
    """
    return domain_classification_loss(
        domain_logits,
        domain_labels,
    )
