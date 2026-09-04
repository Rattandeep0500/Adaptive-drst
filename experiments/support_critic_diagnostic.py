import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torchvision import datasets, transforms

from src.drl.faithful_trainer import FaithfulDRLTrainer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_data(dataset):
    xs = []
    ys = []

    for i in range(len(dataset)):
        x, y = dataset[i]
        xs.append(x.reshape(-1))
        ys.append(y)

    return (
        torch.stack(xs),
        torch.tensor(ys, dtype=torch.long),
    )


def get_locked_batch(dataset, indices, device):
    subset_indices = dataset.indices
    base_dataset = dataset.dataset

    xs = []
    ys = []

    for index in indices:
        real_index = subset_indices[int(index)]
        x, y = base_dataset[real_index]

        xs.append(x.reshape(-1))
        ys.append(y)

    return (
        torch.stack(xs).to(device),
        torch.tensor(
            ys,
            dtype=torch.long,
            device=device,
        ),
    )


class SupportCritic(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return self.encoder(x)


def cosine_support(
    target_features,
    source_features,
    source_labels,
    target_labels=None,
):
    target_norm = F.normalize(
        target_features,
        dim=1,
    )

    source_norm = F.normalize(
        source_features,
        dim=1,
    )

    similarities = (
        target_norm
        @ source_norm.t()
    )

    support_values = similarities.max(
        dim=1
    ).values

    return support_values


def train_support_critic(
    source_x,
    source_y,
    device,
):
    model = SupportCritic(
        input_dim=784,
        hidden_dim=256,
    ).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-3,
        momentum=0.9,
        weight_decay=5e-4,
    )

    model.train()

    for _ in range(5):
        permutation = torch.randperm(
            source_x.size(0),
            device=device,
        )

        for start in range(
            0,
            source_x.size(0),
            64,
        ):
            idx = permutation[
                start:start + 64
            ]

            x = source_x[idx]

            features = model(x)

            normalized = F.normalize(
                features,
                dim=1,
            )

            similarity = (
                normalized
                @ normalized.t()
            )

            labels_equal = (
                source_y[idx].unsqueeze(1)
                == source_y[idx].unsqueeze(0)
            )

            positive_mask = (
                labels_equal
                & ~torch.eye(
                    len(idx),
                    dtype=torch.bool,
                    device=device,
                )
            )

            if not positive_mask.any():
                continue

            positive_scores = similarity[
                positive_mask
            ]

            loss = 1.0 - positive_scores.mean()

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            optimizer.step()

    return model


def cosine_source_support(
    support_model,
    source_features,
    source_labels,
    target_features,
):
    source_norm = F.normalize(
        source_features,
        dim=1,
    )

    target_norm = F.normalize(
        target_features,
        dim=1,
    )

    similarities = (
        target_norm
        @ source_norm.t()
    )

    support_scores = []

    for cls in range(10):
        class_mask = source_labels == cls

        if not class_mask.any():
            continue

        class_similarities = similarities[
            :,
            class_mask,
        ]

        support_scores.append(
            (
                cls,
                class_similarities.max(
                    dim=1
                ).values,
            )
        )

    return support_scores


def collect_drl_pseudo_labels(
    trainer,
    target_dataset,
    device,
    threshold=0.70,
    portion=0.20,
):
    trainer.alpha.eval()
    trainer.beta.eval()

    all_indices = list(
        range(len(target_dataset))
    )

    probabilities = []
    indices = []

    with torch.no_grad():
        for start in range(
            0,
            len(all_indices),
            128,
        ):
            batch_indices = all_indices[
                start:start + 128
            ]

            xs = []

            for index in batch_indices:
                x, _ = target_dataset[
                    index
                ]
                xs.append(
                    x.reshape(-1)
                )

            x = torch.stack(
                xs,
                dim=0,
            ).to(device)

            domain_logits = trainer.beta(x)

            domain_probs = F.softmax(
                domain_logits,
                dim=1,
            )

            ratio = (
                domain_probs[:, 0:1]
                /
                domain_probs[:, 1:2].clamp_min(
                    1e-8
                )
            )

            dummy = torch.ones(
                x.size(0),
                10,
                device=device,
            )

            logits = trainer.alpha(
                x,
                dummy,
                ratio,
            )

            probs = F.softmax(
                logits,
                dim=1,
            )

            probabilities.append(
                probs.cpu()
            )

            indices.extend(
                batch_indices
            )

    probabilities = torch.cat(
        probabilities,
        dim=0,
    )

    confidence, labels = probabilities.max(
        dim=1
    )

    valid = torch.where(
        confidence >= threshold
    )[0]

    count = min(
        int(len(target_dataset) * portion),
        valid.numel(),
    )

    ranked = valid[
        torch.argsort(
            confidence[valid],
            descending=True,
        )
    ]

    selected = ranked[:count]

    return (
        torch.tensor(
            [indices[int(i)] for i in selected],
            dtype=torch.long,
        ),
        labels[selected],
        confidence[selected],
    )


def compute_auroc(scores, correctness):
    scores = scores.detach().cpu().numpy()
    correctness = correctness.detach().cpu().numpy()

    order = np.argsort(
        -scores
    )

    correctness = correctness[order]

    positives = correctness.sum()
    negatives = len(correctness) - positives

    if positives == 0 or negatives == 0:
        return float("nan")

    tp = 0.0
    fp = 0.0
    prev_tpr = 0.0
    prev_fpr = 0.0
    auc = 0.0

    for value in correctness:
        if value == 1:
            tp += 1
        else:
            fp += 1

        tpr = tp / positives
        fpr = fp / negatives

        auc += (
            fpr - prev_fpr
        ) * (
            tpr + prev_tpr
        ) / 2.0

        prev_tpr = tpr
        prev_fpr = fpr

    return auc


def precision_at_k(
    scores,
    correctness,
    k,
):
    order = torch.argsort(
        scores,
        descending=True,
    )

    selected = correctness[
        order[:k]
    ]

    return selected.float().mean().item()


def main():
    seed = 42

    set_seed(seed)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

    mnist = datasets.MNIST(
        root="data/digits",
        train=True,
        download=True,
        transform=transform,
    )

    usps_train = datasets.USPS(
        root="data/digits",
        train=True,
        download=True,
        transform=transform,
    )

    usps_test = datasets.USPS(
        root="data/digits",
        train=False,
        download=True,
        transform=transform,
    )

    source_dataset = Subset(
        mnist,
        list(range(2000)),
    )

    target_dataset = Subset(
        usps_train,
        list(range(1800)),
    )

    locked_batches = torch.load(
        "data/locked_digits_batches.pt",
        map_location="cpu",
    )

    trainer = FaithfulDRLTrainer(
        input_dim=784,
        hidden_dim=256,
        num_classes=10,
        lr=1e-3,
        beta_lr=1e-3,
        momentum=0.9,
        weight_decay=5e-4,
        device=device,
    )

    remaining_indices = list(
        range(len(target_dataset))
    )

    for epoch in range(5):
        epoch_batches = [
            batch
            for batch in locked_batches
            if int(batch["epoch"]) == epoch
        ]

        for batch in epoch_batches:
            source_x, source_y = get_locked_batch(
                source_dataset,
                batch["source"],
                device,
            )

            target_x, _ = get_locked_batch(
                target_dataset,
                batch["target"],
                device,
            )

            trainer.train_step(
                source_x,
                source_y,
                target_x,
                epoch,
                int(batch["batch"]),
            )

        if epoch < 4:
            continue

    source_x, source_y = get_data(
        source_dataset
    )

    target_x, target_y = get_data(
        target_dataset
    )

    source_x = source_x.to(device)
    source_y = source_y.to(device)
    target_x = target_x.to(device)
    target_y = target_y.to(device)

    support_model = train_support_critic(
        source_x,
        source_y,
        device,
    )

    support_model.eval()

    with torch.no_grad():
        source_features = support_model(
            source_x
        )

        target_features = support_model(
            target_x
        )

    pseudo_indices, pseudo_labels, confidence = (
        collect_drl_pseudo_labels(
            trainer,
            target_dataset,
            device,
            threshold=0.70,
            portion=0.20,
        )
    )

    selected_target_features = target_features[
        pseudo_indices.to(device)
    ]

    selected_true_labels = target_y[
        pseudo_indices.to(device)
    ]

    with torch.no_grad():
        source_norm = F.normalize(
            source_features,
            dim=1,
        )

        target_norm = F.normalize(
            selected_target_features,
            dim=1,
        )

        similarity = (
            target_norm
            @ source_norm.t()
        )

        support_scores = []

        for i in range(
            selected_target_features.size(0)
        ):
            true_label = pseudo_labels[
                i
            ].item()

            mask = (
                source_y == true_label
            )

            if mask.any():
                score = similarity[
                    i,
                    mask,
                ].max()

                support_scores.append(
                    score
                )
            else:
                support_scores.append(
                    torch.tensor(
                        0.0,
                        device=device,
                    )
                )

        support_scores = torch.stack(
            support_scores
        )

    correctness = (
        pseudo_labels.to(device)
        == selected_true_labels
    )

    combined_scores = (
        confidence.to(device)
        * support_scores
    )

    coverage = (
        100.0
        * len(pseudo_indices)
        / len(target_dataset)
    )

    confidence_precision = (
        correctness.float().mean().item()
    )

    support_threshold = torch.quantile(
        support_scores,
        0.5,
    )

    support_selected = (
        support_scores
        >= support_threshold
    )

    combined_threshold = torch.quantile(
        combined_scores,
        0.5,
    )

    combined_selected = (
        combined_scores
        >= combined_threshold
    )

    support_precision = (
        correctness[
            support_selected
        ].float().mean().item()
        if support_selected.any()
        else float("nan")
    )

    combined_precision = (
        correctness[
            combined_selected
        ].float().mean().item()
        if combined_selected.any()
        else float("nan")
    )

    k = min(
        100,
        correctness.numel(),
    )

    confidence_precision_k = precision_at_k(
        confidence.to(device),
        correctness,
        k,
    )

    support_precision_k = precision_at_k(
        support_scores,
        correctness,
        k,
    )

    combined_precision_k = precision_at_k(
        combined_scores,
        correctness,
        k,
    )

    confidence_auc = compute_auroc(
        confidence.to(device),
        correctness,
    )

    support_auc = compute_auroc(
        support_scores,
        correctness,
    )

    combined_auc = compute_auroc(
        combined_scores,
        correctness,
    )

    print()
    print(
        f"Pseudo-labels: "
        f"{len(pseudo_indices)}"
    )

    print(
        f"Coverage: "
        f"{coverage:.2f}%"
    )

    print(
        f"DRL Confidence Mean: "
        f"{confidence.mean().item():.4f}"
    )

    print(
        f"Support Mean: "
        f"{support_scores.mean().item():.4f}"
    )

    print(
        f"Pseudo-label Precision: "
        f"{confidence_precision * 100:.2f}%"
    )

    print(
        f"Confidence-only Precision @50%: "
        f"{confidence_precision_k * 100:.2f}%"
    )

    print(
        f"Support-only Precision @50%: "
        f"{support_precision * 100:.2f}%"
    )

    print(
        f"Combined Precision @50%: "
        f"{combined_precision * 100:.2f}%"
    )

    print(
        f"Top-{k} Confidence Precision: "
        f"{confidence_precision_k * 100:.2f}%"
    )

    print(
        f"Top-{k} Support Precision: "
        f"{support_precision_k * 100:.2f}%"
    )

    print(
        f"Top-{k} Combined Precision: "
        f"{combined_precision_k * 100:.2f}%"
    )

    print(
        f"Confidence AUROC: "
        f"{confidence_auc:.4f}"
    )

    print(
        f"Support AUROC: "
        f"{support_auc:.4f}"
    )

    print(
        f"Combined AUROC: "
        f"{combined_auc:.4f}"
    )


if __name__ == "__main__":
    main()