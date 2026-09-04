import random
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torchvision import datasets, transforms

from src.drl.faithful_trainer import FaithfulDRLTrainer
from experiments.reliability_predictor import (
    build_reliability_features,
    ReliabilityPredictor,
    train_reliability_model,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_all_data(dataset):
    xs = []
    ys = []

    for i in range(len(dataset)):
        x, y = dataset[i]

        xs.append(
            x.reshape(-1)
        )

        ys.append(y)

    return (
        torch.stack(xs),
        torch.tensor(
            ys,
            dtype=torch.long,
        ),
    )


def get_locked_batch(
    dataset,
    indices,
    device,
):
    subset_indices = dataset.indices
    base_dataset = dataset.dataset

    xs = []
    ys = []

    for index in indices:
        real_index = subset_indices[
            int(index)
        ]

        x, y = base_dataset[
            real_index
        ]

        xs.append(
            x.reshape(-1)
        )

        ys.append(y)

    return (
        torch.stack(xs).to(device),
        torch.tensor(
            ys,
            dtype=torch.long,
            device=device,
        ),
    )


def train_drl(
    trainer,
    source_dataset,
    target_dataset,
    locked_batches,
    device,
):
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


def extract_statistics(
    trainer,
    target_x,
    device,
):
    trainer.alpha.eval()
    trainer.beta.eval()

    features = []
    probabilities = []
    confidences = []
    entropies = []
    predictions = []
    ratios = []

    with torch.no_grad():
        for start in range(
            0,
            target_x.size(0),
            128,
        ):
            x = target_x[
                start:start + 128
            ].to(device)

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
                trainer.num_classes,
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

            confidence, prediction = (
                probs.max(dim=1)
            )

            entropy = -torch.sum(
                probs
                * torch.log(
                    probs.clamp_min(1e-8)
                ),
                dim=1,
            )

            feature = trainer.alpha.features(
                x
            )

            features.append(
                feature.cpu()
            )

            probabilities.append(
                probs.cpu()
            )

            confidences.append(
                confidence.cpu()
            )

            entropies.append(
                entropy.cpu()
            )

            predictions.append(
                prediction.cpu()
            )

            ratios.append(
                ratio.squeeze(1).cpu()
            )

    return {
        "features": torch.cat(
            features,
            dim=0,
        ),
        "probabilities": torch.cat(
            probabilities,
            dim=0,
        ),
        "confidence": torch.cat(
            confidences,
            dim=0,
        ),
        "entropy": torch.cat(
            entropies,
            dim=0,
        ),
        "predictions": torch.cat(
            predictions,
            dim=0,
        ),
        "ratios": torch.cat(
            ratios,
            dim=0,
        ),
    }


def evaluate_model(
    trainer,
    dataset,
    device,
):
    trainer.alpha.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for start in range(
            0,
            len(dataset),
            128,
        ):
            end = min(
                start + 128,
                len(dataset),
            )

            xs = []
            ys = []

            for i in range(
                start,
                end,
            ):
                x, y = dataset[i]

                xs.append(
                    x.reshape(-1)
                )

                ys.append(y)

            x = torch.stack(
                xs,
                dim=0,
            ).to(device)

            y = torch.tensor(
                ys,
                dtype=torch.long,
                device=device,
            )

            y_onehot = torch.zeros(
                x.size(0),
                10,
                device=device,
            )

            y_onehot.scatter_(
                1,
                y.unsqueeze(1),
                1.0,
            )

            ratio = torch.ones(
                x.size(0),
                1,
                device=device,
            )

            logits = trainer.alpha(
                x,
                y_onehot,
                ratio,
            )

            correct += (
                logits.argmax(dim=1)
                == y
            ).sum().item()

            total += y.size(0)

    return 100.0 * correct / total


def train_source_epoch(
    trainer,
    source_dataset,
    locked_batches,
    device,
):
    trainer.alpha.train()

    for batch in locked_batches:
        source_x, source_y = get_locked_batch(
            source_dataset,
            batch["source"],
            device,
        )

        y_onehot = torch.zeros(
            source_y.size(0),
            10,
            device=device,
        )

        y_onehot.scatter_(
            1,
            source_y.unsqueeze(1),
            1.0,
        )

        ratio = torch.ones(
            source_y.size(0),
            1,
            device=device,
        )

        logits = trainer.alpha(
            source_x,
            y_onehot,
            ratio,
        )

        loss = torch.sum(logits)

        trainer.optimizer_alpha.zero_grad(
            set_to_none=True
        )

        loss.backward()

        trainer.optimizer_alpha.step()


def select_by_score(
    scores,
    remaining_indices,
    fraction,
):
    if len(remaining_indices) == 0:
        return []

    remaining_scores = scores[
        torch.tensor(
            remaining_indices,
            dtype=torch.long,
        )
    ]

    count = max(
        1,
        int(
            len(remaining_indices)
            * fraction
        ),
    )

    count = min(
        count,
        len(remaining_indices),
    )

    order = torch.argsort(
        remaining_scores,
        descending=True,
    )

    selected = [
        remaining_indices[
            int(i)
        ]
        for i in order[:count]
    ]

    return selected


def train_pseudo_labels(
    trainer,
    target_dataset,
    selected_indices,
    pseudo_labels,
    pseudo_ratios,
    device,
):
    if len(selected_indices) == 0:
        return

    trainer.alpha.train()

    order = list(
        range(
            len(selected_indices)
        )
    )

    random.shuffle(order)

    batch_size = 16

    for start in range(
        0,
        len(order),
        batch_size,
    ):
        batch_order = order[
            start:start + batch_size
        ]

        indices = [
            selected_indices[i]
            for i in batch_order
        ]

        labels = torch.tensor(
            [
                pseudo_labels[i]
                for i in batch_order
            ],
            dtype=torch.long,
            device=device,
        )

        ratios = torch.tensor(
            [
                pseudo_ratios[i]
                for i in batch_order
            ],
            dtype=torch.float32,
            device=device,
        ).reshape(-1, 1)

        xs = []

        for index in indices:
            x, _ = target_dataset[
                int(index)
            ]

            xs.append(
                x.reshape(-1)
            )

        x = torch.stack(
            xs,
            dim=0,
        ).to(device)

        y_onehot = torch.zeros(
            labels.size(0),
            10,
            device=device,
        )

        y_onehot.scatter_(
            1,
            labels.unsqueeze(1),
            1.0,
        )

        logits = trainer.alpha(
            x,
            y_onehot,
            ratios,
        )

        loss = torch.sum(logits)

        trainer.optimizer_alpha.zero_grad(
            set_to_none=True
        )

        loss.backward()

        trainer.optimizer_alpha.step()


def prepare_reliability(
    trainer,
    target_x,
    target_y,
    device,
):
    statistics = extract_statistics(
        trainer,
        target_x,
        device,
    )

    reliability_features = build_reliability_features(
        statistics["features"],
        statistics["probabilities"],
        statistics["confidence"],
        statistics["entropy"],
        statistics["predictions"],
        num_classes=10,
    )

    calibration_indices = torch.arange(
        0,
        600,
    )

    adaptation_indices = torch.arange(
        600,
        1800,
    )

    calibration_x = reliability_features[
        calibration_indices
    ]

    calibration_correctness = (
        statistics["predictions"][
            calibration_indices
        ]
        == target_y[
            calibration_indices
        ]
    )

    reliability_model = train_reliability_model(
        calibration_x,
        calibration_correctness,
        device,
    )

    reliability_model.eval()

    with torch.no_grad():
        reliability_scores = torch.sigmoid(
            reliability_model(
                reliability_features[
                    adaptation_indices
                ].to(device)
            )
        ).cpu()

    adaptation_predictions = (
        statistics["predictions"][
            adaptation_indices
        ]
    )

    adaptation_ratios = (
        statistics["ratios"][
            adaptation_indices
        ]
    )

    adaptation_confidence = (
        statistics["confidence"][
            adaptation_indices
        ]
    )

    adaptation_labels = (
        adaptation_predictions
        .tolist()
    )

    adaptation_ratio_values = (
        adaptation_ratios.tolist()
    )

    absolute_indices = (
        adaptation_indices.tolist()
    )

    correctness = (
        adaptation_predictions
        == target_y[
            adaptation_indices
        ]
    )

    return {
        "absolute_indices": absolute_indices,
        "labels": adaptation_labels,
        "ratios": adaptation_ratio_values,
        "confidence": adaptation_confidence,
        "reliability": reliability_scores,
        "correctness": correctness,
    }


def run_selective_training(
    base_trainer,
    source_dataset,
    target_dataset,
    target_test_dataset,
    locked_batches,
    reliability_data,
    device,
    method,
    fraction=0.20,
):
    trainer = copy.deepcopy(
        base_trainer
    )

    adaptation_indices = (
        reliability_data["absolute_indices"]
    )

    remaining = list(
        adaptation_indices
    )

    labels_by_index = dict(
        zip(
            adaptation_indices,
            reliability_data["labels"],
        )
    )

    ratios_by_index = dict(
        zip(
            adaptation_indices,
            reliability_data["ratios"],
        )
    )

    confidence_scores = dict(
        zip(
            adaptation_indices,
            reliability_data["confidence"].tolist(),
        )
    )

    reliability_scores = dict(
        zip(
            adaptation_indices,
            reliability_data["reliability"].tolist(),
        )
    )

    selected_total = []

    for epoch in range(5):
        train_source_epoch(
            trainer,
            source_dataset,
            [
                batch
                for batch in locked_batches
                if int(batch["epoch"]) == epoch
            ],
            device,
        )

        if not remaining:
            target_accuracy = evaluate_model(
                trainer,
                target_test_dataset,
                device,
            )

            print(
                f"{method} | "
                f"Epoch {epoch + 1}/5 | "
                f"Selected 0 | "
                f"Remaining 0 | "
                f"Coverage 100.00% | "
                f"USPS {target_accuracy:.2f}%"
            )

            continue

        score_tensor = torch.tensor(
            [
                confidence_scores[index]
                if method == "confidence"
                else reliability_scores[index]
                for index in remaining
            ]
        )

        selected = select_by_score(
            score_tensor,
            list(range(len(remaining))),
            fraction,
        )

        selected_indices = [
            remaining[i]
            for i in selected
        ]

        selected_labels = [
            labels_by_index[index]
            for index in selected_indices
        ]

        selected_ratios = [
            ratios_by_index[index]
            for index in selected_indices
        ]

        train_pseudo_labels(
            trainer,
            target_dataset,
            selected_indices,
            selected_labels,
            selected_ratios,
            device,
        )

        selected_total.extend(
            selected_indices
        )

        selected_set = set(
            selected_indices
        )

        remaining = [
            index
            for index in remaining
            if index not in selected_set
        ]

        target_accuracy = evaluate_model(
            trainer,
            target_test_dataset,
            device,
        )

        coverage = (
            100.0
            * len(selected_total)
            / len(adaptation_indices)
        )

        print(
            f"{method} | "
            f"Epoch {epoch + 1}/5 | "
            f"Selected {len(selected_indices)} | "
            f"Remaining {len(remaining)} | "
            f"Coverage {coverage:.2f}% | "
            f"USPS {target_accuracy:.2f}%"
        )

    final_target = evaluate_model(
        trainer,
        target_test_dataset,
        device,
    )

    return final_target


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

    target_test_dataset = Subset(
        usps_test,
        list(range(len(usps_test))),
    )

    locked_batches = torch.load(
        "data/locked_digits_batches.pt",
        map_location="cpu",
    )

    base_trainer = FaithfulDRLTrainer(
        input_dim=784,
        hidden_dim=256,
        num_classes=10,
        lr=1e-3,
        beta_lr=1e-3,
        momentum=0.9,
        weight_decay=5e-4,
        device=device,
    )

    train_drl(
        base_trainer,
        source_dataset,
        target_dataset,
        locked_batches,
        device,
    )

    target_x, target_y = get_all_data(
        target_dataset
    )

    reliability_data = prepare_reliability(
        base_trainer,
        target_x,
        target_y,
        device,
    )

    calibration_accuracy = (
        100.0
        * (
            target_y[:600]
            == extract_predictions(
                base_trainer,
                target_x[:600],
                device,
            )
        )
        .float()
        .mean()
        .item()
    )

    adaptation_correctness = (
        reliability_data["correctness"]
    )

    reliability_top20 = torch.topk(
        reliability_data["reliability"],
        k=int(
            0.20
            * len(
                reliability_data[
                    "reliability"
                ]
            )
        ),
    ).indices

    confidence_top20 = torch.topk(
        reliability_data["confidence"],
        k=int(
            0.20
            * len(
                reliability_data[
                    "confidence"
                ]
            )
        ),
    ).indices

    print()
    print(
        f"Calibration Accuracy: "
        f"{calibration_accuracy:.2f}%"
    )

    print(
        f"Adaptation Pool: "
        f"{len(adaptation_correctness)}"
    )

    print(
        f"Reliability Top20 Precision: "
        f"{adaptation_correctness[reliability_top20].float().mean().item() * 100:.2f}%"
    )

    print(
        f"Confidence Top20 Precision: "
        f"{adaptation_correctness[confidence_top20].float().mean().item() * 100:.2f}%"
    )

    print()
    print(
        "Starting selective self-training"
    )

    confidence_target = run_selective_training(
        base_trainer,
        source_dataset,
        target_dataset,
        target_test_dataset,
        locked_batches,
        reliability_data,
        device,
        "confidence",
        fraction=0.20,
    )

    reliability_target = run_selective_training(
        base_trainer,
        source_dataset,
        target_dataset,
        target_test_dataset,
        locked_batches,
        reliability_data,
        device,
        "reliability",
        fraction=0.20,
    )

    print()
    print(
        f"Confidence-Gated DRST: "
        f"{confidence_target:.2f}%"
    )

    print(
        f"Reliability-Gated DRST: "
        f"{reliability_target:.2f}%"
    )

    print(
        f"Reliability Gain: "
        f"{reliability_target - confidence_target:+.2f} pp"
    )


def extract_predictions(
    trainer,
    x,
    device,
):
    trainer.alpha.eval()
    trainer.beta.eval()

    predictions = []

    with torch.no_grad():
        for start in range(
            0,
            x.size(0),
            128,
        ):
            xb = x[
                start:start + 128
            ].to(device)

            domain_logits = trainer.beta(
                xb
            )

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
                xb.size(0),
                10,
                device=device,
            )

            logits = trainer.alpha(
                xb,
                dummy,
                ratio,
            )

            predictions.append(
                logits.argmax(
                    dim=1
                ).cpu()
            )

    return torch.cat(
        predictions,
        dim=0,
    )


if __name__ == "__main__":
    main()