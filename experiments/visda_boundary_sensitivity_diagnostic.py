import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


SEED = 42
INPUT_DIM = 2048
HIDDEN_DIM = 512
NUM_CLASSES = 12

CACHE_ROOT = Path("checkpoints/visda_feature_cache")
TARGET_CACHE = CACHE_ROOT / "target"

MCD_CHECKPOINT = Path(
    "checkpoints/visda_cached_mcd_corrected/mcd_corrected_seed42.pt"
)

OUTPUT_DIR = Path(
    "checkpoints/visda_boundary_sensitivity"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BATCH_SIZE = 4096
NUM_DIRECTIONS = 8
EPSILONS = [0.01, 0.02, 0.05]

CLASSES = [
    "aeroplane",
    "bicycle",
    "bus",
    "car",
    "horse",
    "knife",
    "motorcycle",
    "person",
    "plant",
    "skateboard",
    "train",
    "truck",
]


class Adapter(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(
                INPUT_DIM,
                HIDDEN_DIM,
            ),
            torch.nn.BatchNorm1d(
                HIDDEN_DIM
            ),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class MCDModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.adapter = Adapter()

        self.classifier1 = torch.nn.Linear(
            HIDDEN_DIM,
            NUM_CLASSES,
        )

        self.classifier2 = torch.nn.Linear(
            HIDDEN_DIM,
            NUM_CLASSES,
        )

    def forward(self, x):
        z = self.adapter(x)

        return (
            self.classifier1(z),
            self.classifier2(z),
            z,
        )


def load_mcd():
    checkpoint = torch.load(
        MCD_CHECKPOINT,
        map_location="cpu",
    )

    model = MCDModel()

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


def load_target_files():
    files = sorted(
        TARGET_CACHE.glob("chunk_*.pt")
    )

    if not files:
        raise RuntimeError(
            f"No target cache chunks found in {TARGET_CACHE}"
        )

    return files


def sample_normalized_directions(
    batch_size,
    dimensions,
    num_directions,
    generator,
):
    directions = torch.randn(
        batch_size,
        num_directions,
        dimensions,
        generator=generator,
        dtype=torch.float32,
    )

    norms = torch.linalg.vector_norm(
        directions,
        dim=2,
        keepdim=True,
    ).clamp_min(1e-12)

    directions = (
        directions
        / norms
    )

    return directions


def collect_statistics(model):
    generator = torch.Generator()
    generator.manual_seed(SEED)

    total = 0

    correct_count = 0
    wrong_count = 0

    base_margin_sum = 0.0
    base_confidence_sum = 0.0
    base_disagreement_sum = 0.0

    flip_counts = {
        epsilon: 0
        for epsilon in EPSILONS
    }

    correct_flip_counts = {
        epsilon: 0
        for epsilon in EPSILONS
    }

    wrong_flip_counts = {
        epsilon: 0
        for epsilon in EPSILONS
    }

    margin_drop_sum = {
        epsilon: 0.0
        for epsilon in EPSILONS
    }

    correct_margin_drop_sum = {
        epsilon: 0.0
        for epsilon in EPSILONS
    }

    wrong_margin_drop_sum = {
        epsilon: 0.0
        for epsilon in EPSILONS
    }

    confidence_drop_sum = {
        epsilon: 0.0
        for epsilon in EPSILONS
    }

    correct_confidence_drop_sum = {
        epsilon: 0.0
        for epsilon in EPSILONS
    }

    wrong_confidence_drop_sum = {
        epsilon: 0.0
        for epsilon in EPSILONS
    }

    class_total = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
    )

    class_correct = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
    )

    class_flip_counts = {
        epsilon: torch.zeros(
            NUM_CLASSES,
            dtype=torch.long,
        )
        for epsilon in EPSILONS
    }

    class_margin_drop_sum = {
        epsilon: torch.zeros(
            NUM_CLASSES,
            dtype=torch.float64,
        )
        for epsilon in EPSILONS
    }

    confusion_flip_counts = {
        epsilon: torch.zeros(
            NUM_CLASSES,
            NUM_CLASSES,
            dtype=torch.long,
        )
        for epsilon in EPSILONS
    }

    pair_total = {
        epsilon: torch.zeros(
            NUM_CLASSES,
            NUM_CLASSES,
            dtype=torch.long,
        )
        for epsilon in EPSILONS
    }

    for path in load_target_files():
        payload = torch.load(
            path,
            map_location="cpu",
        )

        raw_features = payload[
            "features"
        ].float()

        labels = payload[
            "labels"
        ].long()

        for start in range(
            0,
            len(raw_features),
            BATCH_SIZE,
        ):
            end = min(
                start + BATCH_SIZE,
                len(raw_features),
            )

            x = raw_features[
                start:end
            ]

            y = labels[
                start:end
            ]

            with torch.no_grad():
                logits1, logits2, z = model(
                    x
                )

                p1 = F.softmax(
                    logits1,
                    dim=1,
                )

                p2 = F.softmax(
                    logits2,
                    dim=1,
                )

                base_prob = (
                    p1 + p2
                ) / 2.0

                base_top = torch.topk(
                    base_prob,
                    k=2,
                    dim=1,
                )

                base_pred = base_top.indices[:, 0]

                base_conf = base_top.values[:, 0]

                base_margin = (
                    base_top.values[:, 0]
                    - base_top.values[:, 1]
                )

                base_disagreement = (
                    torch.abs(
                        p1 - p2
                    ).mean(dim=1)
                )

            sample_norm = torch.linalg.vector_norm(
                z,
                dim=1,
                keepdim=True,
            ).clamp_min(1e-8)

            directions = sample_normalized_directions(
                z.shape[0],
                HIDDEN_DIM,
                NUM_DIRECTIONS,
                generator,
            )

            base_correct = (
                base_pred == y
            )

            batch_size = y.size(0)

            total += batch_size
            correct_count += int(
                base_correct.sum().item()
            )
            wrong_count += int(
                (~base_correct).sum().item()
            )

            base_margin_sum += (
                base_margin.sum().item()
            )

            base_confidence_sum += (
                base_conf.sum().item()
            )

            base_disagreement_sum += (
                base_disagreement.sum().item()
            )

            for class_id in range(
                NUM_CLASSES
            ):
                class_mask = y == class_id

                if class_mask.any():
                    count = int(
                        class_mask.sum().item()
                    )

                    class_total[
                        class_id
                    ] += count

                    class_correct[
                        class_id
                    ] += int(
                        base_correct[
                            class_mask
                        ]
                        .sum()
                        .item()
                    )

            for epsilon in EPSILONS:
                local_flip = torch.zeros(
                    batch_size,
                    dtype=torch.bool,
                )

                local_margin_drop = torch.zeros(
                    batch_size,
                    dtype=torch.float32,
                )

                local_conf_drop = torch.zeros(
                    batch_size,
                    dtype=torch.float32,
                )

                for direction_id in range(
                    NUM_DIRECTIONS
                ):
                    direction = directions[
                        :,
                        direction_id,
                        :,
                    ]

                    scale = (
                        epsilon
                        * sample_norm
                    )

                    z_perturbed = (
                        z
                        + scale
                        * direction
                    )

                    with torch.no_grad():
                        perturbed_logits1 = (
                            model.classifier1(
                                z_perturbed
                            )
                        )

                        perturbed_logits2 = (
                            model.classifier2(
                                z_perturbed
                            )
                        )

                        perturbed_prob = (
                            F.softmax(
                                perturbed_logits1,
                                dim=1,
                            )
                            + F.softmax(
                                perturbed_logits2,
                                dim=1,
                            )
                        ) / 2.0

                        perturbed_top = torch.topk(
                            perturbed_prob,
                            k=2,
                            dim=1,
                        )

                        perturbed_pred = (
                            perturbed_top.indices[
                                :,
                                0,
                            ]
                        )

                        perturbed_conf = (
                            perturbed_top.values[
                                :,
                                0,
                            ]
                        )

                        perturbed_margin = (
                            perturbed_top.values[
                                :,
                                0,
                            ]
                            - perturbed_top.values[
                                :,
                                1,
                            ]
                        )

                    flipped = (
                        perturbed_pred
                        != base_pred
                    )

                    local_flip |= flipped

                    local_margin_drop = torch.maximum(
                        local_margin_drop,
                        base_margin
                        - perturbed_margin,
                    )

                    local_conf_drop = torch.maximum(
                        local_conf_drop,
                        base_conf
                        - perturbed_conf,
                    )

                    for i in range(
                        batch_size
                    ):
                        if flipped[i]:
                            confusion_flip_counts[
                                epsilon
                            ][
                                int(
                                    y[i].item()
                                ),
                                int(
                                    perturbed_pred[i]
                                    .item()
                                ),
                            ] += 1

                        pair_total[
                            epsilon
                        ][
                            int(
                                base_pred[i]
                                .item()
                            ),
                            int(
                                perturbed_pred[i]
                                .item()
                            ),
                        ] += 1

                flip_counts[
                    epsilon
                ] += int(
                    local_flip.sum().item()
                )

                correct_flip_counts[
                    epsilon
                ] += int(
                    (
                        local_flip
                        & base_correct
                    )
                    .sum()
                    .item()
                )

                wrong_flip_counts[
                    epsilon
                ] += int(
                    (
                        local_flip
                        & (~base_correct)
                    )
                    .sum()
                    .item()
                )

                margin_drop_sum[
                    epsilon
                ] += local_margin_drop.sum().item()

                correct_margin_drop_sum[
                    epsilon
                ] += local_margin_drop[
                    base_correct
                ].sum().item()

                wrong_margin_drop_sum[
                    epsilon
                ] += local_margin_drop[
                    ~base_correct
                ].sum().item()

                confidence_drop_sum[
                    epsilon
                ] += local_conf_drop.sum().item()

                correct_confidence_drop_sum[
                    epsilon
                ] += local_conf_drop[
                    base_correct
                ].sum().item()

                wrong_confidence_drop_sum[
                    epsilon
                ] += local_conf_drop[
                    ~base_correct
                ].sum().item()

                for class_id in range(
                    NUM_CLASSES
                ):
                    mask = y == class_id

                    if mask.any():
                        class_flip_counts[
                            epsilon
                        ][class_id] += int(
                            local_flip[
                                mask
                            ].sum().item()
                        )

                        class_margin_drop_sum[
                            epsilon
                        ][class_id] += (
                            local_margin_drop[
                                mask
                            ]
                            .double()
                            .sum()
                            .item()
                        )

    result = {
        "total": total,
        "correct": correct_count,
        "wrong": wrong_count,
        "base_mean_margin":
            base_margin_sum / total,
        "base_mean_confidence":
            base_confidence_sum / total,
        "base_mean_disagreement":
            base_disagreement_sum / total,
        "class_total":
            class_total,
        "class_correct":
            class_correct,
        "flip_counts":
            flip_counts,
        "correct_flip_counts":
            correct_flip_counts,
        "wrong_flip_counts":
            wrong_flip_counts,
        "margin_drop_sum":
            margin_drop_sum,
        "correct_margin_drop_sum":
            correct_margin_drop_sum,
        "wrong_margin_drop_sum":
            wrong_margin_drop_sum,
        "confidence_drop_sum":
            confidence_drop_sum,
        "correct_confidence_drop_sum":
            correct_confidence_drop_sum,
        "wrong_confidence_drop_sum":
            wrong_confidence_drop_sum,
        "class_flip_counts":
            class_flip_counts,
        "class_margin_drop_sum":
            class_margin_drop_sum,
        "confusion_flip_counts":
            confusion_flip_counts,
        "pair_total":
            pair_total,
    }

    return result


def print_results(result):
    total = result["total"]
    correct = result["correct"]
    wrong = result["wrong"]

    print()
    print("=" * 90)
    print("BASE TARGET STATISTICS")
    print("=" * 90)

    print(
        f"Target samples: {total}"
    )

    print(
        f"Correct predictions: {correct}"
    )

    print(
        f"Wrong predictions: {wrong}"
    )

    print(
        f"Correct rate: "
        f"{100.0 * correct / total:.2f}%"
    )

    print(
        f"Mean confidence: "
        f"{result['base_mean_confidence']:.6f}"
    )

    print(
        f"Mean margin: "
        f"{result['base_mean_margin']:.6f}"
    )

    print(
        f"Mean disagreement: "
        f"{result['base_mean_disagreement']:.6f}"
    )

    print()
    print("=" * 90)
    print("LOCAL BOUNDARY SENSITIVITY")
    print("=" * 90)

    print(
        f"{'Epsilon':10s}"
        f"{'Flip %':>14s}"
        f"{'Correct flip %':>18s}"
        f"{'Wrong flip %':>18s}"
        f"{'Margin drop':>16s}"
        f"{'Conf drop':>16s}"
    )

    for epsilon in EPSILONS:
        flip = (
            100.0
            * result["flip_counts"][epsilon]
            / total
        )

        correct_flip = (
            100.0
            * result["correct_flip_counts"][epsilon]
            / max(correct, 1)
        )

        wrong_flip = (
            100.0
            * result["wrong_flip_counts"][epsilon]
            / max(wrong, 1)
        )

        margin_drop = (
            result["margin_drop_sum"][epsilon]
            / total
        )

        conf_drop = (
            result["confidence_drop_sum"][epsilon]
            / total
        )

        print(
            f"{epsilon:<10.3f}"
            f"{flip:13.2f}%"
            f"{correct_flip:17.2f}%"
            f"{wrong_flip:17.2f}%"
            f"{margin_drop:15.6f}"
            f"{conf_drop:15.6f}"
        )

    print()
    print("=" * 90)
    print("CORRECT VS WRONG LOCAL SENSITIVITY")
    print("=" * 90)

    print(
        f"{'Epsilon':10s}"
        f"{'Correct flip':>18s}"
        f"{'Wrong flip':>18s}"
        f"{'Correct margin':>20s}"
        f"{'Wrong margin':>18s}"
    )

    for epsilon in EPSILONS:
        correct_flip = (
            100.0
            * result["correct_flip_counts"][epsilon]
            / max(correct, 1)
        )

        wrong_flip = (
            100.0
            * result["wrong_flip_counts"][epsilon]
            / max(wrong, 1)
        )

        correct_margin = (
            result["correct_margin_drop_sum"][epsilon]
            / max(correct, 1)
        )

        wrong_margin = (
            result["wrong_margin_drop_sum"][epsilon]
            / max(wrong, 1)
        )

        print(
            f"{epsilon:<10.3f}"
            f"{correct_flip:17.2f}%"
            f"{wrong_flip:17.2f}%"
            f"{correct_margin:19.6f}"
            f"{wrong_margin:17.6f}"
        )

    print()
    print("=" * 90)
    print("PER-CLASS SENSITIVITY")
    print("=" * 90)

    for epsilon in EPSILONS:
        print()
        print(
            f"Epsilon {epsilon:.3f}"
        )

        print(
            f"{'Class':12s}"
            f"{'Accuracy':>12s}"
            f"{'Flip %':>14s}"
            f"{'Margin drop':>18s}"
        )

        for class_id, class_name in enumerate(
            CLASSES
        ):
            total_class = int(
                result["class_total"][class_id]
            )

            correct_class = int(
                result["class_correct"][class_id]
            )

            flip_class = int(
                result["class_flip_counts"][
                    epsilon
                ][class_id]
            )

            margin_class = (
                result["class_margin_drop_sum"][
                    epsilon
                ][class_id].item()
            )

            accuracy = (
                100.0
                * correct_class
                / max(total_class, 1)
            )

            flip_rate = (
                100.0
                * flip_class
                / max(total_class, 1)
            )

            mean_margin_drop = (
                margin_class
                / max(total_class, 1)
            )

            print(
                f"{class_name:12s}"
                f"{accuracy:11.2f}%"
                f"{flip_rate:13.2f}%"
                f"{mean_margin_drop:17.6f}"
            )

    print()
    print("=" * 90)
    print("WRONG-PREDICTION SENSITIVITY")
    print("=" * 90)

    for epsilon in EPSILONS:
        print(
            f"Epsilon {epsilon:.3f} | "
            f"wrong flip rate "
            f"{100.0 * result['wrong_flip_counts'][epsilon] / max(wrong, 1):.2f}% | "
            f"wrong confidence drop "
            f"{result['wrong_confidence_drop_sum'][epsilon] / max(wrong, 1):.6f}"
        )

    print()
    print("=" * 90)
    print("FLIP CONFUSION PATTERNS")
    print("=" * 90)

    for epsilon in EPSILONS:
        matrix = result[
            "confusion_flip_counts"
        ][epsilon]

        pairs = []

        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                if i == j:
                    continue

                count = int(
                    matrix[i, j].item()
                )

                if count > 0:
                    pairs.append(
                        (
                            count,
                            CLASSES[i],
                            CLASSES[j],
                        )
                    )

        pairs.sort(
            reverse=True
        )

        print()
        print(
            f"Epsilon {epsilon:.3f}"
        )

        for count, true_name, pred_name in pairs[:20]:
            print(
                f"{true_name:12s} → "
                f"{pred_name:12s} | "
                f"{count:6d}"
            )


def build_report(result):
    return {
        "experiment":
            "visda_boundary_sensitivity_diagnostic",
        "seed":
            SEED,
        "target_labels_used_only_for_diagnostics":
            True,
        "num_directions":
            NUM_DIRECTIONS,
        "epsilons":
            EPSILONS,
        "total":
            result["total"],
        "correct":
            result["correct"],
        "wrong":
            result["wrong"],
        "base_mean_margin":
            result["base_mean_margin"],
        "base_mean_confidence":
            result["base_mean_confidence"],
        "base_mean_disagreement":
            result["base_mean_disagreement"],
        "flip_rates": {
            str(epsilon):
                result["flip_counts"][epsilon]
                / result["total"]
            for epsilon in EPSILONS
        },
        "correct_flip_rates": {
            str(epsilon):
                result["correct_flip_counts"][epsilon]
                / max(result["correct"], 1)
            for epsilon in EPSILONS
        },
        "wrong_flip_rates": {
            str(epsilon):
                result["wrong_flip_counts"][epsilon]
                / max(result["wrong"], 1)
            for epsilon in EPSILONS
        },
        "mean_margin_drop": {
            str(epsilon):
                result["margin_drop_sum"][epsilon]
                / result["total"]
            for epsilon in EPSILONS
        },
        "correct_mean_margin_drop": {
            str(epsilon):
                result["correct_margin_drop_sum"][epsilon]
                / max(result["correct"], 1)
            for epsilon in EPSILONS
        },
        "wrong_mean_margin_drop": {
            str(epsilon):
                result["wrong_margin_drop_sum"][epsilon]
                / max(result["wrong"], 1)
            for epsilon in EPSILONS
        },
        "mean_confidence_drop": {
            str(epsilon):
                result["confidence_drop_sum"][epsilon]
                / result["total"]
            for epsilon in EPSILONS
        },
        "correct_mean_confidence_drop": {
            str(epsilon):
                result["correct_confidence_drop_sum"][epsilon]
                / max(result["correct"], 1)
            for epsilon in EPSILONS
        },
        "wrong_mean_confidence_drop": {
            str(epsilon):
                result["wrong_confidence_drop_sum"][epsilon]
                / max(result["wrong"], 1)
            for epsilon in EPSILONS
        },
        "class_accuracy": [
            result["class_correct"][i].item()
            / max(
                result["class_total"][i].item(),
                1,
            )
            for i in range(NUM_CLASSES)
        ],
        "class_flip_rates": {
            str(epsilon): [
                result["class_flip_counts"][
                    epsilon
                ][i].item()
                / max(
                    result["class_total"][i].item(),
                    1,
                )
                for i in range(NUM_CLASSES)
            ]
            for epsilon in EPSILONS
        },
        "class_margin_drop": {
            str(epsilon): [
                result["class_margin_drop_sum"][
                    epsilon
                ][i].item()
                / max(
                    result["class_total"][i].item(),
                    1,
                )
                for i in range(NUM_CLASSES)
            ]
            for epsilon in EPSILONS
        },
        "flip_confusion_counts": {
            str(epsilon):
                result["confusion_flip_counts"][
                    epsilon
                ].tolist()
            for epsilon in EPSILONS
        },
    }


def main():
    print("=" * 90)
    print("VISDA-2017 LOCAL BOUNDARY SENSITIVITY DIAGNOSTIC")
    print("=" * 90)

    print(
        "Target labels are used only for diagnostics."
    )

    print()
    print(
        "Loading corrected MCD model..."
    )

    model = load_mcd()

    print(
        f"Directions per sample: "
        f"{NUM_DIRECTIONS}"
    )

    print(
        f"Epsilons: "
        f"{EPSILONS}"
    )

    print()
    print(
        "Running local perturbation analysis..."
    )

    result = collect_statistics(
        model
    )

    print_results(
        result
    )

    report = build_report(
        result
    )

    report_path = (
        OUTPUT_DIR
        / "boundary_sensitivity_seed42.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
        )

    print()
    print("=" * 90)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 90)

    print(
        f"Saved: {report_path}"
    )


if __name__ == "__main__":
    main()