import random
import numpy as np
import torch
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


def get_batch(dataset, indices, device):
    subset_indices = dataset.indices
    base_dataset = dataset.dataset

    xs = []
    ys = []

    for index in indices:
        real_index = subset_indices[int(index)]
        x, y = base_dataset[real_index]
        xs.append(x.reshape(-1))
        ys.append(y)

    x = torch.stack(xs, dim=0).to(device)

    y = torch.tensor(
        ys,
        dtype=torch.long,
        device=device,
    )

    return x, y


def domain_stats(beta, x, y):
    beta.eval()

    with torch.no_grad():
        logits = beta(x)
        probs = F.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)

        accuracy = (
            pred == y
        ).float().mean().item() * 100.0

        loss = F.cross_entropy(
            logits,
            y,
        ).item()

        source_mask = y == 0
        target_mask = y == 1

        source_probability = (
            probs[source_mask, 0].mean().item()
            if source_mask.any()
            else float("nan")
        )

        target_probability = (
            probs[target_mask, 1].mean().item()
            if target_mask.any()
            else float("nan")
        )

        ratio = (
            probs[:, 0:1]
            / probs[:, 1:2].clamp_min(1e-8)
        )

        ratio_mean = ratio.mean().item()
        ratio_median = ratio.median().item()
        ratio_min = ratio.min().item()
        ratio_max = ratio.max().item()

    return {
        "accuracy": accuracy,
        "loss": loss,
        "source_probability": source_probability,
        "target_probability": target_probability,
        "ratio_mean": ratio_mean,
        "ratio_median": ratio_median,
        "ratio_min": ratio_min,
        "ratio_max": ratio_max,
    }


def print_stats(label, stats):
    print(
        f"{label} | "
        f"Domain Acc {stats['accuracy']:.2f}% | "
        f"Loss {stats['loss']:.4f} | "
        f"P(source|source) {stats['source_probability']:.4f} | "
        f"P(target|target) {stats['target_probability']:.4f} | "
        f"Ratio Mean {stats['ratio_mean']:.4f} | "
        f"Ratio Median {stats['ratio_median']:.4f} | "
        f"Ratio Min {stats['ratio_min']:.4f} | "
        f"Ratio Max {stats['ratio_max']:.4f}"
    )


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

    usps = datasets.USPS(
        root="data/digits",
        train=True,
        download=True,
        transform=transform,
    )

    source_dataset = Subset(
        mnist,
        list(range(2000)),
    )

    target_dataset = Subset(
        usps,
        list(range(1800)),
    )

    locked_batches = torch.load(
        "data/locked_digits_batches.pt",
        map_location="cpu",
    )

    first_epoch_batches = [
        batch
        for batch in locked_batches
        if int(batch["epoch"]) == 0
    ][:5]

    source_x, source_y = get_batch(
        source_dataset,
        first_epoch_batches[0]["source"],
        device,
    )

    target_x, _ = get_batch(
        target_dataset,
        first_epoch_batches[0]["target"],
        device,
    )

    input_concat = torch.cat(
        [source_x, target_x],
        dim=0,
    )

    domain_labels = torch.cat(
        [
            torch.zeros(
                source_x.size(0),
                dtype=torch.long,
                device=device,
            ),
            torch.ones(
                target_x.size(0),
                dtype=torch.long,
                device=device,
            ),
        ],
        dim=0,
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

    beta = trainer.beta
    optimizer = trainer.optimizer_beta
    loss_fn = torch.nn.CrossEntropyLoss()

    print("Initial beta state")

    stats = domain_stats(
        beta,
        input_concat,
        domain_labels,
    )

    print_stats(
        "Before updates",
        stats,
    )

    for batch_number, batch in enumerate(
        first_epoch_batches
    ):
        source_x, source_y = get_batch(
            source_dataset,
            batch["source"],
            device,
        )

        target_x, _ = get_batch(
            target_dataset,
            batch["target"],
            device,
        )

        input_concat = torch.cat(
            [source_x, target_x],
            dim=0,
        )

        domain_labels = torch.cat(
            [
                torch.zeros(
                    source_x.size(0),
                    dtype=torch.long,
                    device=device,
                ),
                torch.ones(
                    target_x.size(0),
                    dtype=torch.long,
                    device=device,
                ),
            ],
            dim=0,
        )

        beta.train()

        domain_logits = beta(
            input_concat
        )

        domain_prediction = F.softmax(
            domain_logits,
            dim=1,
        )

        p_s = domain_prediction[:, 0:1]
        p_t = domain_prediction[:, 1:2]

        ratio = (
            p_s
            / p_t.clamp_min(1e-8)
        )

        r_target = ratio[
            source_x.size(0):
        ].detach()

        p_t_target = p_t[
            source_x.size(0):
        ].detach()

        source_onehot = trainer._one_hot(
            source_y
        )

        theta_out = trainer.alpha(
            source_x,
            source_onehot,
            ratio[
                :source_x.size(0)
            ].detach(),
        )

        target_onehot = torch.ones(
            target_x.size(0),
            10,
            device=device,
        )

        nn_out = trainer.alpha(
            target_x,
            target_onehot,
            r_target,
        )

        pred_target = F.softmax(
            nn_out,
            dim=1,
        )

        domain_loss = loss_fn(
            domain_logits,
            domain_labels,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        domain_loss.backward()

        optimizer.step()

        after_domain = domain_stats(
            beta,
            input_concat,
            domain_labels,
        )

        prob_grad_r = beta(
            target_x,
            nn_out.detach(),
            pred_target.detach(),
            p_t_target,
            trainer.sign_variable,
        )

        loss_r = torch.sum(
            prob_grad_r
            * torch.zeros_like(prob_grad_r)
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss_r.backward()

        optimizer.step()

        after_task = domain_stats(
            beta,
            input_concat,
            domain_labels,
        )

        print()
        print(
            f"Batch {batch_number + 1}/5"
        )

        print_stats(
            "Before beta updates",
            domain_stats(
                beta,
                input_concat,
                domain_labels,
            ),
        )

        print_stats(
            "After domain update",
            after_domain,
        )

        print_stats(
            "After task update",
            after_task,
        )

    print()
    print("Final beta state after 5 batches")

    final_stats = domain_stats(
        beta,
        input_concat,
        domain_labels,
    )

    print_stats(
        "Final",
        final_stats,
    )


if __name__ == "__main__":
    main()