import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.drl.faithful_trainer import FaithfulDRLTrainer


class ScheduledDRLTrainer(FaithfulDRLTrainer):
    def __init__(self, *args, beta_schedule="official", **kwargs):
        super().__init__(*args, **kwargs)
        self.beta_schedule = beta_schedule

    def _should_domain_update(self, epoch, batch_index):
        if self.beta_schedule == "official":
            return epoch == 0 and batch_index < 5
        if self.beta_schedule == "domain_every_batch":
            return True
        if self.beta_schedule == "both_every_batch":
            return True
        raise ValueError(
            f"Unknown beta schedule: {self.beta_schedule}"
        )

    def _should_task_update(self, epoch, batch_index):
        if self.beta_schedule == "official":
            return epoch == 0 and batch_index < 5
        if self.beta_schedule == "domain_every_batch":
            return epoch == 0 and batch_index < 5
        if self.beta_schedule == "both_every_batch":
            return True
        raise ValueError(
            f"Unknown beta schedule: {self.beta_schedule}"
        )

    def train_step(
        self,
        source_x,
        source_y,
        target_x,
        epoch,
        batch_index,
    ):
        source_x = source_x.reshape(
            source_x.size(0),
            -1,
        ).to(self.device)

        target_x = target_x.reshape(
            target_x.size(0),
            -1,
        ).to(self.device)

        source_y = source_y.to(self.device)

        batch_size = source_x.size(0)

        input_concat = torch.cat(
            [source_x, target_x],
            dim=0,
        )

        domain_labels = torch.cat(
            [
                torch.zeros(
                    source_x.size(0),
                    dtype=torch.long,
                    device=self.device,
                ),
                torch.ones(
                    target_x.size(0),
                    dtype=torch.long,
                    device=self.device,
                ),
            ],
            dim=0,
        )

        domain_logits = self.beta(input_concat)

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

        r_source = ratio[:batch_size].detach()
        r_target = ratio[batch_size:].detach()

        p_t_target = p_t[batch_size:].detach()

        source_onehot = self._one_hot(source_y)

        theta_out = self.alpha(
            source_x,
            source_onehot,
            r_source,
        )

        target_onehot = torch.ones(
            target_x.size(0),
            self.num_classes,
            device=self.device,
        )

        nn_out = self.alpha(
            target_x,
            target_onehot,
            r_target,
        )

        pred_target = F.softmax(
            nn_out,
            dim=1,
        )

        domain_loss = self.domain_loss_fn(
            domain_logits,
            domain_labels,
        )

        domain_updated = False
        task_updated = False

        if self._should_domain_update(
            epoch,
            batch_index,
        ):
            self.optimizer_beta.zero_grad(
                set_to_none=True
            )

            domain_loss.backward()

            self.optimizer_beta.step()

            domain_updated = True

        if self._should_task_update(
            epoch,
            batch_index,
        ):
            prob_grad_r = self.beta(
                target_x,
                nn_out.detach(),
                pred_target.detach(),
                p_t_target,
                self.sign_variable,
            )

            loss_r = torch.sum(
                prob_grad_r
                * torch.zeros_like(prob_grad_r)
            )

            self.optimizer_beta.zero_grad(
                set_to_none=True
            )

            loss_r.backward()

            self.optimizer_beta.step()

            task_updated = True

        loss_theta = torch.sum(theta_out)

        self.optimizer_alpha.zero_grad(
            set_to_none=True
        )

        loss_theta.backward()

        self.optimizer_alpha.step()

        source_accuracy = (
            (
                theta_out.argmax(dim=1)
                == source_y
            )
            .float()
            .mean()
            .item()
        )

        return {
            "domain_loss": float(
                domain_loss.detach()
            ),
            "source_accuracy": source_accuracy,
            "ratio_mean": float(
                ratio.mean().detach()
            ),
            "ratio_median": float(
                ratio.median().detach()
            ),
            "ratio_min": float(
                ratio.min().detach()
            ),
            "ratio_max": float(
                ratio.max().detach()
            ),
            "domain_updated": domain_updated,
            "task_updated": task_updated,
        }


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

    x = torch.stack(
        xs,
        dim=0,
    ).to(device)

    y = torch.tensor(
        ys,
        dtype=torch.long,
        device=device,
    )

    return x, y


def evaluate_alpha(trainer, loader, device):
    trainer.alpha.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.reshape(
                x.size(0),
                -1,
            ).to(device)

            y = y.to(device)

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
                logits.argmax(dim=1) == y
            ).sum().item()

            total += y.size(0)

    return 100.0 * correct / total


def evaluate_beta(
    trainer,
    source_loader,
    target_loader,
    device,
):
    trainer.beta.eval()

    correct = 0
    total = 0

    ratio_values = []

    with torch.no_grad():
        for x, _ in source_loader:
            x = x.reshape(
                x.size(0),
                -1,
            ).to(device)

            logits = trainer.beta(x)

            probs = F.softmax(
                logits,
                dim=1,
            )

            ratio = (
                probs[:, 0:1]
                / probs[:, 1:2].clamp_min(1e-8)
            )

            correct += (
                logits.argmax(dim=1)
                == 0
            ).sum().item()

            total += x.size(0)

            ratio_values.append(
                ratio.squeeze(1).cpu()
            )

        for x, _ in target_loader:
            x = x.reshape(
                x.size(0),
                -1,
            ).to(device)

            logits = trainer.beta(x)

            probs = F.softmax(
                logits,
                dim=1,
            )

            ratio = (
                probs[:, 0:1]
                / probs[:, 1:2].clamp_min(1e-8)
            )

            correct += (
                logits.argmax(dim=1)
                == 1
            ).sum().item()

            total += x.size(0)

            ratio_values.append(
                ratio.squeeze(1).cpu()
            )

    ratios = torch.cat(
        ratio_values
    )

    return {
        "accuracy": 100.0 * correct / total,
        "ratio_mean": ratios.mean().item(),
        "ratio_median": ratios.median().item(),
        "ratio_min": ratios.min().item(),
        "ratio_max": ratios.max().item(),
    }


def run_schedule(
    schedule,
    seed,
    source_dataset,
    target_dataset,
    source_loader,
    target_loader,
    target_test_loader,
    locked_batches,
    device,
):
    set_seed(seed)

    trainer = ScheduledDRLTrainer(
        input_dim=784,
        hidden_dim=256,
        num_classes=10,
        lr=1e-3,
        beta_lr=1e-3,
        momentum=0.9,
        weight_decay=5e-4,
        device=device,
        beta_schedule=schedule,
    )

    for epoch in range(5):
        epoch_batches = [
            batch
            for batch in locked_batches
            if int(batch["epoch"]) == epoch
        ]

        metrics = []

        for batch in epoch_batches:
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

            result = trainer.train_step(
                source_x,
                source_y,
                target_x,
                epoch,
                int(batch["batch"]),
            )

            metrics.append(result)

        target_accuracy = evaluate_alpha(
            trainer,
            target_test_loader,
            device,
        )

        beta_stats = evaluate_beta(
            trainer,
            source_loader,
            target_loader,
            device,
        )

        mean_ratio = np.mean([
            m["ratio_mean"]
            for m in metrics
        ])

        print(
            f"{schedule} | "
            f"Seed {seed} | "
            f"Epoch {epoch + 1}/5 | "
            f"Target {target_accuracy:.2f}% | "
            f"Beta Acc {beta_stats['accuracy']:.2f}% | "
            f"Ratio {mean_ratio:.4f}"
        )

    final_target = evaluate_alpha(
        trainer,
        target_test_loader,
        device,
    )

    final_source = evaluate_alpha(
        trainer,
        source_loader,
        device,
    )

    final_beta = evaluate_beta(
        trainer,
        source_loader,
        target_loader,
        device,
    )

    return {
        "schedule": schedule,
        "seed": seed,
        "source_accuracy": final_source,
        "target_accuracy": final_target,
        "beta_accuracy": final_beta["accuracy"],
        "ratio_mean": final_beta["ratio_mean"],
        "ratio_median": final_beta["ratio_median"],
        "ratio_min": final_beta["ratio_min"],
        "ratio_max": final_beta["ratio_max"],
    }


def main():
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

    source_loader = DataLoader(
        source_dataset,
        batch_size=128,
        shuffle=False,
    )

    target_loader = DataLoader(
        target_dataset,
        batch_size=128,
        shuffle=False,
    )

    target_test_loader = DataLoader(
        usps_test,
        batch_size=128,
        shuffle=False,
    )

    locked_batches = torch.load(
        "data/locked_digits_batches.pt",
        map_location="cpu",
    )

    schedules = [
        "official",
        "domain_every_batch",
        "both_every_batch",
    ]

    seed = 42

    results = []

    for schedule in schedules:
        print()
        print(
            f"Running schedule: {schedule}"
        )
        print()

        result = run_schedule(
            schedule,
            seed,
            source_dataset,
            target_dataset,
            source_loader,
            target_loader,
            target_test_loader,
            locked_batches,
            device,
        )

        results.append(result)

        print()
        print(
            f"{schedule} | "
            f"Final Source {result['source_accuracy']:.2f}% | "
            f"Final USPS {result['target_accuracy']:.2f}% | "
            f"Beta Acc {result['beta_accuracy']:.2f}% | "
            f"Ratio Mean {result['ratio_mean']:.4f} | "
            f"Ratio Median {result['ratio_median']:.4f} | "
            f"Ratio Min {result['ratio_min']:.4f} | "
            f"Ratio Max {result['ratio_max']:.4f}"
        )

    print()
    print("Summary")
    print()

    for result in results:
        print(
            f"{result['schedule']} | "
            f"USPS {result['target_accuracy']:.2f}% | "
            f"Beta {result['beta_accuracy']:.2f}% | "
            f"Ratio {result['ratio_mean']:.4f}"
        )


if __name__ == "__main__":
    main()