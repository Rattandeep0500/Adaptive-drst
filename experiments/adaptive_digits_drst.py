import copy
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


SEED = 42
DATA_ROOT = "data/digits"

BATCH_SIZE = 128
SOURCE_EPOCHS = 10
ADAPT_EPOCHS = 12

EMA_DECAY = 0.99
CONSISTENCY_WEIGHT = 1.0

MIN_THRESHOLD = 0.80
MAX_THRESHOLD = 0.97

PSEUDO_REFRESH_EVERY = 1
MAX_PER_CLASS = 1000


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        z = self.features(x)
        z = z.flatten(1)
        return self.classifier(z)


class EMATeacher:
    def __init__(self, student, decay=0.99):
        self.student = student
        self.teacher = copy.deepcopy(student)
        self.decay = decay

        for p in self.teacher.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self):
        for tp, sp in zip(
            self.teacher.parameters(),
            self.student.parameters(),
        ):
            tp.mul_(self.decay)
            tp.add_(sp, alpha=1.0 - self.decay)

    def eval(self):
        self.teacher.eval()


def load_data():
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

    mnist = datasets.MNIST(
        DATA_ROOT,
        train=True,
        download=True,
        transform=transform,
    )

    usps_train = datasets.USPS(
        DATA_ROOT,
        train=True,
        download=True,
        transform=transform,
    )

    usps_test = datasets.USPS(
        DATA_ROOT,
        train=False,
        download=True,
        transform=transform,
    )

    return mnist, usps_train, usps_test


def stack_dataset(dataset):
    xs = []
    ys = []

    for i in range(len(dataset)):
        x, y = dataset[i]
        xs.append(x)
        ys.append(y)

    return (
        torch.stack(xs),
        torch.tensor(ys, dtype=torch.long),
    )


def weak_aug(x):
    out = x.clone()

    dx = random.randint(-1, 1)
    dy = random.randint(-1, 1)

    out = torch.roll(
        out,
        shifts=(dy, dx),
        dims=(2, 3),
    )

    return out.clamp(0, 1)


def strong_aug(x):
    out = weak_aug(x)

    out = out + 0.05 * torch.randn_like(out)

    if random.random() < 0.5:
        out = torch.flip(out, dims=[3])

    if random.random() < 0.5:
        h = random.randint(2, 5)
        w = random.randint(2, 5)
        y0 = random.randint(0, 28 - h)
        x0 = random.randint(0, 28 - w)

        out[:, :, y0:y0+h, x0:x0+w] = 0

    return out.clamp(0, 1)


def train_source(model, x, y):
    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=SOURCE_EPOCHS,
    )

    for epoch in range(SOURCE_EPOCHS):
        model.train()
        total_loss = 0.0
        steps = 0

        for bx, by in loader:
            optimizer.zero_grad(set_to_none=True)

            logits = model(bx)
            loss = F.cross_entropy(logits, by)

            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach())
            steps += 1

        scheduler.step()

        print(
            f"[SOURCE] "
            f"{epoch + 1}/{SOURCE_EPOCHS} "
            f"loss={total_loss / steps:.4f}"
        )


@torch.no_grad()
def predict(model, x):
    model.eval()

    loader = DataLoader(
        x,
        batch_size=256,
        shuffle=False,
    )

    probs = []

    for bx in loader:
        probs.append(
            torch.softmax(
                model(bx),
                dim=1,
            )
        )

    return torch.cat(probs)


def accuracy(model, x, y):
    p = predict(model, x)

    return float(
        (p.argmax(1) == y)
        .float()
        .mean()
    )


def adaptive_thresholds(
    probabilities,
    minimum=MIN_THRESHOLD,
    maximum=MAX_THRESHOLD,
):
    confidence, labels = probabilities.max(1)

    thresholds = torch.full(
        (10,),
        minimum,
        dtype=confidence.dtype,
    )

    for cls in range(10):
        mask = labels == cls

        if not mask.any():
            continue

        cls_conf = confidence[mask]

        # Median confidence, bounded globally.
        thresholds[cls] = torch.quantile(
            cls_conf,
            0.50,
        ).clamp(
            minimum,
            maximum,
        )

    return confidence, labels, thresholds


@torch.no_grad()
def build_pseudo_bank(
    teacher,
    target_x,
):
    probabilities = predict(
        teacher,
        target_x,
    )

    confidence, labels, thresholds = adaptive_thresholds(
        probabilities
    )

    selected = torch.zeros(
        len(target_x),
        dtype=torch.bool,
    )

    for cls in range(10):
        mask = (
            (labels == cls)
            & (confidence >= thresholds[cls])
        )

        idx = torch.where(mask)[0]

        if idx.numel() > MAX_PER_CLASS:
            order = torch.argsort(
                confidence[idx],
                descending=True,
            )
            idx = idx[order[:MAX_PER_CLASS]]

        selected[idx] = True

    return {
        "selected": selected,
        "labels": labels,
        "confidence": confidence,
        "thresholds": thresholds,
        "probabilities": probabilities,
    }


def adapt(
    student,
    teacher,
    source_x,
    source_y,
    target_x,
    test_x,
    test_y,
):
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=5e-4,
        weight_decay=1e-4,
    )

    source_loader = DataLoader(
        TensorDataset(source_x, source_y),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    for epoch in range(ADAPT_EPOCHS):
        teacher.eval()

        bank = build_pseudo_bank(
            teacher.teacher,
            target_x,
        )

        selected_mask = bank["selected"]
        pseudo_labels = bank["labels"]

        selected_ids = torch.where(
            selected_mask
        )[0]

        print(
            f"[ADAPT] epoch {epoch + 1}/{ADAPT_EPOCHS} "
            f"pseudo={len(selected_ids)}"
        )

        total_loss = 0.0
        steps = 0

        target_order = torch.randperm(
            len(target_x)
        )

        target_ptr = 0

        student.train()

        for sx, sy in source_loader:
            if target_ptr >= len(target_x):
                target_ptr = 0
                target_order = torch.randperm(
                    len(target_x)
                )

            end = min(
                target_ptr + BATCH_SIZE,
                len(target_x),
            )

            ids = target_order[
                target_ptr:end
            ]

            target_ptr = end

            tx = target_x[ids]

            weak = weak_aug(tx)
            strong = strong_aug(tx)

            source_logits = student(sx)

            source_loss = F.cross_entropy(
                source_logits,
                sy,
            )

            consistency_loss = torch.tensor(
                0.0
            )

            local_selected = (
                selected_mask[ids]
            )

            if local_selected.any():
                global_ids = ids[
                    local_selected
                ]

                pseudo_y = pseudo_labels[
                    global_ids
                ]

                strong_logits = student(
                    strong[local_selected]
                )

                consistency_loss = F.cross_entropy(
                    strong_logits,
                    pseudo_y,
                )

            loss = (
                source_loss
                + CONSISTENCY_WEIGHT
                * consistency_loss
            )

            optimizer.zero_grad(set_to_none=True)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                student.parameters(),
                5.0,
            )

            optimizer.step()

            teacher.update()

            total_loss += float(loss.detach())
            steps += 1

        test_accuracy = accuracy(
            teacher.teacher,
            test_x,
            test_y,
        )

        print(
            f"[ADAPT] epoch {epoch + 1}/{ADAPT_EPOCHS} "
            f"loss={total_loss / steps:.4f} "
            f"target_acc={test_accuracy * 100:.2f}%"
        )


def main():
    seed_everything(SEED)

    print("Loading MNIST -> USPS...")

    mnist, usps_train, usps_test = load_data()

    source_x, source_y = stack_dataset(mnist)
    target_x, _ = stack_dataset(usps_train)
    test_x, test_y = stack_dataset(usps_test)

    print(
        f"Source train: {len(source_x)}"
    )
    print(
        f"Target adaptation: {len(target_x)}"
    )
    print(
        f"Target test: {len(test_x)}"
    )

    print("\n=== SOURCE-ONLY ===")

    source_model = CNN()

    train_source(
        source_model,
        source_x,
        source_y,
    )

    source_accuracy = accuracy(
        source_model,
        test_x,
        test_y,
    )

    print(
        f"Source-only target accuracy: "
        f"{source_accuracy * 100:.2f}%"
    )

    print("\n=== ADAPTIVE-DRST-EMA ===")

    student = copy.deepcopy(
        source_model
    )

    teacher = EMATeacher(
        student,
        decay=EMA_DECAY,
    )

    adapt(
        student=student,
        teacher=teacher,
        source_x=source_x,
        source_y=source_y,
        target_x=target_x,
        test_x=test_x,
        test_y=test_y,
    )

    final_accuracy = accuracy(
        teacher.teacher,
        test_x,
        test_y,
    )

    print("\n=== FINAL ===")

    print(
        f"Source-only : "
        f"{source_accuracy * 100:.2f}%"
    )

    print(
        f"Adaptive-DRST: "
        f"{final_accuracy * 100:.2f}%"
    )

    print(
        f"Gain: "
        f"{(final_accuracy - source_accuracy) * 100:+.2f} pp"
    )


if __name__ == "__main__":
    main()