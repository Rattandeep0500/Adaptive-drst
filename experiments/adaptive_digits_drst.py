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

from src.drst.adaptive_selector import adaptive_select_pseudo_labels
from src.drst.dual_selector import select_dual_reliable_pseudo_labels


SEED = 42
DATA_ROOT = "data/digits"

BATCH_SIZE = 128
SOURCE_EPOCHS = 10
ADAPT_EPOCHS = 8

MIN_THRESHOLD = 0.70
MAX_THRESHOLD = 0.95

MAX_PER_CLASS = 500
DUAL_THRESHOLD = 0.70

EMA_DECAY = 0.99
CONSISTENCY_WEIGHT = 0.5


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
        self.teacher = copy.deepcopy(student)
        self.decay = decay

        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, student):
        for teacher_p, student_p in zip(
            self.teacher.parameters(),
            student.parameters(),
        ):
            teacher_p.mul_(self.decay)
            teacher_p.add_(
                student_p,
                alpha=1.0 - self.decay,
            )

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
        torch.tensor(
            ys,
            dtype=torch.long,
        ),
    )


def weak_augment(x):
    out = x.clone()

    shift_x = random.randint(-1, 1)
    shift_y = random.randint(-1, 1)

    out = torch.roll(
        out,
        shifts=(shift_y, shift_x),
        dims=(2, 3),
    )

    return out.clamp(0.0, 1.0)


def strong_augment(x):
    out = weak_augment(x)

    out = out + 0.05 * torch.randn_like(out)

    if random.random() < 0.5:
        out = torch.flip(out, dims=[3])

    return out.clamp(0.0, 1.0)


def make_loader(x, y=None, shuffle=False):
    if y is None:
        dataset = x
    else:
        dataset = TensorDataset(x, y)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
    )


def train_source(model, source_x, source_y):
    loader = make_loader(
        source_x,
        source_y,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    for epoch in range(SOURCE_EPOCHS):
        model.train()

        total_loss = 0.0
        steps = 0

        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)

            loss = F.cross_entropy(
                model(x),
                y,
            )

            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach())
            steps += 1

        print(
            f"[SOURCE] "
            f"epoch={epoch + 1}/{SOURCE_EPOCHS} "
            f"loss={total_loss / max(steps, 1):.4f}"
        )


@torch.no_grad()
def predict(model, x):
    model.eval()

    outputs = []

    loader = make_loader(
        x,
        shuffle=False,
    )

    for batch in loader:
        outputs.append(
            torch.softmax(
                model(batch),
                dim=1,
            )
        )

    return torch.cat(outputs, dim=0)


def accuracy(model, x, y):
    probabilities = predict(model, x)

    return float(
        (
            probabilities.argmax(dim=1)
            == y
        ).float().mean()
    )


@torch.no_grad()
def build_adaptive_selection(
    model,
    target_x,
):
    probabilities = predict(
        model,
        target_x,
    )

    return adaptive_select_pseudo_labels(
        probabilities=probabilities,
        min_threshold=MIN_THRESHOLD,
        max_threshold=MAX_THRESHOLD,
        max_per_class=MAX_PER_CLASS,
    )


@torch.no_grad()
def build_dual_selection(
    model,
    target_x,
):
    logits = []

    domain_inputs = []

    loader = make_loader(
        target_x,
        shuffle=False,
    )

    # Independent support model.
    support_model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 2),
    )

    # Train the support model against a fixed
    # source/target representation distribution.
    # This model is intentionally separate from
    # the task classifier.
    return support_model, loader


def train_support_model(
    source_x,
    target_x,
):
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 2),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    x = torch.cat(
        [source_x, target_x],
        dim=0,
    )

    y = torch.cat(
        [
            torch.zeros(
                len(source_x),
                dtype=torch.long,
            ),
            torch.ones(
                len(target_x),
                dtype=torch.long,
            ),
        ],
        dim=0,
    )

    loader = make_loader(
        x,
        y,
        shuffle=True,
    )

    model.train()

    for _ in range(3):
        for bx, by in loader:
            optimizer.zero_grad(set_to_none=True)

            loss = F.cross_entropy(
                model(bx),
                by,
            )

            loss.backward()
            optimizer.step()

    return model


@torch.no_grad()
def dual_selection(
    task_model,
    support_model,
    target_x,
):
    task_model.eval()
    support_model.eval()

    task_probabilities = predict(
        task_model,
        target_x,
    )

    support_probabilities = []

    loader = make_loader(
        target_x,
        shuffle=False,
    )

    for batch in loader:
        support_probabilities.append(
            torch.softmax(
                support_model(batch),
                dim=1,
            )
        )

    support_probabilities = torch.cat(
        support_probabilities,
        dim=0,
    )

    return select_dual_reliable_pseudo_labels(
        drl_probabilities=task_probabilities,
        domain_probabilities=support_probabilities,
        threshold=DUAL_THRESHOLD,
        max_per_class=MAX_PER_CLASS,
    )


def run_consistency(
    model,
    teacher,
    source_x,
    source_y,
    target_x,
    selector,
    support_model=None,
):
    source_loader = make_loader(
        source_x,
        source_y,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-4,
        weight_decay=1e-4,
    )

    for epoch in range(ADAPT_EPOCHS):
        if selector == "adaptive":
            (
                selected,
                pseudo_y,
                confidence,
                thresholds,
            ) = build_adaptive_selection(
                teacher.teacher,
                target_x,
            )

            support = None
            score = confidence

        else:
            (
                selected,
                pseudo_y,
                confidence,
                support,
                score,
            ) = dual_selection(
                teacher.teacher,
                support_model,
                target_x,
            )

            thresholds = None

        selected_mask = torch.zeros(
            len(target_x),
            dtype=torch.bool,
        )
        selected_mask[selected] = True

        pseudo_bank = torch.full(
            (len(target_x),),
            -1,
            dtype=torch.long,
        )
        pseudo_bank[selected] = pseudo_y

        target_order = torch.randperm(
            len(target_x)
        )

        pointer = 0

        total_loss = 0.0
        selected_seen = 0
        steps = 0

        model.train()

        for sx, sy in source_loader:
            if pointer >= len(target_x):
                pointer = 0
                target_order = torch.randperm(
                    len(target_x)
                )

            end = min(
                pointer + BATCH_SIZE,
                len(target_x),
            )

            ids = target_order[
                pointer:end
            ]

            pointer = end

            tx = target_x[ids]

            source_loss = F.cross_entropy(
                model(sx),
                sy,
            )

            local_mask = selected_mask[ids]

            consistency_loss = torch.zeros(
                (),
                device=sx.device,
            )

            if local_mask.any():
                weak = weak_augment(tx)
                strong = strong_augment(tx)

                with torch.no_grad():
                    weak_teacher = teacher.teacher(
                        weak[local_mask]
                    ).softmax(dim=1)

                strong_logits = model(
                    strong[local_mask]
                )

                local_ids = ids[local_mask]
                labels = pseudo_bank[
                    local_ids
                ]

                consistency_loss = F.cross_entropy(
                    strong_logits,
                    labels,
                )

                selected_seen += int(
                    local_mask.sum()
                )

            loss = (
                source_loss
                + CONSISTENCY_WEIGHT
                * consistency_loss
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            optimizer.step()

            teacher.update(model)

            total_loss += float(
                loss.detach()
            )
            steps += 1

        target_acc = accuracy(
            teacher.teacher,
            TARGET_TEST_X,
            TARGET_TEST_Y,
        )

        print(
            f"[{selector.upper()}] "
            f"epoch={epoch + 1}/{ADAPT_EPOCHS} "
            f"selected={len(selected)} "
            f"used={selected_seen} "
            f"loss={total_loss / max(steps, 1):.4f} "
            f"target_acc={target_acc * 100:.2f}%"
        )

        if thresholds is not None:
            print(
                "  thresholds:",
                [
                    round(float(v), 3)
                    for v in thresholds
                ],
            )

        if selector == "dual" and len(score) > 0:
            print(
                f"  mean dual score="
                f"{float(score.mean()):.4f}"
            )


def main():
    global TARGET_TEST_X
    global TARGET_TEST_Y

    seed_everything(SEED)

    print("Loading MNIST -> USPS...")

    (
        mnist,
        usps_train,
        usps_test,
    ) = load_data()

    source_x, source_y = stack_dataset(
        mnist
    )

    target_x, _ = stack_dataset(
        usps_train
    )

    TARGET_TEST_X, TARGET_TEST_Y = (
        stack_dataset(usps_test)
    )

    print(
        f"Source train: {len(source_x)}"
    )

    print(
        f"Target adaptation: {len(target_x)}"
    )

    print(
        f"Target test: {len(TARGET_TEST_X)}"
    )

    print("\n=== SOURCE-ONLY ===")

    source_model = CNN()

    train_source(
        source_model,
        source_x,
        source_y,
    )

    source_acc = accuracy(
        source_model,
        TARGET_TEST_X,
        TARGET_TEST_Y,
    )

    print(
        f"Source-only target accuracy: "
        f"{source_acc * 100:.2f}%"
    )

    print("\n=== ADAPTIVE-CB + EMA ===")

    adaptive_student = copy.deepcopy(
        source_model
    )

    adaptive_teacher = EMATeacher(
        adaptive_student,
        decay=EMA_DECAY,
    )

    run_consistency(
        adaptive_student,
        adaptive_teacher,
        source_x,
        source_y,
        target_x,
        selector="adaptive",
    )

    adaptive_acc = accuracy(
        adaptive_teacher.teacher,
        TARGET_TEST_X,
        TARGET_TEST_Y,
    )

    print(
        f"Adaptive-CB final: "
        f"{adaptive_acc * 100:.2f}%"
    )

    print("\n=== DUAL-RELIABILITY ===")

    support_model = train_support_model(
        source_x,
        target_x,
    )

    dual_student = copy.deepcopy(
        source_model
    )

    dual_teacher = EMATeacher(
        dual_student,
        decay=EMA_DECAY,
    )

    run_consistency(
        dual_student,
        dual_teacher,
        source_x,
        source_y,
        target_x,
        selector="dual",
        support_model=support_model,
    )

    dual_acc = accuracy(
        dual_teacher.teacher,
        TARGET_TEST_X,
        TARGET_TEST_Y,
    )

    print(
        f"Dual-Reliability final: "
        f"{dual_acc * 100:.2f}%"
    )

    print("\n=== FINAL COMPARISON ===")

    print(
        f"Source-only       : "
        f"{source_acc * 100:.2f}%"
    )

    print(
        f"Adaptive-CB + EMA : "
        f"{adaptive_acc * 100:.2f}%"
    )

    print(
        f"Dual-Reliability  : "
        f"{dual_acc * 100:.2f}%"
    )

    print(
        f"Dual gain vs source: "
        f"{(dual_acc - source_acc) * 100:+.2f} pp"
    )

    print(
        f"Dual gain vs Adaptive-CB: "
        f"{(dual_acc - adaptive_acc) * 100:+.2f} pp"
    )


if __name__ == "__main__":
    main()
