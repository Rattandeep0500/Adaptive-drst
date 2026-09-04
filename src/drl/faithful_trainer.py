import torch
import torch.nn as nn
import torch.nn.functional as F

from src.drl.official_layers import ClassificationLayer, GradLayer


class AlphaDigits(nn.Module):
    def __init__(
        self,
        input_dim=784,
        hidden_dim=256,
        num_classes=10,
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.classifier = ClassificationLayer(
            hidden_dim,
            num_classes,
        )

    def forward(
        self,
        x,
        y_onehot,
        density_ratio,
    ):
        features = self.features(x)

        p_t = torch.ones(
            density_ratio.shape[0],
            1,
            device=density_ratio.device,
        )

        return self.classifier(
            features,
            y_onehot,
            density_ratio,
            p_t,
        )


class BetaDigits(nn.Module):
    def __init__(
        self,
        input_dim=784,
        hidden_dim=256,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )

        self.grad = GradLayer()

    def forward(
        self,
        x,
        nn_output=None,
        prediction=None,
        p_t=None,
        sign_variable=None,
    ):
        logits = self.model(x)

        if nn_output is None:
            return logits

        return self.grad(
            logits,
            nn_output,
            prediction,
            p_t,
            sign_variable,
        )


class FaithfulDRLTrainer:
    def __init__(
        self,
        input_dim=784,
        hidden_dim=256,
        num_classes=10,
        lr=1e-3,
        beta_lr=1e-3,
        momentum=0.9,
        weight_decay=5e-4,
        device=None,
    ):
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.num_classes = num_classes

        self.alpha = AlphaDigits(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
        ).to(self.device)

        self.beta = BetaDigits(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.optimizer_alpha = torch.optim.SGD(
            self.alpha.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

        self.optimizer_beta = torch.optim.SGD(
            self.beta.parameters(),
            lr=beta_lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

        self.domain_loss_fn = nn.BCEWithLogitsLoss()

        self.sign_variable = torch.tensor(
            [0.0],
            dtype=torch.float32,
            device=self.device,
        )

    def _flatten(self, x):
        return x.reshape(
            x.size(0),
            -1,
        )

    def _one_hot(self, labels):
        y = torch.zeros(
            labels.size(0),
            self.num_classes,
            device=self.device,
        )

        y.scatter_(
            1,
            labels.long().unsqueeze(1),
            1.0,
        )

        return y

    def _domain_labels(
        self,
        source_size,
        target_size,
    ):
        source_labels = torch.tensor(
            [[1.0, 0.0]],
            dtype=torch.float32,
            device=self.device,
        ).repeat(
            source_size,
            1,
        )

        target_labels = torch.tensor(
            [[0.0, 1.0]],
            dtype=torch.float32,
            device=self.device,
        ).repeat(
            target_size,
            1,
        )

        return torch.cat(
            [
                source_labels,
                target_labels,
            ],
            dim=0,
        )

    def _density_ratio(self, domain_logits):
        prediction = F.softmax(
            domain_logits,
            dim=1,
        )

        p_s = prediction[:, 0:1]

        p_t = prediction[:, 1:2].clamp_min(
            1e-8
        )

        ratio = p_s / p_t

        return (
            ratio,
            prediction,
            p_s,
            p_t,
        )

    def train_step(
        self,
        source_x,
        source_y,
        target_x,
        epoch,
        batch_index,
    ):
        source_x = self._flatten(
            source_x
        ).to(self.device)

        target_x = self._flatten(
            target_x
        ).to(self.device)

        source_y = source_y.to(
            self.device
        )

        source_size = source_x.size(0)
        target_size = target_x.size(0)

        input_concat = torch.cat(
            [
                source_x,
                target_x,
            ],
            dim=0,
        )

        domain_labels = self._domain_labels(
            source_size,
            target_size,
        )

        domain_logits = self.beta(
            input_concat
        )

        domain_loss = self.domain_loss_fn(
            domain_logits,
            domain_labels,
        )

        ratio, prediction, p_s, p_t = (
            self._density_ratio(
                domain_logits
            )
        )

        r_source = ratio[
            :source_size
        ].detach()

        r_target = ratio[
            source_size:
        ].detach()

        p_t_target = p_t[
            source_size:
        ].detach()

        source_onehot = self._one_hot(
            source_y
        )

        theta_out = self.alpha(
            source_x,
            source_onehot,
            r_source,
        )

        source_pred = F.softmax(
            theta_out,
            dim=1,
        )

        target_onehot = torch.ones(
            target_size,
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

        beta_domain_updated = False
        beta_task_updated = False

        if epoch == 0 and batch_index < 5:
            self.optimizer_beta.zero_grad(
                set_to_none=True
            )

            domain_loss.backward(
                retain_graph=True
            )

            self.optimizer_beta.step()

            beta_domain_updated = True

            prob_grad_r = self.beta(
                target_x,
                nn_out.detach(),
                pred_target.detach(),
                p_t_target,
                self.sign_variable,
            )

            loss_r = torch.sum(
                prob_grad_r
                * torch.zeros_like(
                    prob_grad_r
                )
            )

            self.optimizer_beta.zero_grad(
                set_to_none=True
            )

            loss_r.backward()

            self.optimizer_beta.step()

            beta_task_updated = True

        loss_theta = torch.sum(
            theta_out
        )

        self.optimizer_alpha.zero_grad(
            set_to_none=True
        )

        loss_theta.backward()

        self.optimizer_alpha.step()

        source_accuracy = (
            (
                source_pred.argmax(dim=1)
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
            "alpha_objective": float(
                loss_theta.detach()
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
            "beta_domain_updated": (
                beta_domain_updated
            ),
            "beta_task_updated": (
                beta_task_updated
            ),
        }