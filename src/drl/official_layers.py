import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, y, r_st, p_t, bias=True):
        exp_temp = input.mm(weight.t()).mul(r_st)

        if bias is not None:
            exp_temp += bias.unsqueeze(0).expand_as(exp_temp)

        output = F.softmax(exp_temp, dim=1)
        ctx.save_for_backward(input, weight, bias, output, y, p_t)

        return exp_temp

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, output, y, p_t = ctx.saved_tensors

        grad_input = None
        grad_weight = None
        grad_bias = None
        grad_y = None
        grad_r = None
        grad_p_t = None

        if ctx.needs_input_grad[0]:
            grad_input = (output - y).mm(weight)

        if ctx.needs_input_grad[1]:
            grad_weight = (output.t() - y.t()).mm(input)

        if bias is not None and ctx.needs_input_grad[5]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return (
            grad_input,
            grad_weight,
            grad_y,
            grad_r,
            grad_p_t,
            grad_bias,
        )


class ClassificationLayer(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super().__init__()

        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(
            torch.Tensor(output_features, input_features)
        )

        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter("bias", None)

        self.weight.data.uniform_(
            -1.0 / math.sqrt(input_features),
            1.0 / math.sqrt(input_features),
        )

        if bias:
            self.bias.data.uniform_(
                -1.0 / math.sqrt(input_features),
                1.0 / math.sqrt(input_features),
            )

    def forward(self, input, y, r, p_t):
        return ClassificationFunction.apply(
            input,
            self.weight,
            y,
            r,
            p_t,
            self.bias,
        )


class GradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, nn_output, prediction, p_t, sign_variable):
        ctx.save_for_backward(
            input,
            nn_output,
            prediction,
            p_t,
            sign_variable,
        )
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, nn_output, prediction, p_t, sign_variable = ctx.saved_tensors

        grad_input = None
        grad_out = None
        grad_pred = None
        grad_p_t = None
        grad_sign = None

        if ctx.needs_input_grad[0]:
            if sign_variable is None:
                grad_input = grad_output
            else:
                grad_source = (
                    torch.sum(nn_output.mul(prediction), dim=1)
                    .reshape(-1, 1)
                    / p_t
                )

                grad_target = (
                    torch.sum(nn_output.mul(prediction), dim=1)
                    .reshape(-1, 1)
                    * (-(1.0 - p_t) / (p_t ** 2))
                )

                grad_source /= prediction.shape[0]
                grad_target /= prediction.shape[0]

                grad_input = (
                    1e-1
                    * torch.cat((grad_source, grad_target), dim=1)
                    / p_t.shape[0]
                )

            grad_input = 1e1 * grad_input

        return (
            grad_input,
            grad_out,
            grad_pred,
            grad_p_t,
            grad_sign,
        )


class GradLayer(nn.Module):
    def forward(
        self,
        input,
        nn_output,
        prediction,
        p_t,
        sign_variable,
    ):
        return GradFunction.apply(
            input,
            nn_output,
            prediction,
            p_t,
            sign_variable,
        )