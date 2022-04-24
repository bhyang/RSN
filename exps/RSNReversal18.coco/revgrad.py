"""
Module encapsulation of reverse gradient.

Copied from https://github.com/janfreyberg/pytorch-revgrad/
"""

from torch.nn import Module
from torch import tensor
from reversal_functional import revgrad

class RevGrad(Module):
    def __init__(self, alpha=1., *args, **kwargs):
        """
        A gradient reversal layer.
        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super().__init__(*args, **kwargs)

        self._alpha = tensor(alpha, requires_grad=False)

    def forward(self, input_):
        return revgrad(input_, self._alpha)