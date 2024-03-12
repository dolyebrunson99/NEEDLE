"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.padding = (kernel_size - 1) // 2 # same padding
        self.weight = Parameter(init.kaiming_uniform(
            in_channels * kernel_size * kernel_size,
            out_channels * kernel_size * kernel_size,
            shape=(kernel_size, kernel_size, in_channels, out_channels),
            device=device
        ))
        if bias:
            bound = 1.0/(in_channels*kernel_size**2)**0.5
            self.bias = Parameter(init.rand(
                self.out_channels,
                low=-bound, high=bound,
                device=device
            ))
        else:
            self.bias = None


    def forward(self, x: Tensor) -> Tensor:
        N = x.shape[0]
        out = ops.Conv(self.stride, self.padding)(
            x.permute((0,2,3,1)), 
            self.weight
        ).permute((0,3,1,2))
        if self.bias:
            out += self.bias.reshape((1, self.out_channels, 1, 1))\
                            .broadcast_to(out.shape)
        return out