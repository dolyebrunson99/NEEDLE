from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        max_Z = Z.max(axis=self.axes, keepdims=True)
        max_Z_reduce = Z.max(axis=self.axes)
        out = array_api.log(array_api.exp(Z - max_Z.broadcast_to(Z.shape))\
                                     .sum(axis=self.axes)) + max_Z_reduce
        return out

    def gradient(self, out_grad, node):
        Z = node.inputs[0]
        max_Z = Tensor(Z.cached_data.max(axis=self.axes, keepdims=True))
        exp_Z = exp(Z - Tensor(max_Z))
        sum_Z = reshape(summation(exp_Z, self.axes), max_Z.shape)
        out_grad = reshape(out_grad, max_Z.shape)
        return out_grad * (exp_Z / sum_Z)


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

