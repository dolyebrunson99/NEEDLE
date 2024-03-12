"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND 
from .ops_tuple import *

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * log(a)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a ** self.scalar

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return a ** (self.scalar - 1) * out_grad * self.scalar


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return out_grad / rhs, -lhs * out_grad / (rhs ** 2)


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad, node):
        return out_grad * self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if a.ndim < 2:
            return a
        (ax1, ax2) = (a.ndim-2, a.ndim-1) if self.axes is None else self.axes
        permute_axes = list(range(a.ndim))
        permute_axes[ax1], permute_axes[ax2] = ax2, ax1
        return a.permute(permute_axes)

    def gradient(self, out_grad, node):
        return out_grad.transpose(self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Permute(TensorOp):
    def __init__(self, axes: tuple):
        self.axes = axes
    
    def compute(self, a):
        return a.permute(self.axes)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return permute(out_grad, a.axes)


def permute(a, axes):
    return Permute(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return a.compact().reshape(self.shape)

    def gradient(self, out_grad, node):
        return reshape(out_grad, node.inputs[0].shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        shape = [s1 if s0 == -1 else s0 for s0, s1 in zip(self.shape, a.shape)]
        return a.broadcast_to(shape)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        shape = [s1 if s0 == -1 else s0 for s0, s1 in zip(self.shape, a.shape)]
        # For example a.shape = (1, 3), self.shape = (2, 2, 3)
        # (1, 3) -> (2, 2, 3), before summation we need to pad (1, 3) to (1, 1, 3)
        a_shape = (1,) * (len(shape) - len(a.shape)) + a.shape # pad input shape
        sum_axes = tuple(i for i, d in enumerate(a_shape) if d == 1)
        grad_a = summation(out_grad, sum_axes)
        return reshape(grad_a, a.shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        axes = self.axes
        if axes is None:
            axes = tuple(numpy.arange(a.ndim))
        if isinstance(axes, int):
            axes = (axes,)
        out = a
        for axis in reversed(sorted(axes)):
            out = array_api.sum(out, axis)
        return out

    def gradient(self, out_grad, node):
        # dims are not kept 
        # a.shape = (1, 2, 3) -> out.shape(1, 3), self.axes = (1,)
        a = node.inputs[0]
        if self.axes is None:
            return broadcast_to(out_grad.reshape((1,)*len(a.shape)), a.shape)
        a_shape = list(a.shape) # (1, 2, 3)
        axes = self.axes
        if isinstance(self.axes, int):
            axes = (self.axes,)
        for i in axes:
            a_shape[i] = 1 # a_shape = (1, 1, 3)
        return broadcast_to(out_grad.reshape(tuple(a_shape)), a.shape)



def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return a @ b

    def gradient(self, out_grad, node):
        a, b = node.inputs
        grad_a = out_grad @ b.T
        grad_b = a.T @ out_grad
        if len(a.shape) < len(b.shape): # b contains batch axes
            grad_a = summation(grad_a, axes=tuple(range(len(b.shape) - len(a.shape))))
        elif len(a.shape) > len(b.shape): # a contains batch axes
            grad_b = summation(grad_b, axes=tuple(range(len(a.shape) - len(b.shape))))
        return grad_a, grad_b


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return -out_grad


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        return out_grad / node.inputs[0]


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        return out_grad * exp(node.inputs[0])


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad, node):
        node_input = node.inputs[0]
        return out_grad * Tensor(node.cached_data > 0,\
                                 dtype="float32",\
                                 device=node.device,\
                                 requires_grad=False)


def relu(a):
    return ReLU()(a)

class Tanh(TensorOp):
    def compute(self, a):
        return array_api.tanh(a)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return out_grad * (init.ones(*out_grad.shape,
                                     device=out_grad.device,
                                     requires_grad=False) - power_scalar(tanh(a), 2.))



def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        # Indexing a negative axis of a `TensorTuple` of tensors does not make
        # sense.
        assert self.axis >= 0, "can not stack negative axis"
        # check shapes
        shape = args[0].shape
        for tensor in args:
            assert tensor.shape == shape, 'tensor dimension mismatch'
        device = args[0].device
        # check device
        for tensor in args:
            assert tensor.device == device, 'tensor device mismatch'
        # stack tensors
        ndim = len(shape)
        out = array_api.empty(shape=shape[ :self.axis] + (len(args),) + shape[self.axis: ],
                              device=device)
        for i, tensor in enumerate(args):
            idx = (slice(None),) * self.axis + (i,) + (slice(None),) * (ndim - self.axis)
            out[idx] = tensor
        return out


    def gradient(self, out_grad, node):
        return split(out_grad,
                     axis=self.axis)


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        # handle negative axis
        if self.axis < 0:
            self.axis += A.ndim

        shape = A.shape[ :self.axis] + A.shape[self.axis+1: ]
        out = []
        for i in range(A.shape[self.axis]):
            idx = (slice(None),) * self.axis + (i,) + (slice(None),) * (len(shape) - self.axis)
            # Do not forget to compact a tensor before reshaping it.
            out.append(array_api.reshape(A[idx].compact(),
                                         new_shape=shape))
        return tuple(out)

    def gradient(self, out_grad, node):
        return stack(out_grad,
                     axis=self.axis)


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return a.flip(self.axes)

    def gradient(self, out_grad, node):
        return flip(out_grad, self.axes)


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        if self.dilation == 0:
            return a
        stride = self.dilation + 1
        out_shape = tuple(s if i not in self.axes 
                            else s * stride
                            for i, s in enumerate(a.shape))
        out = array_api.full(out_shape, 0., device=a.device)
        idx = tuple(slice(None) if i not in self.axes
                                else slice(None, None, stride) 
                                for i in range(a.ndim))
        out[idx] = a
        return out

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return undilate(out_grad, self.axes, self.dilation)


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        stride = self.dilation + 1
        out_shape = tuple(s if i not in self.axes 
                            else s//stride 
                            for i, s in enumerate(a.shape))
        out = array_api.full(out_shape, 0., device=a.device)
        idx = tuple(slice(None) if i not in self.axes 
                                else slice(None, None, stride)
                                for i in range(a.ndim))
        out = a[idx]
        return out

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return dilate(out_grad, self.axes, self.dilation)


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        if self.padding:
            A = A.pad(((0, 0),
                       (self.padding, self.padding),
                       (self.padding, self.padding),
                       (0, 0)))
        N, H_in, W_in, C_in = A.shape
        K, _, _, C_out = B.shape
        Ns, Hs, Ws, Cs = A.strides
        H_out, W_out = (H_in - K) // self.stride + 1,\
                       (W_in - K) // self.stride + 1
        
        A_col = A.as_strided(shape=(N, H_out, W_out, K, K, C_in),
                             strides=(Ns, 
                                      Hs * self.stride,
                                      Ws * self.stride,
                                      Hs, Ws, Cs))
        inner_dims = K * K * C_in
        out = A_col.compact().reshape((-1, inner_dims)) @\
              B.compact().reshape((inner_dims, -1))
        return out.reshape((N, H_out, W_out, C_out))

    def gradient(self, out_grad, node):
        A, B = node.inputs
        K = B.shape[0]

        out_grad = out_grad.dilate((1, 2), self.stride - 1)
        B_permuted = B.flip(axes=(0, 1)).permute((0,1,3,2))
        A_permuted = A.permute((3,1,2,0))
        out_grad_permuted = out_grad.permute((1,2,0,3))

        return conv(out_grad, B_permuted, padding=K-1-self.padding),\
               conv(A_permuted, out_grad_permuted, padding=self.padding).permute((1,2,0,3))


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
