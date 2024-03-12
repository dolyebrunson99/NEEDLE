"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return (1 + ops.exp(-x))**(-1)
        ### END YOUR SOLUTION

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.device = device
        self.dtype = dtype
        bound = np.sqrt(1 / hidden_size)
        self.W_ih = Parameter(init.rand(input_size, hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
        self.bias_ih = None
        self.bias_hh = None
        if bias:
            self.bias_ih = Parameter(init.rand(hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
            self.bias_hh = Parameter(init.rand(hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
        if nonlinearity == "tanh":
            self.nonlinearity = ops.Tanh()
        elif nonlinearity == "relu":
            self.nonlinearity = ops.ReLU()
        else:
            raise ValueError("unsupported nonlinearity function. Only support ReLU and Tanh.")
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs, input_size = X.shape
        hidden_size = self.hidden_size
        if h is None:
            h = init.zeros(bs, hidden_size)
        if self.bias:
            b_ih = self.bias_ih.reshape((1, -1)).broadcast_to((bs, -1))
            b_hh = self.bias_hh.reshape((1, -1)).broadcast_to((bs, -1))
            h = self.nonlinearity(X@self.W_ih + b_ih + h@self.W_hh + b_hh)
        else:
            h = self.nonlinearity(X@self.W_ih + h@self.W_hh)
        return h
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype

        self.rnn_cells = [ RNNCell(input_size, hidden_size, bias, nonlinearity, device, dtype) ] +\
                         [ RNNCell(hidden_size, hidden_size, bias, nonlinearity, device, dtype) for _ in range(num_layers - 1) ]
        ### END YOUR SOLUTION

    def forward(self, X:Tensor, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, input_size = X.shape
        inputs = list(X.split(axis=0)) # list of length seq_len, inputs[0].shape = bs, input_size
        if h0 is None:
            h0 = init.zeros(self.num_layers, bs, self.hidden_size, device=self.device, dtype=self.dtype)
        h0s = h0.split(axis=0) # tuple of length num_layers, h0s[k].shape = bs, hidden_size
        h_n = []
        for k in range(self.num_layers):
            rnn_cell = self.rnn_cells[k]
            h = h0s[k]
            for t in range(seq_len):
                h = rnn_cell(inputs[t], h) # bs, hidden_size
                inputs[t] = h
            h_n.append(h)
        return ops.stack(inputs, 0), ops.stack(h_n, 0)
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.device = device
        self.dtype = dtype
        bound = np.sqrt(1 / hidden_size)
        self.W_ih = Parameter(init.rand(input_size, 4*hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
        self.W_hh = Parameter(init.rand(hidden_size, 4*hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
        self.bias_ih = None
        self.bias_hh = None
        if bias:
            self.bias_ih = Parameter(init.rand(4*hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
            self.bias_hh = Parameter(init.rand(4*hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
        self.tanh = ops.Tanh()
        self.sigmoid = Sigmoid()
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs = X.shape[0]
        if h is None:
            h = (
                init.zeros(bs, self.hidden_size, device=self.device, dtype=self.dtype),
                init.zeros(bs, self.hidden_size, device=self.device, dtype=self.dtype)
            )
        h0, c0 = h
        ifgo = None
        if self.bias:
            b_ih = self.bias_ih.reshape((1, -1)).broadcast_to((bs, -1))
            b_hh = self.bias_hh.reshape((1, -1)).broadcast_to((bs, -1))
            ifgo = X@self.W_ih + b_ih + h0@self.W_hh + b_hh
        else:
            ifgo = X@self.W_ih + h0@self.W_hh
        
        i,f,g,o = ifgo.reshape((bs, 4, -1)).split(axis=1)
        i,f,g,o = self.sigmoid(i), self.sigmoid(f), self.tanh(g), self.sigmoid(o)
        c_out = f*c0 + i*g
        h_out = o*self.tanh(c_out)
        return h_out, c_out
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype

        self.lstm_cells = [ LSTMCell(input_size, hidden_size, bias, device, dtype) ] +\
                         [ LSTMCell(hidden_size, hidden_size, bias, device, dtype) for _ in range(num_layers - 1) ]
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            c_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, input_size = X.shape
        inputs = list(X.split(axis=0)) # list of length seq_len, inputs[0].shape = bs, input_size
        if h is None:
            h = (
                init.zeros(self.num_layers, bs, self.hidden_size, device=self.device, dtype=self.dtype),
                init.zeros(self.num_layers, bs, self.hidden_size, device=self.device, dtype=self.dtype)
            )
        h0, c0 = h
        hs = list(zip(h0.split(axis=0), c0.split(axis=0)))
        h_n, c_n = [], []
        for k in range(self.num_layers):
            lstm_cell = self.lstm_cells[k]
            h = hs[k]
            for t in range(seq_len):
                h = lstm_cell(inputs[t], h) # bs, hidden_size
                inputs[t] = h[0]
            h_n.append(h[0])
            c_n.append(h[1])
        return ops.stack(inputs, 0), (ops.stack(h_n, 0), ops.stack(c_n, 0))
        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION