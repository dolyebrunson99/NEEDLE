import os, sys

# Set environment variables
os.environ['PYTHONPATH'] = '../python'
os.environ['NEEDLE_BACKEND'] = 'nd'

sys.path.append('../python')

import needle as ndl
import numpy as np
from needle import backend_ndarray as nd


_A = np.arange(16).reshape((4,4))
A = nd.array(_A, device=nd.cpu())
A_perm = A.permute((1, 0))
lhs = A.permute((1, 0)).compact()
print(f"A.shape = {A.shape}, A.stride = {A.strides}, A.offset = {A._offset}")
print(f"A =\n{A}")
print(f"A_perm.shape = {A_perm.shape}, A.stride = {A_perm.strides}, A_perm.offset = {A_perm._offset}")
print(f"A_perm =\n{A_perm}")
print("Compacted Array")
print(f"lhs.shape = {lhs.shape}, lhs.stride = {lhs.strides}, lhs.offset = {lhs._offset}")
print(f"lhs =\n{lhs}")
