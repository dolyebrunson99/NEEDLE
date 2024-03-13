import os, sys

# Set environment variables
os.environ['PYTHONPATH'] = '../python'
os.environ['NEEDLE_BACKEND'] = 'nd'

sys.path.append('../python')

import needle as ndl
import numpy as np
from needle import backend_ndarray as nd

_A = np.arange(3)
_B = np.arange(3)
A = nd.array(_A, device=nd.cpu())
B = nd.array(_B, device=nd.cpu())
C = nd.maximum(A, B)

print(f"A = {A}")
print(f"B = {B}")
print(f"C = {C}")


