#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>
#include <math.h>


namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

CudaDims CudaTwoDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) out[tid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides

CudaVec getCompactStrides(const std::vector<int32_t> &shape){
  uint32_t ndims = shape.size();
  CudaVec compact_strides{.size = ndims};
  compact_strides.data[ndims - 1] = 1;
  for (int i = ndims - 2; i >= 0; --i) {
    compact_strides.data[i] = shape[i + 1] * compact_strides.data[i + 1];
  }
  return compact_strides;
}


__device__ size_t getMemIdx(size_t tid, CudaVec &shape, CudaVec &strides, 
                            CudaVec &compact_strides, size_t offset) {
  int32_t idx = offset;
  for (size_t i = 0; i < shape.size; ++i)
    idx += (tid / compact_strides.data[i]) % shape.data[i] * strides.data[i];
  return idx;
}

__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, CudaVec compact_strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location tid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN SOLUTION
  size_t idx = getMemIdx(tid, shape, strides, compact_strides, offset);
  out[tid] = a[idx];
  /// END SOLUTION
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CudaVec compact_strides = getCompactStrides(shape);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), compact_strides, offset);
}


__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                                   CudaVec strides, CudaVec compact_strides, size_t offset) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t idx = getMemIdx(tid, shape, strides, compact_strides, offset);
  out[idx] = a[tid];
}

void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  You will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(a.size);
  CudaVec compact_strides = getCompactStrides(shape);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, VecToCuda(shape),
                                              VecToCuda(strides), compact_strides, offset);
  /// END SOLUTION
}

__global__ void ScalarSetitemKernel(scalar_t val, scalar_t *out, size_t size,
                                    CudaVec shape, CudaVec strides, 
                                    CudaVec compact_strides, size_t offset) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    size_t idx = getMemIdx(tid, shape, strides, compact_strides, offset);
    out[idx] = val;
  }
}

void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will not be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(size);
  CudaVec compact_strides = getCompactStrides(shape);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(val, out->ptr, size,
                                               VecToCuda(shape), VecToCuda(strides), 
                                               compact_strides, offset);
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////


#define EWISE_BOP(OpFunc, KernelFunc)\
  void OpFunc(const CudaArray& a, const CudaArray& b, CudaArray* out) {\
    CudaDims dim = CudaOneDim(out->size);\
    KernelFunc<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);\
  }

#define EWISE_UOP(OpFunc, KernelFunc)\
  void OpFunc(const CudaArray& a, CudaArray* out) {\
    CudaDims dim = CudaOneDim(out->size);\
    KernelFunc<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);\
  }

#define SCALAR_BOP(OpFunc, KernelFunc)\
  void OpFunc(const CudaArray& a, scalar_t val, CudaArray* out) {\
    CudaDims dim = CudaOneDim(out->size);\
    KernelFunc<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);\
  }

#define EWISE_BOP_KERNEL(KernelFunc, Bop)\
  __global__ void KernelFunc(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {\
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;\
    if (tid < size) out[tid] = Bop(a[tid], b[tid]);\
  }

#define EWISE_UOP_KERNEL(KernelFunc, Uop)\
  __global__ void KernelFunc(const scalar_t* a, scalar_t* out, size_t size) {\
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;\
    if (tid < size) out[tid] = Uop(a[tid]);\
  }

#define SCALAR_BOP_KERNEL(KernelFunc, Bop)\
  __global__ void KernelFunc(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {\
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;\
    if (tid < size) out[tid] = Bop(a[tid], val);\
  }

EWISE_BOP_KERNEL(EwiseAddKernel, [] __device__ (scalar_t a, scalar_t b) { return a + b; })
EWISE_BOP_KERNEL(EwiseMulKernel, [] __device__ (scalar_t a, scalar_t b) { return a * b; })
EWISE_BOP_KERNEL(EwiseDivKernel, [] __device__ (scalar_t a, scalar_t b) { return a / b; })
EWISE_BOP_KERNEL(EwiseEqKernel,  [] __device__ (scalar_t a, scalar_t b) { return a == b; })
EWISE_BOP_KERNEL(EwiseGeKernel,  [] __device__ (scalar_t a, scalar_t b) { return a >= b; })
EWISE_BOP_KERNEL(EwiseMaximumKernel, max)
EWISE_UOP_KERNEL(EwiseLogKernel, logf)
EWISE_UOP_KERNEL(EwiseExpKernel, expf)
EWISE_UOP_KERNEL(EwiseTanhKernel, tanhf)

SCALAR_BOP_KERNEL(ScalarAddKernel, [] __device__ (scalar_t a, scalar_t b) { return a + b; })
SCALAR_BOP_KERNEL(ScalarMulKernel, [] __device__ (scalar_t a, scalar_t b) { return a * b; })
SCALAR_BOP_KERNEL(ScalarDivKernel, [] __device__ (scalar_t a, scalar_t b) { return a / b; })
SCALAR_BOP_KERNEL(ScalarEqKernel,  [] __device__ (scalar_t a, scalar_t b) { return a == b; })
SCALAR_BOP_KERNEL(ScalarGeKernel,  [] __device__ (scalar_t a, scalar_t b) { return a >= b; })
SCALAR_BOP_KERNEL(ScalarMaximumKernel, max)
SCALAR_BOP_KERNEL(ScalarPowerKernel, powf)

EWISE_BOP(EwiseAdd, EwiseAddKernel)
EWISE_BOP(EwiseMul, EwiseMulKernel)
EWISE_BOP(EwiseDiv, EwiseDivKernel)
EWISE_BOP(EwiseEq,  EwiseEqKernel)
EWISE_BOP(EwiseGe,  EwiseGeKernel)
EWISE_BOP(EwiseMaximum, EwiseMaximumKernel)
EWISE_UOP(EwiseLog, EwiseLogKernel)
EWISE_UOP(EwiseExp, EwiseExpKernel)
EWISE_UOP(EwiseTanh, EwiseTanhKernel)

SCALAR_BOP(ScalarAdd, ScalarAddKernel)
SCALAR_BOP(ScalarMul, ScalarMulKernel)
SCALAR_BOP(ScalarDiv, ScalarDivKernel)
SCALAR_BOP(ScalarEq,  ScalarEqKernel)
SCALAR_BOP(ScalarGe,  ScalarGeKernel)
SCALAR_BOP(ScalarMaximum, ScalarMaximumKernel)
SCALAR_BOP(ScalarPower, ScalarPowerKernel)


/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */


////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

#define BLOCK 8 * TILE  // 32 x 32 tile
__global__ void MatmulKernel(scalar_t* A, scalar_t* B, scalar_t* out, 
                       uint32_t M, uint32_t N, uint32_t P) {
  uint32_t bx = blockIdx.x, by = blockIdx.y,
           tx = threadIdx.x, ty = threadIdx.y;
  uint32_t y = by * blockDim.y + ty,
           x = bx * blockDim.x + tx;
  scalar_t val = 0.0;
  __shared__ scalar_t A_tile[BLOCK][BLOCK],
                      B_tile[BLOCK][BLOCK];
  for (int n = 0; n < (N + BLOCK - 1) / BLOCK; ++n) {
    uint32_t xx = n * BLOCK + tx;
    uint32_t yy = n * BLOCK + ty;
    A_tile[ty][tx] = xx < N ? A[y * N + xx] : 0.0f;
    B_tile[ty][tx] = yy < N ? B[yy * P + x] : 0.0f;
    __syncthreads();
    for (int nn = 0; nn < BLOCK; ++nn)
      val += A_tile[ty][nn] * B_tile[nn][tx];
    __syncthreads();
  }
  if (y < M && x < P)
    out[y * P + x] = val;
}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  /// BEGIN SOLUTION
  dim3 grid((M + BLOCK - 1) / BLOCK, (P + BLOCK - 1) / BLOCK),
       block(BLOCK, BLOCK);
  MatmulKernel<<<grid, block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////


void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  assert(false && "Not Implemented");
  /// END SOLUTION
}



void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  assert(false && "Not Implemented");
  /// END SOLUTION
}

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);

  // m.def("reduce_max", ReduceMax);
  // m.def("reduce_sum", ReduceSum);
}
