#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace needle {
namespace cpu {

#define ALIGNMENT 256
#define TILE 8
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);


/**
 * This is a utility structure for maintaining an array aligned to ALIGNMENT boundaries in
 * memory.  This alignment should be at least TILE * ELEM_SIZE, though we make it even larger
 * here by default.
 */
struct AlignedArray {
  AlignedArray(const size_t size) {
    int ret = posix_memalign((void**)&ptr, ALIGNMENT, size * ELEM_SIZE);
    if (ret != 0) throw std::bad_alloc();
    this->size = size;
  }
  ~AlignedArray() { free(ptr); }
  size_t ptr_as_int() {return (size_t)ptr; }
  scalar_t* ptr;
  size_t size;
};



void Fill(AlignedArray* out, scalar_t val) {
  /**
   * Fill the values of an aligned array with val
   */
  for (int i = 0; i < out->size; i++) {
    out->ptr[i] = val;
  }
}

struct CompactIndices {
  CompactIndices(
    int32_t ndim,
    int32_t num_elements,
    std::vector<int32_t> &shape,
    std::vector<int32_t> &strides,
    size_t offset
  ) : ndim(ndim), num_elements(num_elements), shape(shape), strides(strides), offset(offset) {
    idxs = new std::vector<int>(ndim);
  }
  struct Iterator {
    Iterator(CompactIndices *ci, int c) : ci(ci), cnt(c) {}
    Iterator& operator++() {
      ci->idxs->at(ci->ndim - 1)++;
      int carry = 0;
      for (int j = ci->ndim - 1; j >= 0; --j) {
        ci->idxs->at(j) += carry;
        if (ci->idxs->at(j) == ci->shape.at(j)) {
          carry = 1;
          ci->idxs->at(j) = 0;
        } else break;
      }
      cnt++;
      return *this;
    }
    std::pair<int32_t, int32_t> operator*() const {
      int32_t idx = 0;
      for (int j = 0; j < ci->ndim; ++j)
        idx += ci->strides.at(j) * ci->idxs->at(j);
      return { cnt, idx + ci->offset };
    }
    bool operator==(const Iterator& other) const { return cnt == other.cnt; }
    bool operator!=(const Iterator& other) const { return cnt != other.cnt; }
  private:
    int32_t cnt;
    CompactIndices *ci;
  };
  Iterator begin() { return Iterator(this, 0); }
  Iterator end()   { return Iterator(this, num_elements); }
private:
  int32_t ndim;
  int32_t num_elements;
  const std::vector<int32_t> &shape;
  const std::vector<int32_t> &strides;
  const size_t offset;
  std::vector<int> *idxs;
};

void Compact(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory
   *
   * Args:
   *   a: non-compact representation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   *
   * Returns:
   *  void (you need to modify out directly, rather than returning anything; this is true for all the
   *  function will implement here, so we won't repeat this note.)
   */
  int ndim = shape.size();
  int32_t num_elements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int32_t>());
  CompactIndices comp_idxs(ndim, num_elements, shape, strides, offset);
  for (auto [cnt, idx] : comp_idxs)
    out->ptr[cnt] = a.ptr[idx];
}

void EwiseSetitem(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array
   *
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  int32_t ndim = shape.size();
  int32_t num_elements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int32_t>());
  CompactIndices comp_idxs(ndim, num_elements, shape, strides, offset);
  for (auto [cnt, idx] : comp_idxs)
    out->ptr[idx] = a.ptr[cnt];
}    

void ScalarSetitem(const size_t size, scalar_t val, AlignedArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   *
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
   *         product of items in shape, but convenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */

  int32_t ndim = shape.size();
  int32_t num_elements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int32_t>());
  CompactIndices comp_idxs(ndim, num_elements, shape, strides, offset);
  for (auto [cnt, idx] : comp_idxs)
    out->ptr[idx] = val;
}

/**
 * In the code the follows, use the above template to create analogous element-wise
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

#define EWISE_BOP(fnName, op)\
  void fnName(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {\
    for (size_t i = 0; i < a.size; i++) {\
      out->ptr[i] = op(a.ptr[i], b.ptr[i]);\
    }\
  }

#define EWISE_UOP(fnName, op)\
  void fnName(const AlignedArray& a, AlignedArray* out) {\
    for (size_t i = 0; i < a.size; i++) {\
      out->ptr[i] = op(a.ptr[i]);\
    }\
  }

#define SCALAR_BOP(fnName, op)\
  void fnName(const AlignedArray& a, scalar_t val, AlignedArray* out) {\
    for (size_t i = 0; i < a.size; i++) {\
      out->ptr[i] = op(a.ptr[i], val);\
    }\
  }

EWISE_BOP(EwiseAdd, [](scalar_t x, scalar_t y) { return x + y; })
EWISE_BOP(EwiseMul, [](scalar_t x, scalar_t y) { return x * y; })
EWISE_BOP(EwiseDiv, [](scalar_t x, scalar_t y) { return x / y; })
EWISE_BOP(EwiseEq,  [](scalar_t x, scalar_t y) { return x == y; })
EWISE_BOP(EwiseGe,  [](scalar_t x, scalar_t y) { return x >= y; })
EWISE_BOP(EwiseMaximum, std::max)
EWISE_UOP(EwiseLog, std::log)
EWISE_UOP(EwiseExp, std::exp)
EWISE_UOP(EwiseTanh, std::tanh)

SCALAR_BOP(ScalarAdd, [](scalar_t x, scalar_t y) { return x + y; })
SCALAR_BOP(ScalarMul, [](scalar_t x, scalar_t y) { return x * y; })
SCALAR_BOP(ScalarDiv, [](scalar_t x, scalar_t y) { return x / y; })
SCALAR_BOP(ScalarEq,  [](scalar_t x, scalar_t y) { return x == y; })
SCALAR_BOP(ScalarGe,  [](scalar_t x, scalar_t y) { return x >= y; })
SCALAR_BOP(ScalarMaximum, std::max)
SCALAR_BOP(ScalarPower, std::pow)

void Matmul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m, uint32_t n,
            uint32_t p) {
  /**
   * Multiply two (compact) matrices into an output (also compact) matrix.  For this implementation
   * you can use the "naive" three-loop algorithm.
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: compact 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   */

  for (int _m = 0; _m < m; ++_m) {
    for (int _p = 0; _p < p; ++_p) {
      out->ptr[_m * p + _p] = 0;
      for (int _n = 0; _n < n; ++_n) {
        out->ptr[_m * p + _p] += a.ptr[_m * n + _n] * b.ptr[_n * p + _p];
      }
    }
  }
}

inline void AlignedDot(const float* __restrict__ a,
                       const float* __restrict__ b,
                       float* __restrict__ out) {

  /**
   * Multiply together two TILE x TILE matrices, and _add _the result to out (it is important to add
   * the result to the existing out, which you should not set to zero beforehand).  We are including
   * the compiler flags here that enable the compile to properly use vector operators to implement
   * this function.  Specifically, the __restrict__ keyword indicates to the compile that a, b, and
   * out don't have any overlapping memory (which is necessary in order for vector operations to be
   * equivalent to their non-vectorized counterparts (imagine what could happen otherwise if a, b,
   * and out had overlapping memory).  Similarly the __builtin_assume_aligned keyword tells the
   * compiler that the input array will be aligned to the appropriate blocks in memory, which also
   * helps the compiler vectorize the code.
   *
   * Args:
   *   a: compact 2D array of size TILE x TILE
   *   b: compact 2D array of size TILE x TILE
   *   out: compact 2D array of size TILE x TILE to write to
   */

  a = (const float*)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
  b = (const float*)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
  out = (float*)__builtin_assume_aligned(out, TILE * ELEM_SIZE);

  /// BEGIN SOLUTION
  for (int i = 0; i < TILE; ++i) {
    for (int j = 0; j < TILE; ++j) {
      for (int k = 0; k < TILE; ++k) {
        out[i * TILE + j] += a[i * TILE + k] * b[k * TILE + j];
      }
    }
  }
  /// END SOLUTION
}

void MatmulTiled(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m,
                 uint32_t n, uint32_t p) {
  /**
   * Matrix multiplication on tiled representations of array.  In this setting, a, b, and out
   * are all *4D* compact arrays of the appropriate size, e.g. a is an array of size
   *   a[m/TILE][n/TILE][TILE][TILE]
   * You should do the multiplication tile-by-tile to improve performance of the array (i.e., this
   * function should call `AlignedDot()` implemented above).
   *
   * Note that this function will only be called when m, n, p are all multiples of TILE, so you can
   * assume that this division happens without any remainder.
   *
   * Args:
   *   a: compact 4D array of size m/TILE x n/TILE x TILE x TILE
   *   b: compact 4D array of size n/TILE x p/TILE x TILE x TILE
   *   out: compact 4D array of size m/TILE x p/TILE x TILE x TILE to write to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   *
   */
  /// BEGIN SOLUTION
  for (int i = 0; i < m * p; ++i)
    out[i] = 0.0;
  
  for (int _m = 0; _m < m / TILE; ++_m) {
    for (int _n = 0; _n < n / TILE; ++_n) {
      for (int _p = 0; _p < p / TILE; ++_p) {
        AlignedDot(&a.ptr[_m * n * TILE + _n * TILE * TILE], 
                   &b.ptr[_n * p * TILE + _p * TILE * TILE], 
                   &out->ptr[_m * p * TILE + _p * TILE * TILE]);
      }
    }
  }
  /// END SOLUTION
}

void ReduceMax(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  for (int i = 0; i < out->size; i++) {
    scalar_t max = a.ptr[i * reduce_size];
    for (int j = 0; j < reduce_size; j++) {
      max = std::max(max, a.ptr[i * reduce_size + j]);
    }
    out->ptr[i] = max;
  }
}

void ReduceSum(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking sum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  for (int i = 0; i < out->size; i++) {
    scalar_t sum = 0;
    for (int j = 0; j < reduce_size; j++) {
      sum += a.ptr[i * reduce_size + j];
    }
    out->ptr[i] = sum;
  }
}

}  // namespace cpu
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cpu, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cpu;

  m.attr("__device_name__") = "cpu";
  m.attr("__tile_size__") = TILE;

  py::class_<AlignedArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def("ptr", &AlignedArray::ptr_as_int)
      .def_readonly("size", &AlignedArray::size);

  // return numpy array (with copying for simplicity, otherwise garbage
  // collection is a pain)
  m.def("to_numpy", [](const AlignedArray& a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });
    return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset);
  });

  // convert from numpy (with copying)
  m.def("from_numpy", [](py::array_t<scalar_t> a, AlignedArray* out) {
    std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE);
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
  // m.def("matmul_tiled", MatmulTiled);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
