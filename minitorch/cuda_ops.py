# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs) -> Fn:
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn, **kwargs) -> FakeCUDAKernel:
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # TODO: Implement for Task 3.3.
        #raise NotImplementedError("Need to implement for Task 3.3")
        if i < out_size:
            linear_index = i
            for dim in range(len(out_shape) - 1, -1, -1):
                out_index[dim] = linear_index % out_shape[dim]
                linear_index //= out_shape[dim]
                
            for dim in range(len(in_shape)):
                if dim < len(out_shape) and out_shape[dim] == in_shape[dim]:
                    in_index[dim] = out_index[dim]
                else:
                    in_index[dim] = 0
                    
            out_pos = 0
            in_pos = 0
            for dim in range(len(out_shape)):
                out_pos += out_index[dim] * out_strides[dim]
            for dim in range(len(in_shape)):
                in_pos += in_index[dim] * in_strides[dim]

            # Apply the function
            out[out_pos] = fn(in_storage[in_pos])
      
    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)

        # Global thread index
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # TODO: Implement for Task 3.3.
        #raise NotImplementedError("Need to implement for Task 3.3")
        if i < out_size:
            # Map global index to output multidimensional index
            temp = i
            for dim in range(len(out_shape) - 1, -1, -1):
                out_index[dim] = temp % out_shape[dim]
                temp //= out_shape[dim]

            # Compute broadcasted indices for `a` and `b`
            for dim in range(len(a_shape)):
                a_index[dim] = (
                    out_index[dim]
                    if dim < len(out_shape) and out_shape[dim] == a_shape[dim]
                    else 0
                )
            for dim in range(len(b_shape)):
                b_index[dim] = (
                    out_index[dim]
                    if dim < len(out_shape) and out_shape[dim] == b_shape[dim]
                    else 0
                )

            # Compute flattened positions
            a_pos = 0
            for dim in range(len(a_shape)):
                a_pos += a_index[dim] * a_strides[dim]

            b_pos = 0
            for dim in range(len(b_shape)):
                b_pos += b_index[dim] * b_strides[dim]

            # Compute the output value
            out_value = fn(a_storage[a_pos], b_storage[b_pos])

            # Compute the output flattened position
            out_pos = 0
            for dim in range(len(out_shape)):
                out_pos += out_index[dim] * out_strides[dim]

            # Write to the output storage
            out[out_pos] = out_value


    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    """This is a practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32
    # Shared memory for the block
    cache = cuda.shared.array(BLOCK_DIM, numba.float64)

    # Compute global thread index and local position within the block
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    # TODO: Implement for Task 3.3.
    #raise NotImplementedError("Need to implement for Task 3.3")
    if i < size:
        cache[pos] = a[i]
    else:
        cache[pos] = 0.0  # Padding for out-of-bounds threads

    # Synchronize threads to ensure all values are loaded into shared memory
    cuda.syncthreads()

    # Perform block-wide reduction using shared memory
    stride = 1
    while stride < BLOCK_DIM:
        if pos % (2 * stride) == 0 and pos + stride < BLOCK_DIM:
            cache[pos] += cache[pos + stride]
        stride *= 2
        cuda.syncthreads()

    # Write the result for this block to the output tensor
    if pos == 0:
        out[cuda.blockIdx.x] = cache[0]    


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)

        #block_id
        out_pos = cuda.blockIdx.x
        #thread_id
        pos = cuda.threadIdx.x

        # TODO: Implement for Task 3.3.
        #raise NotImplementedError("Need to implement for Task 3.3")
        # Initialize the shared memory for this block
        cache[pos] = reduce_value

        # Compute the reduction size along the reduce dimension
        reduce_size = a_shape[reduce_dim]

        # Map the block index to an output tensor index
        for dim in range(len(out_shape)):
            if dim == reduce_dim:
                out_index[dim] = 0  # We'll iterate over this later
            else:
                out_index[dim] = out_pos // out_strides[dim] % out_shape[dim]

        # Iterate over the reduction dimension in chunks handled by threads
        for i in range(pos, reduce_size, BLOCK_DIM):
            # Compute the input tensor index for this chunk
            input_index = cuda.local.array(MAX_DIMS, numba.int32)
            for dim in range(len(a_shape)):
                if dim == reduce_dim:
                    input_index[dim] = i
                else:
                    input_index[dim] = out_index[dim]

            # Compute flat input position
            input_pos = 0
            for dim in range(len(a_shape)):
                input_pos += input_index[dim] * a_strides[dim]

            # Apply the reduction function to accumulate the value
            cache[pos] = fn(cache[pos], a_storage[input_pos])

        # Synchronize threads to ensure shared memory is ready
        cuda.syncthreads()

        # Perform block-wide reduction in shared memory
        stride = 1
        while stride < BLOCK_DIM:
            if pos % (2 * stride) == 0 and pos + stride < BLOCK_DIM:
                cache[pos] = fn(cache[pos], cache[pos + stride])
            stride *= 2
            cuda.syncthreads()

        # Write the reduced result for this block to the output tensor
        if pos == 0:
            out[out_pos] = cache[0]


    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """This is a practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32
    # TODO: Implement for Task 3.3.
    #raise NotImplementedError("Need to implement for Task 3.3")

    # Shared memory for A and B matrices
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float32)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float32)

    # Thread indices
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    # Global indices
    row = cuda.blockIdx.x * cuda.blockDim.x + tx
    col = cuda.blockIdx.y * cuda.blockDim.y + ty

    # Temporary variable for storing the result
    result = 0.0

    # Loop over tiles
    for k in range(0, size, BLOCK_DIM):
        # Load a and b tiles into shared memory
        if row < size and k + ty < size:
            a_shared[tx, ty] = a[row * size + (k + ty)]
        else:
            a_shared[tx, ty] = 0.0

        if col < size and k + tx < size:
            b_shared[tx, ty] = b[(k + tx) * size + col]
        else:
            b_shared[tx, ty] = 0.0

        # Synchronize threads within a block
        cuda.syncthreads()

        # Compute dot product for this tile
        for n in range(BLOCK_DIM):
            result += a_shared[tx, n] * b_shared[n, ty]

        # Synchronize again to ensure all threads are done before loading new tiles
        cuda.syncthreads()

    # Write result to global memory
    if row < size and col < size:
        out[row * size + col] = result


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The final position c[i, j]
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # The local position in the block.
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Code Plan:
    # 1) Move across shared dimension by block dim.
    #    a) Copy into shared memory for a matrix.
    #    b) Copy into shared memory for b matrix
    #    c) Compute the dot produce for position c[i, j]
    # TODO: Implement for Task 3.4.
    #raise NotImplementedError("Need to implement for Task 3.4")


    # Initialize the result for C[i, j]
    result = 0.0

    # Move across shared dimension in tiles of BLOCK_DIM
    for tile_idx in range(0, a_shape[-1], BLOCK_DIM):
        # Load A tile into shared memory
        if i < out_shape[-2] and tile_idx + pj < a_shape[-1]:
            a_shared[pi, pj] = a_storage[
                batch * a_batch_stride
                + i * a_strides[-2]
                + (tile_idx + pj) * a_strides[-1]
            ]
        else:
            a_shared[pi, pj] = 0.0

        # Load B tile into shared memory
        if tile_idx + pi < b_shape[-2] and j < out_shape[-1]:
            b_shared[pi, pj] = b_storage[
                batch * b_batch_stride
                + (tile_idx + pi) * b_strides[-2]
                + j * b_strides[-1]
            ]
        else:
            b_shared[pi, pj] = 0.0

        # Synchronize threads after loading tiles
        cuda.syncthreads()

        # Compute partial dot product for the tile
        for k in range(BLOCK_DIM):
            result += a_shared[pi, k] * b_shared[k, pj]

        # Synchronize threads before loading the next tile
        #cuda.syncthreads()

    # Write the result to the global memory
    #if i < out_shape[-2] and j < out_shape[-1]:
    #    out[
    #        batch * out_strides[0]
    #        + i * out_strides[-2]
    #        + j * out_strides[-1]
    #    ] = result

tensor_matrix_multiply = jit(_tensor_matrix_multiply)
