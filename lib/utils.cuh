/*
 * Parallel selection algorithm on GPUs
 * Copyright (c) 2018-2019 Tobias Ribizel (oss@ribizel.de)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */
#ifndef UTILS_CUH
#define UTILS_CUH

#include <cuda_definitions.cuh>
#include <limits>
#include <algorithm>

namespace gpu {

struct launch_parameters {
    index block_count;
    index block_size;
    index work_per_thread;
};

constexpr mask full_mask = 0xffffffffu;

template <typename T>
struct max_helper {
    // workaround for ::max being a __host__ function
    constexpr static T value = std::numeric_limits<T>::max();
};

__host__ __device__ inline float add_epsilon(float f) {
    return nextafterf(f, max_helper<float>::value);
}

__host__ __device__ inline double add_epsilon(double f) {
    return nextafter(f, max_helper<double>::value);
}

__host__ __device__ inline constexpr index min(index a, index b) { return a > b ? b : a; }

__host__ __device__ inline constexpr index max(index a, index b) { return a < b ? b : a; }

__host__ __device__ inline constexpr index ceil_div(index a, index b) { return (a + b - 1) / b; }

__device__ inline index ceil_log2(index i) {
    auto high_bit = 31 - __clz(i);
    return __popc(i) <= 1 ? high_bit : high_bit + 1;
}

namespace kernels {

template <typename T>
__device__ void swap(T& a, T& b) {
    auto tmp = b;
    b = a;
    a = tmp;
}

template <typename T>
__device__ T shfl(mask amask, T el, index source, index width = warp_size) {
#if (__CUDACC_VER_MAJOR__ >= 9)
    return __shfl_sync(amask, el, source, width);
#else
    return __shfl(el, source);
#endif
}

template <typename T>
__device__ T shfl_xor(mask amask, T el, index lanemask, index width = warp_size) {
#if (__CUDACC_VER_MAJOR__ >= 9)
    return __shfl_xor_sync(amask, el, lanemask, width);
#else
    return __shfl_xor(el, lanemask);
#endif
}

__device__ inline mask ballot(mask amask, bool predicate) {
#if (__CUDACC_VER_MAJOR__ >= 9)
    return __ballot_sync(amask, predicate);
#else
    return __ballot(predicate) & amask;
#endif
}

__device__ inline void sync_dist(int dist) {
#if (__CUDACC_VER_MAJOR__ >= 9)
    if (dist >= warp_size) {
        __syncthreads();
    } else {
        __syncwarp();
    }
#else
    __syncthreads();
#endif
}

} // namespace kernels
} // namespace gpu

#endif // UTILS_CUH
