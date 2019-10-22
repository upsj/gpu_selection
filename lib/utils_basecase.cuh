#ifndef UTILS_BASECASE_CUH
#define UTILS_BASECASE_CUH
#include "utils_sort.cuh"

namespace gpu {
namespace kernels {

template <typename T, typename Config>
__device__ void load_local(const T* in, T* local, index size) {
    index idx = threadIdx.x;
    for (index i = 0; i < Config::basecase::local_size; ++i) {
        auto lidx = idx * Config::basecase::local_size + i;
        local[i] = lidx < size ? in[lidx] : max_helper<T>::value;
    }
}

template <typename T, typename Config>
__device__ void small_sort_warp(const T* in, T* sorted, index size) {
    index idx = threadIdx.x;
    // load data padded with sentinels
    T local[Config::basecase::local_size];
    load_local<T, Config>(in, local, size);
    using sorter = bitonic_helper_warp<T, Config::basecase::local_size_log2, warp_size_log2>;
    sorter::sort(local, false);
    for (index i = 0; i < Config::basecase::local_size; ++i) {
        auto lidx = idx * Config::basecase::local_size + i;
        sorted[lidx] = local[i];
    }
}

template <typename T, typename Config>
__device__ void small_sort(const T* in, T* sorted, index size) {
    // load data padded with sentinels
    T local[Config::basecase::local_size];
    load_local<T, Config>(in, local, size);
    using sorter = bitonic_helper_global<T, Config::basecase::local_size_log2, warp_size_log2, Config::basecase::size_log2 - warp_size_log2 - Config::basecase::local_size_log2, Config::basecase::size_log2>;
    sorter::sort(local, sorted, false);
}

template <typename T, typename Config>
__global__ void select_bitonic_basecase(const T* in, index size, index rank, T* out) {
    __shared__ T data[Config::basecase::size];
    index idx = threadIdx.x;
    if (size <= Config::basecase::local_size * warp_size) {
        if (idx >= warp_size) {
            return;
        }
        small_sort_warp<T, Config>(in, data, size);
    } else {
        small_sort<T, Config>(in, data, size);
    }
    __syncthreads();
    // store result
    if (idx == 0) {
        *out = data[rank];
    }
}

template <typename T, typename Config>
__device__ void select_bitonic_multiple_basecase_impl(const T* in, index size,
                                                      const index* ranks, index ranks_size,
                                                      index rank_base, T* out) {
    __shared__ T data[Config::basecase::size];
    index idx = threadIdx.x;
    if (size <= Config::basecase::local_size * warp_size) {
        if (idx >= warp_size) {
            return;
        }
        small_sort_warp<T, Config>(in, data, size);
    } else {
        small_sort<T, Config>(in, data, size);
    }
    __syncthreads();
    for (index i = 0; i < Config::basecase::local_size; i++) {
        auto gi = idx * Config::basecase::local_size + i;
        if (gi < ranks_size) {
            auto pos = ranks[gi] - rank_base;
            out[gi] = data[pos];
        }
    }
}

template <typename T, typename Config>
__global__ void select_bitonic_multiple_basecase(const T* in, index size,
                                                        const index* ranks, index ranks_size,
                                                        index rank_base, T* out) {
    select_bitonic_multiple_basecase_impl<T, Config>(in, size, ranks, ranks_size, rank_base, out);
}

} // namespace kernels
} // namespace gpu

#endif // UTILS_BASECASE_CUH