#ifndef QS_RECURSION_CUH
#define QS_RECURSION_CUH

#include "qs_reduce.cuh"
#include "qs_scan.cuh"
#include "utils_prefixsum.cuh"
#include "utils_sampling.cuh"
#include "utils_basecase.cuh"
#include "utils_search.cuh"

namespace gpu {
namespace kernels {

template <typename T, typename Config>
__global__ void quickselect_tailcall(T* in, T* tmp,
                                     index* count_tmp, index size, index rank, T pivot,
                                     T* out);

template <typename T, typename Config>
__device__ __forceinline__ void launch_quickselect(T* in, T* tmp,
                                                   index* count_tmp, index size,
                                                   index rank, T* out) {
    auto idx = threadIdx.x;
    // assert blockDim.x == warp_size

    if (size <= Config::basecase::size) {
        if (idx == 0) {
            select_bitonic_basecase<T, Config><<<1, Config::basecase::launch_size>>>(in, size, rank, out);
        }
    } else {
        // find sample median
        auto pick_idx = random_pick_idx(idx, warp_size, size);
        auto pick = in[pick_idx];
        auto local = pick;
        bitonic_helper_warp<T, 0, warp_size_log2>::sort(&local, false);
        auto pivot = shfl(full_mask, local, warp_size / 2);

        // determine the index of the sample median
        auto mask = ballot(full_mask, pick == pivot);
        auto pivot_idx = shfl(full_mask, pick_idx, __ffs(mask) - 1);
        if (idx > 0) {
            return;
        }
        // swap the sample median to the first position
        swap(in[pivot_idx], in[0]);
        // reset atomic counters
        if (!Config::algorithm::shared_memory) {
            count_tmp[0] = 0;
            count_tmp[1] = 0;
        }
        gpu::partition<T, Config>(in + 1, tmp, count_tmp, size - 1, pivot);
        quickselect_tailcall<T, Config>
                <<<1, warp_size>>>(in + 1, tmp, count_tmp, size - 1, rank, pivot, out);
    }
}

template <typename T, typename Config>
__global__ void quickselect_tailcall(T* in, T* tmp,
                                     index* count_tmp, index size, index rank, T pivot,
                                     T* out) {
    if (threadIdx.x >= warp_size) {
        return;
    }

    auto lcount = count_tmp[0];
    auto rcount = count_tmp[1];
    if (rank == lcount) {
        if (threadIdx.x == 0) {
            *out = pivot;
        }
    } else if (rank < lcount) {
        launch_quickselect<T, Config>(tmp, in, count_tmp, lcount, rank, out);
    } else {
        launch_quickselect<T, Config>(tmp + lcount, in, count_tmp, rcount, rank - lcount - 1, out);
    }
}

template <typename T, typename Config>
__global__ void quickselect(T* in, T* tmp, index* count_tmp,
                            index size, index rank, T* out) {
    launch_quickselect<T, Config>(in, tmp, count_tmp, size, rank, out);
}

} // namespace kernels

template <typename T, typename Config>
void quickselect(T* in, T* tmp, index* count_tmp, index size, index rank, T* out) {
    kernels::quickselect<T, Config><<<1, warp_size>>>(in, tmp, count_tmp, size, rank, out);
}

template <typename Config>
index quickselect_alloc_size(index size) {
    return sizeof(index) * (Config::algorithm::max_block_count * 2 + 2);
}

} // namespace gpu

#endif // QS_RECURSION_CUH
