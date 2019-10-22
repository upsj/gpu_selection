#ifndef QS_RECURSION_MULTI_CUH
#define QS_RECURSION_MULTI_CUH

#include "qs_recursion.cuh"

namespace gpu {
namespace kernels {

template <typename T, typename Config>
__global__ void quickselect_tailcall_multi(T* in, T* tmp,
                                           index* count_tmp, index size, const index* ranks, index rank_count, index rank_base, T pivot,
                                           T* out);

template <typename T, typename Config>
__device__ __forceinline__ void launch_quickselect_multi(T* in, T* tmp,
                                                         index* count_tmp, index size,
                                                         const index* ranks, index rank_count, index rank_base, T* out) {
    if (rank_count == 0) {
        return;
    }
    auto idx = threadIdx.x;
    // assert blockDim.x == warp_size

    if (size <= Config::basecase::size) {
        if (idx == 0) {
            select_bitonic_multiple_basecase<T, Config><<<1, Config::basecase::launch_size>>>(in, size, ranks, rank_count, rank_base, out);
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
        quickselect_tailcall_multi<T, Config>
                <<<2, warp_size>>>(in + 1, tmp, count_tmp, size - 1, ranks, rank_count, rank_base, pivot, out);
    }
}

template <typename T, typename Config>
__global__ void quickselect_tailcall_multi(T* in, T* tmp,
                                           index* count_tmp, index size, const index* ranks, index rank_count, index rank_base, T pivot,
                                           T* out) {
    // assert blockDim.x == warp_size

    auto lcount = count_tmp[0];
    auto rcount = count_tmp[1];
    auto middle = binary_search(ranks, rank_count, lcount + rank_base);
    if (blockIdx.x == 0) {
        if (middle < rank_count && ranks[middle] == lcount + rank_base) {
            if (threadIdx.x == 0) {
                out[middle] = pivot;
            }
            if (middle < rank_count - 1) {
                launch_quickselect_multi<T, Config>(tmp + lcount, in + lcount, count_tmp + lcount, rcount,
                    ranks + middle + 1, rank_count - middle - 1, rank_base + (lcount + 1), out + middle + 1);
            }
            } else {
            if (middle < rank_count) {
                launch_quickselect_multi<T, Config>(tmp + lcount, in + lcount, count_tmp + lcount, rcount,
                    ranks + middle, rank_count - middle, rank_base + (lcount + 1), out + middle);
            }
        }
    } else {
        if (middle > 0) {
            launch_quickselect_multi<T, Config>(tmp, in, count_tmp, lcount, ranks, middle, rank_base, out);
        }
    }
}

template <typename T, typename Config>
__global__ void quickselect_multi(T* in, T* tmp, index* count_tmp,
                                  index size, const index* ranks, index rank_count, T* out) {
    launch_quickselect_multi<T, Config>(in, tmp, count_tmp, size, ranks, rank_count, 0, out);
}

} // namespace kernels

template <typename T, typename Config>
void quickselect_multi(T* in, T* tmp, index* count_tmp, index size, const index* ranks, index rank_count, T* out) {
    kernels::quickselect_multi<T, Config><<<1, warp_size>>>(in, tmp, count_tmp, size, ranks, rank_count, out);
}

template <typename Config>
index quickselect_alloc_size_multi(index size) {
    return sizeof(index) * (std::max(Config::algorithm::max_block_count * 2 + 2, size));
}

} // namespace gpu

#endif // QS_RECURSION_MULTI_CUH
