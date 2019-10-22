#ifndef UTILS_PREFIXSUM_CUH
#define UTILS_PREFIXSUM_CUH

#include "utils.cuh"
#include "utils_mask.cuh"

namespace gpu {
namespace kernels {

template <index size_log2>
__device__ void small_prefix_sum_upward(index* data) {
    constexpr auto size = 1 << size_log2;
    auto idx = threadIdx.x;
    // upward phase: reduce
    // here we build an implicit reduction tree, overwriting values
    // the entry at the end of a power-of-two block stores the sum of this block
    // the block sizes are increased stepwise
    for (index blocksize = 2; blocksize <= size; blocksize *= 2) {
        index base_idx = idx * blocksize;
        __syncthreads();
        if (base_idx < size) {
            data[base_idx + blocksize - 1] += data[base_idx + blocksize / 2 - 1];
        }
    }
}

template <index size_log2>
__device__ void small_prefix_sum_downward(index* data) {
    constexpr auto size = 1 << size_log2;
    auto idx = threadIdx.x;
    // downward phase: build prefix sum
    // every right child stores the sum of its left sibling
    // every left child stores its own sum
    // thus we store zero at the root
    if (idx == 0) {
        data[size - 1] = 0;
    }
    for (auto blocksize = size; blocksize != 1; blocksize /= 2) {
        auto base_idx = idx * blocksize;
        __syncthreads();
        if (base_idx < size) {
            // we preserve the invariant for the next level
            auto r = data[base_idx + blocksize - 1];
            auto l = data[base_idx + blocksize / 2 - 1];
            data[base_idx + blocksize / 2 - 1] = r;
            data[base_idx + blocksize - 1] = l + r;
        }
    }
}

template <index size_log2>
__device__ void small_prefix_sum(index* data) {
    small_prefix_sum_upward<size_log2>(data);
    __syncthreads();
    small_prefix_sum_downward<size_log2>(data);
}

template <index size_log2>
__device__ void small_prefix_sum_sentinel(index* data) {
    auto size = 1 << size_log2;
    gpu::index tmp{};
    if (threadIdx.x == size - 1) tmp = data[size - 1];
    __syncthreads();
    small_prefix_sum<size_log2>(data);
    __syncthreads();
    // append sentinel
    if (threadIdx.x == size - 1) data[size] = data[size - 1] + tmp;
}

template<index size_log2>
__device__ void masked_prefix_sum(index* counts, const mask* m) {
    index bucket = threadIdx.x;
    constexpr auto size = 1 << size_log2;
    if (bucket < size && !check_mask(bucket, m)) {
        counts[bucket] = 0;
    }
    __syncthreads();
    small_prefix_sum<size_log2>(counts);
}

template<index size_log2>
__device__ void masked_prefix_sum_sentinel(index* counts, const mask* m) {
    index bucket = threadIdx.x;
    constexpr auto size = 1 << size_log2;
    if (bucket < size && !check_mask(bucket, m)) {
        counts[bucket] = 0;
    }
    __syncthreads();
    small_prefix_sum_sentinel<size_log2>(counts);
}

/*
 * Prefix sum selection
 */
template <index size_log2>
__device__ void prefix_sum_select(const index* counts, index rank, poracle* out_bucket,
                                  index* out_rank) {
    constexpr auto size = 1 << size_log2;
    // first compute prefix sum of counts
    auto idx = threadIdx.x;
    __shared__ index sums[size];
    sums[2 * idx] = counts[2 * idx];
    sums[2 * idx + 1] = counts[2 * idx + 1];
    small_prefix_sum<size_log2>(sums);
    __syncthreads();
    if (idx >= warp_size) {
        return;
    }
    // then determine which group of size step the element belongs to
    constexpr auto step = size / warp_size;
    static_assert(step <= warp_size, "need a third selection level");
    auto mask = ballot(full_mask, sums[(warp_size - idx - 1) * step] > rank);
    if (idx >= step) {
        return;
    }
    auto group = __clz(mask) - 1;
    // finally determine which bucket within the group the element belongs to
    auto base_idx = step * group;
    constexpr auto cur_mask = ((1u << (step - 1)) << 1) - 1;
    mask = ballot(cur_mask, sums[base_idx + (step - idx - 1)] > rank);
    // here we need to subtract warp_size - step since we only use a subset of the warp
    if (idx == 0) {
        *out_bucket = __clz(mask) - 1 - (warp_size - step) + base_idx;
        *out_rank = rank - sums[*out_bucket];
    }
}

} // namespace kernels
} // namespace gpu

#endif // UTILS_PREFIXSUM_CUH
