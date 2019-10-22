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
#ifndef UTILS_MASK_CUH
#define UTILS_MASK_CUH

#include <cuda_definitions.cuh>
#include "utils_search.cuh"

namespace gpu {
namespace kernels {

template <int mask_width>
__device__ int select_mask_local(index rank, mask m) {
    static_assert(mask_width <= 32, "mask too wide");
    constexpr auto amask = ~mask{0} >> (32 - mask_width);
    auto masked_m = m << (31 - threadIdx.x);
    auto result = ballot(amask, __popc(masked_m) >= rank + 1);
    return __ffs(result) - 1;
}

template <int mask_width>
__device__ int select_mask(index rank, const mask* m) {
    constexpr auto mask_blocks = ceil_div(mask_width, 32);
    constexpr auto local_mask_size = mask_width >= 32 ? 32 : mask_width;
    static_assert(mask_blocks <= 32, "mask too wide");
    // we have few enough blocks so we can do this naively
    index count{};
    index block{};
    for (; block < mask_blocks; ++block) {
        auto partial = __popc(m[block]);
        if (rank >= count && rank < count + partial) {
            return select_mask_local<local_mask_size>(rank - count, m[block]) + block * 32;
        }
        count += partial;
    }
    // should never be reached
    return 0xDEADBEEF;
}    

inline __device__ bool check_mask(index idx, const mask* m) {
    static_assert(sizeof(mask) * 8 == warp_size, "Mask and warp size inconsistent");
    auto mask_block = idx / (sizeof(mask) * 8);
    auto mask_bit = mask(idx % (sizeof(mask) * 8));
    auto masked_bit = mask(1) << mask_bit;
    return bool(m[mask_block] & masked_bit);
}

inline __device__ void compute_bucket_mask_impl(const index* ranks, index rank_count, index rank_base, const index* bucket_prefixsum, mask* bucket_mask, index* range_begins) {
    auto bucket = threadIdx.x;
    auto lb = bucket_prefixsum[bucket] + rank_base;
    auto ub = bucket_prefixsum[bucket + 1] + rank_base;
    auto lb_start = binary_search(ranks, rank_count, lb);
    auto ub_start = binary_search(ranks, rank_count, ub);
    auto local_mask = ballot(full_mask, lb_start != ub_start);
    if (bucket % warp_size == 0) {
        bucket_mask[bucket / warp_size] = local_mask;
    }
    range_begins[bucket] = lb_start;
    // this is a deliberate race condition, as both threads compute the same result :)
    range_begins[bucket + 1] = ub_start;
}

} // namespace kernels
} // namespace gpu
#endif // UTILS_MASK_CUH