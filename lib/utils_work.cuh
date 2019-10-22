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
#ifndef UTILS_WORK_CUH
#define UTILS_WORK_CUH

#include "utils.cuh"

namespace gpu {
namespace kernels {

template <typename Config, typename F>
__device__ void blockwise_work(index local_work, index size, F function) {
    auto stride = gridDim.x * blockDim.x;
    auto base_idx = threadIdx.x + blockDim.x * blockIdx.x;
    constexpr auto unroll = Config::algorithm::unroll;
    for (index i = 0; i < local_work; i += unroll) {
        // will any thread in the current iteration be without work?
        auto warp_last_idx = (base_idx / warp_size) * warp_size + warp_size - 1;
        if (warp_last_idx + (i + unroll - 1) * stride >= size) {
            // then the compiler cannot benefit from unrolling
            for (auto j = 0; j < unroll; ++j) {
                auto idx = base_idx + (i + j) * stride;
                auto amask = ballot(full_mask, idx < size);
                if (idx < size) {
                    function(idx, amask);
                }
            }
        } else {
// otherwise all predicates above will be true, so we can unroll
#pragma unroll
            for (auto j = 0; j < unroll; ++j) {
                auto idx = base_idx + (i + j) * stride;
                function(idx, full_mask);
            }
        }
    }
}
template <typename Config, typename F>
__device__ void blockwise_work_local_large(index local_work, index size, F function) {
    auto stride = blockDim.x;
    auto base_idx = threadIdx.x;
    constexpr auto unroll = Config::algorithm::unroll;
    for (index i = 0; i < local_work; i += unroll) {
        // will any thread in the current iteration be without work?
        auto warp_last_idx = (base_idx / warp_size) * warp_size + warp_size - 1;
        if (warp_last_idx + (i + unroll - 1) * stride >= size) {
            // then the compiler cannot benefit from unrolling
            for (auto j = 0; j < unroll; ++j) {
                auto idx = base_idx + (i + j) * stride;
                auto amask = ballot(full_mask, idx < size);
                if (idx < size) {
                    function(idx, amask);
                }
            }
        } else {
// otherwise all predicates above will be true, so we can unroll
#pragma unroll
            for (auto j = 0; j < unroll; ++j) {
                auto idx = base_idx + (i + j) * stride;
                function(idx, full_mask);
            }
        }
    }
}

template <typename F>
__device__ void blockwise_work_local(index size, F function) {
    for (index i = threadIdx.x; i < size; i += blockDim.x) {
        function(i);
    }
}

} // namespace kernels
} // namespace gpu

#endif // UTILS_WORK_CUH