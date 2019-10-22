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
#ifndef QS_REDUCE_CUH
#define QS_REDUCE_CUH

#include "utils_prefixsum.cuh"

namespace gpu {
namespace kernels {

template <typename Config>
__global__ void partition_prefixsum(index* counts, index block_count) {
    __shared__ index local_lcounts[Config::algorithm::max_block_count];
    __shared__ index local_rcounts[Config::algorithm::max_block_count];
    auto i = threadIdx.x;
    auto l = i >= block_count ? 0 : counts[2 * i];
    auto r = i >= block_count ? 0 : counts[2 * i + 1];
    local_lcounts[i] = l;
    local_rcounts[i] = r;
    small_prefix_sum<Config::algorithm::max_block_count_log2>(local_lcounts);
    small_prefix_sum<Config::algorithm::max_block_count_log2>(local_rcounts);
    __syncthreads();
    if (i < block_count) {
        counts[2 * i + 2] = local_lcounts[i];
        counts[2 * i + 3] = local_rcounts[i];
    }
    // store the total sum at the beginning
    if (i == block_count - 1) {
        counts[0] = l + local_lcounts[i];
        counts[1] = r + local_rcounts[i];
    }
}

} // namespace kernels
} // namespace gpu

#endif // QS_REDUCE_CUH