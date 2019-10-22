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
#ifndef QS_LAUNCHERS_CUH
#define QS_LAUNCHERS_CUH

#include "qs_reduce.cuh"
#include "qs_scan.cuh"

namespace gpu {

using kernels::partition;
using kernels::partition_count;
using kernels::partition_distr;
using kernels::partition_prefixsum;

template <typename T, typename Config>
__device__ __host__ void partition(const T* in, T* out, index* counts, index size, T pivot) {
    auto bsize = Config::algorithm::max_block_size;
    auto nblocks = min(ceil_div(size, bsize), Config::algorithm::max_block_count);
    auto per_thread = ceil_div(size, nblocks * bsize);
    if (Config::algorithm::shared_memory) {
        partition_count<T, Config><<<nblocks, bsize>>>(in, counts, size, pivot, per_thread);
        partition_prefixsum<Config><<<1, Config::algorithm::max_block_count>>>(counts, nblocks);
        partition_distr<T, Config><<<nblocks, bsize>>>(in, out, counts, size, pivot, per_thread);
    } else {
        partition<T, Config><<<nblocks, bsize>>>(in, out, counts, size, pivot, per_thread);
    }
}

} // namespace gpu

#endif // QS_LAUNCHERS_CUH