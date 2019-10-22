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
#ifndef SSSS_LAUNCHERS_CUH
#define SSSS_LAUNCHERS_CUH

#include "ssss_build_searchtree.cuh"
#include "ssss_collect.cuh"
#include "ssss_collect_multi.cuh"
#include "ssss_count.cuh"
#include "ssss_reduce.cuh"
#include "ssss_merged.cuh"

namespace gpu {

template <typename T, typename Config>
__host__ __device__ launch_parameters get_launch_parameters(index size) {
    launch_parameters result{};
    result.block_size = Config::algorithm::max_block_size;
    result.block_count = min(ceil_div(size, result.block_size), Config::algorithm::max_block_count);
    auto threads = result.block_size * result.block_count;
    result.work_per_thread = ceil_div(size, threads);
    return result;
}

template <typename T, typename Config>
__host__ __device__ void build_searchtree(const T* in, T* out, index size) {
    constexpr auto threads = Config::searchtree_kernel_size;
    static_assert(threads <= max_block_size, "Work won't fit into a single thread block");
    kernels::build_searchtree<T, Config><<<1, threads>>>(in, out, size);
}

template <typename T, typename Config>
__host__ __device__ void count_buckets(const T* in, const T* tree, index* localcounts,
                                       index* counts, poracle* oracles, index size) {
    auto params = get_launch_parameters<T, Config>(size);
    if (Config::algorithm::shared_memory) {
        kernels::count_buckets<T, Config><<<params.block_count, params.block_size>>>(
                in, tree, localcounts, oracles, size, params.work_per_thread);
        constexpr auto reduce_bsize =
                min(Config::searchtree::width, Config::algorithm::max_block_count);
        constexpr auto reduce_blocks = ceil_div(Config::searchtree::width, reduce_bsize);
        if (Config::algorithm::write) {
            kernels::prefix_sum_counts<Config>
                    <<<reduce_blocks, reduce_bsize>>>(localcounts, counts, params.block_count);
        } else {
            kernels::reduce_counts<Config>
                    <<<reduce_blocks, reduce_bsize>>>(localcounts, counts, params.block_count);
        }
    } else {
        kernels::count_buckets<T, Config><<<params.block_count, params.block_size>>>(
                in, tree, counts, oracles, size, params.work_per_thread);
    }
}

template <typename T, typename Config>
__host__ __device__ void collect_bucket(const T* data, const poracle* oracles_packed,
                                        const index* prefix_sum, T* out, index size, oracle bucket,
                                        index* atomic) {
    auto params = get_launch_parameters<T, Config>(size);
    kernels::collect_bucket<T, Config><<<params.block_count, params.block_size>>>(
            data, oracles_packed, prefix_sum, out, size, bucket, atomic, params.work_per_thread);
}

template <typename T, typename Config>
__host__ __device__ void collect_bucket_indirect(const T* data, const poracle* oracles_packed,
                                                 const index* prefix_sum, T* out, index size,
                                                 const oracle* bucket, index* atomic) {
    auto params = get_launch_parameters<T, Config>(size);
    kernels::collect_bucket_indirect<T, Config><<<params.block_count, params.block_size>>>(
            data, oracles_packed, prefix_sum, out, size, bucket, atomic, params.work_per_thread);
}

template <typename T, typename Config>
__host__ __device__ void collect_buckets(const T* data, const poracle* oracles_packed,
                                         const index* block_prefix_sum, const index* bucket_out_ranges,
                                         T* out, index size, mask* buckets, index* atomic) {
    auto params = get_launch_parameters<T, Config>(size);
    kernels::collect_buckets<T, Config><<<params.block_count, params.block_size>>>(
            data, oracles_packed, block_prefix_sum, bucket_out_ranges, out, size, buckets, atomic, params.work_per_thread);
}

template <typename T, typename Config>
__host__ __device__ void ssss_merged(
    const T* in,
    T* out,
    poracle* oracles,
    index offset,
    const index* ranks,
    index rank_offset,
    index rank_base,
    const kernels::ssss_multi_aux<T, Config>* aux_in,
    kernels::ssss_multi_aux<T, Config>* aux_outs,
    T* out_trees) {
    kernels::ssss_merged_kernel<T, Config><<<Config::searchtree::width, Config::algorithm::max_block_size>>>(
        in, out, oracles, offset, ranks, rank_offset, rank_base, aux_in, aux_outs, out_trees);
}

} // namespace gpu

#endif // SSSS_LAUNCHERS_CUH