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
#ifndef LAUNCHER_FWD_CUH
#define LAUNCHER_FWD_CUH

#include "cuda_definitions.cuh"
#include "ssss_merged_memory.cuh"

namespace gpu {

namespace kernels {
template <typename T, typename Config>
struct ssss_multi_aux;

template <typename T, typename Config>
__global__ void partition(const T* in, T* out, index* atomic, index size, T pivot, index workcount);

template <typename T, typename Config>
__global__ void partition_count(const T* in, index* counts, index size, T pivot, index workcount);

template <typename T, typename Config>
__global__ void partition_distr(const T* in, T* out, const index* counts, index size, T pivot, index workcount);

template <typename Config>
__global__ void reduce_counts(const index* in, index* out, index num_blocks);

template <typename Config>
__global__ void prefix_sum_counts(index* in, index* out, index num_blocks);

template <typename Config>
__global__ void partition_prefixsum(index* counts, index block_count);

template <typename T, typename Config>
__global__ void count_buckets(const T* in, const T* tree, index* counts, poracle* oracles, index size, index workcount);

template<index size_log2>
__device__ void masked_prefix_sum(index* counts, const mask* m);
}
    
template <typename T, typename Config>
__host__ __device__ void build_searchtree(const T* in, T* out, index size);
    
    template <typename T, typename Config>
__host__ __device__ void count_buckets(const T* in, const T* tree, index* localcounts,
                                        index* counts, poracle* oracles, index size);
    
template <typename T, typename Config>
__host__ __device__ void collect_bucket(const T* data, const poracle* oracles_packed,
                                        const index* prefix_sum, T* out, index size, oracle bucket,
                                        index* atomic);
    
template <typename T, typename Config>
__host__ __device__ void collect_bucket_indirect(const T* data, const poracle* oracles_packed,
                                                 const index* prefix_sum, T* out, index size,
                                                 const oracle* bucket, index* atomic);
    
template <typename T, typename Config>
__host__ __device__ void collect_buckets(const T* data, const poracle* oracles_packed,
                                         const index* block_prefix_sum, const index* bucket_out_ranges,
                                         T* out, index size, mask* buckets, index* atomic);
    
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
    T* out_trees);

template <typename T, typename Config>
void sampleselect(T* in, T* tmp, T* tree, index* count_tmp, index size, index rank, T* out);

template <typename T, typename Config>
void sampleselect_host(T* in, T* tmp, T* tree, index* count_tmp, index size, index rank, T* out);

template <typename T, typename Config>
void sampleselect_multi(T* in, T* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, T* out);

template <typename T, typename Config>
__device__ __host__ void partition(const T* in, T* out, index* counts, index size, T pivot);

template <typename T, typename Config>
void quickselect_multi(T* in, T* tmp, index* count_tmp, index size, const index* ranks, index rank_count, T* out);

template <typename T, typename Config>
void quickselect(T* in, T* tmp, index* count_tmp, index size, index rank, T* out);

template <typename T, typename Config>
__host__ __device__ launch_parameters get_launch_parameters(index size);
    
} // namespace gpu

#endif // LAUNCHER_FWD_CUH