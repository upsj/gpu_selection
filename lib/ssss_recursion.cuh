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
#ifndef SSSS_RECURSION_CUH
#define SSSS_RECURSION_CUH

#include <vector>

#include "ssss_build_searchtree.cuh"
#include "ssss_collect.cuh"
#include "ssss_count.cuh"
#include "ssss_launchers.cuh"
#include "ssss_reduce.cuh"
#include "utils_prefixsum.cuh"
#include "utils_basecase.cuh"

namespace gpu {
namespace kernels {

template <typename T>
__global__ void set_zero(T* in, index size) {
    if (threadIdx.x < size) {
        in[threadIdx.x] = 0;
    }
}

template <typename T, typename Config>
__global__ void sampleselect_tailcall(T* in, T* tmp, T* tree,
                                      index* count_tmp, T* out);

template <typename Config>
__global__ void sampleselect_findbucket(index* totalcounts, index rank,
                                        oracle* out_bucket,
                                        index* out_rank) {
    prefix_sum_select<Config::searchtree::height>(totalcounts, rank, out_bucket, out_rank);
}

template <typename T, typename Config>
__device__ void launch_sampleselect(T* in, T* tmp, T* tree,
                                    index* count_tmp, index size, index rank,
                                    T* out) {
    if (threadIdx.x != 0) {
        return;
    }

    if (size <= Config::basecase::size) {
        select_bitonic_basecase<T, Config><<<1, Config::basecase::launch_size>>>(in, size, rank, out);
        return;
    }

    // launch kernels:
    // sample and build searchtree
    gpu::build_searchtree<T, Config>(in, tree, size);

    // Usage of tmp storage:
    // | bucket_idx | rank_out | atomic | totalcounts... | oracles... | localcounts... |
    auto bucket_idx = (oracle*)count_tmp;
    auto rank_out = ((index*)bucket_idx) + 1;
    auto atomic = rank_out + 1;
    auto totalcounts = atomic + 1;
    auto oracles = (poracle*)(totalcounts + Config::searchtree::width);
    auto localcounts = (index*)(oracles + ceil_div(size, 4));

    if (!Config::algorithm::shared_memory) {
        *atomic = 0;
        set_zero<<<1, Config::searchtree::width>>>(totalcounts, Config::searchtree::width);
    }

    // count buckets
    gpu::count_buckets<T, Config>(in, tree, localcounts, totalcounts, oracles, size);
    sampleselect_findbucket<Config>
            <<<1, Config::searchtree::width / 2>>>(totalcounts, rank, bucket_idx, rank_out);
    gpu::collect_bucket_indirect<T, Config>(in, oracles, localcounts, tmp, size, bucket_idx,
                                            atomic);
    sampleselect_tailcall<T, Config><<<1, 1>>>(tmp, in, tree, count_tmp, out);
}

template <typename T, typename Config>
__global__ void sampleselect_tailcall(T* in, T* tmp, T* tree,
                                      index* count_tmp, T* out) {
    if (threadIdx.x != 0) {
        return;
    }
    auto bucket_idx_p = count_tmp;
    auto rank_out = bucket_idx_p + 1;
    auto atomic = rank_out + 1;
    auto totalcounts = atomic + 1;
    auto bucket_idx = *bucket_idx_p;

    auto size = totalcounts[bucket_idx];
    auto rank = *rank_out;
    auto leaves = tree + Config::searchtree::width - 1;
    if (!Config::algorithm::bucket_select && is_equality_bucket<T, Config>(leaves, bucket_idx)) {
        *out = leaves[bucket_idx];
    } else {
        launch_sampleselect<T, Config>(in, tmp, tree, count_tmp, size, rank, out);
    }
}

template <typename T, typename Config>
__global__ void sampleselect(T* in, T* tmp, T* tree,
                             index* count_tmp, index size, index rank,
                             T* out) {
    static_assert(Config::algorithm::write, "Recursive algorithm needs algorithm::write");
    launch_sampleselect<T, Config>(in, tmp, tree, count_tmp, size, rank, out);
}

} // namespace kernels

template <typename T, typename Config>
void sampleselect(T* in, T* tmp, T* tree, index* count_tmp, index size, index rank, T* out) {
    kernels::sampleselect<T, Config><<<1, 1>>>(in, tmp, tree, count_tmp, size, rank, out);
}

template <typename T, typename Config>
void sampleselect_host(T* in, T* tmp, T* tree, index* count_tmp, index size, index rank, T* out) {
    std::vector<index> host_count_tmp(3 + Config::searchtree::width);
    auto host_bucket_idx = (oracle*)host_count_tmp.data();
    auto host_rank_out = ((index*)host_bucket_idx) + 1;
    auto host_atomic = host_rank_out + 1;
    auto host_totalcounts = host_atomic + 1;
    std::vector<T> host_tree(Config::searchtree::size);
    T host_out{};
    auto host_leaves = host_tree.data() + Config::searchtree::width - 1;
    do {
        if (size <= Config::basecase::size) {
            kernels::select_bitonic_basecase<T, Config><<<1, Config::basecase::launch_size>>>(in, size, rank, out);
            return;
        }

        // launch kernels:
        // sample and build searchtree
        gpu::build_searchtree<T, Config>(in, tree, size);

        // Usage of tmp storage:
        // | bucket_idx | rank_out | atomic | totalcounts... | oracles... | localcounts... |
        auto bucket_idx = (oracle*)count_tmp;
        auto rank_out = ((index*)bucket_idx) + 1;
        auto atomic = rank_out + 1;
        auto totalcounts = atomic + 1;
        auto oracles = (poracle*)(totalcounts + Config::searchtree::width);
        auto localcounts = (index*)(oracles + ceil_div(size, 4));

        if (!Config::algorithm::shared_memory) {
            cudaMemset(atomic, 0, sizeof(*atomic));
            kernels::set_zero<<<1, Config::searchtree::width>>>(totalcounts, Config::searchtree::width);
        }

        // count buckets
        gpu::count_buckets<T, Config>(in, tree, localcounts, totalcounts, oracles, size);
        kernels::sampleselect_findbucket<Config>
                <<<1, Config::searchtree::width / 2>>>(totalcounts, rank, bucket_idx, rank_out);
        gpu::collect_bucket_indirect<T, Config>(in, oracles, localcounts, tmp, size, bucket_idx,
                                                atomic);
        // tailcall
        cudaMemcpy(host_count_tmp.data(), count_tmp, sizeof(index) * host_count_tmp.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_tree.data(), tree, sizeof(T) * host_tree.size(), cudaMemcpyDeviceToHost);
        
        if (!Config::algorithm::bucket_select && kernels::is_equality_bucket<T, Config>(host_leaves, *host_bucket_idx)) {
            host_out = host_leaves[*host_bucket_idx];
            cudaMemcpy(out, &host_out, sizeof(T), cudaMemcpyHostToDevice);
            return;
        } else {
            std::swap(in, tmp);
            rank = *host_rank_out;
            size = host_totalcounts[*host_bucket_idx];
        }
    } while(true);
}

template <typename Config>
index sampleselect_alloc_size(index size) {
    return ceil_div(sizeof(index) * (1   // bucket index
                                     + 1 // rank
                                     + 1 // atomic
                                     + (Config::algorithm::max_block_count + 1) *
                                               Config::searchtree::width) // totalcount + localcount
                            + sizeof(poracle) * ceil_div(size, 4),
                    sizeof(index)); // oracles
}

} // namespace gpu

#endif // SSSS_RECURSION_CUH