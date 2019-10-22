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
#ifndef SSSS_COUNT_CUH
#define SSSS_COUNT_CUH

#include "ssss_reduce.cuh"
#include "utils_bytestorage.cuh"
#include "utils_warpaggr.cuh"
#include "utils_work.cuh"

namespace gpu {
namespace kernels {

template <typename T, typename Config>
__device__ oracle searchtree_traversal(const T* searchtree, T el, mask amask, mask& equal_mask, T min_split, T max_split) {
    index i = 0;
    equal_mask = amask;
    if (Config::algorithm::bucket_select) {
        auto maxbucket = Config::searchtree::width - 1;
        auto floatbucket = (el - min_split) / (max_split - min_split) * Config::searchtree::width - T(0.5);
        floatbucket = floatbucket > maxbucket ? maxbucket : floatbucket;
        floatbucket = floatbucket < 0 ? 0 : floatbucket;
        auto bucket = oracle(floatbucket);
        for (index lvl = 0; lvl < Config::searchtree::height; ++lvl) {
            auto bit = (bucket >> lvl) & 1;
            equal_mask &= ballot(amask, bit) ^ (bit - 1);
        }
        return oracle(floatbucket);
    } else {
        auto root_splitter = searchtree[0];
        bool next_smaller = el < root_splitter;
        for (index lvl = 0; lvl < Config::searchtree::height; ++lvl) {
            // compute next node index
            bool smaller = next_smaller;
            i = 2 * i + 2 - smaller;
            next_smaller = el < searchtree[i];
            // update equality mask
            auto local_mask = ballot(amask, smaller) ^ (smaller - 1);
            equal_mask &= local_mask;
        }
        // return leaf rank
        return i - (Config::searchtree::width - 1);
    }
}

template <typename T, typename Config, typename BucketCallback>
__device__ __forceinline__ void ssss_impl(const T* in, const T* tree,
                                          index size, index workcount, BucketCallback bucket_cb) {
    __shared__ T local_tree[Config::algorithm::bucket_select ? 2 : Config::searchtree::size];
    // Load searchtree into shared memory
    if (Config::algorithm::bucket_select) {
        if (threadIdx.x == 0) {
            local_tree[0] = tree[Config::searchtree::width];
            local_tree[1] = tree[Config::searchtree::size - 1];
        }
    } else {
        blockwise_work_local(Config::searchtree::size, [&](index i) { local_tree[i] = tree[i]; });
    }
    __syncthreads();
    // only for bucket select
    auto min_split = local_tree[0];
    auto max_split = local_tree[1];

    // Determine the bucket and equality mask for every entry
    blockwise_work<Config>(workcount, size, [&](index idx, mask amask) {
        mask equal_mask{};
        auto bucket_idx = searchtree_traversal<T, Config>(local_tree, in[idx], amask, equal_mask, min_split, max_split);
        bucket_cb(idx, bucket_idx, amask, equal_mask);
    });
}

template <typename T, typename Config>
__global__ void count_buckets(const T* in, const T* tree,
                              index* counts, poracle* oracles, index size,
                              index workcount) {
    __shared__ index local_counts[Config::searchtree::width];
    // Initialize shared-memory counts
    if (Config::algorithm::shared_memory) {
        blockwise_work_local(Config::searchtree::width, [&](index i) { local_counts[i] = 0; });
        __syncthreads();
    }
    // Traverse searchtree for every entry
    ssss_impl<T, Config>(
            in, tree, size, workcount, [&](index idx, oracle bucket, mask amask, mask mask) {
                static_assert(!Config::algorithm::write || Config::searchtree::height <= 8,
                              "can't pack bucket idx into byte");
                // Store oracles
                if (Config::algorithm::write) {
                    store_packed_bytes(oracles, amask, bucket, idx);
                }
                // Increment bucket count
                index add = Config::algorithm::warp_aggr ? __popc(mask) : 1;
                if (!Config::algorithm::warp_aggr || is_group_leader(mask)) {
                    if (Config::algorithm::shared_memory) {
                        atomicAdd(&local_counts[bucket], add);
                    } else {
                        atomicAdd(&counts[bucket], add);
                    }
                }
            });
    // Write shared-memory counts to global memory
    if (Config::algorithm::shared_memory) {
        __syncthreads();
        // store the local counts grouped by block idx
        blockwise_work_local(Config::searchtree::width, [&](oracle bucket) {
            auto idx = partial_sum_idx(blockIdx.x, bucket, gridDim.x, Config::searchtree::width);
            counts[idx] = local_counts[bucket];
        });
    }
}

} // namespace kernels
} // namespace gpu

#endif // SSSS_COUNT_CUH
