#ifndef SSSS_MERGED_CUH
#define SSSS_MERGED_CUH

#include "ssss_count.cuh"
#include "ssss_collect_multi.cuh"
#include "ssss_build_searchtree.cuh"
#include "ssss_merged_memory.cuh"
#include "ssss_count.cuh"
#include "utils_work.cuh"
#include "utils_prefixsum.cuh"

namespace gpu {
namespace kernels {

template <typename T, typename Config, typename EqCallback>
__device__ void ssss_merged_impl(
    const T* in,
    T* tmp,
    index* count_tmp,
    index offset,
    index size,
    const ssss_multi_aux<T, Config>* aux_in,
    ssss_multi_aux<T, Config>* aux_out,
    const index* ranks,
    index rank_offset,
    index rank_count,
    index rank_base,
    EqCallback equality_bucket_callback) {
    __shared__ T tree[Config::searchtree::size];
    __syncthreads();
    auto bucket = threadIdx.x;
    auto workcount = ceil_div(size, blockDim.x);
    auto oracles = (poracle*)(count_tmp + offset);
    // build searchtree
    build_searchtree_shared<T, Config>(in + offset, size, tree);
    // Initialize shared-memory counts
    __shared__ index counts[Config::searchtree::width];
    if (bucket < Config::searchtree::width) {
        counts[bucket] = 0;
    }
    __syncthreads();
    // count buckets
    T min_split{};
    T max_split{};
    if (Config::algorithm::bucket_select) {
        min_split = tree[Config::searchtree::width];
        max_split = tree[Config::searchtree::size - 1];
    }
    blockwise_work_local_large<Config>(workcount, size, [&](index idx, mask amask) {
        mask equal_mask{};
        auto bucket = searchtree_traversal<T, Config>(tree, in[idx + offset], amask, equal_mask, min_split, max_split);
        static_assert(Config::searchtree::height <= 8, "can't pack bucket idx into byte");
        // Store oracles
        store_packed_bytes(oracles, amask, bucket, idx);
        // Increment bucket count
        index add = Config::algorithm::warp_aggr ? __popc(equal_mask) : 1;
        if (!Config::algorithm::warp_aggr || is_group_leader(equal_mask)) {
            atomicAdd(&counts[bucket], add);
        }
    });
    __syncthreads();
    // compute bucket prefix sum and build mask and masked prefix sum
    constexpr index mask_size = ceil_div(Config::searchtree::width, sizeof(mask) * 8);
    __shared__ mask new_bucket_mask[mask_size];
    __shared__ index prefixsum[Config::searchtree::width + 1];
    __shared__ index new_rank_ranges[Config::searchtree::width + 1];
    if (bucket < Config::searchtree::width) {
        prefixsum[bucket] = counts[bucket];
    }
    __syncthreads();
    small_prefix_sum_sentinel<Config::searchtree::height>(prefixsum);
    __syncthreads();
    index rank_local_base{};
    if (bucket < Config::searchtree::width) {
        rank_local_base = prefixsum[bucket];
        compute_bucket_mask_impl(ranks + rank_offset, rank_count, rank_base, prefixsum, new_bucket_mask, new_rank_ranges);
    }
    __syncthreads();
    if (bucket < Config::searchtree::width) {
        // assert is_unmasked == check_mask(bucket, new_bucket_mask);
        if (!Config::algorithm::bucket_select) {
            auto is_unmasked = new_rank_ranges[bucket + 1] > new_rank_ranges[bucket];
            // compute equality bucket mask
            auto leaves = tree + (Config::searchtree::width - 1);
            auto is_unmasked_equality = is_unmasked && is_equality_bucket<T, Config>(leaves, bucket);
            if (is_unmasked_equality) {
                // launch set_value
                if (!equality_bucket_callback(rank_offset + new_rank_ranges[bucket], leaves[bucket], new_rank_ranges[bucket + 1] - new_rank_ranges[bucket])) {
                    // false: treat as normal bucket, don't remove from mask
                    is_unmasked_equality = false;
                }
            }
            mask equality{};
            // convoluted way to avoid warnings about overflowing shifts
            constexpr auto partial_mask = warp_size > Config::searchtree::width ? (mask{1} << (Config::searchtree::width % warp_size)) - 1 : full_mask;
            equality = ballot(partial_mask, is_unmasked_equality);
            // remove equality buckets from mask
            if (bucket % warp_size == 0) {
                new_bucket_mask[bucket / warp_size] &= ~equality;
            }
        }
        prefixsum[bucket] = counts[bucket];
    }
    __syncthreads();
    masked_prefix_sum_sentinel<Config::searchtree::height>(prefixsum, new_bucket_mask);
    __syncthreads();
    // extract elements from the specified buckets
    blockwise_work_local_large<Config>(workcount, size, [&](index idx, mask amask) {
        // load bucket index
        auto bucket = load_packed_bytes(oracles, amask, idx);
        if (check_mask(bucket, new_bucket_mask)) {
            auto ofs = atomicAdd(&prefixsum[bucket], 1);
            // store element
            tmp[ofs + offset] = in[idx + offset];
        }
    });
    __syncthreads();
    // store results in aux
    if (bucket < Config::searchtree::width) {
        aux_out->stage2.bucket_prefixsum[bucket] = rank_local_base;
        // we used them as counters, thus need to move one bucket back
        aux_out->stage2.bucket_masked_prefixsum[bucket] = bucket == 0 ? 0 : prefixsum[bucket - 1];
        aux_out->stage2.rank_ranges[bucket] = new_rank_ranges[bucket];
        if (bucket == Config::searchtree::width - 1) {
            aux_out->stage2.bucket_prefixsum[bucket + 1] = rank_local_base + counts[bucket];
            aux_out->stage2.bucket_masked_prefixsum[bucket + 1] = prefixsum[bucket];
            aux_out->stage2.rank_ranges[bucket + 1] = new_rank_ranges[bucket + 1];
        }
        if (!Config::algorithm::shared_memory) {
            aux_out->stage2.atomic[bucket] = 0;
        }
        if (bucket < mask_size) {
            aux_out->stage2.bucket_mask[bucket] = new_bucket_mask[bucket];
        }
    }
}

template <typename T, typename Config>
__global__ void ssss_merged_kernel(
    const T* in,
    T* out,
    poracle* oracles,
    index base_offset,
    const index* ranks,
    index rank_base_offset,
    index rank_base_base,
    const ssss_multi_aux<T, Config>* aux_in,
    ssss_multi_aux<T, Config>* aux_out,
    T* trees_out) {
    __shared__ index offset;
    __shared__ index size;
    __shared__ index rank_offset;
    __shared__ index rank_count;
    __shared__ index rank_base;
        
    auto bucket = blockIdx.x;
    if (threadIdx.x == 0) {
        offset = aux_in->stage2.bucket_masked_prefixsum[bucket] + base_offset;
        size = aux_in->stage2.bucket_masked_prefixsum[bucket + 1] - (offset - base_offset);
        rank_offset = aux_in->stage2.rank_ranges[bucket] + rank_base_offset;
        rank_count = aux_in->stage2.rank_ranges[bucket + 1] - (rank_offset - rank_base_offset);
        rank_base = aux_in->stage2.bucket_prefixsum[bucket] + rank_base_base;
    }
    __syncthreads();
    if (size == 0) {
        return;
    }
    // for testing purposes
    build_searchtree_shared<T, Config>(in + offset, size, trees_out + bucket * Config::searchtree::size);
    auto aux_new = aux_out + bucket;
    ssss_merged_impl(in, out, oracles, offset, size, aux_in, aux_new, ranks, rank_offset, rank_count, rank_base, [](index, T, index) {
        return false;
    });
}

} // namespace kernels
} // namespace gpu

#endif // SSSS_MERGED_CUH