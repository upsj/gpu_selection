#ifndef SSSS_RECURSION_MULTI_CUH
#define SSSS_RECURSION_MULTI_CUH

#include "ssss_recursion.cuh"
#include "ssss_merged.cuh"
#include "utils_prefixsum.cuh"

namespace gpu {
namespace kernels {

template <typename T>
__global__ void set_value(T* in, T val, index size) {
    auto idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < size) {
        in[idx] = val;
    }
}

template <typename T, typename Config>
__global__ void sampleselect_tailcall_multi(T* in, T* tmp, index offset, index* tmp_storage, ssss_multi_aux<T, Config>* aux, ssss_multi_aux<T, Config>* aux_storage,
                                            index* aux_atomic, const index* ranks, index rank_offset, index rank_base, T* out);

template <typename T, typename Config>
__device__ void sampleselect_tailcall_multi_impl(T* in, T* tmp, index offset, index* tmp_storage, ssss_multi_aux<T, Config>* aux, ssss_multi_aux<T, Config>* aux_storage,
                                                 index* aux_atomic, const index* ranks, index rank_offset, index rank_base, T* out);
                                            
template <typename T, typename Config>
__global__ void ssss_merged_recursive(
    T* in,
    T* tmp,
    index* count_tmp,
    index base_offset,
    ssss_multi_aux<T, Config>* aux, 
    ssss_multi_aux<T, Config>* aux_storage,
    index* aux_atomic,
    const index* ranks,
    index rank_base_offset, index rank_base_base,
    T* out) {
    __shared__ index offset;
    __shared__ index size;
    __shared__ index rank_offset;
    __shared__ index rank_count;
    __shared__ index rank_base;
    __shared__ index aux_idx;
    
    index bucket{};
    if (threadIdx.x < warp_size) {
        bucket = select_mask<Config::searchtree::width>(blockIdx.x, aux->stage2.bucket_mask);
    }
    if (threadIdx.x == 0) {
        offset = aux->stage2.bucket_masked_prefixsum[bucket] + base_offset;
        size = aux->stage2.bucket_masked_prefixsum[bucket + 1] - (offset - base_offset);
        rank_offset = aux->stage2.rank_ranges[bucket] + rank_base_offset;
        rank_count = aux->stage2.rank_ranges[bucket + 1] - (rank_offset - rank_base_offset);
        rank_base = aux->stage2.bucket_prefixsum[bucket] + rank_base_base;
        if (size > Config::basecase::size) {
            aux_idx = atomicAdd(aux_atomic, 1);
        }
    }
    __syncthreads();
    if (size == 0) {
        return;
    }
    if (size <= Config::basecase::size) {
        if (threadIdx.x >= Config::basecase::launch_size) {
            return;
        }
        select_bitonic_multiple_basecase_impl<T, Config>(in + offset, size, ranks + rank_offset, rank_count, rank_base, out + rank_offset);
        return;
    }
    auto aux_new = aux_storage + aux_idx;
    ssss_merged_impl(in, tmp, count_tmp, offset, size, aux, aux_new, ranks, rank_offset, rank_count, rank_base, [&](index out_offset, T val, index size) {
        auto blocksize = size > max_block_size ? max_block_size : size;
        auto blockcount = ceil_div(size, blocksize);
        set_value<<<blockcount, blocksize>>>(out + out_offset, val, size);
        return true;
    });
    __syncthreads();
    if (threadIdx.x < Config::searchtree::width) {
        sampleselect_tailcall_multi_impl<T, Config>(tmp, in, offset, count_tmp, aux_new, aux_storage, aux_atomic, ranks, rank_offset, rank_base, out);
    }
}

// launch with searchtree::width threads in a single block:
// computes full and masked prefixsum as well as rank ranges and bucket mask
// treats equality buckets
template <typename T, typename Config>
__global__ void prepare_collect(const index* ranks, index rank_count, index rank_base, ssss_multi_aux<T, Config>* aux, T* out) {
    auto bucket = threadIdx.x;
    __shared__ index shared_prefix_sum[Config::searchtree::width + 1];
    __shared__ index shared_rank_ranges[Config::searchtree::width + 1];
    // STAGE 1
    // compute full prefix sum
    {
        // load to shared memory
        shared_prefix_sum[bucket] = aux->stage1.bucket_counts[bucket];
        __syncthreads();
        // compute prefix sum
        small_prefix_sum_sentinel<Config::searchtree::height>(shared_prefix_sum);
        __syncthreads();
    }
    // compute bucket mask and rank ranges
    constexpr auto mask_size = ssss_multi_aux<T, Config>::mask_size;
    __shared__ mask shared_mask[mask_size];
    compute_bucket_mask_impl(ranks, rank_count, rank_base, shared_prefix_sum, shared_mask, shared_rank_ranges);
    __syncthreads();
    // remove equality buckets from mask and launch corresponding set_value kernels
    if (!Config::algorithm::bucket_select) {
        // compute equality bucket mask
        auto leaves = aux->stage1.tree + (Config::searchtree::width - 1);
        auto is_unmasked_equality = check_mask(bucket, shared_mask) && is_equality_bucket<T, Config>(leaves, bucket);
        auto equality = ballot(full_mask, is_unmasked_equality);
        if (is_unmasked_equality) {
            // launch set_value
            auto begin = shared_rank_ranges[bucket];
            auto end = shared_rank_ranges[bucket + 1];
            auto size = end - begin;
            auto blocksize = size > max_block_size ? max_block_size : size;
            auto blockcount = ceil_div(size, blocksize);
            set_value<<<blockcount, blocksize>>>(out + begin, leaves[bucket], size);
        }
        // remove equality buckets from mask
        if (bucket % warp_size == 0) {
            shared_mask[bucket / warp_size] &= ~equality;
        }
        __syncthreads();
    }
    // STAGE 2
    // compute masked bucket prefixsum
    {
        // store mask to global memory
        if (bucket < mask_size) {
            aux->stage2.bucket_mask[bucket] = shared_mask[bucket];
        }
        // store full prefixsum in global memory
        aux->stage2.bucket_prefixsum[bucket] = shared_prefix_sum[bucket];
        if (bucket == Config::searchtree::width - 1) {
            aux->stage2.bucket_prefixsum[bucket + 1] = shared_prefix_sum[bucket + 1];
        }
        // again load counts to shared memory, this time masked
        auto size = shared_prefix_sum[bucket + 1] - shared_prefix_sum[bucket];
        __syncthreads();
        shared_prefix_sum[bucket] = check_mask(bucket, shared_mask) ? size : 0;
        __syncthreads();
        // compute prefix sum
        small_prefix_sum_sentinel<Config::searchtree::height>(shared_prefix_sum);
        __syncthreads();
        // store prefix sum in global memory
        aux->stage2.bucket_masked_prefixsum[bucket] = shared_prefix_sum[bucket];
        if (bucket == Config::searchtree::width - 1) {
            aux->stage2.bucket_masked_prefixsum[bucket + 1] = shared_prefix_sum[bucket + 1];
        }
    }
    // store rank ranges to global memory
    {
        aux->stage2.rank_ranges[bucket] = shared_rank_ranges[bucket];
        if (bucket == Config::searchtree::width - 1) {
            aux->stage2.rank_ranges[bucket + 1] = shared_rank_ranges[bucket + 1];
        }
    }
}

template <typename T, typename Config>
__device__ void launch_sampleselect_multi(T* in, T* tmp, index offset, index size,
                                          const index* ranks, index rank_offset, index rank_count, index rank_base, index* count_tmp,
                                          ssss_multi_aux<T, Config>* aux_storage, index* aux_atomic, T* out) {
    if (size <= Config::basecase::size) {
        select_bitonic_multiple_basecase<T, Config><<<1, Config::basecase::launch_size>>>(in + offset, size, ranks + rank_offset, rank_count, rank_base, out + rank_offset);
        return;
    }
    auto aux_idx = atomicAdd(aux_atomic, 1);
    auto aux = aux_storage + aux_idx;

    auto oracles = (poracle*)(count_tmp + offset);
    auto localcounts = (index*)(oracles + ceil_div(size, sizeof(poracle)));

    // launch kernels:
    // sample and build searchtree
    gpu::build_searchtree<T, Config>(in + offset, aux->stage1.tree, size);
    if (!Config::algorithm::shared_memory) {
        set_zero<<<1, Config::searchtree::width>>>(aux->stage1.bucket_counts, Config::searchtree::width);
    }

    // count buckets
    gpu::count_buckets<T, Config>(in + offset, aux->stage1.tree, localcounts, aux->stage1.bucket_counts, oracles, size);
    prepare_collect<T, Config><<<1, Config::searchtree::width>>>(ranks + rank_offset, rank_count, rank_base, aux, out + rank_offset);
    if (!Config::algorithm::shared_memory) {
        set_zero<<<1, Config::searchtree::width>>>(aux->stage2.atomic, Config::searchtree::width);
    }
    gpu::collect_buckets<T, Config>(in + offset, oracles, localcounts, aux->stage2.bucket_masked_prefixsum, tmp + offset, size, aux->stage2.bucket_mask, aux->stage2.atomic);
    sampleselect_tailcall_multi<T, Config><<<1, Config::searchtree::width>>>(tmp, in, offset, count_tmp, aux, aux_storage, aux_atomic, ranks, rank_offset, rank_base, out);
}

template <typename T, typename Config>
__device__ void sampleselect_tailcall_multi_impl(T* in, T* tmp, index offset, index* tmp_storage, ssss_multi_aux<T, Config>* aux, ssss_multi_aux<T, Config>* aux_storage,
                                                 index* aux_atomic, const index* ranks, index rank_offset, index rank_base, T* out) {
    auto bucket = threadIdx.x;
    __shared__ index subcall_count;
    static_assert(ssss_multi_aux<T, Config>::mask_size < warp_size, "insufficient synchronization");
    if (threadIdx.x == 0) {
        subcall_count = 0;
    }
    __syncwarp();
    if (threadIdx.x < ssss_multi_aux<T, Config>::mask_size) {
        auto mask_block = aux->stage2.bucket_mask[bucket];
        atomicAdd(&subcall_count, __popc(mask_block));
    }
    __syncthreads();
    if (subcall_count > Config::algorithm::merged_limit) {
        if (threadIdx.x == 0) {
            ssss_merged_recursive<T, Config><<<subcall_count, Config::algorithm::max_block_size>>>(in, tmp, tmp_storage, offset, aux, aux_storage, aux_atomic, ranks, rank_offset, rank_base, out);
        }
    } else {
        auto bucket_begin = aux->stage2.bucket_masked_prefixsum[bucket];
        auto bucket_end = aux->stage2.bucket_masked_prefixsum[bucket + 1];
        auto bucket_size = bucket_end - bucket_begin;
        auto rank_begin = aux->stage2.rank_ranges[bucket];
        auto rank_end = aux->stage2.rank_ranges[bucket + 1];
        auto rank_count_new = rank_end - rank_begin;
        auto rank_base_new = rank_base + aux->stage2.bucket_prefixsum[bucket];
        if (check_mask(bucket, aux->stage2.bucket_mask)) {
            launch_sampleselect_multi<T, Config>(in, tmp, offset + bucket_begin, bucket_size, ranks, rank_offset + rank_begin, rank_count_new, rank_base_new, tmp_storage, aux_storage, aux_atomic, out);
        }
    }
}

template <typename T, typename Config>
__global__ void sampleselect_tailcall_multi(T* in, T* tmp, index offset, index* tmp_storage, ssss_multi_aux<T, Config>* aux, ssss_multi_aux<T, Config>* aux_storage,
                                            index* aux_atomic, const index* ranks, index rank_offset, index rank_base, T* out) {
    sampleselect_tailcall_multi_impl<T, Config>(in, tmp, offset, tmp_storage, aux, aux_storage, aux_atomic, ranks, rank_offset, rank_base, out);
}


template <typename T, typename Config>
__global__ void sampleselect_multi(T* in, T* tmp, index size, const index* ranks,
                                   index rank_count, index* tmp_storage, ssss_multi_aux<T, Config>* aux_storage, index* aux_atomic, T* out) {
    static_assert(Config::algorithm::write, "Recursive algorithm needs algorithm::write");
    launch_sampleselect_multi<T, Config>(in, tmp, 0, size, ranks, 0, rank_count, 0, tmp_storage, aux_storage, aux_atomic, out);
}

} // namespace kernels

template <typename T, typename Config>
void sampleselect_multi(T* in, T* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, T* out) {
    kernels::sampleselect_multi<T, Config><<<1, 1>>>(in, tmp, size, ranks, rank_count, tmp_storage, (kernels::ssss_multi_aux<T, Config>*)aux_storage, aux_atomic, out);
}

template <typename Config>
index sampleselect_alloc_size_multi(index size) {
    return std::max(ceil_div(sizeof(index) * (1   // bucket index
                                     + 1 // rank
                                     + 1 // atomic
                                     + (Config::algorithm::max_block_count + 1) *
                                               Config::searchtree::width) // totalcount + localcount
                            + sizeof(poracle) * ceil_div(size, 4), // oracles
                    sizeof(index)), size); // rough upper bound for memory usage
}

} // namespace gpu

#endif // SSSS_RECURSION_MULTI_CUH