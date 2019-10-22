#ifndef SSSS_COLLECT_MULTI_CUH
#define SSSS_COLLECT_MULTI_CUH

#include "ssss_reduce.cuh"
#include "utils_bytestorage.cuh"
#include "utils_warpaggr.cuh"
#include "utils_work.cuh"
#include "utils_mask.cuh"
#include "utils_search.cuh"

namespace gpu {
namespace kernels {

template <typename T, typename Config>
__global__ void
collect_buckets(const T* data, const poracle* oracles_packed,
    const index* block_prefix_sum, const index* bucket_out_ranges,
    T* out, index size, const mask* buckets,
    index* atomic, index workcount) {
        // initialize mask cache in shared memory
        constexpr auto mask_size = ceil_div(Config::searchtree::width, sizeof(mask) * 8);
        __shared__ mask shared_mask[mask_size];
        static_assert(mask_size < 32, "mask too big, just a misconfiguration failsafe");
        if (threadIdx.x < mask_size) {
            shared_mask[threadIdx.x] = buckets[threadIdx.x];
        }

        // initialize block-local count from prefix sum
        __shared__ index count[Config::searchtree::width];
        if (Config::algorithm::shared_memory) {
            blockwise_work_local(Config::searchtree::width, [&](index bucket) {
                auto base_idx = partial_sum_idx(blockIdx.x, bucket, gridDim.x, Config::searchtree::width);
                count[bucket] = bucket_out_ranges[bucket] + block_prefix_sum[base_idx];
            });
        } else {
            blockwise_work_local(Config::searchtree::width, [&](index bucket) {
                count[bucket] = bucket_out_ranges[bucket];
            });
        }

        __syncthreads();

        // extract elements from the specified bucket
        blockwise_work<Config>(workcount, size, [&](index idx, mask amask) {
            // load bucket index
            auto bucket = load_packed_bytes(oracles_packed, amask, idx);
            // determine target location
            index ofs{};
            if (check_mask(bucket, shared_mask)) {
                if (Config::algorithm::shared_memory) {
                    ofs = atomicAdd(&count[bucket], 1);
                } else {
                    ofs = atomicAdd(&atomic[bucket], 1) + count[bucket];
                }
                // store element
                out[ofs] = data[idx];
            }
        });
}

} // namespace kernels
} // namespace gpu

#endif // SSSS_COLLECT_MULTI_CUH