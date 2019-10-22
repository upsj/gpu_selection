#ifndef SSSS_MERGED_MEMORY_CUH
#define SSSS_MERGED_MEMORY_CUH

#include <cuda_definitions.cuh>
#include "utils.cuh"

namespace gpu {
namespace kernels {

template <typename T, typename Config>
struct ssss_multi_aux {
    constexpr static auto mask_size = ceil_div(Config::searchtree::width, sizeof(mask) * 8);
    union {
        struct {
            T tree[Config::searchtree::size];
            index bucket_counts[Config::searchtree::width + 1];
        } stage1;
        struct {
            mask bucket_mask[mask_size];
            index bucket_prefixsum[Config::searchtree::width + 1];
            index bucket_masked_prefixsum[Config::searchtree::width + 1];
            index rank_ranges[Config::searchtree::width + 1];
            index atomic[Config::algorithm::shared_memory ? 1 : Config::searchtree::width];
        } stage2;
    };
};

} // namespace kernels
} // namespace gpu

#endif