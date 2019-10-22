#ifndef SSSS_REDUCE_CUH
#define SSSS_REDUCE_CUH

#include "utils.cuh"

namespace gpu {
namespace kernels {

__device__ inline index partial_sum_idx(index block, oracle bucket, int num_blocks,
                                        int num_buckets) {
    return bucket + block * num_buckets;
}

template <typename Config>
__global__ void reduce_counts(const index* in, index* out,
                              index num_blocks) {
    index bucket = blockIdx.x * blockDim.x + threadIdx.x;
    if (bucket < Config::searchtree::width) {
        index sum{};
        for (index block = 0; block < num_blocks; ++block) {
            sum += in[partial_sum_idx(block, bucket, num_blocks, Config::searchtree::width)];
        }
        out[bucket] = sum;
    }
}

template <typename Config>
__global__ void prefix_sum_counts(index* in, index* out,
                                  index num_blocks) {
    index bucket = blockIdx.x * blockDim.x + threadIdx.x;
    if (bucket < Config::searchtree::width) {
        index sum{};
        for (index block = 0; block < num_blocks; ++block) {
            auto idx = partial_sum_idx(block, bucket, num_blocks, Config::searchtree::width);
            auto tmp = in[idx];
            in[idx] = sum;
            sum += tmp;
        }
        out[bucket] = sum;
    }
}

} // namespace kernels
} // namespace gpu

#endif // SSSS_REDUCE_CUH
