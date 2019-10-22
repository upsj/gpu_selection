#ifndef UTILS_SAMPLING_CUH
#define UTILS_SAMPLING_CUH

#include "utils.cuh"

namespace gpu {
namespace kernels {

__device__ inline index uniform_pick_idx(index idx, index samplesize, index size) {
    auto stride = size / samplesize;
    if (stride == 0) {
        return idx * size / samplesize;
    } else {
        return idx * stride + stride / 2;
    }
}

__device__ inline index random_pick_idx(index idx, index samplesize, index size) {
    // TODO
    return uniform_pick_idx(idx, samplesize, size);
}

} // namespace kernels
} // namespace gpu

#endif // UTILS_SAMPLING_CUH