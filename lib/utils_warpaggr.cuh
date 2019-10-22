#ifndef UTILS_WARPAGGR_CUH
#define UTILS_WARPAGGR_CUH

#include "utils.cuh"

namespace gpu {
namespace kernels {

__device__ inline bool is_group_leader(mask amask) {
    return (__ffs(amask) - 1) == (threadIdx.x % warp_size);
}

__device__ inline index prefix_popc(mask amask, index shift) {
    mask prefix_mask = (1u << shift) - 1;
    return __popc(amask & prefix_mask);
}

__device__ inline index warp_aggr_atomic_count_mask(index* atomic, mask amask, mask cmask) {
    auto lane_idx = threadIdx.x % warp_size;
    index ofs{};
    if (lane_idx == 0) {
        ofs = atomicAdd(atomic, __popc(cmask));
    }
    ofs = shfl(amask, ofs, 0);
    auto local_ofs = prefix_popc(cmask, lane_idx);
    return ofs + local_ofs;
}

__device__ inline index warp_aggr_atomic_count_predicate(index* atomic, mask amask,
                                                         bool predicate) {
    auto mask = ballot(amask, predicate);
    return warp_aggr_atomic_count_mask(atomic, amask, mask);
}

} // namespace kernels
} // namespace gpu

#endif // UTILS_WARPAGGR_CUH