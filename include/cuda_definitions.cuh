#ifndef GPU_SELECTION_CUDA_DEFINITIONS_CUH
#define GPU_SELECTION_CUDA_DEFINITIONS_CUH

#include <cstdint>

namespace gpu {

using index = std::uint32_t;
using poracle = std::uint32_t;
using oracle = std::uint32_t;
using mask = std::uint32_t;

constexpr index warp_size_log2 = 5;
constexpr index warp_size = 1 << warp_size_log2;
constexpr index max_block_size_log2 = 10;
constexpr index max_block_size = 1 << max_block_size_log2;

} // namespace gpu

#endif // GPU_SELECTION_CUDA_DEFINITIONS_CUH
