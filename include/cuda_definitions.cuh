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
