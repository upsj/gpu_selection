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