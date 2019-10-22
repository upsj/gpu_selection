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
#ifndef UTILS_SEARCH_CUH
#define UTILS_SEARCH_CUH

#include "utils.cuh"

namespace gpu {
namespace kernels {

// finds the smallest element >= needle in sorted haystack
inline __device__ index binary_search(const index* haystack, index haystack_size, index needle) {
    auto range_begin = 0;
    auto range_size = haystack_size;
    while (range_size > 0) {
        auto half_size = range_size / 2;
        auto middle = range_begin + half_size;
        // if the middle is already a candidate: discard everything right of it
        auto go_left = haystack[middle] >= needle;
        range_begin = go_left ? range_begin : middle + 1;
        range_size = go_left ? half_size : (range_size - half_size - 1);
    }
    return range_begin;
}

} // namespace kernels
} // namespace gpu

#endif // UTILS_SEARCH_CUH