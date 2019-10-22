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
#ifndef UTILS_BYTESTORAGE_CUH
#define UTILS_BYTESTORAGE_CUH

#include "utils.cuh"

namespace gpu {
namespace kernels {

__device__ inline void store_packed_bytes(poracle* output, mask amask, oracle byte, index idx) {
    // pack 4 consecutive bytes into a dword
    poracle result = byte;
    // ------00 -> ----1100
    result |= shfl_xor(amask, result, 1, 4) << 8;
    // ----1100 -> 33221100
    result |= shfl_xor(amask, result, 2, 4) << 16;
    if (idx % 4 == 0) {
        output[idx / 4] = result;
    }
}

__device__ inline oracle load_packed_bytes(const poracle* input, mask amask, index idx) {
    auto char_idx = idx % 4;
    auto pack_idx = idx / 4;
    poracle packed{};
    // first thread in quartet loads the data
    if (char_idx == 0) {
        packed = input[pack_idx];
    }
    // distribute the data onto all threads
    packed = shfl(amask, packed, (pack_idx * 4) % warp_size, 4);
    packed >>= char_idx * 8;
    packed &= 0xff;
    return packed;
}

} // namespace kernels
} // namespace gpu

#endif // UTILS_BYTESTORAGE_CUH