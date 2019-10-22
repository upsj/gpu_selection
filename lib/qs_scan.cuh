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
#ifndef QS_SCAN_CUH
#define QS_SCAN_CUH

#include "utils_warpaggr.cuh"
#include "utils_work.cuh"

namespace gpu {
namespace kernels {

template <typename T, typename Config, typename Callback>
__device__ void partition_impl(const T* in, index size, T pivot, index workcount,
                               Callback callback) {
    blockwise_work<Config>(workcount, size, [&](index idx, mask amask) {
        auto el = in[idx];
        bool left = el < pivot;
        auto lmask = ballot(amask, left);
        auto rmask = lmask ^ amask;

        callback(el, left, amask, lmask, rmask);
    });
}

template <typename T, typename Config>
__global__ void partition(const T* in, T* out, index* atomic,
                          index size, T pivot, index workcount) {
    partition_impl<T, Config>(in, size, pivot, workcount,
                              [&](T el, bool l, mask amask, mask lm, mask rm) {
                                  auto lofs = warp_aggr_atomic_count_mask(atomic, amask, lm);
                                  auto rofs = warp_aggr_atomic_count_mask(atomic + 1, amask, rm);
                                  auto target_idx = l ? lofs : size - 1 - rofs;
                                  out[target_idx] = el;
                              });
}

template <typename T, typename Config>
__global__ void partition_count(const T* in, index* counts, index size,
                                T pivot, index workcount) {
    __shared__ index lcount, rcount;
    if (threadIdx.x == 0) {
        lcount = 0;
        rcount = 0;
    }
    __syncthreads();
    partition_impl<T, Config>(in, size, pivot, workcount,
                              [&](T el, bool l, mask amask, mask lm, mask rm) {
                                  if (threadIdx.x % warp_size == 0) {
                                      atomicAdd(&lcount, __popc(lm));
                                      atomicAdd(&rcount, __popc(rm));
                                  }
                              });
    __syncthreads();
    if (threadIdx.x == 0) {
        counts[2 * blockIdx.x] = lcount;
        counts[2 * blockIdx.x + 1] = rcount;
    }
}

template <typename T, typename Config>
__global__ void partition_distr(const T* in, T* out,
                                const index* counts, index size, T pivot,
                                index workcount) {
    __shared__ index lcount, rcount;
    if (threadIdx.x == 0) {
        lcount = counts[2 * blockIdx.x + 2];
        rcount = counts[2 * blockIdx.x + 3];
    }
    __syncthreads();
    partition_impl<T, Config>(in, size, pivot, workcount,
                              [&](T el, bool l, mask amask, mask lm, mask rm) {
                                  auto lofs = warp_aggr_atomic_count_mask(&lcount, amask, lm);
                                  auto rofs = warp_aggr_atomic_count_mask(&rcount, amask, rm);
                                  auto target_idx = l ? lofs : size - 1 - rofs;
                                  out[target_idx] = el;
                              });
}

} // namespace kernels
} // namespace gpu

#endif // QS_SCAN_CUH