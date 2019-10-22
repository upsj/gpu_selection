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
#ifndef UTILS_SORT_CUH
#define UTILS_SORT_CUH

#include "utils.cuh"

namespace gpu {
namespace kernels {
template <typename T>
__host__ __device__ void bitonic_cas(T& a, T& b, bool odd) {
    auto tmp = a;
    bool cmp = (a < b) != odd;
    a = cmp ? a : b;
    b = cmp ? b : tmp;
}

template <typename T, int N2>
struct bitonic_helper_local {
    // # local elements
    constexpr static auto n = 1 << N2;
    using half_helper = bitonic_helper_local<T, N2 - 1>;

    // merges two bitonic sequences els[0, n / 2), els[n / 2, n)
    __host__ __device__ static void merge(T* els, bool odd) {
        for (auto i = 0; i < n / 2; ++i) {
            bitonic_cas(els[i], els[i + n / 2], odd);
        }
        half_helper::merge(els, odd);
        half_helper::merge(els + (n / 2), odd);
    }

    // sorts an unsorted sequence els[0, n)
    __host__ __device__ static void sort(T* els, bool odd) {
        half_helper::sort(els, odd);
        half_helper::sort(els + (n / 2), !odd);
        merge(els, odd);
    }
};

template <typename T>
struct bitonic_helper_local<T, 0> {
    __host__ __device__ static void merge(T*, bool) {}
    __host__ __device__ static void sort(T*, bool) {}
};

template <typename T, int NL2, int NT2>
struct bitonic_helper_warp {
    // # local elements
    constexpr static auto nl = 1 << NL2;
    // # warps
    constexpr static auto nt = 1 << NT2;
    constexpr static auto n = nl * nt;
    using half_helper = bitonic_helper_warp<T, NL2, NT2 - 1>;
    static_assert(nt <= 32, "warp only has 32 threads");

    // check if we are at the sending or receiving side of a CAS
    // also check if if we subdivide our sort, we are in the forward- or backward group
    __device__ static bool is_odd() {
        return bool(threadIdx.x & (nt / 2));
    }

    __device__ static void merge(T* els, bool odd) {
        auto new_odd = odd != is_odd();
        for (auto i = 0; i < nl; ++i) {
            auto other = __shfl_xor_sync(full_mask, els[i], nt / 2);
            // is_odd is true iff we exchange with a larger lane
            bitonic_cas(els[i], other, odd != is_odd());
        }
        half_helper::merge(els, odd);
    }

    __device__ static void sort(T* els, bool odd) {
        // is_odd is true iff we are in the upper half of the group
        half_helper::sort(els, odd != is_odd());
        merge(els, odd);
    }
};

template <typename T, int NL2>
struct bitonic_helper_warp<T, NL2, 0> {
    using local = bitonic_helper_local<T, NL2>;
    __device__ static void merge(T* els, bool odd) {
        local::merge(els, odd);
    }
    __device__ static void sort(T* els, bool odd) {
        local::sort(els, odd);
    }
};

template <typename T, int NL2, int NT2, int NW2, int NTOTAL2>
struct bitonic_helper_global {
    static_assert(NL2 + NT2 + NW2 <= NTOTAL2, "inconsistent sizes");
    // # local elements
    constexpr static auto nl = 1 << NL2;
    // # threads per warp
    constexpr static auto nt = 1 << NT2;
    // # warps
    constexpr static auto nw = 1 << NW2;
    // total # elements
    constexpr static auto ntotal = 1 << NTOTAL2;
    constexpr static auto n = nw * nt * nl;
    using half_helper = bitonic_helper_global<T, NL2, NT2, NW2 - 1, NTOTAL2>;

    // check if if we subdivide our sort, we are in the forward- or backward group
    __device__ static bool is_odd() {
        return bool(threadIdx.x & (nw * nt / 2));
    }

    __device__ static void merge(T* local_els, T* shared_els, bool odd) {
        auto base = threadIdx.x * nl;
        __syncthreads();
        if (!(base & (n / 2)) && base < ntotal) {
            for (auto i = 0; i < nl; ++i) {
                bitonic_cas(shared_els[i + base], shared_els[i + base + n / 2], odd);
            }
        }
        half_helper::merge(local_els, shared_els, odd);
    }

    __device__ static void sort(T* local_els, T* shared_els, bool odd) {
        half_helper::sort(local_els, shared_els, odd != is_odd());
        merge(local_els, shared_els, odd);
    }
};

template <typename T, int NL2, int NT2, int NTOTAL2>
struct bitonic_helper_global<T, NL2, NT2, 0, NTOTAL2> {
    // # local elements
    constexpr static auto nl = 1 << NL2;
    // # threads per warp
    constexpr static auto nt = 1 << NT2;
    // total # elements
    constexpr static auto ntotal = 1 << NTOTAL2;
    using warp = bitonic_helper_warp<T, NL2, NT2>;
    __device__ static void merge(T* local_els, T* shared_els, bool odd) {
        __syncthreads();
        auto base = threadIdx.x * nl;
        if (base < ntotal) {
            for (auto i = 0; i < nl; ++i) {
                local_els[i] = shared_els[i + base];
            }
            warp::merge(local_els, odd);
            for (auto i = 0; i < nl; ++i) {
                shared_els[i + base] = local_els[i];
            }
        }
    }

    __device__ static void sort(T* local_els, T* shared_els, bool odd) {
        // This is the first step executed, so we don't need to load from shared memory
        auto base = threadIdx.x * nl;
        if (base < ntotal) {
            warp::sort(local_els, odd);
            for (auto i = 0; i < nl; ++i) {
                shared_els[i + base] = local_els[i];
            }
        }
    }
};

} // namespace kernels
} // namespace gpu

#endif // UTILS_SORT_CUH