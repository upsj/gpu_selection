#ifndef SSSS_BUILD_SEARCHTREE_CUH
#define SSSS_BUILD_SEARCHTREE_CUH

#include "utils_sampling.cuh"
#include "utils_sort.cuh"
#include "utils_work.cuh"

namespace gpu {
namespace kernels {

template <typename Config>
__device__ __forceinline__ index searchtree_entry(index idx) {
    // determine the level by the node index
    // rationale: a complete binary tree with 2^k leaves has 2^k - 1 inner nodes
    // lvl == log2(idx + 1)
    auto lvl = 31 - __clz(idx + 1);
    // step == n / 2^lvl
    auto step = Config::searchtree::width >> lvl;
    // index within the level
    auto lvl_idx = idx - (1 << lvl) + 1;
    return lvl_idx * step + step / 2;
}

template <typename T, typename Config>
__device__ bool is_equality_bucket(const T* leaves, index bucket_idx) {
    // first and last bucket can't definitely be checked to be equality buckets
    return bucket_idx > 0 && bucket_idx < Config::searchtree::width - 1 && leaves[bucket_idx + 1] == add_epsilon(leaves[bucket_idx]);
}

template <typename T, typename Config>
__device__ void equality_bucket(T* leaves) {
    auto idx = threadIdx.x;
    if (idx < Config::searchtree::width && idx > 0) {
        // If we are the last in a sequence of equal elements, we add a small epsilon
        bool equality = leaves[idx] == leaves[idx - 1] &&
                        (idx == Config::searchtree::width - 1 || leaves[idx] < leaves[idx + 1]);
        if (equality) {
            leaves[idx] = add_epsilon(leaves[idx]);
        }
    }
}

template <typename T, typename Config>
__device__ void build_searchtree_shared(const T* in, index size, T* tree) {
    __shared__ T sample_buffer[Config::sample::size];
    static_assert(Config::sample::size >= Config::searchtree::width, "sample too small");
    auto idx = threadIdx.x;
    
    // pick sample
    T local_buffer[Config::sample::local_size];
    if (threadIdx.x * Config::sample::local_size < Config::sample::size) {
        for (auto i = 0; i < Config::sample::local_size; ++i) {
            local_buffer[i] = in[random_pick_idx(threadIdx.x * Config::sample::local_size + i, Config::sample::size, size)];
        }
    }
    // sort sample
    using sorter = bitonic_helper_global<T, Config::sample::local_size_log2, warp_size_log2, Config::sample::size_log2 - warp_size_log2 - Config::sample::local_size_log2, Config::sample::size_log2>;
    sorter::sort(local_buffer, sample_buffer, false);
    __syncthreads();
    // pick splitters from sorted sample
    if (idx < Config::searchtree::width) {
        tree[idx + Config::searchtree::width - 1] = sample_buffer[uniform_pick_idx(idx,
            Config::searchtree::width, Config::sample::size)];
    }
    __syncthreads();
    // create equality bucket if necessary
    equality_bucket<T, Config>(tree + (Config::searchtree::width - 1));
    __syncthreads();
    // inner nodes
    if (idx < Config::searchtree::width - 1) {
        tree[idx] = tree[searchtree_entry<Config>(idx) + Config::searchtree::width - 1];
    }
}

template <typename T, typename Config>
__global__ void build_searchtree(const T* in, T* out, index size) {
    build_searchtree_shared<T, Config>(in, size, out);
}

} // namespace kernels
} // namespace gpu

#endif // SSSS_BUILD_SEARCHTREE_CU