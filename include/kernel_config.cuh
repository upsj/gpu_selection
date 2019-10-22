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
#include "cuda_definitions.cuh"
#include <algorithm>

namespace gpu {

template <index Size_log2, index Local_size_log2>
struct bitonic_basecase_config {
    constexpr static index size_log2 = Size_log2;
    constexpr static index size = 1 << size_log2;
    constexpr static index local_size_log2 = Local_size_log2;
    constexpr static index local_size = 1 << local_size_log2;
    constexpr static index launch_size = size / local_size;
};

template <index Size_log2>
struct sample_config {
    constexpr static index size_log2 = Size_log2;
    constexpr static index size = 1 << size_log2;
    constexpr static index local_size_log2 = size_log2 > max_block_size_log2 ? size_log2 - max_block_size_log2 : 0;
    constexpr static index local_size = 1 << local_size_log2;
};

template <index Height>
struct searchtree_config {
    constexpr static index height = Height;
    constexpr static index width = 1 << height;
    constexpr static index size = 2 * width - 1;
};

template <bool Shared_memory, bool Warp_aggr, bool Write, index Unroll, index Max_block_size_log2,
          index Max_block_count_log2, bool Bucket_select, index Merged_limit>
struct algorithm_config {
    constexpr static bool shared_memory = Shared_memory;
    constexpr static bool warp_aggr = Warp_aggr;
    constexpr static bool write = Write;
    constexpr static index unroll = Unroll;
    constexpr static index max_block_size_log2 = Max_block_size_log2;
    constexpr static index max_block_size = 1 << max_block_size_log2;
    constexpr static index max_block_count_log2 = Max_block_count_log2;
    constexpr static index max_block_count = 1 << max_block_count_log2;
    constexpr static index merged_limit = Merged_limit;
    constexpr static bool bucket_select = Bucket_select;
};

template <index basecase_log2 = 10, index sample_log2 = 10, index searchtree_height = 8,
          bool shared_memory = true, bool warp_aggr = true, bool write = true, index unroll = 8,
          index max_block_size_log2 = 10, index max_block_count_log2 = 10, bool bucket_select = false, index merged_limit = 8, index sort_local_log2 = 2>
struct select_config {
    using basecase = bitonic_basecase_config<basecase_log2, sort_local_log2>;
    using sample = sample_config<sample_log2>;
    using searchtree = searchtree_config<searchtree_height>;
    using algorithm = algorithm_config<shared_memory, warp_aggr, write, unroll, max_block_size_log2,
                                       max_block_count_log2, bucket_select, merged_limit>;
    constexpr static auto searchtree_kernel_size = std::max(std::min(max_block_size, sample::size), searchtree::width);    
};

} // namespace gpu