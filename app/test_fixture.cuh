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
#include <algorithm>
#include <cuda_definitions.cuh>
#include <cuda_error.cuh>
#include <cuda_memory.cuh>
#include <random>
#include <tuple>
#include <utils.cuh>
#include <vector>

namespace gpu {

constexpr auto max_tree_width = 4096;
constexpr auto max_tree_size = 2 * max_tree_width * 2;
constexpr auto max_block_count = 1024;

template <typename Pair, index Size = 1, index Valsize = 1>
struct basic_test_data {
    using T = typename Pair::first_type;
    using Config = typename Pair::second_type;
    index size;
    std::vector<T> data;
    std::vector<T> tree;
    std::vector<T> data_out;
    std::vector<poracle> oracles;
    std::vector<index> count_out;
    std::vector<index> atomic;
    std::vector<index> zeros;
    std::vector<index> ranks;
    std::vector<mask> bucket_mask;
    cuda_resettable_array<T> gpu_data;
    cuda_array<T> gpu_data_tmp;
    cuda_array<T> gpu_tree;
    cuda_array<T> gpu_data_out;
    cuda_array<poracle> gpu_oracles;
    cuda_array<index> gpu_aux;
    cuda_resettable_array<index> gpu_atomic;
    cuda_array<index> gpu_count_tmp;
    cuda_resettable_array<index> gpu_count_out;
    cuda_array<index> gpu_bucket_ranges;
    cuda_array<index> gpu_rank_ranges;
    cuda_array<index> gpu_ranks;
    cuda_array<mask> gpu_bucket_mask;
    index rank;
    T pivot;

    basic_test_data(index size = Size, index valsize = Valsize, index seed = 0)
            : size{size}, data(size), tree(max_tree_size), oracles(size), count_out(max_block_count * 2 + 2),
              zeros(size + max_block_count * max_tree_width * 16), ranks(287), bucket_mask(max_tree_size / (sizeof(mask) * 8)),
              atomic(max_tree_width) {
        std::default_random_engine random(seed);
        std::uniform_int_distribution<std::size_t> dist(0, valsize - 1);
        std::uniform_int_distribution<std::size_t> idist(0, size - 1);
        std::uniform_int_distribution<mask> maskdist(mask(0), ~mask(0));
        std::vector<index> smallzeros(max_tree_size);
        for (auto& el : data) {
            el = dist(random);
        }
        for (auto& el : ranks) {
            el = idist(random);
        }
        ranks.back() = size - 1;
        std::sort(ranks.begin(), ranks.end());
        rank = idist(random);
        pivot = data[rank];
        gpu_data.copy_from(data);
        gpu_tree.copy_from(tree);
        gpu_data_tmp.copy_from(data);
        gpu_data_out.copy_from(data);
        gpu_atomic.copy_from(atomic);
        gpu_count_tmp.copy_from(zeros);
        gpu_aux.copy_from(zeros);
        gpu_count_out.copy_from(count_out);
        gpu_oracles.copy_from(oracles);
        gpu_bucket_ranges.copy_from(smallzeros);
        gpu_rank_ranges.copy_from(smallzeros);
        gpu_ranks.copy_from(ranks);
        gpu_bucket_mask.copy_from(bucket_mask);
    }

    void reset() {
        gpu_data.reset();
        gpu_atomic.reset();
        gpu_count_out.reset();
    }

    void copy_from_gpu() {
        gpu_data_out.copy_to(data_out);
        gpu_count_out.copy_to(count_out);
        gpu_tree.copy_to(tree);
        gpu_oracles.copy_to(oracles);
        gpu_atomic.copy_to(atomic);
    }

    template <typename F>
    void run(F f) {
        cudaChecked(f);
        copy_from_gpu();
    }
};

inline std::vector<unsigned char> unpack(const std::vector<poracle>& in, int size) {
    using uc = unsigned char;
    std::vector<uc> result;
    result.reserve(in.size() * 4);
    for (auto el : in) {
        result.insert(result.end(), {uc(el), uc(el >> 8), uc(el >> 16), uc(el >> 24)});
    }
    result.resize(size);
    return result;
}

inline std::vector<index> build_ranks_uniform(index size, index count) {
    std::vector<index> result;
    for (index i = 0; i < count; ++i) {
        result.push_back(int(double(i) * size / count));
    }
    return result;
}

inline std::vector<index> build_ranks_clustered(index size) {
    std::vector<index> result;
    auto step = size / 2;
    while (step >= 1) {
        result.push_back(step);
        step = step / 2;
    }
    std::reverse(result.begin(), result.end());
    return result;
}

} // namespace gpu