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
#include <numeric>
#include <cassert>
#include <cmath>
#include <cpu_reference.hpp>

namespace cpu {

template <typename T>
std::pair<int, int> partition(const std::vector<T>& data, int begin, int end, std::vector<T>& ref,
                              int pivot_idx) {
    auto pivot = data[pivot_idx];
    int left = begin;
    int right = end - 1;
    for (int i = begin; i < end; ++i) {
        if (i == pivot_idx) {
            continue;
        }
        bool gt = data[i] >= pivot;
        ref[gt ? right : left] = data[i];
        left += !gt;
        right -= gt;
    }
    ref[left] = pivot;
    return {left, data.size() - right};
}

template std::pair<int, int> partition<float>(const std::vector<float>&, int, int,
                                              std::vector<float>&, int);
template std::pair<int, int> partition<double>(const std::vector<double>&, int, int,
                                               std::vector<double>&, int);

template <typename T>
T quickselect(std::vector<T>& in, std::vector<T>& out, int rank) {
    int begin = 0;
    int end = in.size();
    int l;
    while ((l = partition(in, begin, end, out, in[begin]).first) != rank) {
        std::swap(in, out);
        if (rank < l) {
            end = l;
        } else {
            begin = l + 1;
        }
    }
    return out[rank];
}

template double quickselect<double>(std::vector<double>&, std::vector<double>&, int);
template float quickselect<float>(std::vector<float>&, std::vector<float>&, int);

namespace {

template <typename T>
void build_searchtree_impl(int ofs, int size, int lvl, int tree_idx, std::vector<T>& tree) {
    auto searchtree_width = tree.size() / 2 + 1;
    tree[tree_idx] = tree[searchtree_width - 1 + ofs + size / 2];
    if (size > 1) {
        build_searchtree_impl(ofs, size / 2, lvl + 1, 2 * tree_idx + 1, tree);
        build_searchtree_impl(ofs + size / 2, size / 2, lvl + 1, 2 * tree_idx + 2, tree);
    }
}

inline float add_epsilon(float f) { return std::nextafterf(f, std::numeric_limits<float>::max()); }

inline double add_epsilon(double f) {
    return std::nextafter(f, std::numeric_limits<double>::max());
}

} // anonymous namespace

template <typename T>
std::vector<T> build_searchtree(const std::vector<T>& data, int sample_size, int searchtree_size) {
    // sample elements
    std::vector<T> splitters(sample_size);
    auto searchtree_width = searchtree_size / 2 + 1;
    auto stride = data.size() / sample_size;
    for (int i = 0; i < sample_size; ++i) {
        splitters[i] = data[stride == 0 ? i * data.size() / sample_size : i * stride + stride / 2];
    }
    // sort them
    std::sort(splitters.begin(), splitters.end());
    // sample leaves from splitters
    std::vector<T> tree(searchtree_size);
    auto input_stride = sample_size / searchtree_width;
    for (int i = 0; i < searchtree_width; ++i) {
        tree[i + searchtree_width - 1] =
                splitters[input_stride == 0 ? i * sample_size / searchtree_width
                                            : i * input_stride + input_stride / 2];
    }
    // apply equality bucket fix
    for (int i = 1; i < searchtree_width; ++i) {
        auto l0 = tree[i + searchtree_width - 2];
        auto l1 = tree[i + searchtree_width - 1];
        auto l2 = tree[i + searchtree_width];
        bool equality = l0 == l1 && (i == searchtree_width - 1 || l1 < l2);
        if (equality) {
            tree[i + searchtree_width - 1] = add_epsilon(tree[i + searchtree_width - 1]);
        }
    }
    // build remaining tree
    build_searchtree_impl(0, searchtree_width, 0, 0, tree);
    return tree;
}

template std::vector<float> build_searchtree<float>(const std::vector<float>&, int, int);
template std::vector<double> build_searchtree<double>(const std::vector<double>&, int, int);

template <typename T>
std::pair<std::vector<index>, std::vector<unsigned char>> ssss(const std::vector<T>& data,
                                                               const std::vector<T>& _tree, bool write) {
    auto tree = _tree;
    std::vector<index> histogram(tree.size() / 2 + 1);
    assert(histogram.size() <= 256 || !write);
    std::vector<unsigned char> oracles(data.size());
    tree[tree.size() / 2] = -std::numeric_limits<T>::max();
    int i = 0;
    for (auto f : data) {
        auto it = std::upper_bound(tree.begin() + tree.size() / 2, tree.end(), f);
        int bucket = std::distance(tree.begin() + tree.size() / 2, it) - 1;
        oracles[i] = bucket;
        ++histogram[bucket];
        ++i;
    }
    return {std::move(histogram), std::move(oracles)};
}

template std::pair<std::vector<index>, std::vector<unsigned char>>
ssss<float>(const std::vector<float>&, const std::vector<float>&, bool);
template std::pair<std::vector<index>, std::vector<unsigned char>>
ssss<double>(const std::vector<double>&, const std::vector<double>&, bool);

std::vector<index> grouped_reduce(const std::vector<index>& data, int searchtree_size) {
    std::vector<index> result(searchtree_size / 2 + 1);
    for (int i = 0; i < data.size(); ++i) {
        result[i % result.size()] += data[i];
    }
    return result;
}

std::vector<index> grouped_prefix_sum(const std::vector<index>& data, int searchtree_size) {
    std::vector<index> result(data.size());
    for (int i = searchtree_size / 2 + 1; i < data.size(); ++i) {
        result[i] = result[i - (searchtree_size / 2 + 1)] + data[i - (searchtree_size / 2 + 1)];
    }
    return result;
}

std::pair<std::vector<index>, index> masked_prefix_sum(const std::vector<index>& counts, const std::vector<mask>& m) {
    auto res = counts;
    index count{};
    for (index i = 0; i < counts.size(); ++i) {
        auto mask_block = i / (sizeof(mask) * 8);
        auto mask_bit = mask(i % (sizeof(mask) * 8));
        auto masked_bit = mask(1) << mask_bit;
        res[i] = count;
        if (m[mask_block] & masked_bit) {
            count += counts[i];
        }
    }
    return {res, count};
}

std::vector<index> compute_rank_ranges(std::vector<index> counts, const std::vector<index>& ranks) {
    counts.insert(counts.begin(), 0);
    std::partial_sum(counts.begin(), counts.end(), counts.begin());
    std::vector<index> result;
    for (auto i = 0; i < counts.size(); ++i) {
        result.push_back(std::distance(ranks.begin(), std::lower_bound(ranks.begin(), ranks.end(), counts[i])));
    }
    return result;
}

std::vector<mask> compute_bucket_mask(const std::vector<index>& rank_ranges) {
    auto mask_size = sizeof(gpu::mask) * 8;
    std::vector<mask> result((rank_ranges.size() + mask_size - 2) / mask_size);
    for (index i = 0; i < rank_ranges.size() - 1; ++i) {
        if (rank_ranges[i] < rank_ranges[i + 1]) {
            result[i / (sizeof(gpu::mask) * 8)] |= gpu::mask{1} << (i % (sizeof(gpu::mask) * 8));
        }
    }
    return result;
}

} // namespace cpu
