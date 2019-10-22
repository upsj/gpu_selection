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
#include <verification.hpp>

#include <algorithm>
#include <iostream>

namespace verification {

template <typename T>
std::pair<int, int> count_mispartitioned(const std::vector<T>& data, int pivot_rank, T pivot) {
    int lcount{};
    int rcount{};
    for (int i = 0; i < pivot_rank; ++i) {
        if (data[i] >= pivot) {
            ++lcount;
        }
    }
    for (int i = pivot_rank; i < data.size(); ++i) {
        if (data[i] < pivot) {
            ++rcount;
        }
    }
    return {lcount, rcount};
}

template std::pair<int, int> count_mispartitioned<float>(const std::vector<float>&, int, float);
template std::pair<int, int> count_mispartitioned<double>(const std::vector<double>&, int, double);

template <typename T>
std::vector<T> nth_elements(const std::vector<T>& data, std::vector<gpu::index> ranks) {
    auto tmp = data;
    std::sort(tmp.begin(), tmp.end());
    std::vector<T> result;
    for (auto el : ranks) {
        result.push_back(tmp[el]);
    }
    return result;
}

template <typename T>
T nth_element(const std::vector<T>& data, int rank) {
    auto tmp = data;
    std::sort(tmp.begin(), tmp.end());
    return tmp[rank];
}

template float nth_element<float>(const std::vector<float>&, int);
template double nth_element<double>(const std::vector<double>&, int);

template std::vector<float> nth_elements<float>(const std::vector<float>&, std::vector<gpu::index>);
template std::vector<double> nth_elements<double>(const std::vector<double>&, std::vector<gpu::index>);

template <typename T>
int count_not_in_bucket(const std::vector<T>& data, T lower, T upper) {
    return std::count_if(data.begin(), data.end(),
                         [&](T val) { return val < lower || val >= upper; });
}

template int count_not_in_bucket<float>(const std::vector<float>&, float, float);
template int count_not_in_bucket<double>(const std::vector<double>&, double, double);

template <typename T>
std::vector<index> count_not_in_buckets(const std::vector<T>& data, std::vector<index> prefix_sum, const std::vector<T>& searchtree) {
    auto splitter_count = prefix_sum.size() - 1;
    std::vector<index> result(splitter_count);
    for (index bucket = 0; bucket < splitter_count; ++bucket) {
        // we don't use the smallest splitter
        auto lower = bucket == 0 ? 0 : searchtree[bucket + splitter_count - 1];
        // we don't store the sentinels
        auto upper = bucket == splitter_count - 1 ? std::numeric_limits<T>::max() : searchtree[bucket + splitter_count];
        result[bucket] = std::count_if(data.begin() + prefix_sum[bucket], data.begin() + prefix_sum[bucket + 1], [&](T val) {
            return val < lower || val >= upper;
        });
    }
    return result;
}

template std::vector<index> count_not_in_buckets<float>(const std::vector<float>& data, std::vector<index> prefix_sum, const std::vector<float>& searchtree);
template std::vector<index> count_not_in_buckets<double>(const std::vector<double>& data, std::vector<index> prefix_sum, const std::vector<double>& searchtree);

bool verify_rank_ranges(const std::vector<index>& ranks, const std::vector<index>& index_ranges, const std::vector<index>& rank_ranges) {
    auto searchtree_width = rank_ranges.size() - 1;
    if (!std::is_sorted(rank_ranges.begin(), rank_ranges.end())) return false;
    for (gpu::index i = 0; i < searchtree_width; ++i) {
        auto lb = index_ranges[i];
        auto ub = index_ranges[i + 1];
        for (auto j = rank_ranges[i]; j < rank_ranges[i + 1]; ++j) {
            if (ranks[j] < lb || ranks[j] >= ub) return false;
        }
    }
    return rank_ranges[0] == 0 && rank_ranges.back() == ranks.size();
}

} // namespace verification
