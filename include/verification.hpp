#ifndef GPU_SELECTION_VERIFICATION_HPP
#define GPU_SELECTION_VERIFICATION_HPP

#include <tuple>
#include <vector>
#include <cuda_definitions.cuh>

namespace verification {

using gpu::index;
using gpu::mask;

template <typename T>
std::pair<int, int> count_mispartitioned(const std::vector<T>& data, int pivot_rank, T pivot);

template <typename T>
T nth_element(const std::vector<T>& data, int rank);

template <typename T>
std::vector<T> nth_elements(const std::vector<T>& data, std::vector<gpu::index> ranks);

template <typename T>
int count_not_in_bucket(const std::vector<T>& data, T lower, T upper);

template <typename T>
std::vector<index> count_not_in_buckets(const std::vector<T>& data, std::vector<index> prefix_sum, const std::vector<T>& searchtree);

bool verify_rank_ranges(const std::vector<index>& ranks, const std::vector<index>& index_ranges, const std::vector<index>& rank_ranges);

} // namespace verification

#endif // GPU_SELECTION_VERIFICATION_HPP
