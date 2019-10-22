#ifndef GPU_SELECTION_CPU_REFERENCE_HPP
#define GPU_SELECTION_CPU_REFERENCE_HPP

#include <cuda_definitions.cuh>
#include <vector>

namespace cpu {

using gpu::index;
using gpu::mask;

template <typename T>
std::pair<int, int> partition(const std::vector<T>& data, int begin, int end, std::vector<T>& out,
                              int pivot_idx);

template <typename T>
T quickselect(std::vector<T>& in, std::vector<T>& out, int rank);

template <typename T>
std::vector<T> build_searchtree(const std::vector<T>& in, int sample_size, int searchtree_size);

template <typename T>
std::pair<std::vector<index>, std::vector<unsigned char>> ssss(const std::vector<T>& data,
                                                               const std::vector<T>& tree, bool write = true);

std::vector<index> grouped_reduce(const std::vector<index>& data, int searchtree_size);
std::vector<index> grouped_prefix_sum(const std::vector<index>& data, int searchtree_size);

std::vector<index> compute_rank_ranges(std::vector<index> counts, const std::vector<index>& ranks);
std::vector<mask> compute_bucket_mask(const std::vector<index>& rank_ranges);

std::pair<std::vector<index>, index> masked_prefix_sum(const std::vector<index>& counts, const std::vector<mask>& m);

} // namespace cpu

#endif // GPU_SELECTION_CPU_REFERENCE_HPP
