#include <kernel_config.cuh>
#include <qs_launchers.cuh>
#include <qs_recursion.cuh>
#include <qs_recursion_multi.cuh>
#include <ssss_recursion.cuh>
#include <ssss_recursion_multi.cuh>
#include <ssss_launchers.cuh>
namespace gpu {
template void sampleselect_multi<float,select_config<10, 10, 8, true, true, true, 8, 10, 10>>(float* in, float* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, float* out);
template __global__ void kernels::partition<float,select_config<10, 10, 8, false, true, true, 8, 10, 10>>(const float* in, float* out, index* atomic, index size, float pivot, index workcount);
template __global__ void kernels::reduce_counts<select_config<10, 8, 6, false, true, false, 8, 10, 10>>(const index* in, index* out, index);
template __host__ __device__ void collect_buckets<float,select_config<10, 10, 8, false, false, true, 8, 10, 10>>(const float* data, const poracle* oracles_packed, const index* block_prefix_sum, const index* bucket_out_ranges, float* out, index size, mask* buckets, index* atomic);
template __host__ __device__ void count_buckets<double,select_config<10, 11, 9, true, true, false, 8, 10, 10>>(const double* in, const double* tree, index* localcounts, index* counts, poracle* oracles, index size);
template void quickselect_multi<float,select_config<10, 10, 8, false, true, true, 8, 10, 10>>(float* in, float* tmp, index* count_tmp, index size, const index* ranks, index rank_count, float* out);
template __host__ __device__ void ssss_merged<float,select_config<10, 10, 8, false, false, true, 8, 10, 10>>(const float* in, float* out, poracle* oracles, index offset, const index* ranks, index rank_offset, index rank_base, const kernels::ssss_multi_aux<float, select_config<10, 10, 8, false, false, true, 8, 10, 10>>* aux_in, kernels::ssss_multi_aux<float, select_config<10, 10, 8, false, false, true, 8, 10, 10>>* aux_outs, float* out_tree);
template void quickselect<double,select_config<10, 10, 8, true, true, true, 8, 10, 10>>(double* in, double* tmp, index* count_tmp, index size, index rank, double* out);
template void sampleselect<double,select_config<10, 10, 8, true, true, true, 8, 8, 10>>(double* in, double* tmp, double* tree, index* count_tmp, index size, index rank, double* out);
}