#include <kernel_config.cuh>
#include <qs_launchers.cuh>
#include <qs_recursion.cuh>
#include <qs_recursion_multi.cuh>
#include <ssss_recursion.cuh>
#include <ssss_recursion_multi.cuh>
#include <ssss_launchers.cuh>
namespace gpu {
template void sampleselect_multi<float,select_config<10, 10, 8, true, true, true, 4, 10, 10>>(float* in, float* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, float* out);
template void sampleselect_multi<double,select_config<10, 10, 7, true, true, true, 8, 10, 10>>(double* in, double* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, double* out);
template __host__ __device__ launch_parameters get_launch_parameters<double,select_config<10, 11, 9, true, true, false, 8, 10, 10>>(index size);
template __host__ __device__ void collect_buckets<double,select_config<10, 10, 8, false, true, true, 8, 10, 10>>(const double* data, const poracle* oracles_packed, const index* block_prefix_sum, const index* bucket_out_ranges, double* out, index size, mask* buckets, index* atomic);
template void sampleselect<float,select_config<8, 10, 8, true, true, true, 8, 10, 10>>(float* in, float* tmp, float* tree, index* count_tmp, index size, index rank, float* out);
template __global__ void kernels::partition<double,select_config<10, 5, 8, false, true, true, 8, 10, 10>>(const double* in, double* out, index* atomic, index size, double pivot, index workcount);
template void quickselect<double,select_config<10, 10, 8, false, true, true, 2, 10, 10>>(double* in, double* tmp, index* count_tmp, index size, index rank, double* out);
template __host__ __device__ void ssss_merged<float,select_config<10, 10, 8, true, false, true, 8, 10, 10>>(const float* in, float* out, poracle* oracles, index offset, const index* ranks, index rank_offset, index rank_base, const kernels::ssss_multi_aux<float, select_config<10, 10, 8, true, false, true, 8, 10, 10>>* aux_in, kernels::ssss_multi_aux<float, select_config<10, 10, 8, true, false, true, 8, 10, 10>>* aux_outs, float* out_tree);
template __global__ void kernels::prefix_sum_counts<select_config<10, 12, 10, true, true, false, 8, 10, 10>>(index* in, index* out, index);
}