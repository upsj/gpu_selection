#include <kernel_config.cuh>
#include <qs_launchers.cuh>
#include <qs_recursion.cuh>
#include <qs_recursion_multi.cuh>
#include <ssss_recursion.cuh>
#include <ssss_recursion_multi.cuh>
#include <ssss_launchers.cuh>
namespace gpu {
template void sampleselect_multi<float,select_config<10, 10, 8, true, true, true, 8, 10, 10, false, 16>>(float* in, float* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, float* out);
template __host__ __device__ void ssss_merged<double,select_config<10, 10, 8, true, false, true, 8, 10, 10>>(const double* in, double* out, poracle* oracles, index offset, const index* ranks, index rank_offset, index rank_base, const kernels::ssss_multi_aux<double, select_config<10, 10, 8, true, false, true, 8, 10, 10>>* aux_in, kernels::ssss_multi_aux<double, select_config<10, 10, 8, true, false, true, 8, 10, 10>>* aux_outs, double* out_tree);
template __host__ __device__ void count_buckets<double,select_config<10, 10, 8, false, true, true, 8, 10, 10>>(const double* in, const double* tree, index* localcounts, index* counts, poracle* oracles, index size);
template __global__ void kernels::reduce_counts<select_config<10, 10, 8, false, false, false, 8, 10, 10>>(const index* in, index* out, index);
template __host__ __device__ launch_parameters get_launch_parameters<double,select_config<10, 10, 8, false, false, false, 8, 10, 10>>(index size);
template void sampleselect<double,select_config<10, 10, 8, false, true, true, 8, 10, 10, false, 8, 0>>(double* in, double* tmp, double* tree, index* count_tmp, index size, index rank, double* out);
template void sampleselect<float,select_config<10, 10, 8, true, true, true, 8, 10, 10, false, 8, 0>>(float* in, float* tmp, float* tree, index* count_tmp, index size, index rank, float* out);
template __global__ void kernels::partition_prefixsum<select_config<10, 10, 8, true, true, true, 8, 10, 10>>(index* counts, index block_count);
template __host__ __device__ launch_parameters get_launch_parameters<double,select_config<10, 9, 7, false, true, false, 8, 10, 10>>(index size);
}