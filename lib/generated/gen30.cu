#include <kernel_config.cuh>
#include <qs_launchers.cuh>
#include <qs_recursion.cuh>
#include <qs_recursion_multi.cuh>
#include <ssss_recursion.cuh>
#include <ssss_recursion_multi.cuh>
#include <ssss_launchers.cuh>
namespace gpu {
template void quickselect<double,select_config<10, 5, 8, false, true, true, 8, 10, 10>>(double* in, double* tmp, index* count_tmp, index size, index rank, double* out);
template __global__ void kernels::partition_distr<double,select_config<10, 5, 8, false, true, true, 8, 10, 10>>(const double* in, double* out, const index* counts, index size, double pivot, index workcount);
template __global__ void kernels::reduce_counts<select_config<10, 10, 8, true, false, false, 8, 10, 10>>(const index* in, index* out, index);
template __host__ __device__ void collect_bucket<float,select_config<10, 10, 8, true, false, true, 8, 10, 10>>(const float* data, const poracle* oracles_packed, const index* prefix_sum, float* out, index size, oracle bucket, index* atomic);
template void quickselect<float,select_config<10, 10, 8, true, true, true, 4, 10, 10>>(float* in, float* tmp, index* count_tmp, index size, index rank, float* out);
template void sampleselect<double,select_config<10, 10, 8, true, false, true, 8, 10, 10>>(double* in, double* tmp, double* tree, index* count_tmp, index size, index rank, double* out);
template void quickselect_multi<float,select_config<10, 10, 8, true, true, true, 8, 10, 10>>(float* in, float* tmp, index* count_tmp, index size, const index* ranks, index rank_count, float* out);
template __host__ __device__ void ssss_merged<double,select_config<10, 10, 8, false, false, true, 8, 10, 10>>(const double* in, double* out, poracle* oracles, index offset, const index* ranks, index rank_offset, index rank_base, const kernels::ssss_multi_aux<double, select_config<10, 10, 8, false, false, true, 8, 10, 10>>* aux_in, kernels::ssss_multi_aux<double, select_config<10, 10, 8, false, false, true, 8, 10, 10>>* aux_outs, double* out_tree);
template __host__ __device__ launch_parameters get_launch_parameters<double,select_config<10, 10, 8, true, true, true, 8, 10, 10>>(index size);
}