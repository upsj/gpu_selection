#include <kernel_config.cuh>
#include <qs_launchers.cuh>
#include <qs_recursion.cuh>
#include <qs_recursion_multi.cuh>
#include <ssss_recursion.cuh>
#include <ssss_recursion_multi.cuh>
#include <ssss_launchers.cuh>
namespace gpu {
template __host__ __device__ void ssss_merged<float,select_config<10, 10, 8, false, true, true, 8, 10, 10>>(const float* in, float* out, poracle* oracles, index offset, const index* ranks, index rank_offset, index rank_base, const kernels::ssss_multi_aux<float, select_config<10, 10, 8, false, true, true, 8, 10, 10>>* aux_in, kernels::ssss_multi_aux<float, select_config<10, 10, 8, false, true, true, 8, 10, 10>>* aux_outs, float* out_tree);
template __host__ __device__ void collect_bucket<float,select_config<10, 11, 9, true, true, false, 8, 10, 10>>(const float* data, const poracle* oracles_packed, const index* prefix_sum, float* out, index size, oracle bucket, index* atomic);
template __host__ __device__ void collect_bucket<double,select_config<10, 10, 8, false, false, true, 8, 10, 10>>(const double* data, const poracle* oracles_packed, const index* prefix_sum, double* out, index size, oracle bucket, index* atomic);
template __host__ __device__ launch_parameters get_launch_parameters<float,select_config<10, 9, 7, false, true, false, 8, 10, 10>>(index size);
template __host__ __device__ void collect_bucket_indirect<float,select_config<10, 10, 8, true, true, true, 8, 10, 10>>(const float* data, const poracle* oracles_packed, const index* prefix_sum, float* out, index size, const oracle* bucket, index* atomic);
template void quickselect<float,select_config<9, 10, 8, true, true, true, 8, 10, 10>>(float* in, float* tmp, index* count_tmp, index size, index rank, float* out);
template void sampleselect<double,select_config<10, 10, 6, true, true, true, 8, 10, 10, false, 1024>>(double* in, double* tmp, double* tree, index* count_tmp, index size, index rank, double* out);
template __global__ void kernels::partition_distr<double,select_config<10, 10, 8, false, true, true, 8, 10, 10>>(const double* in, double* out, const index* counts, index size, double pivot, index workcount);
template void sampleselect<float,select_config<10, 10, 6, false, false, true, 8, 10, 10>>(float* in, float* tmp, float* tree, index* count_tmp, index size, index rank, float* out);
template __host__ __device__ launch_parameters get_launch_parameters<float,select_config<10, 10, 8, false, true, true, 8, 10, 10>>(index size);
}