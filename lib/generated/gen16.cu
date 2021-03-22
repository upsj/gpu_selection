#include <kernel_config.cuh>
#include <qs_launchers.cuh>
#include <qs_recursion.cuh>
#include <qs_recursion_multi.cuh>
#include <ssss_recursion.cuh>
#include <ssss_recursion_multi.cuh>
#include <ssss_launchers.cuh>
namespace gpu {
template void sampleselect<double,select_config<10, 10, 8, false, true, true, 8, 10, 10, false, 8, 0>>(double* in, double* tmp, double* tree, index* count_tmp, index size, index rank, double* out);
template void sampleselect<float,select_config<10, 10, 8, true, true, true, 8, 10, 10, false, 16>>(float* in, float* tmp, float* tree, index* count_tmp, index size, index rank, float* out);
template __host__ __device__ void build_searchtree<double,select_config<10, 10, 8, false, true, false, 8, 10, 10>>(const double* in, double* out, index size);
template __global__ void kernels::count_buckets<float, select_config<10, 11, 9, true, true, false, 8, 10, 10>>(const float* in, const float* tree, index* counts, poracle* oracles, index size, index workcount);
template void sampleselect_host<float,select_config<10, 10, 6, false, true, true, 8, 10, 10>>(float* in, float* tmp, float* tree, index* count_tmp, index size, index rank, float* out);
template __host__ __device__ void build_searchtree<double,select_config<10, 8, 6, false, true, false, 8, 10, 10>>(const double* in, double* out, index size);
template __host__ __device__ launch_parameters get_launch_parameters<double,select_config<10, 11, 9, true, true, false, 8, 10, 10>>(index size);
template __host__ __device__ void collect_bucket<float,select_config<10, 10, 8, true, false, true, 8, 10, 10>>(const float* data, const poracle* oracles_packed, const index* prefix_sum, float* out, index size, oracle bucket, index* atomic);
template __host__ __device__ void build_searchtree<float,select_config<10, 12, 10, false, true, false, 8, 10, 10>>(const float* in, float* out, index size);
template __host__ __device__ void ssss_merged<float,select_config<10, 10, 8, true, false, true, 8, 10, 10>>(const float* in, float* out, poracle* oracles, index offset, const index* ranks, index rank_offset, index rank_base, const kernels::ssss_multi_aux<float, select_config<10, 10, 8, true, false, true, 8, 10, 10>>* aux_in, kernels::ssss_multi_aux<float, select_config<10, 10, 8, true, false, true, 8, 10, 10>>* aux_outs, float* out_tree);
template void quickselect<double,select_config<10, 10, 8, false, true, true, 2, 10, 10>>(double* in, double* tmp, index* count_tmp, index size, index rank, double* out);
}