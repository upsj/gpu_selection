#include <kernel_config.cuh>
#include <qs_launchers.cuh>
#include <qs_recursion.cuh>
#include <qs_recursion_multi.cuh>
#include <ssss_recursion.cuh>
#include <ssss_recursion_multi.cuh>
#include <ssss_launchers.cuh>
namespace gpu {
template __global__ void kernels::reduce_counts<select_config<10, 10, 8, false, true, true, 8, 10, 10>>(const index* in, index* out, index);
template __global__ void kernels::prefix_sum_counts<select_config<10, 12, 10, false, true, false, 8, 10, 10>>(index* in, index* out, index);
template __host__ __device__ void count_buckets<double,select_config<10, 10, 8, false, false, true, 8, 10, 10>>(const double* in, const double* tree, index* localcounts, index* counts, poracle* oracles, index size);
template __global__ void kernels::partition_prefixsum<select_config<10, 5, 8, false, true, true, 8, 10, 10>>(index* counts, index block_count);
template __host__ __device__ launch_parameters get_launch_parameters<float,select_config<10, 8, 6, false, true, false, 8, 10, 10>>(index size);
template __host__ __device__ void build_searchtree<float,select_config<10, 10, 8, false, true, false, 8, 10, 10>>(const float* in, float* out, index size);
template __host__ __device__ launch_parameters get_launch_parameters<float,select_config<10, 10, 8, true, false, false, 8, 10, 10>>(index size);
template void quickselect<double,select_config<9, 10, 8, true, true, true, 8, 10, 10>>(double* in, double* tmp, index* count_tmp, index size, index rank, double* out);
template __global__ void kernels::count_buckets<double, select_config<10, 9, 7, false, true, false, 8, 10, 10>>(const double* in, const double* tree, index* counts, poracle* oracles, index size, index workcount);
template void quickselect_multi<float,select_config<10, 5, 8, false, true, true, 8, 10, 10>>(float* in, float* tmp, index* count_tmp, index size, const index* ranks, index rank_count, float* out);
}