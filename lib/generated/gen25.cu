#include <kernel_config.cuh>
#include <qs_launchers.cuh>
#include <qs_recursion.cuh>
#include <qs_recursion_multi.cuh>
#include <ssss_recursion.cuh>
#include <ssss_recursion_multi.cuh>
#include <ssss_launchers.cuh>
namespace gpu {
template void sampleselect<double,select_config<10, 10, 8, false, true, true, 8, 9, 10>>(double* in, double* tmp, double* tree, index* count_tmp, index size, index rank, double* out);
template __host__ __device__ launch_parameters get_launch_parameters<double,select_config<10, 10, 8, false, true, true, 8, 10, 10>>(index size);
template __host__ __device__ void collect_bucket<double,select_config<10, 10, 8, false, true, false, 8, 10, 10>>(const double* data, const poracle* oracles_packed, const index* prefix_sum, double* out, index size, oracle bucket, index* atomic);
template __global__ void kernels::reduce_counts<select_config<10, 8, 6, true, true, false, 8, 10, 10>>(const index* in, index* out, index);
template __global__ void kernels::count_buckets<float, select_config<10, 10, 8, true, false, false, 8, 10, 10>>(const float* in, const float* tree, index* counts, poracle* oracles, index size, index workcount);
template void quickselect<double,select_config<9, 10, 8, false, true, true, 8, 10, 10>>(double* in, double* tmp, index* count_tmp, index size, index rank, double* out);
template __global__ void kernels::prefix_sum_counts<select_config<10, 8, 6, false, true, false, 8, 10, 10>>(index* in, index* out, index);
template __global__ void kernels::partition_distr<float,select_config<10, 10, 8, true, true, true, 8, 10, 10>>(const float* in, float* out, const index* counts, index size, float pivot, index workcount);
template __global__ void kernels::count_buckets<double, select_config<10, 11, 9, true, true, false, 8, 10, 10>>(const double* in, const double* tree, index* counts, poracle* oracles, index size, index workcount);
template __host__ __device__ void count_buckets<float,select_config<10, 10, 8, false, true, true, 8, 10, 10>>(const float* in, const float* tree, index* localcounts, index* counts, poracle* oracles, index size);
}