#include <kernel_config.cuh>
#include <qs_launchers.cuh>
#include <qs_recursion.cuh>
#include <qs_recursion_multi.cuh>
#include <ssss_recursion.cuh>
#include <ssss_recursion_multi.cuh>
#include <ssss_launchers.cuh>
namespace gpu {
template void sampleselect<float,select_config<10, 10, 8, false, true, true, 8, 8, 10>>(float* in, float* tmp, float* tree, index* count_tmp, index size, index rank, float* out);
template void quickselect<double,select_config<8, 10, 8, true, true, true, 8, 10, 10>>(double* in, double* tmp, index* count_tmp, index size, index rank, double* out);
template __global__ void partition_count<double,select_config<10, 10, 8, false, true, true, 8, 10, 10>>(const double* in, index* counts, index size, double pivot, index workcount);
template __host__ __device__ void build_searchtree<float,select_config<10, 10, 8, false, false, false, 8, 10, 10>>(const float* in, float* out, index size);
template __global__ void kernels::partition_prefixsum<select_config<10, 10, 8, false, true, true, 8, 10, 10>>(index* counts, index block_count);
template __global__ void kernels::reduce_counts<select_config<10, 10, 8, false, true, false, 8, 10, 10>>(const index* in, index* out, index);
template void quickselect_multi<float,select_config<10, 5, 8, true, true, true, 8, 10, 10>>(float* in, float* tmp, index* count_tmp, index size, const index* ranks, index rank_count, float* out);
template void sampleselect<float,select_config<10, 10, 8, false, true, true, 8, 10, 10>>(float* in, float* tmp, float* tree, index* count_tmp, index size, index rank, float* out);
template __host__ __device__ void build_searchtree<double,select_config<10, 10, 8, false, true, false, 8, 10, 10>>(const double* in, double* out, index size);
template __global__ void kernels::reduce_counts<select_config<10, 11, 9, true, true, false, 8, 10, 10>>(const index* in, index* out, index);
}