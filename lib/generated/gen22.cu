#include <kernel_config.cuh>
#include <qs_launchers.cuh>
#include <qs_recursion.cuh>
#include <qs_recursion_multi.cuh>
#include <ssss_recursion.cuh>
#include <ssss_recursion_multi.cuh>
#include <ssss_launchers.cuh>
namespace gpu {
template void sampleselect<float,select_config<10, 10, 8, true, true, true, 4, 10, 10>>(float* in, float* tmp, float* tree, index* count_tmp, index size, index rank, float* out);
template void quickselect<float,select_config<10, 10, 8, true, true, true, 8, 10, 10>>(float* in, float* tmp, index* count_tmp, index size, index rank, float* out);
template void sampleselect<float,select_config<9, 10, 8, true, true, true, 8, 10, 10>>(float* in, float* tmp, float* tree, index* count_tmp, index size, index rank, float* out);
template void sampleselect<float,select_config<10, 10, 8, false, true, true, 4, 10, 10>>(float* in, float* tmp, float* tree, index* count_tmp, index size, index rank, float* out);
template __device__ __host__ void partition<double,select_config<10, 5, 8, false, true, true, 8, 10, 10>>(const double* in, double* out, index* counts, index size, double pivot);
template __host__ __device__ void collect_bucket<float,select_config<10, 11, 9, false, true, false, 8, 10, 10>>(const float* data, const poracle* oracles_packed, const index* prefix_sum, float* out, index size, oracle bucket, index* atomic);
template __host__ __device__ void collect_bucket<double,select_config<10, 8, 6, true, true, false, 8, 10, 10>>(const double* data, const poracle* oracles_packed, const index* prefix_sum, double* out, index size, oracle bucket, index* atomic);
template __host__ __device__ void build_searchtree<double,select_config<10, 10, 8, true, false, true, 8, 10, 10>>(const double* in, double* out, index size);
template __global__ void kernels::reduce_counts<select_config<10, 9, 7, false, true, false, 8, 10, 10>>(const index* in, index* out, index);
template __global__ void kernels::prefix_sum_counts<select_config<10, 9, 7, false, true, false, 8, 10, 10>>(index* in, index* out, index);
}