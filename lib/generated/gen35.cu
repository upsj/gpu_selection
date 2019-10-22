#include <kernel_config.cuh>
#include <qs_launchers.cuh>
#include <qs_recursion.cuh>
#include <qs_recursion_multi.cuh>
#include <ssss_recursion.cuh>
#include <ssss_recursion_multi.cuh>
#include <ssss_launchers.cuh>
namespace gpu {
template void sampleselect<float,select_config<10, 10, 6, true, false, true, 8, 10, 10>>(float* in, float* tmp, float* tree, index* count_tmp, index size, index rank, float* out);
template void sampleselect<double,select_config<10, 10, 8, false, true, true, 4, 10, 10>>(double* in, double* tmp, double* tree, index* count_tmp, index size, index rank, double* out);
template __global__ void kernels::count_buckets<double, select_config<10, 10, 8, false, true, false, 8, 10, 10>>(const double* in, const double* tree, index* counts, poracle* oracles, index size, index workcount);
template __host__ __device__ void build_searchtree<float,select_config<10, 9, 7, false, true, false, 8, 10, 10>>(const float* in, float* out, index size);
template __host__ __device__ void collect_bucket<float,select_config<10, 8, 6, false, true, false, 8, 10, 10>>(const float* data, const poracle* oracles_packed, const index* prefix_sum, float* out, index size, oracle bucket, index* atomic);
template void sampleselect<double,select_config<10, 10, 8, false, true, true, 8, 10, 10, true>>(double* in, double* tmp, double* tree, index* count_tmp, index size, index rank, double* out);
template __global__ void kernels::count_buckets<double, select_config<10, 10, 8, false, false, false, 8, 10, 10>>(const double* in, const double* tree, index* counts, poracle* oracles, index size, index workcount);
template __host__ __device__ void build_searchtree<float,select_config<10, 8, 6, true, true, false, 8, 10, 10>>(const float* in, float* out, index size);
template __device__ __host__ void partition<double,select_config<10, 5, 8, true, true, true, 8, 10, 10>>(const double* in, double* out, index* counts, index size, double pivot);
}