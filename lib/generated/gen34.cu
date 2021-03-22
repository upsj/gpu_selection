#include <kernel_config.cuh>
#include <qs_launchers.cuh>
#include <qs_recursion.cuh>
#include <qs_recursion_multi.cuh>
#include <ssss_recursion.cuh>
#include <ssss_recursion_multi.cuh>
#include <ssss_launchers.cuh>
namespace gpu {
template void sampleselect_host<float,select_config<10, 10, 8, false, true, true, 2, 10, 10>>(float* in, float* tmp, float* tree, index* count_tmp, index size, index rank, float* out);
template __host__ __device__ void collect_bucket<float,select_config<10, 8, 6, true, true, false, 8, 10, 10>>(const float* data, const poracle* oracles_packed, const index* prefix_sum, float* out, index size, oracle bucket, index* atomic);
template __host__ __device__ void build_searchtree<double,select_config<10, 10, 8, true, true, true, 8, 10, 10>>(const double* in, double* out, index size);
template void sampleselect_host<double,select_config<10, 10, 6, false, true, true, 8, 10, 10>>(double* in, double* tmp, double* tree, index* count_tmp, index size, index rank, double* out);
template void sampleselect_host<double,select_config<10, 10, 8, false, true, true, 8, 10, 10, false, 4>>(double* in, double* tmp, double* tree, index* count_tmp, index size, index rank, double* out);
template __host__ __device__ void collect_bucket<float,select_config<10, 10, 8, false, true, false, 8, 10, 10>>(const float* data, const poracle* oracles_packed, const index* prefix_sum, float* out, index size, oracle bucket, index* atomic);
template void quickselect<double,select_config<10, 10, 8, true, true, true, 8, 9, 10>>(double* in, double* tmp, index* count_tmp, index size, index rank, double* out);
template __host__ __device__ void build_searchtree<double,select_config<10, 9, 7, false, true, false, 8, 10, 10>>(const double* in, double* out, index size);
template __global__ void partition_count<double,select_config<10, 5, 8, false, true, true, 8, 10, 10>>(const double* in, index* counts, index size, double pivot, index workcount);
template void sampleselect<float,select_config<10, 10, 8, false, true, true, 8, 10, 10>>(float* in, float* tmp, float* tree, index* count_tmp, index size, index rank, float* out);
template __device__ __host__ void partition<float,select_config<10, 5, 8, true, true, true, 8, 10, 10>>(const float* in, float* out, index* counts, index size, float pivot);
}