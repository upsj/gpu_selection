#include <kernel_config.cuh>
#include <qs_launchers.cuh>
#include <qs_recursion.cuh>
#include <qs_recursion_multi.cuh>
#include <ssss_recursion.cuh>
#include <ssss_recursion_multi.cuh>
#include <ssss_launchers.cuh>
namespace gpu {
template __host__ __device__ void build_searchtree<float,select_config<10, 10, 8, true, true, true, 8, 10, 10>>(const float* in, float* out, index size);
template __host__ __device__ void build_searchtree<double,select_config<10, 10, 8, true, false, false, 8, 10, 10>>(const double* in, double* out, index size);
template __host__ __device__ void collect_bucket<double,select_config<10, 10, 8, false, false, false, 8, 10, 10>>(const double* data, const poracle* oracles_packed, const index* prefix_sum, double* out, index size, oracle bucket, index* atomic);
template __global__ void kernels::partition_distr<float,select_config<10, 5, 8, true, true, true, 8, 10, 10>>(const float* in, float* out, const index* counts, index size, float pivot, index workcount);
template __host__ __device__ void build_searchtree<float,select_config<10, 12, 10, true, true, false, 8, 10, 10>>(const float* in, float* out, index size);
template void sampleselect<double,select_config<10, 10, 8, false, true, true, 2, 10, 10>>(double* in, double* tmp, double* tree, index* count_tmp, index size, index rank, double* out);
template __global__ void partition_count<float,select_config<10, 5, 8, false, true, true, 8, 10, 10>>(const float* in, index* counts, index size, float pivot, index workcount);
template __host__ __device__ void collect_buckets<double,select_config<10, 10, 8, true, false, true, 8, 10, 10>>(const double* data, const poracle* oracles_packed, const index* block_prefix_sum, const index* bucket_out_ranges, double* out, index size, mask* buckets, index* atomic);
template __device__ __host__ void partition<double,select_config<10, 10, 8, false, true, true, 8, 10, 10>>(const double* in, double* out, index* counts, index size, double pivot);
template void quickselect<float,select_config<10, 10, 8, false, true, true, 2, 10, 10>>(float* in, float* tmp, index* count_tmp, index size, index rank, float* out);
}