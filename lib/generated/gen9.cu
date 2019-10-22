#include <kernel_config.cuh>
#include <qs_launchers.cuh>
#include <qs_recursion.cuh>
#include <qs_recursion_multi.cuh>
#include <ssss_recursion.cuh>
#include <ssss_recursion_multi.cuh>
#include <ssss_launchers.cuh>
namespace gpu {
template void sampleselect<float,select_config<10, 10, 8, false, true, true, 8, 10, 10, false, 16>>(float* in, float* tmp, float* tree, index* count_tmp, index size, index rank, float* out);
template __global__ void kernels::prefix_sum_counts<select_config<10, 10, 8, false, true, true, 8, 10, 10>>(index* in, index* out, index);
template __global__ void kernels::prefix_sum_counts<select_config<10, 10, 8, true, true, false, 8, 10, 10>>(index* in, index* out, index);
template __device__ void kernels::masked_prefix_sum<8>(index* counts, const mask* m);
template void sampleselect<double,select_config<8, 10, 8, false, true, true, 8, 10, 10>>(double* in, double* tmp, double* tree, index* count_tmp, index size, index rank, double* out);
template __host__ __device__ void collect_buckets<float,select_config<10, 10, 8, false, true, true, 8, 10, 10>>(const float* data, const poracle* oracles_packed, const index* block_prefix_sum, const index* bucket_out_ranges, float* out, index size, mask* buckets, index* atomic);
template void sampleselect<double,select_config<10, 10, 8, true, true, true, 2, 10, 10>>(double* in, double* tmp, double* tree, index* count_tmp, index size, index rank, double* out);
template void quickselect<float,select_config<10, 5, 8, true, true, true, 8, 10, 10>>(float* in, float* tmp, index* count_tmp, index size, index rank, float* out);
template __host__ __device__ void build_searchtree<double,select_config<10, 10, 8, false, true, true, 8, 10, 10>>(const double* in, double* out, index size);
template void quickselect<float,select_config<8, 10, 8, true, true, true, 8, 10, 10>>(float* in, float* tmp, index* count_tmp, index size, index rank, float* out);
}