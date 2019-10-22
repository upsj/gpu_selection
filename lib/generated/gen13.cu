#include <kernel_config.cuh>
#include <qs_launchers.cuh>
#include <qs_recursion.cuh>
#include <qs_recursion_multi.cuh>
#include <ssss_recursion.cuh>
#include <ssss_recursion_multi.cuh>
#include <ssss_launchers.cuh>
namespace gpu {
template void sampleselect<double,select_config<10, 10, 7, false, true, true, 8, 10, 10>>(double* in, double* tmp, double* tree, index* count_tmp, index size, index rank, double* out);
template __global__ void kernels::partition_prefixsum<select_config<10, 5, 8, true, true, true, 8, 10, 10>>(index* counts, index block_count);
template void sampleselect<float,select_config<10, 10, 8, true, true, true, 8, 9, 10>>(float* in, float* tmp, float* tree, index* count_tmp, index size, index rank, float* out);
template __device__ void kernels::masked_prefix_sum<6>(index* counts, const mask* m);
template __global__ void kernels::count_buckets<float, select_config<10, 10, 8, false, false, false, 8, 10, 10>>(const float* in, const float* tree, index* counts, poracle* oracles, index size, index workcount);
template __global__ void kernels::reduce_counts<select_config<10, 10, 8, true, true, false, 8, 10, 10>>(const index* in, index* out, index);
template __global__ void partition_count<float,select_config<10, 10, 8, true, true, true, 8, 10, 10>>(const float* in, index* counts, index size, float pivot, index workcount);
template void quickselect<double,select_config<10, 10, 8, false, true, true, 8, 9, 10>>(double* in, double* tmp, index* count_tmp, index size, index rank, double* out);
template __host__ __device__ void collect_bucket<float,select_config<10, 10, 8, false, true, false, 8, 10, 10>>(const float* data, const poracle* oracles_packed, const index* prefix_sum, float* out, index size, oracle bucket, index* atomic);
template void sampleselect<double,select_config<10, 10, 7, true, true, true, 8, 10, 10>>(double* in, double* tmp, double* tree, index* count_tmp, index size, index rank, double* out);
}