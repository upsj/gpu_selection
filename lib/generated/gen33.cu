#include <kernel_config.cuh>
#include <qs_launchers.cuh>
#include <qs_recursion.cuh>
#include <qs_recursion_multi.cuh>
#include <ssss_recursion.cuh>
#include <ssss_recursion_multi.cuh>
#include <ssss_launchers.cuh>
namespace gpu {
template __global__ void kernels::partition<float,select_config<10, 10, 8, false, true, true, 8, 10, 10>>(const float* in, float* out, index* atomic, index size, float pivot, index workcount);
template void sampleselect_host<float,select_config<10, 10, 8, false, true, true, 8, 10, 10, false, 16>>(float* in, float* tmp, float* tree, index* count_tmp, index size, index rank, float* out);
template void sampleselect<float,select_config<9, 10, 8, false, true, true, 8, 10, 10>>(float* in, float* tmp, float* tree, index* count_tmp, index size, index rank, float* out);
template __device__ void kernels::masked_prefix_sum<6>(index* counts, const mask* m);
template __host__ __device__ void collect_bucket<float,select_config<10, 8, 6, false, true, false, 8, 10, 10>>(const float* data, const poracle* oracles_packed, const index* prefix_sum, float* out, index size, oracle bucket, index* atomic);
template __global__ void kernels::prefix_sum_counts<select_config<10, 11, 9, true, true, false, 8, 10, 10>>(index* in, index* out, index);
template __global__ void kernels::partition<float,select_config<10, 5, 8, false, true, true, 8, 10, 10>>(const float* in, float* out, index* atomic, index size, float pivot, index workcount);
template void sampleselect_multi<float,select_config<10, 10, 6, false, true, true, 8, 10, 10>>(float* in, float* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, float* out);
template __global__ void kernels::partition_prefixsum<select_config<10, 5, 8, false, true, true, 8, 10, 10>>(index* counts, index block_count);
template void sampleselect_multi<float,select_config<10, 10, 6, true, false, true, 8, 10, 10>>(float* in, float* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, float* out);
template void sampleselect_host<double,select_config<10, 10, 8, false, false, true, 8, 10, 10>>(double* in, double* tmp, double* tree, index* count_tmp, index size, index rank, double* out);
}