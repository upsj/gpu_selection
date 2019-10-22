#include <kernel_config.cuh>
#include <qs_launchers.cuh>
#include <qs_recursion.cuh>
#include <qs_recursion_multi.cuh>
#include <ssss_recursion.cuh>
#include <ssss_recursion_multi.cuh>
#include <ssss_launchers.cuh>
namespace gpu {
template __global__ void kernels::count_buckets<float, select_config<10, 10, 8, true, true, false, 8, 10, 10>>(const float* in, const float* tree, index* counts, poracle* oracles, index size, index workcount);
template __global__ void partition_count<double,select_config<10, 10, 8, true, true, true, 8, 10, 10>>(const double* in, index* counts, index size, double pivot, index workcount);
template __host__ __device__ void build_searchtree<float,select_config<10, 8, 6, false, true, false, 8, 10, 10>>(const float* in, float* out, index size);
template void sampleselect_multi<float,select_config<10, 10, 8, false, true, true, 8, 10, 10, false, 4>>(float* in, float* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, float* out);
template void sampleselect_multi<double,select_config<10, 10, 8, false, true, true, 8, 10, 10, true>>(double* in, double* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, double* out);
template __global__ void kernels::reduce_counts<select_config<10, 11, 9, false, true, false, 8, 10, 10>>(const index* in, index* out, index);
template void sampleselect<double,select_config<10, 10, 8, true, true, true, 8, 9, 10>>(double* in, double* tmp, double* tree, index* count_tmp, index size, index rank, double* out);
template __host__ __device__ void build_searchtree<float,select_config<10, 11, 9, true, true, false, 8, 10, 10>>(const float* in, float* out, index size);
template __host__ __device__ void count_buckets<float,select_config<10, 11, 9, true, true, false, 8, 10, 10>>(const float* in, const float* tree, index* localcounts, index* counts, poracle* oracles, index size);
template __device__ void kernels::masked_prefix_sum<10>(index* counts, const mask* m);
}