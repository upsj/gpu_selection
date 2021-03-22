#include <kernel_config.cuh>
#include <qs_launchers.cuh>
#include <qs_recursion.cuh>
#include <qs_recursion_multi.cuh>
#include <ssss_recursion.cuh>
#include <ssss_recursion_multi.cuh>
#include <ssss_launchers.cuh>
namespace gpu {
template __global__ void kernels::partition_distr<float,select_config<10, 5, 8, false, true, true, 8, 10, 10>>(const float* in, float* out, const index* counts, index size, float pivot, index workcount);
template void sampleselect_host<double,select_config<10, 10, 8, true, true, true, 8, 10, 10>>(double* in, double* tmp, double* tree, index* count_tmp, index size, index rank, double* out);
template void quickselect<float,select_config<10, 10, 8, true, true, true, 4, 10, 10>>(float* in, float* tmp, index* count_tmp, index size, index rank, float* out);
template void sampleselect_multi<double,select_config<10, 10, 6, false, false, true, 8, 10, 10>>(double* in, double* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, double* out);
template __global__ void partition_count<float,select_config<10, 10, 8, false, true, true, 8, 10, 10>>(const float* in, index* counts, index size, float pivot, index workcount);
template __global__ void kernels::reduce_counts<select_config<10, 10, 8, true, false, false, 8, 10, 10>>(const index* in, index* out, index);
template void sampleselect_multi<double,select_config<10, 10, 8, false, true, true, 2, 10, 10>>(double* in, double* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, double* out);
template __host__ __device__ void ssss_merged<float,select_config<10, 10, 8, true, true, true, 8, 10, 10>>(const float* in, float* out, poracle* oracles, index offset, const index* ranks, index rank_offset, index rank_base, const kernels::ssss_multi_aux<float, select_config<10, 10, 8, true, true, true, 8, 10, 10>>* aux_in, kernels::ssss_multi_aux<float, select_config<10, 10, 8, true, true, true, 8, 10, 10>>* aux_outs, float* out_tree);
template __global__ void partition_count<float,select_config<10, 10, 8, true, true, true, 8, 10, 10>>(const float* in, index* counts, index size, float pivot, index workcount);
template __global__ void kernels::reduce_counts<select_config<10, 8, 6, true, true, false, 8, 10, 10>>(const index* in, index* out, index);
template __host__ __device__ void count_buckets<float,select_config<10, 10, 8, false, false, true, 8, 10, 10>>(const float* in, const float* tree, index* localcounts, index* counts, poracle* oracles, index size);
}