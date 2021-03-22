#include <kernel_config.cuh>
#include <qs_launchers.cuh>
#include <qs_recursion.cuh>
#include <qs_recursion_multi.cuh>
#include <ssss_recursion.cuh>
#include <ssss_recursion_multi.cuh>
#include <ssss_launchers.cuh>
namespace gpu {
template __host__ __device__ void ssss_merged<float,select_config<10, 10, 8, false, false, true, 8, 10, 10>>(const float* in, float* out, poracle* oracles, index offset, const index* ranks, index rank_offset, index rank_base, const kernels::ssss_multi_aux<float, select_config<10, 10, 8, false, false, true, 8, 10, 10>>* aux_in, kernels::ssss_multi_aux<float, select_config<10, 10, 8, false, false, true, 8, 10, 10>>* aux_outs, float* out_tree);
template void quickselect<float,select_config<10, 10, 8, true, true, true, 2, 10, 10>>(float* in, float* tmp, index* count_tmp, index size, index rank, float* out);
template __global__ void kernels::reduce_counts<select_config<10, 11, 9, true, true, false, 8, 10, 10>>(const index* in, index* out, index);
template __global__ void kernels::count_buckets<float, select_config<10, 9, 7, false, true, false, 8, 10, 10>>(const float* in, const float* tree, index* counts, poracle* oracles, index size, index workcount);
template void sampleselect_multi<float,select_config<10, 10, 8, false, true, true, 8, 10, 10, false, 8, 0>>(float* in, float* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, float* out);
template __global__ void kernels::count_buckets<double, select_config<10, 10, 8, true, true, true, 8, 10, 10>>(const double* in, const double* tree, index* counts, poracle* oracles, index size, index workcount);
template void sampleselect_host<double,select_config<10, 10, 6, true, false, true, 8, 10, 10>>(double* in, double* tmp, double* tree, index* count_tmp, index size, index rank, double* out);
template __global__ void partition_count<float,select_config<10, 5, 8, true, true, true, 8, 10, 10>>(const float* in, index* counts, index size, float pivot, index workcount);
template __global__ void kernels::prefix_sum_counts<select_config<10, 10, 8, false, true, false, 8, 10, 10>>(index* in, index* out, index);
template __host__ __device__ void collect_bucket<double,select_config<10, 10, 8, true, true, true, 8, 10, 10>>(const double* data, const poracle* oracles_packed, const index* prefix_sum, double* out, index size, oracle bucket, index* atomic);
template void sampleselect_host<double,select_config<10, 10, 8, false, true, true, 2, 10, 10>>(double* in, double* tmp, double* tree, index* count_tmp, index size, index rank, double* out);
}