#include <kernel_config.cuh>
#include <qs_launchers.cuh>
#include <qs_recursion.cuh>
#include <qs_recursion_multi.cuh>
#include <ssss_recursion.cuh>
#include <ssss_recursion_multi.cuh>
#include <ssss_launchers.cuh>
namespace gpu {
template __global__ void kernels::count_buckets<float, select_config<10, 12, 10, false, true, false, 8, 10, 10>>(const float* in, const float* tree, index* counts, poracle* oracles, index size, index workcount);
template void sampleselect<float,select_config<10, 10, 6, false, true, true, 8, 10, 10, false, 1024>>(float* in, float* tmp, float* tree, index* count_tmp, index size, index rank, float* out);
template __global__ void kernels::partition_distr<double,select_config<10, 10, 8, true, true, true, 8, 10, 10>>(const double* in, double* out, const index* counts, index size, double pivot, index workcount);
template void sampleselect_multi<double,select_config<10, 10, 6, false, true, true, 8, 10, 10, false, 1024>>(double* in, double* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, double* out);
template void sampleselect<double,select_config<10, 10, 8, false, false, true, 8, 10, 10>>(double* in, double* tmp, double* tree, index* count_tmp, index size, index rank, double* out);
template __host__ __device__ void ssss_merged<float,select_config<10, 10, 8, true, true, true, 8, 10, 10>>(const float* in, float* out, poracle* oracles, index offset, const index* ranks, index rank_offset, index rank_base, const kernels::ssss_multi_aux<float, select_config<10, 10, 8, true, true, true, 8, 10, 10>>* aux_in, kernels::ssss_multi_aux<float, select_config<10, 10, 8, true, true, true, 8, 10, 10>>* aux_outs, float* out_tree);
template __global__ void kernels::prefix_sum_counts<select_config<10, 10, 8, false, true, false, 8, 10, 10>>(index* in, index* out, index);
template void sampleselect_multi<float,select_config<10, 10, 8, true, true, true, 8, 10, 10, false, 4>>(float* in, float* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, float* out);
template void sampleselect_multi<double,select_config<10, 10, 8, true, true, true, 2, 10, 10>>(double* in, double* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, double* out);
}