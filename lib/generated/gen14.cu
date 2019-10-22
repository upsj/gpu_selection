#include <kernel_config.cuh>
#include <qs_launchers.cuh>
#include <qs_recursion.cuh>
#include <qs_recursion_multi.cuh>
#include <ssss_recursion.cuh>
#include <ssss_recursion_multi.cuh>
#include <ssss_launchers.cuh>
namespace gpu {
template __host__ __device__ void build_searchtree<float,select_config<10, 10, 8, false, false, true, 8, 10, 10>>(const float* in, float* out, index size);
template void sampleselect_multi<double,select_config<10, 10, 8, false, true, true, 2, 10, 10>>(double* in, double* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, double* out);
template __global__ void kernels::reduce_counts<select_config<10, 10, 8, true, true, true, 8, 10, 10>>(const index* in, index* out, index);
template void sampleselect<double,select_config<10, 10, 6, false, true, true, 8, 10, 10>>(double* in, double* tmp, double* tree, index* count_tmp, index size, index rank, double* out);
template void sampleselect<double,select_config<10, 10, 8, false, true, true, 8, 10, 10, false, 4>>(double* in, double* tmp, double* tree, index* count_tmp, index size, index rank, double* out);
template __global__ void kernels::count_buckets<double, select_config<10, 11, 9, false, true, false, 8, 10, 10>>(const double* in, const double* tree, index* counts, poracle* oracles, index size, index workcount);
template __device__ void kernels::masked_prefix_sum<9>(index* counts, const mask* m);
template __host__ __device__ void collect_bucket<double,select_config<10, 11, 9, false, true, false, 8, 10, 10>>(const double* data, const poracle* oracles_packed, const index* prefix_sum, double* out, index size, oracle bucket, index* atomic);
template __global__ void kernels::prefix_sum_counts<select_config<10, 10, 8, true, true, true, 8, 10, 10>>(index* in, index* out, index);
template void sampleselect_multi<double,select_config<10, 10, 6, false, false, true, 8, 10, 10>>(double* in, double* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, double* out);
}