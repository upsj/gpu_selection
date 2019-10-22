#include <kernel_config.cuh>
#include <qs_launchers.cuh>
#include <qs_recursion.cuh>
#include <qs_recursion_multi.cuh>
#include <ssss_recursion.cuh>
#include <ssss_recursion_multi.cuh>
#include <ssss_launchers.cuh>
namespace gpu {
template void sampleselect_multi<double,select_config<10, 10, 8, true, true, true, 8, 8, 10>>(double* in, double* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, double* out);
template __host__ __device__ void build_searchtree<double,select_config<10, 8, 6, true, true, false, 8, 10, 10>>(const double* in, double* out, index size);
template __global__ void kernels::prefix_sum_counts<select_config<10, 11, 9, false, true, false, 8, 10, 10>>(index* in, index* out, index);
template __host__ __device__ void collect_bucket<float,select_config<10, 10, 8, true, false, false, 8, 10, 10>>(const float* data, const poracle* oracles_packed, const index* prefix_sum, float* out, index size, oracle bucket, index* atomic);
template __host__ __device__ launch_parameters get_launch_parameters<float,select_config<10, 9, 7, true, true, false, 8, 10, 10>>(index size);
template __global__ void kernels::count_buckets<double, select_config<10, 10, 8, false, true, true, 8, 10, 10>>(const double* in, const double* tree, index* counts, poracle* oracles, index size, index workcount);
template __global__ void kernels::reduce_counts<select_config<10, 12, 10, false, true, false, 8, 10, 10>>(const index* in, index* out, index);
template void sampleselect<float,select_config<10, 10, 8, true, true, true, 8, 10, 10, false, 16>>(float* in, float* tmp, float* tree, index* count_tmp, index size, index rank, float* out);
template __global__ void kernels::prefix_sum_counts<select_config<10, 9, 7, true, true, false, 8, 10, 10>>(index* in, index* out, index);
template void sampleselect_multi<float,select_config<10, 10, 8, false, true, true, 2, 10, 10>>(float* in, float* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, float* out);
}