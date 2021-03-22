#include <kernel_config.cuh>
#include <qs_launchers.cuh>
#include <qs_recursion.cuh>
#include <qs_recursion_multi.cuh>
#include <ssss_recursion.cuh>
#include <ssss_recursion_multi.cuh>
#include <ssss_launchers.cuh>
namespace gpu {
template void quickselect_multi<double,select_config<10, 5, 8, false, true, true, 8, 10, 10>>(double* in, double* tmp, index* count_tmp, index size, const index* ranks, index rank_count, double* out);
template __host__ __device__ void build_searchtree<double,select_config<10, 10, 8, true, false, false, 8, 10, 10>>(const double* in, double* out, index size);
template void sampleselect<float,select_config<8, 10, 8, false, true, true, 8, 10, 10>>(float* in, float* tmp, float* tree, index* count_tmp, index size, index rank, float* out);
template void sampleselect_host<double,select_config<9, 10, 8, false, true, true, 8, 10, 10>>(double* in, double* tmp, double* tree, index* count_tmp, index size, index rank, double* out);
template void sampleselect_host<float,select_config<10, 10, 6, false, false, true, 8, 10, 10>>(float* in, float* tmp, float* tree, index* count_tmp, index size, index rank, float* out);
template __global__ void kernels::partition_distr<double,select_config<10, 5, 8, true, true, true, 8, 10, 10>>(const double* in, double* out, const index* counts, index size, double pivot, index workcount);
template __host__ __device__ void collect_buckets<double,select_config<10, 10, 8, true, true, true, 8, 10, 10>>(const double* data, const poracle* oracles_packed, const index* block_prefix_sum, const index* bucket_out_ranges, double* out, index size, mask* buckets, index* atomic);
template void sampleselect_multi<float,select_config<10, 10, 6, false, false, true, 8, 10, 10>>(float* in, float* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, float* out);
template __host__ __device__ void collect_bucket<float,select_config<10, 11, 9, true, true, false, 8, 10, 10>>(const float* data, const poracle* oracles_packed, const index* prefix_sum, float* out, index size, oracle bucket, index* atomic);
template void sampleselect_multi<double,select_config<10, 10, 8, false, true, true, 8, 10, 10, false, 4>>(double* in, double* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, double* out);
template __host__ __device__ void collect_bucket<float,select_config<10, 9, 7, true, true, false, 8, 10, 10>>(const float* data, const poracle* oracles_packed, const index* prefix_sum, float* out, index size, oracle bucket, index* atomic);
}