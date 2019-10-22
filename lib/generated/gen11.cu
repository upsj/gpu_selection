#include <kernel_config.cuh>
#include <qs_launchers.cuh>
#include <qs_recursion.cuh>
#include <qs_recursion_multi.cuh>
#include <ssss_recursion.cuh>
#include <ssss_recursion_multi.cuh>
#include <ssss_launchers.cuh>
namespace gpu {
template void sampleselect<float,select_config<10, 10, 8, false, true, true, 8, 10, 10, false, 8, 0>>(float* in, float* tmp, float* tree, index* count_tmp, index size, index rank, float* out);
template __host__ __device__ void collect_bucket_indirect<float,select_config<10, 10, 8, false, false, true, 8, 10, 10>>(const float* data, const poracle* oracles_packed, const index* prefix_sum, float* out, index size, const oracle* bucket, index* atomic);
template __host__ __device__ void collect_buckets<double,select_config<10, 10, 8, true, true, true, 8, 10, 10>>(const double* data, const poracle* oracles_packed, const index* block_prefix_sum, const index* bucket_out_ranges, double* out, index size, mask* buckets, index* atomic);
template void sampleselect_multi<float,select_config<10, 10, 8, true, true, true, 8, 9, 10>>(float* in, float* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, float* out);
template __global__ void kernels::count_buckets<float, select_config<10, 9, 7, true, true, false, 8, 10, 10>>(const float* in, const float* tree, index* counts, poracle* oracles, index size, index workcount);
template __global__ void kernels::count_buckets<float, select_config<10, 10, 8, false, true, false, 8, 10, 10>>(const float* in, const float* tree, index* counts, poracle* oracles, index size, index workcount);
template void sampleselect<float,select_config<10, 10, 6, true, true, true, 8, 10, 10, false, 1024>>(float* in, float* tmp, float* tree, index* count_tmp, index size, index rank, float* out);
template void sampleselect_multi<float,select_config<10, 10, 8, false, true, true, 8, 10, 10, false, 8, 0>>(float* in, float* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, float* out);
template void sampleselect<float,select_config<10, 10, 8, false, false, true, 8, 10, 10>>(float* in, float* tmp, float* tree, index* count_tmp, index size, index rank, float* out);
template __host__ __device__ void count_buckets<double,select_config<10, 10, 8, true, true, false, 8, 10, 10>>(const double* in, const double* tree, index* localcounts, index* counts, poracle* oracles, index size);
}