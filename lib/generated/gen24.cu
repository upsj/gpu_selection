#include <kernel_config.cuh>
#include <qs_launchers.cuh>
#include <qs_recursion.cuh>
#include <qs_recursion_multi.cuh>
#include <ssss_recursion.cuh>
#include <ssss_recursion_multi.cuh>
#include <ssss_launchers.cuh>
namespace gpu {
template void sampleselect_multi<double,select_config<10, 10, 8, true, true, true, 8, 10, 10, true>>(double* in, double* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, double* out);
template __global__ void kernels::partition<double,select_config<10, 10, 8, true, true, true, 8, 10, 10>>(const double* in, double* out, index* atomic, index size, double pivot, index workcount);
template void sampleselect_multi<double,select_config<9, 10, 8, true, true, true, 8, 10, 10>>(double* in, double* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, double* out);
template void sampleselect_multi<double,select_config<10, 10, 8, true, true, true, 8, 10, 10, false, 16>>(double* in, double* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, double* out);
template __host__ __device__ void collect_bucket_indirect<double,select_config<10, 10, 8, false, true, true, 8, 10, 10>>(const double* data, const poracle* oracles_packed, const index* prefix_sum, double* out, index size, const oracle* bucket, index* atomic);
template __global__ void kernels::prefix_sum_counts<select_config<10, 11, 9, true, true, false, 8, 10, 10>>(index* in, index* out, index);
template __global__ void kernels::prefix_sum_counts<select_config<10, 10, 8, true, false, false, 8, 10, 10>>(index* in, index* out, index);
template void sampleselect_multi<float,select_config<9, 10, 8, true, true, true, 8, 10, 10>>(float* in, float* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, float* out);
template __host__ __device__ void count_buckets<double,select_config<10, 10, 8, true, true, true, 8, 10, 10>>(const double* in, const double* tree, index* localcounts, index* counts, poracle* oracles, index size);
template __host__ __device__ void build_searchtree<double,select_config<10, 11, 9, false, true, false, 8, 10, 10>>(const double* in, double* out, index size);
}