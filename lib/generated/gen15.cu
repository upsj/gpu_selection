#include <kernel_config.cuh>
#include <qs_launchers.cuh>
#include <qs_recursion.cuh>
#include <qs_recursion_multi.cuh>
#include <ssss_recursion.cuh>
#include <ssss_recursion_multi.cuh>
#include <ssss_launchers.cuh>
namespace gpu {
template __global__ void kernels::partition_distr<float,select_config<10, 5, 8, true, true, true, 8, 10, 10>>(const float* in, float* out, const index* counts, index size, float pivot, index workcount);
template void quickselect_multi<float,select_config<10, 5, 8, false, true, true, 8, 10, 10>>(float* in, float* tmp, index* count_tmp, index size, const index* ranks, index rank_count, float* out);
template __global__ void kernels::partition<double,select_config<10, 10, 8, true, true, true, 8, 10, 10>>(const double* in, double* out, index* atomic, index size, double pivot, index workcount);
template void sampleselect_multi<double,select_config<8, 10, 8, false, true, true, 8, 10, 10>>(double* in, double* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, double* out);
template __host__ __device__ void collect_bucket<float,select_config<10, 9, 7, false, true, false, 8, 10, 10>>(const float* data, const poracle* oracles_packed, const index* prefix_sum, float* out, index size, oracle bucket, index* atomic);
template __host__ __device__ void collect_bucket_indirect<float,select_config<10, 10, 8, true, true, true, 8, 10, 10>>(const float* data, const poracle* oracles_packed, const index* prefix_sum, float* out, index size, const oracle* bucket, index* atomic);
template __global__ void kernels::reduce_counts<select_config<10, 10, 8, false, true, true, 8, 10, 10>>(const index* in, index* out, index);
template void sampleselect_multi<double,select_config<10, 10, 8, false, true, true, 8, 10, 10, false, 8, 0>>(double* in, double* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, double* out);
template void sampleselect_host<float,select_config<10, 10, 8, false, false, true, 8, 10, 10>>(float* in, float* tmp, float* tree, index* count_tmp, index size, index rank, float* out);
template void quickselect<float,select_config<8, 10, 8, true, true, true, 8, 10, 10>>(float* in, float* tmp, index* count_tmp, index size, index rank, float* out);
template __global__ void kernels::count_buckets<double, select_config<10, 11, 9, true, true, false, 8, 10, 10>>(const double* in, const double* tree, index* counts, poracle* oracles, index size, index workcount);
}