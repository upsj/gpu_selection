#include <kernel_config.cuh>
#include <qs_launchers.cuh>
#include <qs_recursion.cuh>
#include <qs_recursion_multi.cuh>
#include <ssss_recursion.cuh>
#include <ssss_recursion_multi.cuh>
#include <ssss_launchers.cuh>
namespace gpu {
template __host__ __device__ void count_buckets<float,select_config<10, 10, 8, true, true, true, 8, 10, 10>>(const float* in, const float* tree, index* localcounts, index* counts, poracle* oracles, index size);
template __global__ void kernels::partition<float,select_config<10, 10, 8, true, true, true, 8, 10, 10>>(const float* in, float* out, index* atomic, index size, float pivot, index workcount);
template void sampleselect<double,select_config<10, 10, 8, true, false, true, 8, 10, 10>>(double* in, double* tmp, double* tree, index* count_tmp, index size, index rank, double* out);
template void sampleselect<float,select_config<10, 10, 8, true, true, true, 8, 10, 10, true>>(float* in, float* tmp, float* tree, index* count_tmp, index size, index rank, float* out);
template void sampleselect_multi<float,select_config<10, 10, 6, true, true, true, 8, 10, 10>>(float* in, float* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, float* out);
template void sampleselect_multi<float,select_config<10, 10, 8, true, true, true, 8, 10, 10, false, 4>>(float* in, float* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, float* out);
template void sampleselect_multi<double,select_config<10, 10, 8, true, true, true, 8, 10, 10, false, 4>>(double* in, double* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, double* out);
template __host__ __device__ void collect_bucket<float,select_config<10, 11, 9, false, true, false, 8, 10, 10>>(const float* data, const poracle* oracles_packed, const index* prefix_sum, float* out, index size, oracle bucket, index* atomic);
template void sampleselect<double,select_config<10, 10, 6, false, false, true, 8, 10, 10>>(double* in, double* tmp, double* tree, index* count_tmp, index size, index rank, double* out);
template __host__ __device__ void collect_bucket<double,select_config<10, 10, 8, false, false, false, 8, 10, 10>>(const double* data, const poracle* oracles_packed, const index* prefix_sum, double* out, index size, oracle bucket, index* atomic);
template void sampleselect_host<float,select_config<10, 10, 8, true, true, true, 8, 8, 10>>(float* in, float* tmp, float* tree, index* count_tmp, index size, index rank, float* out);
}