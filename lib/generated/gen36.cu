#include <kernel_config.cuh>
#include <qs_launchers.cuh>
#include <qs_recursion.cuh>
#include <qs_recursion_multi.cuh>
#include <ssss_recursion.cuh>
#include <ssss_recursion_multi.cuh>
#include <ssss_launchers.cuh>
namespace gpu {
template void sampleselect_multi<double,select_config<10, 10, 8, true, true, true, 4, 10, 10>>(double* in, double* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, double* out);
template __global__ void partition_count<float,select_config<10, 5, 8, true, true, true, 8, 10, 10>>(const float* in, index* counts, index size, float pivot, index workcount);
template void sampleselect_multi<double,select_config<10, 10, 6, true, false, true, 8, 10, 10>>(double* in, double* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, double* out);
template void quickselect<float,select_config<10, 10, 8, false, true, true, 8, 10, 10>>(float* in, float* tmp, index* count_tmp, index size, index rank, float* out);
template void sampleselect<float,select_config<10, 10, 8, false, true, true, 8, 10, 10, false, 4>>(float* in, float* tmp, float* tree, index* count_tmp, index size, index rank, float* out);
template __host__ __device__ void collect_bucket<double,select_config<10, 10, 8, false, true, true, 8, 10, 10>>(const double* data, const poracle* oracles_packed, const index* prefix_sum, double* out, index size, oracle bucket, index* atomic);
template void sampleselect<float,select_config<10, 10, 7, false, true, true, 8, 10, 10>>(float* in, float* tmp, float* tree, index* count_tmp, index size, index rank, float* out);
template __global__ void kernels::count_buckets<double, select_config<10, 10, 8, true, true, true, 8, 10, 10>>(const double* in, const double* tree, index* counts, poracle* oracles, index size, index workcount);
template void sampleselect<double,select_config<10, 10, 8, true, true, true, 8, 10, 10, false, 16>>(double* in, double* tmp, double* tree, index* count_tmp, index size, index rank, double* out);
}