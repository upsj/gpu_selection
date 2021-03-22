#include <kernel_config.cuh>
#include <qs_launchers.cuh>
#include <qs_recursion.cuh>
#include <qs_recursion_multi.cuh>
#include <ssss_recursion.cuh>
#include <ssss_recursion_multi.cuh>
#include <ssss_launchers.cuh>
namespace gpu {
template __host__ __device__ void build_searchtree<double,select_config<10, 8, 6, true, true, false, 8, 10, 10>>(const double* in, double* out, index size);
template __global__ void kernels::partition_prefixsum<select_config<10, 5, 8, true, true, true, 8, 10, 10>>(index* counts, index block_count);
template void sampleselect<double,select_config<10, 10, 8, false, true, true, 8, 8, 10>>(double* in, double* tmp, double* tree, index* count_tmp, index size, index rank, double* out);
template void sampleselect_host<double,select_config<10, 10, 8, true, true, true, 8, 10, 10, false, 8, 0>>(double* in, double* tmp, double* tree, index* count_tmp, index size, index rank, double* out);
template void quickselect_multi<float,select_config<10, 5, 8, true, true, true, 8, 10, 10>>(float* in, float* tmp, index* count_tmp, index size, const index* ranks, index rank_count, float* out);
template void sampleselect_host<double,select_config<10, 10, 8, true, true, true, 8, 9, 10>>(double* in, double* tmp, double* tree, index* count_tmp, index size, index rank, double* out);
template void sampleselect_host<float,select_config<10, 10, 8, false, true, true, 4, 10, 10>>(float* in, float* tmp, float* tree, index* count_tmp, index size, index rank, float* out);
template __host__ __device__ void build_searchtree<double,select_config<10, 11, 9, true, true, false, 8, 10, 10>>(const double* in, double* out, index size);
template __global__ void kernels::count_buckets<double, select_config<10, 8, 6, false, true, false, 8, 10, 10>>(const double* in, const double* tree, index* counts, poracle* oracles, index size, index workcount);
template void sampleselect_multi<float,select_config<10, 10, 8, true, true, true, 2, 10, 10>>(float* in, float* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, float* out);
template void sampleselect<double,select_config<10, 10, 8, true, true, true, 4, 10, 10>>(double* in, double* tmp, double* tree, index* count_tmp, index size, index rank, double* out);
}