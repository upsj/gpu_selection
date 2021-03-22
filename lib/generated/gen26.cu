#include <kernel_config.cuh>
#include <qs_launchers.cuh>
#include <qs_recursion.cuh>
#include <qs_recursion_multi.cuh>
#include <ssss_recursion.cuh>
#include <ssss_recursion_multi.cuh>
#include <ssss_launchers.cuh>
namespace gpu {
template void sampleselect<double,select_config<10, 10, 8, true, true, true, 2, 10, 10>>(double* in, double* tmp, double* tree, index* count_tmp, index size, index rank, double* out);
template void sampleselect_host<double,select_config<10, 10, 8, false, true, true, 8, 8, 10>>(double* in, double* tmp, double* tree, index* count_tmp, index size, index rank, double* out);
template void sampleselect_host<float,select_config<10, 10, 8, false, true, true, 8, 10, 10, true>>(float* in, float* tmp, float* tree, index* count_tmp, index size, index rank, float* out);
template void sampleselect_multi<float,select_config<10, 10, 8, true, true, true, 8, 10, 10, true>>(float* in, float* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, float* out);
template __global__ void partition_count<double,select_config<10, 10, 8, false, true, true, 8, 10, 10>>(const double* in, index* counts, index size, double pivot, index workcount);
template __global__ void kernels::reduce_counts<select_config<10, 8, 6, false, true, false, 8, 10, 10>>(const index* in, index* out, index);
template void sampleselect<double,select_config<10, 10, 7, false, true, true, 8, 10, 10>>(double* in, double* tmp, double* tree, index* count_tmp, index size, index rank, double* out);
template void sampleselect_host<float,select_config<10, 10, 8, false, true, true, 8, 10, 10, false, 8, 0>>(float* in, float* tmp, float* tree, index* count_tmp, index size, index rank, float* out);
template void sampleselect<float,select_config<10, 10, 6, false, false, true, 8, 10, 10>>(float* in, float* tmp, float* tree, index* count_tmp, index size, index rank, float* out);
template void quickselect<double,select_config<10, 5, 8, false, true, true, 8, 10, 10>>(double* in, double* tmp, index* count_tmp, index size, index rank, double* out);
template __host__ __device__ void build_searchtree<float,select_config<10, 12, 10, true, true, false, 8, 10, 10>>(const float* in, float* out, index size);
}