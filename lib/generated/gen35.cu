#include <kernel_config.cuh>
#include <qs_launchers.cuh>
#include <qs_recursion.cuh>
#include <qs_recursion_multi.cuh>
#include <ssss_recursion.cuh>
#include <ssss_recursion_multi.cuh>
#include <ssss_launchers.cuh>
namespace gpu {
template void sampleselect_multi<float,select_config<10, 10, 8, false, true, true, 4, 10, 10>>(float* in, float* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, float* out);
template void sampleselect_multi<double,select_config<10, 10, 8, false, false, true, 8, 10, 10>>(double* in, double* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, double* out);
template __host__ __device__ launch_parameters get_launch_parameters<double,select_config<10, 10, 8, true, true, false, 8, 10, 10>>(index size);
template __host__ __device__ void build_searchtree<float,select_config<10, 10, 8, false, false, false, 8, 10, 10>>(const float* in, float* out, index size);
template __global__ void kernels::partition<double,select_config<10, 5, 8, false, true, true, 8, 10, 10>>(const double* in, double* out, index* atomic, index size, double pivot, index workcount);
template void sampleselect<float,select_config<10, 10, 6, true, false, true, 8, 10, 10>>(float* in, float* tmp, float* tree, index* count_tmp, index size, index rank, float* out);
template void sampleselect<double,select_config<10, 10, 6, true, true, true, 8, 10, 10>>(double* in, double* tmp, double* tree, index* count_tmp, index size, index rank, double* out);
template void quickselect<double,select_config<10, 5, 8, true, true, true, 8, 10, 10>>(double* in, double* tmp, index* count_tmp, index size, index rank, double* out);
template void quickselect<double,select_config<10, 10, 8, false, true, true, 8, 9, 10>>(double* in, double* tmp, index* count_tmp, index size, index rank, double* out);
template __global__ void kernels::prefix_sum_counts<select_config<10, 10, 8, true, true, true, 8, 10, 10>>(index* in, index* out, index);
template void sampleselect_multi<float,select_config<10, 10, 8, true, true, true, 8, 10, 10, false, 16>>(float* in, float* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, float* out);
}