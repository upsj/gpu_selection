#include <kernel_config.cuh>
#include <qs_launchers.cuh>
#include <qs_recursion.cuh>
#include <qs_recursion_multi.cuh>
#include <ssss_recursion.cuh>
#include <ssss_recursion_multi.cuh>
#include <ssss_launchers.cuh>
namespace gpu {
template __device__ __host__ void partition<float,select_config<10, 5, 8, false, true, true, 8, 10, 10>>(const float* in, float* out, index* counts, index size, float pivot);
template __device__ __host__ void partition<float,select_config<10, 5, 8, true, true, true, 8, 10, 10>>(const float* in, float* out, index* counts, index size, float pivot);
template void quickselect<double,select_config<10, 10, 8, true, true, true, 8, 8, 10>>(double* in, double* tmp, index* count_tmp, index size, index rank, double* out);
template void sampleselect_multi<double,select_config<10, 10, 8, true, true, true, 8, 10, 10, false, 4>>(double* in, double* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, double* out);
template void quickselect<float,select_config<10, 10, 8, false, true, true, 8, 8, 10>>(float* in, float* tmp, index* count_tmp, index size, index rank, float* out);
template __global__ void kernels::partition_distr<double,select_config<10, 5, 8, true, true, true, 8, 10, 10>>(const double* in, double* out, const index* counts, index size, double pivot, index workcount);
template __host__ __device__ void collect_bucket<float,select_config<10, 10, 8, false, false, true, 8, 10, 10>>(const float* data, const poracle* oracles_packed, const index* prefix_sum, float* out, index size, oracle bucket, index* atomic);
template __host__ __device__ launch_parameters get_launch_parameters<float,select_config<10, 10, 8, true, true, false, 8, 10, 10>>(index size);
template void quickselect<float,select_config<8, 10, 8, false, true, true, 8, 10, 10>>(float* in, float* tmp, index* count_tmp, index size, index rank, float* out);
template void sampleselect_multi<double,select_config<10, 10, 8, false, true, true, 8, 10, 10>>(double* in, double* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, double* out);
}