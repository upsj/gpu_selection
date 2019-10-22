#include <kernel_config.cuh>
#include <qs_launchers.cuh>
#include <qs_recursion.cuh>
#include <qs_recursion_multi.cuh>
#include <ssss_recursion.cuh>
#include <ssss_recursion_multi.cuh>
#include <ssss_launchers.cuh>
namespace gpu {
template __host__ __device__ launch_parameters get_launch_parameters<float,select_config<10, 12, 10, false, true, false, 8, 10, 10>>(index size);
template void sampleselect<float,select_config<10, 10, 8, true, true, true, 2, 10, 10>>(float* in, float* tmp, float* tree, index* count_tmp, index size, index rank, float* out);
template __global__ void kernels::count_buckets<double, select_config<10, 9, 7, true, true, false, 8, 10, 10>>(const double* in, const double* tree, index* counts, poracle* oracles, index size, index workcount);
template void quickselect<float,select_config<10, 10, 8, false, true, true, 4, 10, 10>>(float* in, float* tmp, index* count_tmp, index size, index rank, float* out);
template __host__ __device__ launch_parameters get_launch_parameters<float,select_config<10, 10, 8, true, true, true, 8, 10, 10>>(index size);
template void sampleselect_multi<float,select_config<10, 10, 6, true, false, true, 8, 10, 10>>(float* in, float* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, float* out);
template __host__ __device__ void collect_bucket<double,select_config<10, 9, 7, true, true, false, 8, 10, 10>>(const double* data, const poracle* oracles_packed, const index* prefix_sum, double* out, index size, oracle bucket, index* atomic);
template __global__ void kernels::partition<double,select_config<10, 10, 8, false, true, true, 8, 10, 10>>(const double* in, double* out, index* atomic, index size, double pivot, index workcount);
template void quickselect<double,select_config<10, 10, 8, false, true, true, 4, 10, 10>>(double* in, double* tmp, index* count_tmp, index size, index rank, double* out);
}