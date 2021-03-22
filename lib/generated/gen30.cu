#include <kernel_config.cuh>
#include <qs_launchers.cuh>
#include <qs_recursion.cuh>
#include <qs_recursion_multi.cuh>
#include <ssss_recursion.cuh>
#include <ssss_recursion_multi.cuh>
#include <ssss_launchers.cuh>
namespace gpu {
template __host__ __device__ launch_parameters get_launch_parameters<double,select_config<10, 8, 6, false, true, false, 8, 10, 10>>(index size);
template void sampleselect<float,select_config<10, 10, 8, true, true, true, 8, 8, 10>>(float* in, float* tmp, float* tree, index* count_tmp, index size, index rank, float* out);
template __global__ void kernels::count_buckets<double, select_config<10, 9, 7, true, true, false, 8, 10, 10>>(const double* in, const double* tree, index* counts, poracle* oracles, index size, index workcount);
template void quickselect<float,select_config<10, 5, 8, true, true, true, 8, 10, 10>>(float* in, float* tmp, index* count_tmp, index size, index rank, float* out);
template void sampleselect_host<double,select_config<10, 10, 8, true, true, true, 2, 10, 10>>(double* in, double* tmp, double* tree, index* count_tmp, index size, index rank, double* out);
template __host__ __device__ void collect_buckets<float,select_config<10, 10, 8, true, false, true, 8, 10, 10>>(const float* data, const poracle* oracles_packed, const index* block_prefix_sum, const index* bucket_out_ranges, float* out, index size, mask* buckets, index* atomic);
template __host__ __device__ launch_parameters get_launch_parameters<double,select_config<10, 10, 8, false, true, false, 8, 10, 10>>(index size);
template void sampleselect<float,select_config<10, 10, 8, false, true, true, 2, 10, 10>>(float* in, float* tmp, float* tree, index* count_tmp, index size, index rank, float* out);
template __host__ __device__ launch_parameters get_launch_parameters<float,select_config<10, 10, 8, false, true, true, 8, 10, 10>>(index size);
template void sampleselect_multi<double,select_config<10, 10, 8, false, true, true, 4, 10, 10>>(double* in, double* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, double* out);
template void sampleselect_multi<float,select_config<8, 10, 8, true, true, true, 8, 10, 10>>(float* in, float* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, float* out);
}