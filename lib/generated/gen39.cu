#include <kernel_config.cuh>
#include <qs_launchers.cuh>
#include <qs_recursion.cuh>
#include <qs_recursion_multi.cuh>
#include <ssss_recursion.cuh>
#include <ssss_recursion_multi.cuh>
#include <ssss_launchers.cuh>
namespace gpu {
template __global__ void kernels::reduce_counts<select_config<10, 12, 10, true, true, false, 8, 10, 10>>(const index* in, index* out, index);
template __device__ void kernels::masked_prefix_sum<7>(index* counts, const mask* m);
template void sampleselect<float,select_config<8, 10, 8, false, true, true, 8, 10, 10>>(float* in, float* tmp, float* tree, index* count_tmp, index size, index rank, float* out);
template __device__ __host__ void partition<float,select_config<10, 10, 8, false, true, true, 8, 10, 10>>(const float* in, float* out, index* counts, index size, float pivot);
template __host__ __device__ void collect_bucket_indirect<double,select_config<10, 10, 8, true, true, true, 8, 10, 10>>(const double* data, const poracle* oracles_packed, const index* prefix_sum, double* out, index size, const oracle* bucket, index* atomic);
template void sampleselect<float,select_config<10, 10, 8, true, true, true, 8, 10, 10>>(float* in, float* tmp, float* tree, index* count_tmp, index size, index rank, float* out);
template __host__ __device__ launch_parameters get_launch_parameters<float,select_config<10, 11, 9, true, true, false, 8, 10, 10>>(index size);
template __host__ __device__ void collect_bucket<double,select_config<10, 8, 6, false, true, false, 8, 10, 10>>(const double* data, const poracle* oracles_packed, const index* prefix_sum, double* out, index size, oracle bucket, index* atomic);
template void sampleselect<double,select_config<10, 10, 8, true, true, true, 8, 10, 10, true>>(double* in, double* tmp, double* tree, index* count_tmp, index size, index rank, double* out);
}