#include <kernel_config.cuh>
#include <qs_launchers.cuh>
#include <qs_recursion.cuh>
#include <qs_recursion_multi.cuh>
#include <ssss_recursion.cuh>
#include <ssss_recursion_multi.cuh>
#include <ssss_launchers.cuh>
namespace gpu {
template __host__ __device__ void collect_bucket<float,select_config<10, 10, 8, true, true, true, 8, 10, 10>>(const float* data, const poracle* oracles_packed, const index* prefix_sum, float* out, index size, oracle bucket, index* atomic);
template __host__ __device__ void collect_bucket_indirect<float,select_config<10, 10, 8, true, false, true, 8, 10, 10>>(const float* data, const poracle* oracles_packed, const index* prefix_sum, float* out, index size, const oracle* bucket, index* atomic);
template __host__ __device__ launch_parameters get_launch_parameters<float,select_config<10, 10, 8, false, false, false, 8, 10, 10>>(index size);
template void sampleselect<double,select_config<10, 10, 6, true, false, true, 8, 10, 10>>(double* in, double* tmp, double* tree, index* count_tmp, index size, index rank, double* out);
template void sampleselect<float,select_config<10, 10, 7, true, true, true, 8, 10, 10>>(float* in, float* tmp, float* tree, index* count_tmp, index size, index rank, float* out);
template __global__ void kernels::count_buckets<float, select_config<10, 12, 10, true, true, false, 8, 10, 10>>(const float* in, const float* tree, index* counts, poracle* oracles, index size, index workcount);
template __device__ __host__ void partition<float,select_config<10, 10, 8, true, true, true, 8, 10, 10>>(const float* in, float* out, index* counts, index size, float pivot);
template __global__ void kernels::partition<float,select_config<10, 5, 8, false, true, true, 8, 10, 10>>(const float* in, float* out, index* atomic, index size, float pivot, index workcount);
template __host__ __device__ void collect_bucket<float,select_config<10, 12, 10, false, true, false, 8, 10, 10>>(const float* data, const poracle* oracles_packed, const index* prefix_sum, float* out, index size, oracle bucket, index* atomic);
template __host__ __device__ void collect_bucket_indirect<double,select_config<10, 10, 8, true, false, true, 8, 10, 10>>(const double* data, const poracle* oracles_packed, const index* prefix_sum, double* out, index size, const oracle* bucket, index* atomic);
}