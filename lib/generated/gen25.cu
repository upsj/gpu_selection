#include <kernel_config.cuh>
#include <qs_launchers.cuh>
#include <qs_recursion.cuh>
#include <qs_recursion_multi.cuh>
#include <ssss_recursion.cuh>
#include <ssss_recursion_multi.cuh>
#include <ssss_launchers.cuh>
namespace gpu {
template void sampleselect_host<float,select_config<10, 10, 8, true, true, true, 8, 10, 10, false, 4>>(float* in, float* tmp, float* tree, index* count_tmp, index size, index rank, float* out);
template void sampleselect<double,select_config<10, 10, 6, false, true, true, 8, 10, 10, false, 1024>>(double* in, double* tmp, double* tree, index* count_tmp, index size, index rank, double* out);
template __device__ void kernels::masked_prefix_sum<9>(index* counts, const mask* m);
template void quickselect_multi<double,select_config<10, 5, 8, true, true, true, 8, 10, 10>>(double* in, double* tmp, index* count_tmp, index size, const index* ranks, index rank_count, double* out);
template __host__ __device__ void collect_bucket<float,select_config<10, 10, 8, true, false, false, 8, 10, 10>>(const float* data, const poracle* oracles_packed, const index* prefix_sum, float* out, index size, oracle bucket, index* atomic);
template __device__ void kernels::masked_prefix_sum<10>(index* counts, const mask* m);
template void sampleselect_multi<double,select_config<10, 10, 8, false, true, true, 8, 10, 10, true>>(double* in, double* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, double* out);
template void sampleselect_multi<float,select_config<10, 10, 8, true, true, true, 8, 10, 10, false, 8, 0>>(float* in, float* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, float* out);
template __host__ __device__ void build_searchtree<float,select_config<10, 10, 8, false, true, false, 8, 10, 10>>(const float* in, float* out, index size);
template __host__ __device__ launch_parameters get_launch_parameters<float,select_config<10, 10, 8, false, false, false, 8, 10, 10>>(index size);
template void sampleselect_host<double,select_config<10, 10, 6, true, true, true, 8, 10, 10>>(double* in, double* tmp, double* tree, index* count_tmp, index size, index rank, double* out);
}