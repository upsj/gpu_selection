#include <kernel_config.cuh>
#include <qs_launchers.cuh>
#include <qs_recursion.cuh>
#include <qs_recursion_multi.cuh>
#include <ssss_recursion.cuh>
#include <ssss_recursion_multi.cuh>
#include <ssss_launchers.cuh>
namespace gpu {
template __host__ __device__ launch_parameters get_launch_parameters<float,select_config<10, 10, 8, false, true, false, 8, 10, 10>>(index size);
template void sampleselect_multi<float,select_config<10, 10, 8, false, true, true, 8, 9, 10>>(float* in, float* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, float* out);
template __host__ __device__ void build_searchtree<float,select_config<10, 9, 7, true, true, false, 8, 10, 10>>(const float* in, float* out, index size);
template __host__ __device__ void collect_bucket<float,select_config<10, 10, 8, true, true, false, 8, 10, 10>>(const float* data, const poracle* oracles_packed, const index* prefix_sum, float* out, index size, oracle bucket, index* atomic);
template void sampleselect_multi<float,select_config<10, 10, 8, true, false, true, 8, 10, 10>>(float* in, float* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, float* out);
template __host__ __device__ void collect_bucket<double,select_config<10, 10, 8, true, true, true, 8, 10, 10>>(const double* data, const poracle* oracles_packed, const index* prefix_sum, double* out, index size, oracle bucket, index* atomic);
template __host__ __device__ void build_searchtree<double,select_config<10, 8, 6, false, true, false, 8, 10, 10>>(const double* in, double* out, index size);
template void sampleselect_multi<double,select_config<10, 10, 6, false, true, true, 8, 10, 10>>(double* in, double* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, double* out);
template void sampleselect_multi<float,select_config<10, 10, 8, true, true, true, 2, 10, 10>>(float* in, float* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, float* out);
}