#include <kernel_config.cuh>
#include <qs_launchers.cuh>
#include <qs_recursion.cuh>
#include <qs_recursion_multi.cuh>
#include <ssss_recursion.cuh>
#include <ssss_recursion_multi.cuh>
#include <ssss_launchers.cuh>
namespace gpu {
template void sampleselect_multi<double,select_config<10, 10, 8, false, true, true, 8, 8, 10>>(double* in, double* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, double* out);
template __host__ __device__ void collect_bucket_indirect<float,select_config<10, 10, 8, true, false, true, 8, 10, 10>>(const float* data, const poracle* oracles_packed, const index* prefix_sum, float* out, index size, const oracle* bucket, index* atomic);
template void sampleselect_multi<double,select_config<10, 10, 6, true, false, true, 8, 10, 10>>(double* in, double* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, double* out);
template void sampleselect_host<double,select_config<8, 10, 8, true, true, true, 8, 10, 10>>(double* in, double* tmp, double* tree, index* count_tmp, index size, index rank, double* out);
template __host__ __device__ void build_searchtree<float,select_config<10, 11, 9, false, true, false, 8, 10, 10>>(const float* in, float* out, index size);
template __host__ __device__ void ssss_merged<float,select_config<10, 10, 8, false, true, true, 8, 10, 10>>(const float* in, float* out, poracle* oracles, index offset, const index* ranks, index rank_offset, index rank_base, const kernels::ssss_multi_aux<float, select_config<10, 10, 8, false, true, true, 8, 10, 10>>* aux_in, kernels::ssss_multi_aux<float, select_config<10, 10, 8, false, true, true, 8, 10, 10>>* aux_outs, float* out_tree);
template void quickselect<double,select_config<10, 10, 8, true, true, true, 2, 10, 10>>(double* in, double* tmp, index* count_tmp, index size, index rank, double* out);
template void sampleselect<float,select_config<10, 10, 6, false, true, true, 8, 10, 10>>(float* in, float* tmp, float* tree, index* count_tmp, index size, index rank, float* out);
template __host__ __device__ void collect_bucket_indirect<double,select_config<10, 10, 8, false, true, true, 8, 10, 10>>(const double* data, const poracle* oracles_packed, const index* prefix_sum, double* out, index size, const oracle* bucket, index* atomic);
template void sampleselect<double,select_config<10, 10, 8, true, true, true, 8, 9, 10>>(double* in, double* tmp, double* tree, index* count_tmp, index size, index rank, double* out);
template __host__ __device__ void count_buckets<double,select_config<10, 10, 8, true, false, true, 8, 10, 10>>(const double* in, const double* tree, index* localcounts, index* counts, poracle* oracles, index size);
template __host__ __device__ void collect_bucket<float,select_config<10, 10, 8, false, false, false, 8, 10, 10>>(const float* data, const poracle* oracles_packed, const index* prefix_sum, float* out, index size, oracle bucket, index* atomic);
}