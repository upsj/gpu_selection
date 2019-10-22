#!/usr/bin/env python3

header = "#include <kernel_config.cuh>\n#include <qs_launchers.cuh>\n#include <qs_recursion.cuh>\n#include <qs_recursion_multi.cuh>\n#include <ssss_recursion.cuh>\n#include <ssss_recursion_multi.cuh>\n#include <ssss_launchers.cuh>\nnamespace gpu {\n"
footer = "}"

ssss_count_part_line = "template __host__ __device__ void build_searchtree<{0},{1}>(const {0}* in, {0}* out, index size);\ntemplate __host__ __device__ void count_buckets<{0},{1}>(const {0}* in, const {0}* tree, index* localcounts, index* counts, poracle* oracles, index size);\n"
ssss_part_line = "template __host__ __device__ void build_searchtree<{0},{1}>(const {0}* in, {0}* out, index size);\ntemplate __host__ __device__ void count_buckets<{0},{1}>(const {0}* in, const {0}* tree, index* localcounts, index* counts, poracle* oracles, index size);\ntemplate __host__ __device__ void collect_bucket<{0},{1}>(const {0}* data, const poracle* oracles_packed, const index* prefix_sum, {0}* out, index size, oracle bucket, index* atomic);\ntemplate __host__ __device__ void collect_bucket_indirect<{0},{1}>(const {0}* data, const poracle* oracles_packed, const index* prefix_sum, {0}* out, index size, const oracle* bucket, index* atomic);\ntemplate __host__ __device__ void collect_buckets<{0},{1}>(const {0}* data, const poracle* oracles_packed, const index* block_prefix_sum, const index* bucket_out_ranges, {0}* out, index size, mask* buckets, index* atomic);\ntemplate __host__ __device__ void ssss_merged<{0},{1}>(const {0}* in, {0}* out, poracle* oracles, index offset, const index* ranks, index rank_offset, index rank_base, const kernels::ssss_multi_aux<{0}, {1}>* aux_in, kernels::ssss_multi_aux<{0}, {1}>* aux_outs, {0}* out_tree);\n"
ssss_line = "template void sampleselect<{0},{1}>({0}* in, {0}* tmp, {0}* tree, index* count_tmp, index size, index rank, {0}* out);\ntemplate void sampleselect_multi<{0},{1}>({0}* in, {0}* tmp, index size, const index* ranks, index rank_count, index* tmp_storage, index* aux_storage, index* aux_atomic, {0}* out);\n"
ssss_part_bench_line = "template __device__ void kernels::masked_prefix_sum<{2}>(index* counts, const mask* m);\ntemplate __host__ __device__ launch_parameters get_launch_parameters<{0},{1}>(index size);\ntemplate __host__ __device__ void collect_bucket<{0},{1}>(const {0}* data, const poracle* oracles_packed, const index* prefix_sum, {0}* out, index size, oracle bucket, index* atomic);\ntemplate __host__ __device__ void build_searchtree<{0},{1}>(const {0}* in, {0}* out, index size);\ntemplate __global__ void kernels::count_buckets<{0}, {1}>(const {0}* in, const {0}* tree, index* counts, poracle* oracles, index size, index workcount);\ntemplate __global__ void kernels::prefix_sum_counts<{1}>(index* in, index* out, index);\ntemplate __global__ void kernels::reduce_counts<{1}>(const index* in, index* out, index);\n"
qs_line1 = "template __device__ __host__ void partition<{0},{1}>(const {0}* in, {0}* out, index* counts, index size, {0} pivot);\ntemplate __global__ void kernels::partition<{0},{1}>(const {0}* in, {0}* out, index* atomic, index size, {0} pivot, index workcount);\ntemplate __global__ void partition_count<{0},{1}>(const {0}* in, index* counts, index size, {0} pivot, index workcount);\ntemplate __global__ void kernels::partition_prefixsum<{1}>(index* counts, index block_count);\ntemplate __global__ void kernels::partition_distr<{0},{1}>(const {0}* in, {0}* out, const index* counts, index size, {0} pivot, index workcount);\n"
qs_line2 = "template void quickselect<{0},{1}>({0}* in, {0}* tmp, index* count_tmp, index size, index rank, {0}* out);\n"
qs_line3 = "template void quickselect_multi<{0},{1}>({0}* in, {0}* tmp, index* count_tmp, index size, const index* ranks, index rank_count, {0}* out);\n"

ssss_test1_identifier = 'TEMPLATE_PRODUCT_TEST_CASE_METHOD(test_data, "count-distribute"'
ssss_test2_identifier = 'TEMPLATE_PRODUCT_TEST_CASE_METHOD(test_data, "sampleselect"'
ssss_test3_identifier = 'TEMPLATE_PRODUCT_TEST_CASE_METHOD(test_data, "count"'
qs_test1_identifier = 'TEMPLATE_PRODUCT_TEST_CASE_METHOD(test_data, "bipartition"'
qs_test2_identifier = 'TEMPLATE_PRODUCT_TEST_CASE_METHOD(test_data, "quickselect"'
qs_test3_identifier = 'TEMPLATE_PRODUCT_TEST_CASE_METHOD(test_data, "quickselect_multi"'
qs_benchmark1_identifier = 'void qs_partition('
qs_benchmark2_identifier = 'void qs_recursive('
qs_benchmark3_identifier = 'void qs_multi('
ssss_benchmark1_identifier = 'ssss_partition_impl<'
ssss_benchmark2_identifier = 'ssss_recursive_impl<'
ssss_benchmark3_identifier = 'ssss_multi_impl<'

def extract_configs(filename, begin):
    lines = list(open(filename))
    begin_line = next(i for i in range(len(lines)) if lines[i].startswith(begin))
    end_line = next(i for i in range(begin_line, len(lines)) if '}' in lines[i])
    matches = [lines[i][lines[i].find('select_config'):lines[i].find('>') + 1] for i in range(begin_line, end_line) if 'select_config' in lines[i]]
    return matches

def extract_configs_direct(filename, linepattern):
    lines = list(open(filename))
    matches = [l[l.find('select_config'):l.find('>') + 1] for l in lines if 'select_config' in l and linepattern in l]
    return matches

instantiations = []

def generate(params, line):
    for c in params:
        instantiations.extend(line.format("float", c, c.split(',')[2].strip()).split('\n'))
        if not ("12," in c):
            instantiations.extend(line.format("double", c, c.split(',')[2].strip()).split('\n'))

qs_partition_lines = list(set(extract_configs("../app/test_qs.cu", qs_test1_identifier) + extract_configs("../app/benchmark.cu", qs_benchmark1_identifier)))
qs_lines           = list(set(extract_configs("../app/test_qs.cu", qs_test2_identifier) + extract_configs("../app/benchmark.cu", qs_benchmark2_identifier)))
qs_multi_lines     = list(set(extract_configs("../app/test_qs.cu", qs_test3_identifier) + extract_configs("../app/benchmark.cu", qs_benchmark3_identifier)))
ssss_lines = list(set(extract_configs("../app/test_ssss.cu", ssss_test2_identifier) + extract_configs_direct("../app/benchmark.cu", ssss_benchmark2_identifier) + extract_configs_direct("../app/benchmark.cu", ssss_benchmark3_identifier)))

generate(extract_configs("../app/test_ssss.cu", ssss_test3_identifier), ssss_count_part_line)
generate(extract_configs("../app/test_ssss.cu", ssss_test1_identifier), ssss_part_line)
generate(extract_configs_direct("../app/benchmark.cu", ssss_benchmark1_identifier), ssss_part_bench_line)
generate(ssss_lines, ssss_line)
generate(qs_partition_lines, qs_line1)
generate(qs_lines, qs_line2)
generate(qs_multi_lines, qs_line3)

instantiations = list(set(instantiations))

for i in range(40):
    filename = "generated/gen{}.cu".format(i)
    f = open(filename, "w")
    print(filename)
    f.write(header)
    f.write("\n".join(instantiations[i:len(instantiations):40]) + "\n")
    f.write(footer)

open("generated/gen-full.cu", "w").write("\n".join(sorted(instantiations)))
