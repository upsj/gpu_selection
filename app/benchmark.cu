/*
 * Parallel selection algorithm on GPUs
 * Copyright (c) 2018-2019 Tobias Ribizel (oss@ribizel.de)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */
#include "catch.hpp"
#include "test_fixture.cuh"
#include <launcher_fwd.cuh>
#include <verification.hpp>
#include <cpu_reference.hpp>
#include <cuda_timer.cuh>
#include <kernel_config.cuh>
#include <limits>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

namespace gpu {

constexpr auto num_runs = 10;

template <typename T, typename Config>
void qs_recursive_impl(std::string name, index n, index d, basic_test_data<std::pair<T,int>>& data, cuda_timer& timer, T ref) {
    INFO(name);
    timer.timed(name, num_runs, [&](auto event) {
        data.reset();
        event(0);
        quickselect<T, Config>(data.gpu_data, data.gpu_data_tmp, data.gpu_count_tmp, data.size, data.rank,
                               data.gpu_data_out);
        event(1);
    });
    data.copy_from_gpu();
    CHECK(ref == data.data_out[0]);
}

template <typename T, typename Config>
void qs_multi_impl(std::string name, index n, index d, basic_test_data<std::pair<T,int>>& data, cuda_timer& timer, const std::vector<T>& ref) {
    INFO(name);
    timer.timed(name, num_runs, [&](auto event) {
        data.reset();
        event(0);
        quickselect_multi<T, Config>(data.gpu_data, data.gpu_data_tmp, data.gpu_count_tmp, data.size, data.gpu_ranks, ref.size(), data.gpu_data_out);
        event(1);
    });
    data.copy_from_gpu();
    CHECK(std::equal(ref.begin(), ref.end(), data.data_out.begin()));
}

template <typename T, typename Config>
void qs_partition_impl(std::string name, index n, index d, basic_test_data<std::pair<T,int>>& data, cuda_timer& timer) {
    INFO(name);
    timer.timed(name, num_runs, [&](auto event) {
        data.reset();
        auto bsize = Config::algorithm::max_block_size;
        auto nblocks = min(ceil_div(n, bsize), Config::algorithm::max_block_count);
        auto per_thread = ceil_div(n, nblocks * bsize);
        if (Config::algorithm::shared_memory) {
            event(0);
            kernels::partition_count<T, Config><<<nblocks, bsize>>>(data.gpu_data, data.gpu_count_out, n,
                                                                    data.pivot, per_thread);
            event(1);
            kernels::partition_prefixsum<Config>
                    <<<1, Config::algorithm::max_block_count>>>(data.gpu_count_out, nblocks);
            event(2);
            kernels::partition_distr<T, Config><<<nblocks, bsize>>>(data.gpu_data, data.gpu_data_out,
                                                                    data.gpu_count_out, n, data.pivot,
                                                                    per_thread);
            event(3);
        } else {
            event(0);
            kernels::partition<T, Config><<<nblocks, bsize>>>(data.gpu_data, data.gpu_data_out,
                                                              data.gpu_count_out, n, data.pivot, per_thread);
            event(1);
        }
    });
    data.copy_from_gpu();
    // Check correctness: counts
    auto lsize = data.count_out[0];
    auto rsize = data.count_out[1];
    CAPTURE(lsize);
    CAPTURE(rsize);
    CHECK(lsize + rsize == data.size);
    // Check correctness: partition
    auto counts = verification::count_mispartitioned(data.data_out, lsize, data.pivot);
    auto lcount = counts.first;
    auto rcount = counts.second;
    CHECK(lcount == 0);
    CHECK(rcount == 0);
}

template <typename T, typename Config>
void ssss_recursive_impl(std::string name, index n, index d, basic_test_data<std::pair<T,int>>& data, cuda_timer& timer, T ref) {
    INFO(name);
    timer.timed(name, num_runs, [&](auto event) {
        data.reset();
        event(0);
        sampleselect<T, Config>(data.gpu_data, data.gpu_data_tmp, data.gpu_tree, data.gpu_count_tmp, data.size,
                                data.rank, data.gpu_data_out);
        event(1);
    });
    data.copy_from_gpu();
    CHECK(ref == data.data_out[0]);
}

template <typename T, typename Config>
void ssss_host_impl(std::string name, index n, index d, basic_test_data<std::pair<T,int>>& data, cuda_timer& timer, T ref) {
    INFO(name);
    timer.timed(name, num_runs, [&](auto event) {
        data.reset();
        event(0);
        sampleselect_host<T, Config>(data.gpu_data, data.gpu_data_tmp, data.gpu_tree, data.gpu_count_tmp, data.size,
                                data.rank, data.gpu_data_out);
        event(1);
    });
    data.copy_from_gpu();
    CHECK(ref == data.data_out[0]);
}

template <typename T, typename Config>
void ssss_multi_impl(std::string name, index n, index d, basic_test_data<std::pair<T,int>>& data, cuda_timer& timer, const std::vector<T>& ref) {
    INFO(name);
    timer.timed(name, num_runs, [&](auto event) {
        data.reset();
        event(0);
        sampleselect_multi<T, Config>(data.gpu_data, data.gpu_data_tmp, n, data.gpu_ranks, ref.size(), data.gpu_count_tmp, data.gpu_aux, data.gpu_atomic, data.gpu_data_out);
        event(1);
    });
    data.copy_from_gpu();
    CHECK(std::equal(ref.begin(), ref.end(), data.data_out.begin()));
}

template <typename T, typename Config>
void ssss_partition_impl(std::string name, index n, index d, basic_test_data<std::pair<T,int>>& data, cuda_timer& timer) {
    auto params = get_launch_parameters<T, Config>(n);
    // always extract the last bucket, the choice should not matter in the end
    auto bucket = Config::searchtree::width - 1;
    INFO(name);
    timer.timed(name, num_runs, [&](auto event) {
        data.reset();
        event(0);
        build_searchtree<T, Config>(data.gpu_data, data.gpu_tree, data.size);
        event(1);
        if (Config::algorithm::shared_memory) {
            kernels::count_buckets<T, Config><<<params.block_count, params.block_size>>>(
                    data.gpu_data, data.gpu_tree, data.gpu_count_tmp, data.gpu_oracles, n, params.work_per_thread);
            event(2);
            constexpr auto reduce_bsize =
                    min(Config::searchtree::width, Config::algorithm::max_block_count);
            constexpr auto reduce_blocks = ceil_div(Config::searchtree::width, reduce_bsize);
            if (Config::algorithm::write) {
                kernels::prefix_sum_counts<Config>
                        <<<reduce_blocks, reduce_bsize>>>(data.gpu_count_tmp, data.gpu_count_out, params.block_count);
                event(3);
                collect_bucket<T, Config>(data.gpu_data, data.gpu_oracles, data.gpu_count_tmp,
                        data.gpu_data_out, n, bucket, data.gpu_atomic);
            } else {
                kernels::reduce_counts<Config>
                        <<<reduce_blocks, reduce_bsize>>>(data.gpu_count_tmp, data.gpu_count_out, params.block_count);
                event(3);
            }
            event(4);
        } else {
            kernels::count_buckets<T, Config><<<params.block_count, params.block_size>>>(
                    data.gpu_data, data.gpu_tree, data.gpu_count_out, data.gpu_oracles, n, params.work_per_thread);
            event(2);
            event(3);
            if (Config::algorithm::write) {
                collect_bucket<T, Config>(data.gpu_data, data.gpu_oracles, data.gpu_count_tmp,
                        data.gpu_data_out, n, bucket, data.gpu_atomic);
                event(4);
            }
        }
    });
    data.copy_from_gpu();
    // check correctness: tree
    auto tree = data.tree;
    tree.resize(Config::searchtree::size);
    auto ref_tree =
            cpu::build_searchtree(data.data, Config::sample::size, Config::searchtree::size);
    CHECK(tree == ref_tree);
    // check correctness: counts & oracles
    auto counts = data.count_out;
    counts.resize(Config::searchtree::width);
    auto ref_ssss = cpu::ssss(data.data, tree, Config::algorithm::write);
    auto ref_counts = ref_ssss.first;
    CHECK(counts == ref_counts);
    if (Config::algorithm::write) {
        auto ref_oracles = ref_ssss.second;
        CHECK(unpack(data.oracles, data.size) == ref_oracles);
        // check correctness: extracted bucket
        if (!Config::algorithm::shared_memory) {
            CHECK(data.atomic[0] == data.count_out[bucket]);
        }
        auto out = data.data_out;
        out.resize(counts[bucket]);
        // add sentinel
        tree.push_back(std::numeric_limits<T>::infinity());
        auto lb = tree[bucket + Config::searchtree::width - 1];
        auto ub = tree[bucket + Config::searchtree::width];
        auto count_misplaced = verification::count_not_in_bucket(out, lb, ub);
        CHECK(count_misplaced == 0);
    }
    // compute approximation error
    std::partial_sum(counts.begin(), counts.end(), counts.begin());
    auto res_bucket = std::distance(counts.begin(),
                                    std::upper_bound(counts.begin(), counts.end(), data.rank));
    std::cout << name << ';' << counts[res_bucket] << ';' << data.rank << '\n';
}

template<index size_log2>
__global__ void masked_prefixsum_helper(index* data, const mask* m) {
    constexpr auto size = 1 << size_log2;
    __shared__ index sh_data[size];
    auto idx = threadIdx.x;
    if (idx < size) {
        sh_data[idx] = data[idx];
    }
    __syncthreads();
    kernels::masked_prefix_sum<size_log2>(sh_data, m);
    __syncthreads();
    if (idx < size) {
        data[idx] = sh_data[idx];
    }
}

template <typename T, typename Config>
void ssss_partition_multi_impl(std::string name, index n, index d, basic_test_data<std::pair<T,int>>& data, cuda_timer& timer) {
    auto params = get_launch_parameters<T, Config>(n);
    static_assert(Config::algorithm::write, "multi without write");
    // select every second bucket
    auto mask_size = ceil_div(Config::searchtree::width, (sizeof(mask) * 8));
    auto alternating_mask = mask(0x55555555ull);
    std::vector<mask> bucket_mask(mask_size, alternating_mask);
    data.gpu_bucket_mask.copy_from(bucket_mask);
    INFO(name);
    timer.timed(name, num_runs, [&](auto event) {
        data.reset();
        event(0);
        build_searchtree<T, Config>(data.gpu_data, data.gpu_tree, data.size);
        event(1);
        if (Config::algorithm::shared_memory) {
            kernels::count_buckets<T, Config><<<params.block_count, params.block_size>>>(
                    data.gpu_data, data.gpu_tree, data.gpu_count_tmp, data.gpu_oracles, n, params.work_per_thread);
            event(2);
            constexpr auto reduce_bsize =
                    min(Config::searchtree::width, Config::algorithm::max_block_count);
            constexpr auto reduce_blocks = ceil_div(Config::searchtree::width, reduce_bsize);
            kernels::prefix_sum_counts<Config>
                    <<<reduce_blocks, reduce_bsize>>>(data.gpu_count_tmp, data.gpu_count_out, params.block_count);
            masked_prefixsum_helper<Config::searchtree::height><<<1, Config::searchtree::width>>>(data.gpu_count_out, data.gpu_bucket_mask);
            event(3);
            collect_buckets<T, Config>(data.gpu_data, data.gpu_oracles, data.gpu_count_tmp,
                    data.gpu_count_out, data.gpu_data_out, n, data.gpu_bucket_mask, data.gpu_atomic);
            event(4);
        } else {
            kernels::count_buckets<T, Config><<<params.block_count, params.block_size>>>(
                    data.gpu_data, data.gpu_tree, data.gpu_count_out, data.gpu_oracles, n, params.work_per_thread);
            event(2);
            masked_prefixsum_helper<Config::searchtree::height><<<1, Config::searchtree::width>>>(data.gpu_count_out, data.gpu_bucket_mask);
            event(3);
            collect_buckets<T, Config>(data.gpu_data, data.gpu_oracles, data.gpu_count_tmp,
                    data.gpu_count_out, data.gpu_data_out, n, data.gpu_bucket_mask, data.gpu_atomic);
            event(4);
        }
    });
    data.copy_from_gpu();
    // check correctness: tree
    auto tree = data.tree;
    tree.resize(Config::searchtree::size);
    auto ref_tree =
            cpu::build_searchtree(data.data, Config::sample::size, Config::searchtree::size);
    CHECK(tree == ref_tree);
    // check correctness: counts & oracles
    auto counts = data.count_out;
    counts.resize(Config::searchtree::width);
    auto ref_ssss = cpu::ssss(data.data, tree, Config::algorithm::write);
    auto ref_counts = ref_ssss.first;
    auto ref_oracles = ref_ssss.second;
    CHECK(unpack(data.oracles, data.size) == ref_oracles);    
    auto ref_pair = cpu::masked_prefix_sum(ref_counts, bucket_mask);
    CHECK(counts == ref_pair.first);
    data.data_out.resize(ref_pair.second);
    ref_pair.first.push_back(ref_pair.second);
    auto count_misplaced_multi = verification::count_not_in_buckets(data.data_out, ref_pair.first, ref_tree);
    std::vector<index> zeros(Config::searchtree::width);
    CHECK(count_misplaced_multi == zeros);
}

template <typename T>
void qs_recursive(std::string name, index n, index d, basic_test_data<std::pair<T,int>>& data, cuda_timer& timer, T ref) {
    qs_recursive_impl<T, select_config<10, 10, 8, true, true, true, 8, 10, 10>>(name + "-sdefault", n, d, data, timer, ref);
    qs_recursive_impl<T, select_config<10, 10, 8, false, true, true, 8, 10, 10>>(name + "-gdefault", n, d, data, timer, ref);
    // Different base case sizes
    qs_recursive_impl<T, select_config<9, 10, 8, true, true, true, 8, 10, 10>>(name + "-sb9", n, d, data, timer, ref);
    qs_recursive_impl<T, select_config<8, 10, 8, true, true, true, 8, 10, 10>>(name + "-sb8", n, d, data, timer, ref);
    qs_recursive_impl<T, select_config<9, 10, 8, false, true, true, 8, 10, 10>>(name + "-gb9", n, d, data, timer, ref);
    qs_recursive_impl<T, select_config<8, 10, 8, false, true, true, 8, 10, 10>>(name + "-gb8", n, d, data, timer, ref);
    // Different block sizes
    qs_recursive_impl<T, select_config<10, 10, 8, true, true, true, 8, 9, 10>>(name + "-sbl9", n, d, data, timer, ref);
    qs_recursive_impl<T, select_config<10, 10, 8, true, true, true, 8, 8, 10>>(name + "-sbl8", n, d, data, timer, ref);
    qs_recursive_impl<T, select_config<10, 10, 8, false, true, true, 8, 9, 10>>(name + "-gbl9", n, d, data, timer, ref);
    qs_recursive_impl<T, select_config<10, 10, 8, false, true, true, 8, 8, 10>>(name + "-gbl8", n, d, data, timer, ref);
    // Different unrolling widths
    qs_recursive_impl<T, select_config<10, 10, 8, true, true, true, 4, 10, 10>>(name + "-su4", n, d, data, timer, ref);
    qs_recursive_impl<T, select_config<10, 10, 8, true, true, true, 2, 10, 10>>(name + "-su2", n, d, data, timer, ref);
    qs_recursive_impl<T, select_config<10, 10, 8, false, true, true, 4, 10, 10>>(name + "-gu4", n, d, data, timer, ref);
    qs_recursive_impl<T, select_config<10, 10, 8, false, true, true, 2, 10, 10>>(name + "-gu2", n, d, data, timer, ref);
}

template <typename T>
void qs_multi(std::string name, index n, index d, basic_test_data<std::pair<T,int>>& data, cuda_timer& timer, const std::vector<T>& ref) {
    qs_multi_impl<T, select_config<10, 10, 8, true, true, true, 8, 10, 10>>(name + "-sdefault", n, d, data, timer, ref);
    qs_multi_impl<T, select_config<10, 10, 8, false, true, true, 8, 10, 10>>(name + "-gdefault", n, d, data, timer, ref);
}

template <typename T>
void qs_partition(std::string name, index n, index d, basic_test_data<std::pair<T,int>>& data, cuda_timer& timer) {
    qs_partition_impl<T, select_config<10, 10, 8, true, true, true, 8, 10, 10>>(name + "-sdefault", n, d, data, timer);
    qs_partition_impl<T, select_config<10, 10, 8, false, true, true, 8, 10, 10>>(name + "-gdefault", n, d, data, timer);
}

template <typename T>
void ssss_recursive(std::string name, index n, index d, basic_test_data<std::pair<T,int>>& data, cuda_timer& timer, T ref) {
    ssss_recursive_impl<T, select_config<10, 10, 8, true, true, true, 8, 10, 10>>(name + "-sdefault", n, d, data, timer, ref);
    ssss_recursive_impl<T, select_config<10, 10, 8, false, true, true, 8, 10, 10>>(name + "-gdefault", n, d, data, timer, ref);
    if (d >= n) {
        // bucket select
        ssss_recursive_impl<T, select_config<10, 10, 8, true, true, true, 8, 10, 10, true>>(name + "-sbucket", n, d, data, timer, ref);
        ssss_recursive_impl<T, select_config<10, 10, 8, false, true, true, 8, 10, 10, true>>(name + "-gbucket", n, d, data, timer, ref);
    }
    // Different base case sizes
    ssss_recursive_impl<T, select_config<9, 10, 8, true, true, true, 8, 10, 10>>(name + "-sb9", n, d, data, timer, ref);
    ssss_recursive_impl<T, select_config<8, 10, 8, true, true, true, 8, 10, 10>>(name + "-sb8", n, d, data, timer, ref);
    ssss_recursive_impl<T, select_config<9, 10, 8, false, true, true, 8, 10, 10>>(name + "-gb9", n, d, data, timer, ref);
    ssss_recursive_impl<T, select_config<8, 10, 8, false, true, true, 8, 10, 10>>(name + "-gb8", n, d, data, timer, ref);
    // Different searchtree sizes
    ssss_recursive_impl<T, select_config<10, 10, 7, true, true, true, 8, 10, 10>>(name + "-st7", n, d, data, timer, ref);
    ssss_recursive_impl<T, select_config<10, 10, 6, true, true, true, 8, 10, 10>>(name + "-st6", n, d, data, timer, ref);
    ssss_recursive_impl<T, select_config<10, 10, 7, false, true, true, 8, 10, 10>>(name + "-gt7", n, d, data, timer, ref);
    ssss_recursive_impl<T, select_config<10, 10, 6, false, true, true, 8, 10, 10>>(name + "-gt6", n, d, data, timer, ref);
    // Different block sizes
    ssss_recursive_impl<T, select_config<10, 10, 8, true, true, true, 8, 9, 10>>(name + "-sbl9", n, d, data, timer, ref);
    ssss_recursive_impl<T, select_config<10, 10, 8, true, true, true, 8, 8, 10>>(name + "-sbl8", n, d, data, timer, ref);
    ssss_recursive_impl<T, select_config<10, 10, 8, false, true, true, 8, 9, 10>>(name + "-gbl9", n, d, data, timer, ref);
    ssss_recursive_impl<T, select_config<10, 10, 8, false, true, true, 8, 8, 10>>(name + "-gbl8", n, d, data, timer, ref);
    // Different unrolling widths
    ssss_recursive_impl<T, select_config<10, 10, 8, true, true, true, 4, 10, 10>>(name + "-su4", n, d, data, timer, ref);
    ssss_recursive_impl<T, select_config<10, 10, 8, true, true, true, 2, 10, 10>>(name + "-su2", n, d, data, timer, ref);
    ssss_recursive_impl<T, select_config<10, 10, 8, false, true, true, 4, 10, 10>>(name + "-gu4", n, d, data, timer, ref);
    ssss_recursive_impl<T, select_config<10, 10, 8, false, true, true, 2, 10, 10>>(name + "-gu2", n, d, data, timer, ref);
}

template <typename T>
void ssss_host(std::string name, index n, index d, basic_test_data<std::pair<T,int>>& data, cuda_timer& timer, T ref) {
    ssss_host_impl<T, select_config<10, 10, 8, true, true, true, 8, 10, 10>>(name + "-sdefault", n, d, data, timer, ref);
    ssss_host_impl<T, select_config<10, 10, 8, false, true, true, 8, 10, 10>>(name + "-gdefault", n, d, data, timer, ref);
    if (d >= n) {
        // bucket select
        ssss_host_impl<T, select_config<10, 10, 8, true, true, true, 8, 10, 10, true>>(name + "-sbucket", n, d, data, timer, ref);
        ssss_host_impl<T, select_config<10, 10, 8, false, true, true, 8, 10, 10, true>>(name + "-gbucket", n, d, data, timer, ref);
    }
    // Different base case sizes
    ssss_host_impl<T, select_config<9, 10, 8, true, true, true, 8, 10, 10>>(name + "-sb9", n, d, data, timer, ref);
    ssss_host_impl<T, select_config<8, 10, 8, true, true, true, 8, 10, 10>>(name + "-sb8", n, d, data, timer, ref);
    ssss_host_impl<T, select_config<9, 10, 8, false, true, true, 8, 10, 10>>(name + "-gb9", n, d, data, timer, ref);
    ssss_host_impl<T, select_config<8, 10, 8, false, true, true, 8, 10, 10>>(name + "-gb8", n, d, data, timer, ref);
    // Different searchtree sizes
    ssss_host_impl<T, select_config<10, 10, 7, true, true, true, 8, 10, 10>>(name + "-st7", n, d, data, timer, ref);
    ssss_host_impl<T, select_config<10, 10, 6, true, true, true, 8, 10, 10>>(name + "-st6", n, d, data, timer, ref);
    ssss_host_impl<T, select_config<10, 10, 7, false, true, true, 8, 10, 10>>(name + "-gt7", n, d, data, timer, ref);
    ssss_host_impl<T, select_config<10, 10, 6, false, true, true, 8, 10, 10>>(name + "-gt6", n, d, data, timer, ref);
    // Different block sizes
    ssss_host_impl<T, select_config<10, 10, 8, true, true, true, 8, 9, 10>>(name + "-sbl9", n, d, data, timer, ref);
    ssss_host_impl<T, select_config<10, 10, 8, true, true, true, 8, 8, 10>>(name + "-sbl8", n, d, data, timer, ref);
    ssss_host_impl<T, select_config<10, 10, 8, false, true, true, 8, 9, 10>>(name + "-gbl9", n, d, data, timer, ref);
    ssss_host_impl<T, select_config<10, 10, 8, false, true, true, 8, 8, 10>>(name + "-gbl8", n, d, data, timer, ref);
    // Different unrolling widths
    ssss_host_impl<T, select_config<10, 10, 8, true, true, true, 4, 10, 10>>(name + "-su4", n, d, data, timer, ref);
    ssss_host_impl<T, select_config<10, 10, 8, true, true, true, 2, 10, 10>>(name + "-su2", n, d, data, timer, ref);
    ssss_host_impl<T, select_config<10, 10, 8, false, true, true, 4, 10, 10>>(name + "-gu4", n, d, data, timer, ref);
    ssss_host_impl<T, select_config<10, 10, 8, false, true, true, 2, 10, 10>>(name + "-gu2", n, d, data, timer, ref);
}

template <typename T>
void ssss_multi(std::string name, index n, index d, basic_test_data<std::pair<T,int>>& data, cuda_timer& timer, const std::vector<T>& ref) {
    ssss_multi_impl<T, select_config<10, 10, 8, true, true, true, 8, 10, 10>>(name + "-sdefault", n, d, data, timer, ref);
    ssss_multi_impl<T, select_config<10, 10, 8, false, true, true, 8, 10, 10>>(name + "-gdefault", n, d, data, timer, ref);
    ssss_multi_impl<T, select_config<9, 10, 8, true, true, true, 8, 10, 10>>(name + "-sbase9", n, d, data, timer, ref);
    ssss_multi_impl<T, select_config<9, 10, 8, false, true, true, 8, 10, 10>>(name + "-gbase9", n, d, data, timer, ref);
    ssss_multi_impl<T, select_config<10, 10, 8, true, true, true, 8, 10, 10, false, 8, 0>>(name + "-slocal0", n, d, data, timer, ref);
    ssss_multi_impl<T, select_config<10, 10, 8, false, true, true, 8, 10, 10, false, 8, 0>>(name + "-glocal0", n, d, data, timer, ref);
    if (d >= n) {
        // bucket select
        ssss_multi_impl<T, select_config<10, 10, 8, true, true, true, 8, 10, 10, true>>(name + "-sbucket", n, d, data, timer, ref);
        ssss_multi_impl<T, select_config<10, 10, 8, false, true, true, 8, 10, 10, true>>(name + "-gbucket", n, d, data, timer, ref);
    }
}

template <typename T>
void ssss_partition(std::string name, index n, index d, basic_test_data<std::pair<T,int>>& data, cuda_timer& timer) {
    ssss_partition_impl<T, select_config<10, 10, 8, true, true, true, 8, 10, 10>>(name + "-s", n, d, data, timer);
    ssss_partition_impl<T, select_config<10, 10, 8, false, true, true, 8, 10, 10>>(name + "-g", n, d, data, timer);
    ssss_partition_multi_impl<T, select_config<10, 10, 8, true, true, true, 8, 10, 10>>(name + "-smulti", n, d, data, timer);
    ssss_partition_multi_impl<T, select_config<10, 10, 8, false, true, true, 8, 10, 10>>(name + "-gmulti", n, d, data, timer);
    // Without writing
    ssss_partition_impl<T, select_config<10, 10, 8, true, true, false, 8, 10, 10>>(name + "-csc", n, d, data, timer);
    ssss_partition_impl<T, select_config<10, 10, 8, false, true, false, 8, 10, 10>>(name + "-cgc", n, d, data, timer);
    ssss_partition_impl<T, select_config<10, 10, 8, true, false, false, 8, 10, 10>>(name + "-csn", n, d, data, timer);
    ssss_partition_impl<T, select_config<10, 10, 8, false, false, false, 8, 10, 10>>(name + "-cgn", n, d, data, timer);
    // Larger searchtree sizes and oversampling factors
    ssss_partition_impl<T, select_config<10, 11, 9, true, true, false, 8, 10, 10>>(name + "-cst9", n, d, data, timer);
    if (sizeof(T) <= 4) ssss_partition_impl<T, select_config<10, 12, 10, true, true, false, 8, 10, 10>>(name + "-cst10", n, d, data, timer);
    ssss_partition_impl<T, select_config<10, 11, 9, false, true, false, 8, 10, 10>>(name + "-cgt9", n, d, data, timer);
    if (sizeof(T) <= 4) ssss_partition_impl<T, select_config<10, 12, 10, false, true, false, 8, 10, 10>>(name + "-cgt10", n, d, data, timer);
}

TEMPLATE_TEST_CASE("full", "[.],[full]", float, double) {
    using T = TestType;
    auto n = GENERATE(as<index>{}, 65536, 262144, 524288, 1048576, 2097152, 4194304, 8388608,
                      16777216, 33554432, 67108864, 134217728);
    auto d = GENERATE(as<index>{}, 1 << 30, 1024, 128, 16, 1);
    auto seed = GENERATE(take(2, Catch::Generators::random(0, 1000000)));
    basic_test_data<std::pair<T, int>> data{n, d, index(seed)};
    CAPTURE(n);
    CAPTURE(d);
    CAPTURE(seed);
    auto ref = verification::nth_element(data.data, data.rank);
    cuda_timer timer{std::cerr};
    auto suffix = "-" + std::to_string(n) + "-" + std::to_string(d) + "-" + typeid(T).name();
    if (d >= n) {
        qs_recursive<T>("quickselect" + suffix, n, d, data, timer, ref);
        qs_partition<T>("bipartition" + suffix, n, d, data, timer);
        auto ranks1 = build_ranks_uniform(n, 32);
        auto refs1 = verification::nth_elements(data.data, ranks1);
        auto ranks2 = build_ranks_clustered(n);
        auto refs2 = verification::nth_elements(data.data, ranks2);
        data.gpu_ranks.copy_from(ranks1);
        qs_multi<T>("quickselectmultiuniform" + suffix, n, d, data, timer, refs1);
        ssss_multi<T>("sampleselectmultiuniform" + suffix, n, d, data, timer, refs1);
        data.gpu_ranks.copy_from(ranks2);
        qs_multi<T>("quickselectmulticlustered" + suffix, n, d, data, timer, refs2);
        ssss_multi<T>("sampleselectmulticlustered" + suffix, n, d, data, timer, refs2);
    }
    ssss_host<T>("sampleselect_host" + suffix, n, d, data, timer, ref);
    ssss_recursive<T>("sampleselect" + suffix, n, d, data, timer, ref);
    ssss_partition<T>("kpartition" + suffix, n, d, data, timer);    
}

TEMPLATE_TEST_CASE("full-multionly", "[.],[full-multionly]", float, double) {
    using T = TestType;
    auto n = GENERATE(as<index>{}, 65536, 262144, 524288, 1048576, 2097152, 4194304, 8388608,
                      16777216, 33554432, 67108864, 134217728);
    auto d = GENERATE(as<index>{}, 1 << 30);
    auto seed = GENERATE(take(2, Catch::Generators::random(0, 1000000)));
    basic_test_data<std::pair<T, int>> data{n, d, index(seed)};
    CAPTURE(n);
    CAPTURE(d);
    CAPTURE(seed);
    auto ref = verification::nth_element(data.data, data.rank);
    cuda_timer timer{std::cerr};
    auto suffix = "-" + std::to_string(n) + "-" + std::to_string(d) + "-" + typeid(T).name();
    auto ranks1 = build_ranks_uniform(n, 32);
    auto refs1 = verification::nth_elements(data.data, ranks1);
    auto ranks2 = build_ranks_clustered(n);
    auto refs2 = verification::nth_elements(data.data, ranks2);
    data.gpu_ranks.copy_from(ranks1);
    qs_multi<T>("quickselectmultiuniform" + suffix, n, d, data, timer, refs1);
    ssss_multi<T>("sampleselectmultiuniform" + suffix, n, d, data, timer, refs1);
    data.gpu_ranks.copy_from(ranks2);
    qs_multi<T>("quickselectmulticlustered" + suffix, n, d, data, timer, refs2);
    ssss_multi<T>("sampleselectmulticlustered" + suffix, n, d, data, timer, refs2);
}

TEMPLATE_TEST_CASE("approx", "[.],[approx]", float, double) {
    using T = TestType;
    auto n = GENERATE(as<index>{}, 65536, 262144, 524288, 1048576, 2097152, 4194304, 8388608,
                      16777216, 33554432, 67108864, 134217728);
    auto d = index{1 << 30};
    auto seed = GENERATE(take(10, Catch::Generators::random(0, 1000000)));
    basic_test_data<std::pair<T, int>> data{n, d, index(seed)};
    CAPTURE(n);
    CAPTURE(d);
    CAPTURE(seed);
    auto ref = verification::nth_element(data.data, data.rank);
    cuda_timer timer{std::cerr};
    auto suffix = "-" + std::to_string(n) + "-" + std::to_string(d) + "-" + typeid(T).name();
    auto name = "kpartition" + suffix;
    ssss_partition_impl<T, select_config<10, 10, 8, true, true, false, 8, 10, 10>>(name + "-csc", n, d, data, timer);
    ssss_partition_impl<T, select_config<10, 8, 6, true, true, false, 8, 10, 10>>(name + "-cst6", n, d, data, timer);
    ssss_partition_impl<T, select_config<10, 9, 7, true, true, false, 8, 10, 10>>(name + "-cst7", n, d, data, timer);
    ssss_partition_impl<T, select_config<10, 11, 9, true, true, false, 8, 10, 10>>(name + "-cst9", n, d, data, timer);
    if (sizeof(T) <= 4) ssss_partition_impl<T, select_config<10, 12, 10, true, true, false, 8, 10, 10>>(name + "-cst10", n, d, data, timer);
    name = "sampleselect" + suffix;
    ssss_recursive_impl<T, select_config<10, 10, 8, true, true, true, 8, 10, 10>>(name + "-sdefault", n, d, data, timer, ref);
}

TEMPLATE_TEST_CASE("approx-g", "[.],[approx-g]", float, double) {
    using T = TestType;
    auto n = GENERATE(as<index>{}, 65536, 262144, 524288, 1048576, 2097152, 4194304, 8388608,
                      16777216, 33554432, 67108864, 134217728);
    auto d = index{1 << 30};
    auto seed = GENERATE(take(10, Catch::Generators::random(0, 1000000)));
    basic_test_data<std::pair<T, int>> data{n, d, index(seed)};
    CAPTURE(n);
    CAPTURE(d);
    CAPTURE(seed);
    auto ref = verification::nth_element(data.data, data.rank);
    cuda_timer timer{std::cerr};
    auto suffix = "-" + std::to_string(n) + "-" + std::to_string(d) + "-" + typeid(T).name();
    auto name = "kpartition" + suffix;
    ssss_partition_impl<T, select_config<10, 10, 8, false, true, false, 8, 10, 10>>(name + "-csc", n, d, data, timer);
    ssss_partition_impl<T, select_config<10, 8, 6, false, true, false, 8, 10, 10>>(name + "-cst6", n, d, data, timer);
    ssss_partition_impl<T, select_config<10, 9, 7, false, true, false, 8, 10, 10>>(name + "-cst7", n, d, data, timer);
    ssss_partition_impl<T, select_config<10, 11, 9, false, true, false, 8, 10, 10>>(name + "-cst9", n, d, data, timer);
    if (sizeof(T) <= 4) ssss_partition_impl<T, select_config<10, 12, 10, false, true, false, 8, 10, 10>>(name + "-cst10", n, d, data, timer);
    name = "sampleselect" + suffix;
    ssss_recursive_impl<T, select_config<10, 10, 8, false, true, true, 8, 10, 10>>(name + "-sdefault", n, d, data, timer, ref);
}


TEMPLATE_TEST_CASE("multi", "[.],[multi]", float, double) {
    using T = TestType;
    auto n = GENERATE(as<index>{}, 134217728);
    auto d = index{1 << 30};
    auto seed = GENERATE(take(2, Catch::Generators::random(0, 1000000)));
    basic_test_data<std::pair<T, int>> data{n, d, index(seed)};
    CAPTURE(n);
    CAPTURE(d);
    CAPTURE(seed);
    auto ref = verification::nth_element(data.data, data.rank);
    cuda_timer timer{std::cerr};
    auto suffix = "-" + std::to_string(n) + "-" + std::to_string(d) + "-" + typeid(T).name();
    for (index r = 1; r <= 1024 * 1024; r *= 2) {
        CAPTURE(r);
        auto ranks = build_ranks_uniform(n, r);
        auto refs = verification::nth_elements(data.data, ranks);
        data.gpu_ranks.copy_from(ranks);
        if (r < 1024) {
            auto name = "quickmulti" + std::to_string(r) + suffix;
            qs_multi_impl<T, select_config<10, 10, 8, true, true, true, 8, 10, 10>>(name + "-s", n, d, data, timer, refs);
            qs_multi_impl<T, select_config<10, 10, 8, false, true, true, 8, 10, 10>>(name + "-g", n, d, data, timer, refs);
        }
        auto name = "samplemulti" + std::to_string(r) + suffix;
        ssss_multi_impl<T, select_config<10, 10, 8, true, true, true, 8, 10, 10, false, 16>>(name + "-s16", n, d, data, timer, refs);
        ssss_multi_impl<T, select_config<10, 10, 8, false, true, true, 8, 10, 10, false, 16>>(name + "-g16", n, d, data, timer, refs);
        ssss_multi_impl<T, select_config<10, 10, 8, true, true, true, 8, 10, 10>>(name + "-s8", n, d, data, timer, refs);
        ssss_multi_impl<T, select_config<10, 10, 8, false, true, true, 8, 10, 10>>(name + "-g8", n, d, data, timer, refs);
        ssss_multi_impl<T, select_config<10, 10, 8, true, true, true, 8, 10, 10, false, 4>>(name + "-s4", n, d, data, timer, refs);
        ssss_multi_impl<T, select_config<10, 10, 8, false, true, true, 8, 10, 10, false, 4>>(name + "-g4", n, d, data, timer, refs);
    }
}

TEMPLATE_TEST_CASE("test", "[.],[test]", float, double) {
    using T = TestType;
    auto n = GENERATE(as<index>{}, 8388608);
    auto d = GENERATE(as<index>{}, 1 << 30, 3);
    auto seed = GENERATE(take(1, Catch::Generators::random(0, 1000000)));
    basic_test_data<std::pair<T, int>> data{n, d, index(seed)};
    CAPTURE(seed);
    auto ref = verification::nth_element(data.data, data.rank);
    cuda_timer timer{std::cerr};
    auto suffix = "-" + std::to_string(n) + "-" + std::to_string(d) + "-" + typeid(T).name();
    if (d >= n) {
        qs_recursive<T>("quickselect" + suffix, n, d, data, timer, ref);
        qs_partition<T>("bipartition" + suffix, n, d, data, timer);
        auto ranks1 = build_ranks_uniform(n, 128);
        auto refs1 = verification::nth_elements(data.data, ranks1);
        auto ranks2 = build_ranks_clustered(n);
        auto refs2 = verification::nth_elements(data.data, ranks2);
        data.gpu_ranks.copy_from(ranks1);
        qs_multi<T>("quickselectmultiuniform" + suffix, n, d, data, timer, refs1);
        ssss_multi<T>("sampleselectmultiuniform" + suffix, n, d, data, timer, refs1);
        data.gpu_ranks.copy_from(ranks2);
        qs_multi<T>("quickselectmulticlustered" + suffix, n, d, data, timer, refs2);
        ssss_multi<T>("sampleselectmulticlustered" + suffix, n, d, data, timer, refs2);
    }
    ssss_host<T>("sampleselect_host" + suffix, n, d, data, timer, ref);
    ssss_recursive<T>("sampleselect" + suffix, n, d, data, timer, ref);
    ssss_partition<T>("kpartition" + suffix, n, d, data, timer);
}

} // namespace gpu
