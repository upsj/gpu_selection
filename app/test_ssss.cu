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

#include <cpu_reference.hpp>
#include <kernel_config.cuh>
#include <launcher_fwd.cuh>
#include <utils_mask.cuh>
#include <utils_prefixsum.cuh>
#include <verification.hpp>

namespace gpu {

template <typename A, typename B, int Size, int Replsize>
struct extended_pair {
    constexpr static int size = Size;
    constexpr static int replsize = Replsize;
    using first_type = A;
    using second_type = B;
};

template <typename Pair>
using test_data = basic_test_data<Pair, Pair::size, Pair::replsize>;

template <typename Config>
using float_pair = extended_pair<float, Config, 362934, 362934>;

template <typename Config>
using float_pair_repl = extended_pair<float, Config, 957498, 472>;

template <typename Config>
using double_pair = extended_pair<double, Config, 362934, 362934>;

template <typename Config>
using double_pair_repl = extended_pair<double, Config, 957498, 472>;

__global__ void compute_bucket_mask_inplace(const index* ranks, index rank_count, const index* bucket_prefixsum, mask* bucket_mask, index* range_begins) {
    gpu::kernels::compute_bucket_mask_impl(ranks, rank_count, 0, bucket_prefixsum, bucket_mask, range_begins);
}

template <typename Config>
__global__ void small_prefix_sum_inplace(gpu::index* data) {
    gpu::kernels::small_prefix_sum_sentinel<Config::searchtree::height>(data);
}

template <typename Config>
__global__ void masked_prefix_sum_inplace(index* data, const mask* m) {
    gpu::kernels::masked_prefix_sum_sentinel<Config::searchtree::height>(data, m);
}

TEMPLATE_PRODUCT_TEST_CASE_METHOD(test_data, "count", "[sampleselect]",
                                  (float_pair, double_pair),
                                  ((select_config<10, 10, 8, true, true, false, 8, 10, 10>),
                                   (select_config<10, 11, 9, true, true, false, 8, 10, 10>),
                                   (select_config<10, 12, 10, true, true, false, 8, 10, 10>))) {
    using T = typename TestType::first_type;
    using Config = typename TestType::second_type;
    if (sizeof(T) * Config::sample::size <= 16384) {
        this->run([&]() { build_searchtree<T, Config>(this->gpu_data, this->gpu_tree, this->size); });
        this->tree.resize(Config::searchtree::size);
        auto ref_tree =
                cpu::build_searchtree(this->data, Config::sample::size, Config::searchtree::size);
        CHECK(this->tree == ref_tree);
        this->run([&]() {
            count_buckets<T, Config>(this->gpu_data, this->gpu_tree, this->gpu_count_tmp,
                                    this->gpu_count_out, this->gpu_oracles, this->size);
        });
        this->count_out.resize(Config::searchtree::width);
        this->tree.resize(Config::searchtree::size);
        auto counts = this->count_out;
        auto ref_ssss = cpu::ssss(this->data, this->tree);
        auto ref_counts = ref_ssss.first;
        REQUIRE(this->count_out == ref_counts);
    }
}

TEMPLATE_PRODUCT_TEST_CASE_METHOD(test_data, "count-distribute", "[sampleselect]",
                                  (float_pair, double_pair),
                                  ((select_config<10, 10, 8, true, true, true, 8, 10, 10>),
                                   (select_config<10, 10, 8, false, true, true, 8, 10, 10>),
                                   (select_config<10, 10, 8, true, false, true, 8, 10, 10>),
                                   (select_config<10, 10, 8, false, false, true, 8, 10, 10>))) {
    using T = typename TestType::first_type;
    using Config = typename TestType::second_type;
    this->run([&]() { build_searchtree<T, Config>(this->gpu_data, this->gpu_tree, this->size); });
    this->tree.resize(Config::searchtree::size);
    auto ref_tree =
            cpu::build_searchtree(this->data, Config::sample::size, Config::searchtree::size);
    CHECK(this->tree == ref_tree);
    this->run([&]() {
        count_buckets<T, Config>(this->gpu_data, this->gpu_tree, this->gpu_count_tmp,
                                 this->gpu_count_out, this->gpu_oracles, this->size);
    });
    this->count_out.resize(Config::searchtree::width);
    this->tree.resize(Config::searchtree::size);
    auto counts = this->count_out;
    auto ref_ssss = cpu::ssss(this->data, this->tree);
    auto ref_counts = ref_ssss.first;
    auto ref_oracles = ref_ssss.second;
    CHECK(unpack(this->oracles, this->size) == ref_oracles);
    REQUIRE(this->count_out == ref_counts);
    // select and extract bucket
    std::partial_sum(counts.begin(), counts.end(), counts.begin());
    auto bucket = std::distance(counts.begin(),
                                std::upper_bound(counts.begin(), counts.end(), this->pivot));
    this->run([&]() {
        collect_bucket<T, Config>(this->gpu_data, this->gpu_oracles, this->gpu_count_tmp,
                                  this->gpu_data_out, this->size, bucket, this->gpu_atomic);
    });
    if (!Config::algorithm::shared_memory) {
        CHECK(this->atomic[0] == this->count_out[bucket]);
    }
    this->data_out.resize(this->count_out[bucket]);
    // add sentinel
    this->tree.push_back(std::numeric_limits<T>::infinity());
    auto lb = this->tree[bucket + Config::searchtree::width - 1];
    auto ub = this->tree[bucket + Config::searchtree::width];
    auto count_misplaced = verification::count_not_in_bucket(this->data_out, lb, ub);
    CHECK(count_misplaced == 0);

    // select and extract multiple buckets:
    // compute mask - gpu
    this->run([&]() {
        small_prefix_sum_inplace<Config><<<1, Config::searchtree::width>>>(this->gpu_count_out);
        compute_bucket_mask_inplace<<<1, Config::searchtree::width>>>(this->gpu_ranks, this->ranks.size(), this->gpu_count_out, this->gpu_bucket_mask, this->gpu_rank_ranges);
    });
    // verify computed mask
    std::vector<gpu::index> rank_ranges;
    this->gpu_bucket_mask.copy_to(this->bucket_mask);
    this->gpu_rank_ranges.copy_to(rank_ranges);
    constexpr auto mask_size = ceil_div(Config::searchtree::width, sizeof(gpu::mask) * 8);
    this->bucket_mask.resize(mask_size);
    rank_ranges.resize(Config::searchtree::width + 1);
    this->count_out.resize(Config::searchtree::width + 1);
    auto bucket_prefixsum = this->count_out;
    ref_counts.push_back(0);
    this->gpu_count_out.copy_from(ref_counts);
    ref_counts.pop_back();
    REQUIRE(verification::verify_rank_ranges(this->ranks, this->count_out, rank_ranges));
    auto ref_mask = cpu::compute_bucket_mask(rank_ranges);
    CHECK(ref_mask == this->bucket_mask);
    // compute masked bucket count prefix sum - reference
    auto ref_pair = cpu::masked_prefix_sum(ref_counts, this->bucket_mask);
    auto& m_pfs_ref = ref_pair.first;
    // compute masked bucket count prefix sum - gpu
    this->run([&]() {
        static_assert(Config::searchtree::width <= max_block_size, "Such large searchtrees are not supported");
        masked_prefix_sum_inplace<Config><<<1, Config::searchtree::width>>>(this->gpu_count_out, this->gpu_bucket_mask);
    });
    this->count_out.resize(Config::searchtree::width + 1);
    m_pfs_ref.push_back(ref_pair.second);
    REQUIRE(m_pfs_ref == this->count_out);
    // extract buckets
    this->atomic.assign(Config::searchtree::width, 0);
    this->gpu_atomic.copy_from(this->atomic);
    this->run([&]() {
        collect_buckets<T, Config>(this->gpu_data, this->gpu_oracles, this->gpu_count_tmp,
            this->gpu_count_out, this->gpu_data_out, this->size, this->gpu_bucket_mask, this->gpu_atomic);
    });
    this->data_out.resize(ref_pair.second);
    auto count_misplaced_multi = verification::count_not_in_buckets(this->data_out, m_pfs_ref, ref_tree);
    std::vector<index> zeros(Config::searchtree::width);
    CHECK(count_misplaced_multi == zeros);

    // run multiple ssss kernels in parallel
    std::vector<kernels::ssss_multi_aux<T, Config>> auxs{Config::searchtree::width + 1};
    auto& aux = auxs[0];
    std::copy_n(ref_mask.begin(), mask_size, aux.stage2.bucket_mask);
    std::copy_n(bucket_prefixsum.begin(), Config::searchtree::width + 1, aux.stage2.bucket_prefixsum);
    std::copy_n(m_pfs_ref.begin(), Config::searchtree::width + 1, aux.stage2.bucket_masked_prefixsum);
    std::copy_n(rank_ranges.begin(), Config::searchtree::width + 1, aux.stage2.rank_ranges);
    cuda_array<kernels::ssss_multi_aux<T, Config>> gpu_aux;
    gpu_aux.copy_from(auxs);
    this->run([&]() {
        ssss_merged<T, Config>(this->gpu_data_out, this->gpu_data, this->gpu_oracles, 0, this->gpu_ranks, 0, 0, gpu_aux, gpu_aux + 1, this->gpu_data_tmp);
    });
    gpu_aux.copy_to(auxs);

    std::vector<T> result;
    this->gpu_data.copy_to(result);
    std::vector<T> out_trees;
    this->gpu_data_tmp.copy_to(out_trees);
    for (gpu::oracle bucket = 0; bucket < Config::searchtree::width; ++bucket) {
        auto range_begin = m_pfs_ref[bucket];
        auto range_end = m_pfs_ref[bucket + 1];
        if (range_begin == range_end) continue;
        std::vector<T> part_data(this->data_out.begin() + range_begin, this->data_out.begin() + range_end);
        std::vector<T> part_result(result.begin() + range_begin, result.begin() + range_end);
        std::vector<index> part_ranks(this->ranks.begin() + rank_ranges[bucket], this->ranks.begin() + rank_ranges[bucket + 1]);
        for (auto& rank : part_ranks) rank -= bucket_prefixsum[bucket];
        auto part_tree = cpu::build_searchtree(part_data, Config::sample::size, Config::searchtree::size);
        auto part_ssss = cpu::ssss(part_data, part_tree);
        auto part_counts = part_ssss.first;
        auto rank_ranges = cpu::compute_rank_ranges(part_counts, part_ranks);
        auto bucket_mask = cpu::compute_bucket_mask(rank_ranges);
        auto masked_prefixsum = cpu::masked_prefix_sum(part_counts, bucket_mask);
        auto full_prefixsum = part_counts;
        full_prefixsum.insert(full_prefixsum.begin(), 0);
        std::partial_sum(full_prefixsum.begin(), full_prefixsum.end(), full_prefixsum.begin());
        std::vector<mask> res_bucket_mask(mask_size);
        std::vector<index> res_bucket_prefixsum(Config::searchtree::width + 1);        
        std::vector<index> res_bucket_masked_prefixsum(Config::searchtree::width + 1);
        std::vector<index> res_rank_ranges(Config::searchtree::width + 1);
        std::vector<T> res_tree(Config::searchtree::size);
        std::copy_n(auxs[bucket + 1].stage2.bucket_mask, mask_size, res_bucket_mask.begin());
        std::copy_n(auxs[bucket + 1].stage2.bucket_prefixsum, Config::searchtree::width + 1, res_bucket_prefixsum.begin());
        std::copy_n(auxs[bucket + 1].stage2.bucket_masked_prefixsum, Config::searchtree::width + 1, res_bucket_masked_prefixsum.begin());
        std::copy_n(auxs[bucket + 1].stage2.rank_ranges, Config::searchtree::width + 1, res_rank_ranges.begin());
        std::copy_n(out_trees.begin() + bucket * Config::searchtree::size, Config::searchtree::size, res_tree.begin());
        masked_prefixsum.first.push_back(masked_prefixsum.second);
        count_misplaced_multi = verification::count_not_in_buckets(part_result, masked_prefixsum.first, part_tree);
        CAPTURE(bucket);
        CHECK(res_tree == part_tree);
        CHECK(res_bucket_prefixsum == full_prefixsum);
        CHECK(res_rank_ranges == rank_ranges);
        CHECK(res_bucket_mask == bucket_mask);
        CHECK(res_bucket_masked_prefixsum == masked_prefixsum.first);
        CHECK(count_misplaced_multi == zeros);
    }
}

TEMPLATE_PRODUCT_TEST_CASE_METHOD(test_data, "sampleselect", "[sampleselect]",
                                  (float_pair, float_pair_repl, double_pair, double_pair_repl),
                                  ((select_config<10, 10, 8, true, true, true, 8, 10, 10>),
                                   (select_config<10, 10, 8, true, true, true, 8, 10, 10, true>), // bucket select
                                   (select_config<10, 10, 8, false, true, true, 8, 10, 10>),
                                   (select_config<10, 10, 8, true, false, true, 8, 10, 10>),
                                   (select_config<10, 10, 8, false, false, true, 8, 10, 10>),
                                   (select_config<10, 10, 6, true, true, true, 8, 10, 10>),
                                   (select_config<10, 10, 6, false, true, true, 8, 10, 10>),
                                   (select_config<10, 10, 6, true, false, true, 8, 10, 10>),
                                   (select_config<10, 10, 6, false, false, true, 8, 10, 10>),
                                   (select_config<10, 10, 6, false, true, true, 8, 10, 10, false, 1024>),
                                   (select_config<10, 10, 6, true, true, true, 8, 10, 10, false, 1024>))) {
    using T = typename TestType::first_type;
    using Config = typename TestType::second_type;
    if (TestType::replsize < TestType::size && Config::algorithm::bucket_select) {
        return;
    }
    SECTION("single") {
        this->run([&]() {
            sampleselect<T, Config>(this->gpu_data, this->gpu_data_tmp, this->gpu_tree, this->gpu_count_tmp, this->size,
                                    this->rank, this->gpu_data_out);
        });
        auto ref = verification::nth_element(this->data, this->rank);
        CHECK(ref == this->data_out[0]);
    }
    SECTION("multi") {
        std::vector<index> ranks;
        SECTION("some ranks") {
            // some, but not all buckets
            for (int i = 0; i < 100; ++i) {
                ranks.push_back(this->size * i / 120);
                ranks.push_back(this->size * i / 120 + 1);
                ranks.push_back(this->size * i / 120 + 2);
                ranks.push_back(this->size * i / 120 + 10);
            }
            // some contiguous area
            for (int i = 0; i < 6000; ++i) {
                ranks.push_back(i + 4000);
            }
            // and a single element to check the single selection base case
            ranks.push_back(this->size - 49);
            std::sort(ranks.begin(), ranks.end());
            ranks.erase(std::unique(ranks.begin(), ranks.end()), ranks.end());
        }
        SECTION("all ranks") {
            ranks.resize(this->size);
            std::iota(ranks.begin(), ranks.end(), 0);
        }
        this->gpu_ranks.copy_from(ranks);
        this->gpu_atomic.copy_from(this->atomic);
        this->gpu_data_out.copy_from(std::vector<T>(ranks.size()));
        this->run([&]() {
            sampleselect_multi<T, Config>(this->gpu_data, this->gpu_data_tmp, this->size, this->gpu_ranks, ranks.size(),
                                          this->gpu_count_tmp, this->gpu_aux, this->gpu_atomic, this->gpu_data_out);
        });
        auto ref = this->data;
        std::sort(ref.begin(), ref.end());
        std::vector<T> result;
        this->gpu_data_out.copy_to(result);
        result.resize(ranks.size());
        std::vector<T> reference;
        for (auto rank : ranks) {
            reference.push_back(ref[rank]);
        }
        int count{};
        for (index i = 0; i < reference.size(); ++i) {
            count += reference[i] != result[i];
        }
        CAPTURE(reference.size());
        CHECK(count == 0);
    }
}

}
