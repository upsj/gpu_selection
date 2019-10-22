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

#include <cuda_memory.cuh>
#include <verification.hpp>
#include <cpu_reference.hpp>

#include <random>
#include <algorithm>
#include <numeric>

#include "../lib/utils_sort.cuh"
#include "../lib/utils_prefixsum.cuh"
#include "../lib/utils_mask.cuh"
#include "../lib/utils.cuh"

constexpr int size_log2 = 10;
constexpr int size = 1 << size_log2;

__global__ void binary_search_tester(const gpu::index* haystack, gpu::index haystack_size, gpu::index needle, gpu::index* result) {
    *result = gpu::kernels::binary_search(haystack, haystack_size, needle);
}

__global__ void compute_bucket_mask_tester(const gpu::index* ranks, gpu::index rank_count, const gpu::index* prefixsum, gpu::mask* mask, gpu::index* begins) {
    gpu::kernels::compute_bucket_mask_impl(ranks, rank_count, 0, prefixsum, mask, begins);
}

__global__ void bitonic_tester(gpu::index* data) {
    gpu::index local[4];
    for (auto i = 0; i < 4; ++i) {
        local[i] = data[threadIdx.x * 4 + i];
    }
    gpu::kernels::bitonic_helper_global<gpu::index, 2, gpu::warp_size_log2, size_log2 - 2 - gpu::warp_size_log2, size_log2>::sort(local, data, false);
    for (auto i = 0; i < 4; ++i) {
        data[threadIdx.x * 4 + i] = local[i];
    }
}

__global__ void small_prefix_sum_up_tester(gpu::index* data) {
    gpu::kernels::small_prefix_sum_upward<size_log2>(data);
}

__global__ void small_prefix_sum_tester(gpu::index* data) { gpu::kernels::small_prefix_sum<size_log2>(data); }

__global__ void masked_prefix_sum_tester(gpu::index* data, const gpu::mask* m) { gpu::kernels::masked_prefix_sum<size_log2>(data, m); }

__global__ void prefix_sum_select_tester(const gpu::index* data, gpu::index rank, gpu::oracle* out_bucket,
    gpu::index* out_rank) {
    gpu::kernels::prefix_sum_select<size_log2>(data, rank, out_bucket, out_rank);
}

__global__ void select_mask_local_helper(gpu::index rank, gpu::mask m, gpu::index* result) {
    *result = gpu::kernels::select_mask_local<32>(rank, m);
}

__global__ void select_mask_local_helper2(gpu::index rank, gpu::mask m, gpu::index* result) {
    *result = gpu::kernels::select_mask_local<16>(rank, m);
}

__global__ void select_mask_helper(gpu::index rank, const gpu::mask* m, gpu::index* result) {
    *result = gpu::kernels::select_mask<1 << size_log2>(rank, m);
}

TEST_CASE("helpers", "") {
    std::vector<gpu::index> input(size);
    std::default_random_engine gen;
    std::uniform_int_distribution<gpu::index> int_dist(0, size - 1);
    for (auto& el : input) {
        el = int_dist(gen);
    }
    auto ref = input;
    cuda_array<gpu::index> gpudata;
    gpudata.copy_from(input);
    SECTION("transformations") {
        SECTION("bitonic_sort") {
            cudaChecked([&]() { bitonic_tester<<<1, size / 2>>>(gpudata); });
            std::sort(ref.begin(), ref.end());
        }
        SECTION("small_prefix_sum_upward") {
            cudaChecked([&]() { small_prefix_sum_up_tester<<<1, size / 2>>>(gpudata); });
            for (int i = 2; i <= size; i *= 2) {
                for (int j = 0; j < size; j += i) {
                    ref[j + i - 1] += ref[j + i / 2 - 1];
                }
            }
        }
        SECTION("small_prefix_sum") {
            cudaChecked([&]() { small_prefix_sum_tester<<<1, size / 2>>>(gpudata); });
            ref.insert(ref.begin(), 0);
            std::partial_sum(ref.begin(), ref.end(), ref.begin());
            ref.pop_back();
        }
        SECTION("masked_prefix_sum") {
            cuda_array<gpu::mask> gpumask;
            std::vector<gpu::mask> cpumask((1 << size_log2) / (sizeof(gpu::mask) * 8));
            std::uniform_int_distribution<gpu::index> dist{gpu::mask{}, ~gpu::mask{}};
            for (auto& el : cpumask) {
                el = dist(gen);
            }
            gpumask.copy_from(cpumask);
            cudaChecked([&]() { masked_prefix_sum_tester<<<1, size>>>(gpudata, gpumask); });
            ref = cpu::masked_prefix_sum(input, cpumask).first;
        }
        gpudata.copy_to(input);
        CHECK(ref == input);
    }
    SECTION("prefix_sum_select") {
        ref.insert(ref.begin(), 0);
        std::partial_sum(ref.begin(), ref.end(), ref.begin());
        auto sum = ref.back();
        ref.pop_back();
        cuda_array<gpu::oracle> out_bucket{1};
        cuda_array<gpu::index> out_rank{1};
        auto rank = std::uniform_int_distribution<gpu::index>{0, sum - 1}(gen);
        cudaChecked([&]() {
            prefix_sum_select_tester<<<1, size / 2>>>(gpudata, rank, out_bucket, out_rank);
        });
        gpu::oracle res_bucket{};
        gpu::index res_rank{};
        out_bucket.copy_to_raw(&res_bucket);
        out_rank.copy_to_raw(&res_rank);
        auto ref_bucket =
                std::distance(ref.begin(), std::upper_bound(ref.begin(), ref.end(), rank)) - 1;
        auto ref_rank = rank - ref[ref_bucket];
        CHECK(ref_bucket == res_bucket);
        CHECK(ref_rank == res_rank);
    }
    SECTION("binary_search") {
        std::partial_sum(input.begin(), input.end(), input.begin());
        ref = input;
        CAPTURE(input);
        gpudata.copy_from(input);
        cuda_array<gpu::index> gpu_out_idx{1};
        gpu::index needle{};
        SECTION("random needle") {
            needle = std::uniform_int_distribution<gpu::index>{0, input.back()}(gen);
            INFO("random needle");
        }
        SECTION("boundary needle") {
            needle = input[std::uniform_int_distribution<gpu::index>{0, size - 1}(gen)];
            INFO("boundary needle");
        }
        cudaChecked([&]() {
            binary_search_tester<<<1, 1>>>(gpudata, size, needle, gpu_out_idx);
        });
        auto ref_idx = std::distance(ref.begin(), std::lower_bound(ref.begin(), ref.end(), needle));
        gpu::index res_idx{};
        gpu_out_idx.copy_to_raw(&res_idx);
        CHECK(ref_idx == res_idx);
    }
    SECTION("compute_bucket_mask") {
        auto original_input = input;
        input.insert(input.begin(), 0);
        std::partial_sum(input.begin(), input.end(), input.begin());
        gpudata.copy_from(input);
        std::vector<gpu::index> ranks;
        for (auto i = 0; i < 100; ++i) {
            ranks.push_back(std::uniform_int_distribution<gpu::index>{0, input.back()}(gen));
        }
        ranks.push_back(input.back() - 1);
        ranks.push_back(input[std::uniform_int_distribution<gpu::index>{0, size - 1}(gen)]);
        ranks.push_back(0);
        std::sort(ranks.begin(), ranks.end());
        cuda_array<gpu::index> gpuranks;
        gpuranks.copy_from(ranks);
        cuda_array<gpu::index> gpuoffsets{size + 1};
        cuda_array<gpu::mask> gpumask{size / (sizeof(gpu::mask) * 8)};
        cudaChecked([&]() {
            compute_bucket_mask_tester<<<1, size>>>(gpuranks, ranks.size(), gpudata, gpumask, gpuoffsets);
        });
        std::vector<gpu::index> offsets;
        std::vector<gpu::mask> masks;
        gpuoffsets.copy_to(offsets);
        gpumask.copy_to(masks);
        CHECK(verification::verify_rank_ranges(ranks, input, offsets));
        CHECK(offsets == cpu::compute_rank_ranges(original_input, ranks));
        CHECK(masks == cpu::compute_bucket_mask(offsets));
    }
    SECTION("select_mask_local") {
        cuda_array<gpu::index> gpuresult{1};
        std::vector<int> ranks{0,1,2,3,4,5,12,15,16,18,21,25,26,27,31};
        gpu::mask m{};
        for (auto el : ranks) {
            m |= 1ull << el;
        }
        int rank{};
        for (auto el : ranks) {
            cudaChecked([&]() {
                select_mask_local_helper<<<1, gpu::warp_size>>>(rank, m, gpuresult);
            });
            gpu::index result{};
            gpuresult.copy_to_raw(&result);
            CHECK(result == el);
            if (el < 16) {
                cudaChecked([&]() {
                    select_mask_local_helper2<<<1, gpu::warp_size>>>(rank, m, gpuresult);
                });
                gpu::index result{};
                gpuresult.copy_to_raw(&result);
                CHECK(result == el);
            }
            ++rank;
        }
    }
    SECTION("select_mask") {
        cuda_array<gpu::mask> gpumask;
        std::vector<gpu::mask> cpumask((1 << size_log2) / (sizeof(gpu::mask) * 8));
        std::uniform_int_distribution<gpu::index> dist{gpu::mask{}, ~gpu::mask{}};
        for (auto& el : cpumask) {
            el = dist(gen);
        }
        gpumask.copy_from(cpumask);
        cuda_array<gpu::index> gpuresult{1};
        auto mask_prefixsum = cpu::masked_prefix_sum(std::vector<gpu::index>(1 << size_log2, 1), cpumask);
        mask_prefixsum.first.push_back(mask_prefixsum.second);
        for (gpu::index i = 0; i < mask_prefixsum.first.size() - 1; ++i) {
            if (mask_prefixsum.first[i + 1] > mask_prefixsum.first[i]) {
                auto rank = mask_prefixsum.first[i];
                cudaChecked([&]() {
                    select_mask_helper<<<1, gpu::warp_size>>>(rank, gpumask, gpuresult);
                });
                gpu::index result{};
                gpuresult.copy_to_raw(&result);
                CHECK(result == i);
            }
        }
    }
}
