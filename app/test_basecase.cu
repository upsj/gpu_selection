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
#include "../lib/utils_basecase.cuh"
#include "test_fixture.cuh"
#include <kernel_config.cuh>
#include <numeric>
#include <algorithm>

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
using float_pair = extended_pair<float, Config, 1024 * 4, 1024 * 4>;
    
TEMPLATE_PRODUCT_TEST_CASE_METHOD(test_data, "basecase", "[basecase]",
                                  (float_pair),
                                  ((select_config<10, 10, 8, true, true, false, 8, 10, 10, false, 8, 0>),
                                   (select_config<10, 10, 8, true, true, false, 8, 10, 10, false, 8, 1>),
                                   (select_config<10, 10, 8, true, true, false, 8, 10, 10, false, 8, 2>),
                                   (select_config<12, 10, 8, true, true, false, 8, 10, 10, false, 8, 2>))) {
    using T = typename TestType::first_type;
    using Config = typename TestType::second_type;
    std::vector<index> ranks(1);
    constexpr auto basecase_size = Config::basecase::size;
    constexpr auto local_size = Config::basecase::local_size;
    constexpr auto cur_launch_size = Config::basecase::launch_size;
    for (auto size : {basecase_size, basecase_size / 5, warp_size * local_size, warp_size * local_size / 5}) {
        for (auto launch_size: {cur_launch_size, max_block_size}) {
            std::string mode;
            SECTION("some ranks") {
                mode = "some ranks";
                ranks.resize(std::min<int>(100, size / 2));
                for (auto i = 0; i < ranks.size(); ++i) {
                    ranks[i] = i * size / ranks.size();
                }
            }
            SECTION("all ranks") {
                mode = "all ranks";
                ranks.resize(size);
                std::iota(ranks.begin(), ranks.end(), 0);
            }
            CAPTURE(size);
            CAPTURE(launch_size);
            CAPTURE(mode);
            this->gpu_ranks.copy_from(ranks);
            this->run([&]() { kernels::select_bitonic_basecase<T, Config><<<1, launch_size>>>(this->gpu_data, size, ranks.back(), this->gpu_data_out); });
            std::vector<T> result;
            this->gpu_data_out.copy_to(result);
            auto data = this->data;
            data.resize(size);
            std::sort(data.begin(), data.end());
            CHECK(data[ranks.back()] == result[0]);
            this->run([&]() { kernels::select_bitonic_multiple_basecase<T, Config><<<1, launch_size>>>(this->gpu_data, size, this->gpu_ranks, ranks.size(), 0, this->gpu_data_out); });
            this->gpu_data_out.copy_to(result);
            index count{};
            for (auto i = 0; i < ranks.size(); ++i) {
                count += result[i] != data[ranks[i]];
            }
            CHECK(count == 0);
        }
    }
}

} // namespace gpu