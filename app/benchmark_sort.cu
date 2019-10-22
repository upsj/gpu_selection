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
#include <cuda_timer.cuh>
#include <cub/cub.cuh>

namespace gpu {

constexpr auto num_runs = 10;

template <typename T>
void cub_sort(std::string name, index n, index d, basic_test_data<std::pair<T,int>>& data, cuda_timer& timer) {
    cub::DoubleBuffer<T> keys{static_cast<T*>(data.gpu_data), static_cast<T*>(data.gpu_data_out)};
    timer.timed(name, num_runs, [&](auto event) {
        data.reset();
        auto tmp_size = sizeof(T) * n;
        event(0);
        cub::DeviceRadixSort::SortKeys(static_cast<T*>(data.gpu_data_tmp), tmp_size, keys, n);
        event(1);
    });
    auto sorted = data.data;
    auto ref = sorted;
    cudaCheckError(cudaMemcpy(ref.data(), keys.Current(), n * sizeof(T), cudaMemcpyDeviceToHost));
    std::sort(sorted.begin(), sorted.end());
    bool is_sorted = sorted == ref;
    CHECK(is_sorted);
}

TEMPLATE_TEST_CASE("sort", "", float, double) {
    using T = TestType;
    auto n = GENERATE(as<index>{}, 65536, 262144, 524288, 1048576, 2097152, 4194304, 8388608,
                      16777216, 33554432, 67108864, 134217728);
    auto d = GENERATE(as<index>{}, 1 << 30);
    auto seed = GENERATE(take(10, Catch::Generators::random(0, 1000000)));
    basic_test_data<std::pair<T, int>> data{n, d, index(seed)};
    CAPTURE(n);
    CAPTURE(d);
    CAPTURE(seed);
    cuda_timer timer{std::cerr};
    auto suffix = "-" + std::to_string(n) + "-" + std::to_string(d) + "-" + typeid(T).name();
    // thrust_sort<T>("thrust_sort" + suffix, n, d, data, timer);
    cub_sort<T>("cub_sort" + suffix, n, d, data, timer);
}

} // namespace gpu
