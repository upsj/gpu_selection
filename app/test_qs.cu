#include "catch.hpp"
#include "test_fixture.cuh"

#include <cpu_reference.hpp>
#include <kernel_config.cuh>
#include <launcher_fwd.cuh>
#include <verification.hpp>

namespace gpu {

template <typename Pair>
using test_data = basic_test_data<Pair, 182934, 182934>;

template <typename Config>
using float_pair = typename std::pair<float, Config>;

template <typename Config>
using double_pair = typename std::pair<double, Config>;

TEMPLATE_PRODUCT_TEST_CASE_METHOD(test_data, "bipartition", "[quickselect]",
                                  (float_pair, double_pair),
                                  ((select_config<10, 5, 8, true, true, true, 8, 10, 10>),
                                   (select_config<10, 5, 8, false, true, true, 8, 10, 10>))) {
    using T = typename TestType::first_type;
    using Config = typename TestType::second_type;
    this->run([&]() {
        partition<T, Config>(this->gpu_data, this->gpu_data_out, this->gpu_count_out, this->size,
                             this->pivot);
    });
    auto lsize = this->count_out[0];
    auto rsize = this->count_out[1];
    CHECK(lsize + rsize == this->size);
    auto counts = verification::count_mispartitioned(this->data_out, lsize, this->pivot);
    auto lcount = counts.first;
    auto rcount = counts.second;
    CHECK(lcount == 0);
    CHECK(rcount == 0);
}

TEMPLATE_PRODUCT_TEST_CASE_METHOD(test_data, "quickselect", "[quickselect]",
                                  (float_pair, double_pair),
                                  ((select_config<10, 5, 8, true, true, true, 8, 10, 10>),
                                   (select_config<10, 5, 8, false, true, true, 8, 10, 10>))) {
    using T = typename TestType::first_type;
    using Config = typename TestType::second_type;
    this->run([&]() {
        quickselect<T, Config>(this->gpu_data, this->gpu_data_tmp, this->gpu_count_tmp, this->size, this->rank,
                               this->gpu_data_out);
    });
    auto ref = verification::nth_element(this->data, this->rank);
    CHECK(ref == this->data_out[0]);
}

TEMPLATE_PRODUCT_TEST_CASE_METHOD(test_data, "quickselect_multi", "[quickselect]",
                                  (float_pair, double_pair),
                                  ((select_config<10, 5, 8, true, true, true, 8, 10, 10>),
                                   (select_config<10, 5, 8, false, true, true, 8, 10, 10>))) {
    using T = typename TestType::first_type;
    using Config = typename TestType::second_type;
    std::vector<index> ranks;
    SECTION("some ranks") {
        for (int i = 0; i < 100; ++i) {
            ranks.push_back(this->size * i / 120);
            ranks.push_back(this->size * i / 120 + 1);
            ranks.push_back(this->size * i / 120 + 2);
            ranks.push_back(this->size * i / 120 + 10);
        }
        for (int i = 0; i < 6000; ++i) {
            ranks.push_back(i + 4000);
        }
        std::sort(ranks.begin(), ranks.end());
        ranks.erase(std::unique(ranks.begin(), ranks.end()), ranks.end());
    }
    SECTION("all ranks") {
        ranks.resize(this->size);
        std::iota(ranks.begin(), ranks.end(), 0);
    }
    std::vector<T> result(ranks.size());
    this->gpu_ranks.copy_from(ranks);
    this->gpu_data_out.copy_from(result);
    this->run([&]() {
        quickselect_multi<T, Config>(this->gpu_data, this->gpu_data_tmp, this->gpu_count_tmp, this->size, this->gpu_ranks, ranks.size(),
                                     this->gpu_data_out);
    });
    auto ref = this->data;
    std::sort(ref.begin(), ref.end());
    this->gpu_data_out.copy_to(result);
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

} // namespace gpu