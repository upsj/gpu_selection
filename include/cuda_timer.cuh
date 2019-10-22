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
#ifndef CUDA_TIMER_CUH
#define CUDA_TIMER_CUH

#include "cuda_error.cuh"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <iterator>
#include <string>

class cuda_timer {
public:
    cuda_timer(std::ostream& output) : m_events(6), m_output{&output} {
        for (auto& event : m_events) {
            cudaCheckError(cudaEventCreate(&event));
        }
    }

    ~cuda_timer() {
        for (auto& event : m_events) {
            cudaEventDestroy(event);
        }
    }

    template <typename Kernel>
    void timed(std::string name, int num_runs, Kernel kernel) {
        std::vector<std::vector<float>> results(num_runs, std::vector<float>(m_events.size() - 1));
        int max_event = -1;
        auto event = [&](int idx_event) {
            cudaCheckError(cudaEventRecord(m_events[idx_event]));
            max_event = std::max(idx_event, max_event);
        };
        for (int i = 0; i < num_runs; ++i) {
            cudaChecked([&]() { kernel(event); });
            cudaCheckError(cudaEventSynchronize(m_events[max_event]));
            for (int j = 0; j < max_event; ++j) {
                cudaCheckError(cudaEventElapsedTime(&results[i][j], m_events[j], m_events[j + 1]));
            }
        }
        auto& out = *m_output;
        out << name;
        for (const auto& run : results) {
            out << ",(";
            std::copy(run.begin(), run.begin() + max_event - 1, std::ostream_iterator<float>(out, ";"));
            out << run[max_event - 1] << ')';
        }
        out << std::endl; // flush output (in case of errors!)
    }

private:
    std::vector<cudaEvent_t> m_events;
    std::ostream* m_output;
};

class cpu_timer {
public:
    void start() { m_start = std::chrono::high_resolution_clock::now(); }
    void stop() { m_end = std::chrono::high_resolution_clock::now(); }
    template <typename F>
    void timed(F f) {
        start();
        f();
        stop();
    }
    double elapsed_us(int repetitions = 1) {
        return std::chrono::duration<double, std::micro>(m_end - m_start).count() / repetitions;
    }

private:
    std::chrono::high_resolution_clock::time_point m_start;
    std::chrono::high_resolution_clock::time_point m_end;
};

#endif // CUDA_TIMER_CUH
