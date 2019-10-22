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
#ifndef CUDA_CHECK_ERROR_CUH
#define CUDA_CHECK_ERROR_CUH

#include <stdexcept>

inline void cudaCheckError(cudaError_t error) {
    if (error != cudaSuccess) {
        std::string msg{"CUDA error "};
        msg += cudaGetErrorName(error);
        msg += ": ";
        msg += cudaGetErrorString(error);
        throw std::runtime_error{msg};
    }
}

template <typename F>
void cudaChecked(F func) {
    func();
    cudaDeviceSynchronize();
    cudaCheckError(cudaGetLastError());
}

#endif // CUDA_CHECK_ERROR_CUH
