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
#ifndef CUDA_MEMORY_CUH
#define CUDA_MEMORY_CUH

#include <iostream>
#include <vector>

#include "cuda_error.cuh"

template <typename T>
class cuda_resettable_array;

template <typename T>
class cuda_array {
    friend class cuda_resettable_array<T>;
public:
    cuda_array() : size{}, storage{nullptr} {}
    cuda_array(std::size_t size) : size{size}, storage{nullptr} {
        cudaCheckError(cudaMalloc(&storage, sizeof(T) * size));
    }
    ~cuda_array() {
        if (storage) {
            try {
                cudaCheckError(cudaFree(storage));
            } catch (std::runtime_error& err) {
                std::cerr << err.what() << std::endl;
            }
        }
    }
    cuda_array(const cuda_array&) = delete;
    cuda_array(cuda_array&& other) {
        storage = other.storage;
        size = other.size;
        other.storage = nullptr;
        other.size = 0;
    }
    cuda_array& operator=(cuda_array&& other) {
        this->~cuda_array();
        storage = other.storage;
        size = other.size;
        other.storage = nullptr;
        other.size = 0;
        return *this;
    }

    operator T*() { return storage; }

    void copy_from_raw(const T* src) {
        cudaCheckError(cudaMemcpy(storage, src, size * sizeof(T), cudaMemcpyHostToDevice));
    }

    void copy_to_raw(T* dst) const {
        cudaCheckError(cudaMemcpy(dst, storage, size * sizeof(T), cudaMemcpyDeviceToHost));
    }

    void copy_from(const std::vector<T>& vec) {
        if (size != vec.size()) {
            *this = cuda_array<T>{vec.size()};
        }
        copy_from_raw(vec.data());
    }

    void copy_to(std::vector<T>& vec) const {
        vec.resize(size);
        copy_to_raw(vec.data());
    }

private:
    std::size_t size;
    T* storage;
};

template <typename T>
class cuda_resettable_array {
public:
    void copy_from_raw(const T* src) {
        storage.copy_from_raw(src);
        refstorage.copy_from_raw(src);
    }

    void copy_to_raw(T* dst) const {
        storage.copy_to_raw(dst);
    }

    void copy_from(const std::vector<T>& vec) {
        storage.copy_from(vec);
        refstorage.copy_from(vec);
    }

    void copy_to(std::vector<T>& vec) const {
        storage.copy_to(vec);
    }

    void reset() {
        cudaCheckError(cudaMemcpy(storage, refstorage, storage.size * sizeof(T), cudaMemcpyDeviceToDevice));
    }

    operator T*() { return storage; }

private:
    cuda_array<T> storage;
    cuda_array<T> refstorage;
};

#endif // CUDA_MEMORY_CUH
