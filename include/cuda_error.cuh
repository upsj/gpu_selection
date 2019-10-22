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
