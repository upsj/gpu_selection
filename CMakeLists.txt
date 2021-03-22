cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(gpu_selection LANGUAGES CXX CUDA)

list(APPEND CMAKE_CUDA_FLAGS "-arch=sm_35 -rdc=true --maxrregcount 64 -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_80,code=compute_80")
add_subdirectory(lib)
add_subdirectory(app)
