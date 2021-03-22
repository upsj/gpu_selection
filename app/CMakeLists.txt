add_executable(benchmark benchmark_main.cu benchmark.cu)
add_executable(benchmark-sort benchmark_main.cu benchmark_sort.cu)
target_include_directories(benchmark-sort PRIVATE ../include ../lib)

add_executable(unittest test_main.cu test_qs.cu test_ssss.cu test_helpers.cu
#    test_basecase.cu
	)

target_link_libraries(benchmark gpu_selection)

target_link_libraries(unittest gpu_selection)

set_target_properties(unittest PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(benchmark-sort PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
