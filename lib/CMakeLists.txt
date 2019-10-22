add_library(gpu_selection
    cpu_reference.cpp
    verification.cpp
generated/gen0.cu
generated/gen1.cu
generated/gen2.cu
generated/gen3.cu
generated/gen4.cu
generated/gen5.cu
generated/gen6.cu
generated/gen7.cu
generated/gen8.cu
generated/gen9.cu
generated/gen10.cu
generated/gen11.cu
generated/gen12.cu
generated/gen13.cu
generated/gen14.cu
generated/gen15.cu
generated/gen16.cu
generated/gen17.cu
generated/gen18.cu
generated/gen19.cu
generated/gen20.cu
generated/gen21.cu
generated/gen22.cu
generated/gen23.cu
generated/gen24.cu
generated/gen25.cu
generated/gen26.cu
generated/gen27.cu
generated/gen28.cu
generated/gen29.cu
generated/gen30.cu
generated/gen31.cu
generated/gen32.cu
generated/gen33.cu
generated/gen34.cu
generated/gen35.cu
generated/gen36.cu
generated/gen37.cu
generated/gen38.cu
generated/gen39.cu
    )

target_compile_features(gpu_selection PUBLIC cxx_std_14)
set_target_properties(gpu_selection PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(gpu_selection PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(gpu_selection PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../include")
