cmake_minimum_required (VERSION 3.1)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/")

SET(CMAKE_C_COMPILER "/usr/bin/gcc-6")   # must be above project
SET(CMAKE_CXX_COMPILER "/usr/bin/g++-6")


project (dma-example LANGUAGES CXX CUDA)

#set(CMAKE_PREFIX_PATH "~/cj/libtorch")
#find_package(Torch REQUIRED)
find_package(CUDA)

SET(CUDA_SEPARABLE_COMPILATION ON)#do not add this line above the find package
include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

set (CMAKE_CXX_STANDARD 14)


#######set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -mavx")
#######set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -pthread -mavx")
#######set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -pthread -mavx")
#######set(CUDA_NVCC_FLAGS -arch=compute_52;-code=sm_52)
#######add_compile_options(-O3)
#######set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -rdc=true ")


add_definitions(-mavx512f)
add_definitions("-Wall -g")
set(CUDA_NVCC_FLAGS --default-stream per-thread)

CUDA_ADD_EXECUTABLE(dma-example
    main.cpp
	common.cpp
    fpga/XDMA.cpp
    fpga/XDMAController.cpp
    fpga/MemoryManager.cpp
	cuda/interface.cu
	cuda/kernel.cu
	cuda/network.cu
	cuda/util.cu
	cuda/network_kernel.cu
	cuda/test.cu
	cuda/app.cu
	cuda/hll.cu
	cuda/kvs.cu
	tool/test.cpp
	tool/input.cpp
	tool/log.cpp
    )
#	cuda/inference.cu
#	cuda/inference_util.cu
target_link_libraries(dma-example
	cuda
	pthread
	gdrapi
    )
#${TORCH_LIBRARIES}
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)


