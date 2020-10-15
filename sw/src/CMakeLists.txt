# Set up project and cmake
cmake_minimum_required (VERSION 3.1)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/")

SET(CMAKE_C_COMPILER "/usr/bin/gcc-6")   # must be above project
SET(CMAKE_CXX_COMPILER "/usr/bin/g++-6")

project (DmaDriverExample)

find_package(CUDA)
SET(CUDA_SEPARABLE_COMPILATION ON)#do not add this line above the find package
include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

# Set up compilation options
set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -mavx")

set (CMAKE_C_STANDARD 11)

set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -pthread -mavx")
set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -pthread -mavx")



add_compile_options(-O0)
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -rdc=true ")

if(CUDA_ENABLE)
    enable_language(CUDA)
endif()
add_definitions(-mavx512f)
add_definitions("-Wall -g")


# Set up target for main executable
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
	tool/test.cpp
	tool/input.cpp
    )
target_link_libraries(dma-example
	${Boost_LIBRARIES}
	cuda
	pthread
	dl
	gdrapi
    )


