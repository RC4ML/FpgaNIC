# Set up project and cmake
cmake_minimum_required (VERSION 3.1)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/")

project (DmaDriverExample)
find_package(CUDA)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

# Set up compilation options
set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -mavx")

set (CMAKE_C_STANDARD 11)

set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -pthread -mavx")
set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -pthread -mavx")
set(CMAKE_C_COMPILER "/usr/bin/nvcc")
#find_package(Boost COMPONENTS program_options REQUIRED)

if(CUDA_ENABLE)
    enable_language(CUDA)
endif()
add_definitions(-mavx512f)

# Set up target for main executable
CUDA_ADD_EXECUTABLE(dma-example
    main.cpp
	common.cpp
    fpga/XDMA.cpp
    fpga/XDMAController.cpp
    fpga/MemoryManager.cpp
	cuda/vectoradd.cu
	tool/test.cpp
    )
target_link_libraries(dma-example
	${Boost_LIBRARIES}
	cuda
	pthread
	dl
	gdrapi
    )


