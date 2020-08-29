# Author:   Huang hongjing (21515069@zju.edu.cn)

cmake_minimum_required(VERSION 3.0)

project(example_module)

# Include custom Find<Module>.cmake scripts to enable searching for Vivado HLS                                
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)  

# Check if parent directory
get_directory_property(hasParent PARENT_DIRECTORY)

if (DEFINED ENV{IPREPO_DIR})
   set(IPREPO_DIR $ENV{IPREPO_DIR})
elseif (NOT IPREPO_DIR)
   set(IPREPO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/iprepo)
   message("set iprepor dir ${IPREPO_DIR}")
endif()


if (NOT DEFINED BYPASS)
   set(BYPASS disable)
endif()







#HLS IPs
add_subdirectory(hls/mem_write_cmd_page_boundary_check_512)
add_subdirectory(hls/tlb)


###VIVADO project
# Find Xilinx Vivado
find_package(Vivado REQUIRED)
if (NOT VIVADO_FOUND)
   message(FATAL_ERROR "Vivado not found.")
endif()
# configure_file(${CMAKE_CURRENT_SOURCE_DIR}/rtl/example_module.vh ${CMAKE_BINARY_DIR}/example_module.vh)
if (BYPASS STREQUAL "enable") 
   file(APPEND ${CMAKE_BINARY_DIR}/example_module.vh "
    `ifndef XDMA_BYPASS
    `define XDMA_BYPASS
    `endif
" )
   endif()
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/scripts/create_xdma_common_ip.tcl.in ${CMAKE_BINARY_DIR}/create_xdma_common_ip.tcl)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/scripts/create_xdma_ip.tcl.in ${CMAKE_BINARY_DIR}/create_xdma_ip.tcl)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/scripts/create_xdma_bypass_ip.tcl.in ${CMAKE_BINARY_DIR}/create_xdma_bypass_ip.tcl)


configure_file(${CMAKE_CURRENT_SOURCE_DIR}/scripts/create_xdma_project.tcl.in ${CMAKE_BINARY_DIR}/create_project.tcl)



set (PROJECT_DEPENDS
      ${CMAKE_CURRENT_SOURCE_DIR}/rtl/example_module.vh
      ${CMAKE_CURRENT_SOURCE_DIR}/scripts/create_xdma_project.tcl.in
      ${CMAKE_CURRENT_SOURCE_DIR}/scripts/create_xdma_common_ip.tcl.in
      ${CMAKE_CURRENT_SOURCE_DIR}/scripts/create_xdma_ip.tcl.in
      ${CMAKE_CURRENT_SOURCE_DIR}/scripts/create_xdma_bypass_ip.tcl.in)

add_custom_target(project
   COMMAND ${VIVADO_BINARY} -mode batch -source ${CMAKE_BINARY_DIR}/create_project.tcl
   DEPENDS ${PROJECT_DEPENDS})
