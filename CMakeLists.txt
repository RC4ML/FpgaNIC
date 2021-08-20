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

   set(NETWORK_BANDWIDTH 100)
   set(NETWORK_INTERFACE 100)
   set(CLOCK_PERIOD 3.2)
   set(DATA_WIDTH 64)







#HLS IPs
add_subdirectory(hls/mem_write_cmd_page_boundary_check_512)
add_subdirectory(hls/tlb)
add_subdirectory(hls/ethernet_frame_padding_512)
add_subdirectory(hls/arp_server_subnet)
add_subdirectory(hls/hash_table)
add_subdirectory(hls/icmp_server)
add_subdirectory(hls/ip_handler)
add_subdirectory(hls/mac_ip_encode)
add_subdirectory(hls/toe)


###VIVADO project
# Find Xilinx Vivado
find_package(Vivado REQUIRED)
if (NOT VIVADO_FOUND)
   message(FATAL_ERROR "Vivado not found.")
endif()
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/scripts/create_xdma_common_ip.tcl.in ${CMAKE_BINARY_DIR}/create_xdma_common_ip.tcl)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/scripts/create_xdma_ip.tcl.in ${CMAKE_BINARY_DIR}/create_xdma_ip.tcl)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/scripts/create_xdma_bypass_ip.tcl.in ${CMAKE_BINARY_DIR}/create_xdma_bypass_ip.tcl)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/scripts/create_xdma_project.tcl.in ${CMAKE_BINARY_DIR}/create_project.tcl)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/scripts/create_tcpip_100g_common_ip.tcl.in ${CMAKE_BINARY_DIR}/create_tcpip_100g_common_ip.tcl)


set (PROJECT_DEPENDS
      ${CMAKE_CURRENT_SOURCE_DIR}/rtl/example_module.vh
      ${CMAKE_CURRENT_SOURCE_DIR}/scripts/create_xdma_project.tcl.in
      ${CMAKE_CURRENT_SOURCE_DIR}/scripts/create_xdma_common_ip.tcl.in
      ${CMAKE_CURRENT_SOURCE_DIR}/scripts/create_xdma_ip.tcl.in
      ${CMAKE_CURRENT_SOURCE_DIR}/scripts/create_xdma_bypass_ip.tcl.in
      ${CMAKE_CURRENT_SOURCE_DIR}/scripts/create_tcpip_100g_common_ip.tcl.in)      

add_custom_target(project
   COMMAND ${VIVADO_BINARY} -mode batch -source ${CMAKE_BINARY_DIR}/create_project.tcl
   DEPENDS ${PROJECT_DEPENDS})
