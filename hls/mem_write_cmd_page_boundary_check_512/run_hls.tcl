
open_project mem_write_cmd_page_boundary_check_512_prj

set_top mem_write_cmd_page_boundary_check_512

add_files mem_write_cmd_page_boundary_check_512.hpp
add_files mem_write_cmd_page_boundary_check_512.cpp


add_files -tb test_mem_write_cmd_page_boundary_check_512.cpp

open_solution "solution1"
set_part {xc7vx690tffg1761-2}
create_clock -period 4 -name default

csynth_design
export_design -format ip_catalog -display_name "Write cmd memory page boundary checker 512" -description "" -vendor "ethz.systems.fpga" -version "0.3"

exit
