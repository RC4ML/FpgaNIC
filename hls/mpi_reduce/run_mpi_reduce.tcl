
open_project mpi_reduce_prj

set_top mpi_reduce

add_files ../axi_utils.cpp
add_files ../packet.hpp
add_files mpi_reduce.hpp
add_files mpi_reduce.cpp


add_files -tb test_mpi_reduce.cpp

open_solution "solution1"
set_part {xc7vx690tffg1761-2}
create_clock -period 4 -name default

csynth_design
export_design -format ip_catalog -display_name "mpi_reduce" -description "" -vendor "zju.rc4ml.fpga" -version "0.01"

exit
