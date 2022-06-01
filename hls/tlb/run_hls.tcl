
open_project tlb_prj

set_top tlb

add_files ../packet.hpp
add_files tlb.hpp
add_files tlb.cpp


add_files -tb test_tlb.cpp

open_solution "solution1"
set_part {xc7vx690tffg1761-2}
create_clock -period 4 -name default

csynth_design
export_design -format ip_catalog -display_name "TLB" -description "" -vendor "ethz.systems.fpga" -version "0.09"

exit
