set prj_dir [file normalize [get_property DIRECTORY [current_project]]]
set project_name [get_property NAME [current_project]]

set src_dir $prj_dir/$project_name.srcs/sources_1/rtl
set ip_repo $prj_dir/$project_name.srcs/sources_1/iprepo
set ip_dir  $prj_dir/$project_name.srcs/sources_1/ip
set constraints_dir $prj_dir/$project_name.srcs/sources_1/constraints

set_property IP_REPO_PATHS $ip_repo [current_fileset]
update_ip_catalog

# Add sources
add_files $src_dir

add_files -fileset constrs_1 $constraints_dir




create_ip -name tlb -vendor ethz.systems.fpga -library hls -version 0.09 -module_name tlb_ip
generate_target {instantiation_template} [get_files $ip_dir/tlb_ip/tlb_ip.xci]
update_compile_order -fileset sources_1

create_ip -name mem_write_cmd_page_boundary_check_512 -vendor ethz.systems.fpga -library hls -version 0.3 -module_name mem_write_cmd_page_boundary_check_512_ip
generate_target {instantiation_template} [get_files $ip_dir/mem_write_cmd_page_boundary_check_512_ip/mem_write_cmd_page_boundary_check_512_ip.xci]
update_compile_order -fileset sources_1

create_ip -name mem_write_cmd_page_boundary_check_512 -vendor ethz.systems.fpga -library hls -version 0.3 -module_name mem_read_cmd_page_boundary_check_512_ip
generate_target {instantiation_template} [get_files $ip_dir/mem_read_cmd_page_boundary_check_512_ip/mem_read_cmd_page_boundary_check_512_ip.xci]
update_compile_order -fileset sources_1


create_ip -name axi_bram_ctrl -vendor xilinx.com -library ip -version 4.1 -module_name axi_bram_ctrl_0
set_property -dict [list CONFIG.PROTOCOL {AXI4LITE} CONFIG.SUPPORTS_NARROW_BURST {0} CONFIG.SINGLE_PORT_BRAM {1} CONFIG.ECC_TYPE {0} CONFIG.MEM_DEPTH {1024} CONFIG.READ_LATENCY {2}] [get_ips axi_bram_ctrl_0]
generate_target {instantiation_template} [get_files $ip_dir/axi_bram_ctrl_0/axi_bram_ctrl_0.xci]
update_compile_order -fileset sources_1

create_ip -name axis_data_fifo -vendor xilinx.com -library ip -version 2.0 -module_name axis_data_fifo_512_cc
set_property -dict [list CONFIG.TDATA_NUM_BYTES {64} CONFIG.IS_ACLK_ASYNC {1} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} CONFIG.Component_Name {axis_data_fifo_512_cc}] [get_ips axis_data_fifo_512_cc]
generate_target {instantiation_template} [get_files $ip_dir/axis_data_fifo_512_cc/axis_data_fifo_512_cc.xci]
update_compile_order -fileset sources_1

create_ip -name axis_clock_converter -vendor xilinx.com -library ip -version 1.1 -module_name axis_clock_converter_32
set_property -dict [list CONFIG.TDATA_NUM_BYTES {4} CONFIG.Component_Name {axis_clock_converter_32}] [get_ips axis_clock_converter_32]
generate_target {instantiation_template} [get_files $ip_dir/axis_clock_converter_32/axis_clock_converter_32.xci]
update_compile_order -fileset sources_1

create_ip -name axis_clock_converter -vendor xilinx.com -library ip -version 1.1 -module_name axis_clock_converter_96
set_property -dict [list CONFIG.TDATA_NUM_BYTES {12} CONFIG.Component_Name {axis_clock_converter_96}] [get_ips axis_clock_converter_96]
generate_target {instantiation_template} [get_files $ip_dir/axis_clock_converter_96/axis_clock_converter_96.xci]
update_compile_order -fileset sources_1

create_ip -name axis_clock_converter -vendor xilinx.com -library ip -version 1.1 -module_name axis_clock_converter_136
set_property -dict [list CONFIG.TDATA_NUM_BYTES {17} CONFIG.Component_Name {axis_clock_converter_136}] [get_ips axis_clock_converter_136]
generate_target {instantiation_template} [get_files $ip_dir/axis_clock_converter_136/axis_clock_converter_136.xci]
update_compile_order -fileset sources_1

create_ip -name xdma -vendor xilinx.com -library ip -version 4.1 -module_name xdma_0
set_property -dict [list CONFIG.mode_selection {Advanced} CONFIG.pl_link_cap_max_link_width {X16} CONFIG.pl_link_cap_max_link_speed {8.0_GT/s} CONFIG.axi_data_width {512_bit} CONFIG.pf0_device_id {903F} CONFIG.axilite_master_en {true} CONFIG.axilite_master_size {4} CONFIG.axilite_master_scale {Kilobytes} CONFIG.xdma_rnum_rids {64} CONFIG.xdma_wnum_rids {32} CONFIG.SYS_RST_N_BOARD_INTERFACE {pcie_perstn} CONFIG.PCIE_BOARD_INTERFACE {pci_express_x16} CONFIG.en_gt_selection {true} CONFIG.coreclk_freq {500} CONFIG.plltype {QPLL1} CONFIG.xdma_axi_intf_mm {AXI_Stream} CONFIG.xdma_pcie_64bit_en {true} CONFIG.dsc_bypass_rd {0001} CONFIG.dsc_bypass_wr {0001} CONFIG.xdma_sts_ports {true} CONFIG.pf0_msix_cap_table_bir {BAR_3:2} CONFIG.pf0_msix_cap_pba_bir {BAR_3:2} CONFIG.cfg_mgmt_if {false} CONFIG.PF0_DEVICE_ID_mqdma {903F} CONFIG.PF2_DEVICE_ID_mqdma {903F} CONFIG.PF3_DEVICE_ID_mqdma {903F}] [get_ips xdma_0]
generate_target {instantiation_template} [get_files $ip_dir/xdma_0/xdma_0.xci]
update_compile_order -fileset sources_1
