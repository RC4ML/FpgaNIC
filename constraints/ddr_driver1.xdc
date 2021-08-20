set_property PACKAGE_PIN BJ43 [get_ports ddr0_sys_100M_p]
set_property PACKAGE_PIN BJ44 [get_ports ddr0_sys_100M_n]
set_property IOSTANDARD  DIFF_SSTL12 [get_ports ddr0_sys_100M_p]
set_property IOSTANDARD  DIFF_SSTL12 [get_ports ddr0_sys_100M_n]

create_clock -name ddr0_sys_clock -period 10 [get_ports ddr0_sys_100M_p]


set_property PACKAGE_PIN BH6 [get_ports ddr1_sys_100M_p]
set_property PACKAGE_PIN BJ6 [get_ports ddr1_sys_100M_n]
set_property IOSTANDARD  DIFF_SSTL12 [get_ports ddr1_sys_100M_p]
set_property IOSTANDARD  DIFF_SSTL12 [get_ports ddr1_sys_100M_n]

create_clock -name ddr1_sys_clock -period 10 [get_ports ddr1_sys_100M_p]



##
set_false_path -to [get_pins -hier *sync_reg[0]/D]
##
set_property C_USER_SCAN_CHAIN 1 [get_debug_cores dbg_hub]
connect_debug_port dbg_hub/clk [get_nets */APB_0_PCLK]

