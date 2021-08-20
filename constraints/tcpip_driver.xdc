set_property PACKAGE_PIN G31 [get_ports sys_100M_p]
set_property PACKAGE_PIN F31 [get_ports sys_100M_n]
set_property IOSTANDARD  DIFF_SSTL12 [get_ports sys_100M_p]
set_property IOSTANDARD  DIFF_SSTL12 [get_ports sys_100M_n]

create_clock -period 10.000 -name sys_100M_clock -add [get_ports sys_100M_p]


###########################set_false_path
set_false_path -from [get_cells network_module_inst/core_reset_reg]
set_false_path -from [get_cells user_rstn_reg]
set_false_path -from [get_cells dma_interface/controller_inst/fpga_control_reg_reg[*]]
set_false_path -to [get_cells dma_interface/controller_inst/fpga_status_reg_r_reg[*]]


set_false_path -from [get_clocks -of_objects [get_pins {network_module_inst/cmac_wrapper_inst/cmac_usplus_axis0_inst/inst/cmac_usplus_axis0_gt_i/inst/gen_gtwizard_gtye4_top.cmac_usplus_axis0_gt_gtwizard_gtye4_inst/gen_gtwizard_gtye4.gen_channel_container[10].gen_enabled_channel.gtye4_channel_wrapper_inst/channel_inst/gtye4_channel_gen.gen_gtye4_channel_inst[0].GTYE4_CHANNEL_PRIM_INST/RXOUTCLK}]] -to [get_clocks -of_objects [get_pins {network_module_inst/cmac_wrapper_inst/cmac_usplus_axis0_inst/inst/cmac_usplus_axis0_gt_i/inst/gen_gtwizard_gtye4_top.cmac_usplus_axis0_gt_gtwizard_gtye4_inst/gen_gtwizard_gtye4.gen_channel_container[10].gen_enabled_channel.gtye4_channel_wrapper_inst/channel_inst/gtye4_channel_gen.gen_gtye4_channel_inst[0].GTYE4_CHANNEL_PRIM_INST/TXOUTCLK}]]
set_false_path -from [get_clocks -of_objects [get_pins {network_module_inst/cmac_wrapper_inst/cmac_usplus_axis0_inst/inst/cmac_usplus_axis0_gt_i/inst/gen_gtwizard_gtye4_top.cmac_usplus_axis0_gt_gtwizard_gtye4_inst/gen_gtwizard_gtye4.gen_channel_container[10].gen_enabled_channel.gtye4_channel_wrapper_inst/channel_inst/gtye4_channel_gen.gen_gtye4_channel_inst[0].GTYE4_CHANNEL_PRIM_INST/TXOUTCLK}]] -to [get_clocks -of_objects [get_pins {network_module_inst/cmac_wrapper_inst/cmac_usplus_axis0_inst/inst/cmac_usplus_axis0_gt_i/inst/gen_gtwizard_gtye4_top.cmac_usplus_axis0_gt_gtwizard_gtye4_inst/gen_gtwizard_gtye4.gen_channel_container[10].gen_enabled_channel.gtye4_channel_wrapper_inst/channel_inst/gtye4_channel_gen.gen_gtye4_channel_inst[0].GTYE4_CHANNEL_PRIM_INST/RXOUTCLK}]]
set_false_path -from [get_clocks -of_objects [get_pins {network_module_inst/cmac_wrapper_inst/cmac_usplus_axis0_inst/inst/cmac_usplus_axis0_gt_i/inst/gen_gtwizard_gtye4_top.cmac_usplus_axis0_gt_gtwizard_gtye4_inst/gen_gtwizard_gtye4.gen_channel_container[10].gen_enabled_channel.gtye4_channel_wrapper_inst/channel_inst/gtye4_channel_gen.gen_gtye4_channel_inst[0].GTYE4_CHANNEL_PRIM_INST/TXOUTCLK}]] -to [get_clocks -of_objects [get_pins user_clk_inst/u_mmcm_0/CLKOUT0]]
set_false_path -from [get_clocks -of_objects [get_pins user_clk_inst/u_mmcm_0/CLKOUT0]] -to [get_clocks -of_objects [get_pins {network_module_inst/cmac_wrapper_inst/cmac_usplus_axis0_inst/inst/cmac_usplus_axis0_gt_i/inst/gen_gtwizard_gtye4_top.cmac_usplus_axis0_gt_gtwizard_gtye4_inst/gen_gtwizard_gtye4.gen_channel_container[10].gen_enabled_channel.gtye4_channel_wrapper_inst/channel_inst/gtye4_channel_gen.gen_gtye4_channel_inst[0].GTYE4_CHANNEL_PRIM_INST/TXOUTCLK}]]

set_false_path -from [get_clocks -of_objects [get_pins inst_ddr0_driver/u_ddr4_0/inst/u_ddr4_infrastructure/gen_mmcme4.u_mmcme_adv_inst/CLKOUT0]] -to [get_clocks -of_objects [get_pins dma_interface/dma_driver_inst/dma_inst/inst/pcie4c_ip_i/inst/gt_top_i/diablo_gt.diablo_gt_phy_wrapper/phy_clk_i/bufg_gt_userclk/O]]


set_false_path -from [get_clocks -of_objects [get_pins user_clk_inst/u_mmcm_0/CLKOUT1]] -to [get_clocks -of_objects [get_pins {network_module_inst/cmac_wrapper_inst/cmac_usplus_axis0_inst/inst/cmac_usplus_axis0_gt_i/inst/gen_gtwizard_gtye4_top.cmac_usplus_axis0_gt_gtwizard_gtye4_inst/gen_gtwizard_gtye4.gen_channel_container[10].gen_enabled_channel.gtye4_channel_wrapper_inst/channel_inst/gtye4_channel_gen.gen_gtye4_channel_inst[0].GTYE4_CHANNEL_PRIM_INST/TXOUTCLK}]]
set_false_path -from [get_clocks -of_objects [get_pins user_clk_inst/u_mmcm_0/CLKOUT1]] -to [get_clocks -of_objects [get_pins user_clk_inst/u_mmcm_0/CLKOUT0]]
set_false_path -from [get_clocks -of_objects [get_pins {network_module_inst/cmac_wrapper_inst/cmac_usplus_axis0_inst/inst/cmac_usplus_axis0_gt_i/inst/gen_gtwizard_gtye4_top.cmac_usplus_axis0_gt_gtwizard_gtye4_inst/gen_gtwizard_gtye4.gen_channel_container[10].gen_enabled_channel.gtye4_channel_wrapper_inst/channel_inst/gtye4_channel_gen.gen_gtye4_channel_inst[0].GTYE4_CHANNEL_PRIM_INST/TXOUTCLK}]] -to [get_clocks -of_objects [get_pins dma_interface/dma_driver_inst/dma_inst/inst/pcie4c_ip_i/inst/gt_top_i/diablo_gt.diablo_gt_phy_wrapper/phy_clk_i/bufg_gt_userclk/O]]

set_false_path -from [get_clocks -of_objects [get_pins dma_interface/dma_driver_inst/dma_inst/inst/pcie4c_ip_i/inst/gt_top_i/diablo_gt.diablo_gt_phy_wrapper/phy_clk_i/bufg_gt_userclk/O]] -to [get_clocks -of_objects [get_pins inst_hbm_driver/u_mmcm_0/CLKOUT0]]

set_false_path -from [get_clocks -of_objects [get_pins inst_hbm_driver/u_mmcm_0/CLKOUT0]] -to [get_clocks -of_objects [get_pins dma_interface/dma_driver_inst/dma_inst/inst/pcie4c_ip_i/inst/gt_top_i/diablo_gt.diablo_gt_phy_wrapper/phy_clk_i/bufg_gt_userclk/O]]

set_false_path -from [get_clocks -of_objects [get_pins dma_interface/dma_driver_inst/dma_inst/inst/pcie4c_ip_i/inst/gt_top_i/diablo_gt.diablo_gt_phy_wrapper/phy_clk_i/bufg_gt_userclk/O]] -to [get_clocks -of_objects [get_pins {network_module_inst/cmac_wrapper_inst/cmac_usplus_axis0_inst/inst/cmac_usplus_axis0_gt_i/inst/gen_gtwizard_gtye4_top.cmac_usplus_axis0_gt_gtwizard_gtye4_inst/gen_gtwizard_gtye4.gen_channel_container[10].gen_enabled_channel.gtye4_channel_wrapper_inst/channel_inst/gtye4_channel_gen.gen_gtye4_channel_inst[0].GTYE4_CHANNEL_PRIM_INST/TXOUTCLK}]]
set_false_path -from [get_clocks -of_objects [get_pins user_clk_inst/u_mmcm_0/CLKOUT2]] -to [get_clocks -of_objects [get_pins dma_interface/dma_driver_inst/dma_inst/inst/pcie4c_ip_i/inst/gt_top_i/diablo_gt.diablo_gt_phy_wrapper/phy_clk_i/bufg_gt_userclk/O]]


######matrix
set_false_path -from [get_clocks -of_objects [get_pins dma_interface/dma_driver_inst/dma_inst/inst/pcie4c_ip_i/inst/gt_top_i/diablo_gt.diablo_gt_phy_wrapper/phy_clk_i/bufg_gt_userclk/O]] -to [get_clocks -of_objects [get_pins inst_hbm_driver/inst_clk_wiz_0/inst/mmcme4_adv_inst/CLKOUT0]]
set_false_path -from [get_clocks -of_objects [get_pins dma_interface/dma_driver_inst/dma_inst/inst/pcie4c_ip_i/inst/gt_top_i/diablo_gt.diablo_gt_phy_wrapper/phy_clk_i/bufg_gt_userclk/O]] -to [get_clocks -of_objects [get_pins inst_hbm_driver/inst_clk_wiz_0/inst/mmcme4_adv_inst/CLKOUT3]]
set_false_path -from [get_clocks -of_objects [get_pins inst_hbm_driver/inst_clk_wiz_0/inst/mmcme4_adv_inst/CLKOUT0]] -to [get_clocks -of_objects [get_pins dma_interface/dma_driver_inst/dma_inst/inst/pcie4c_ip_i/inst/gt_top_i/diablo_gt.diablo_gt_phy_wrapper/phy_clk_i/bufg_gt_userclk/O]]
set_false_path -from [get_clocks -of_objects [get_pins inst_hbm_driver/inst_clk_wiz_0/inst/mmcme4_adv_inst/CLKOUT3]] -to [get_clocks -of_objects [get_pins dma_interface/dma_driver_inst/dma_inst/inst/pcie4c_ip_i/inst/gt_top_i/diablo_gt.diablo_gt_phy_wrapper/phy_clk_i/bufg_gt_userclk/O]]
#######################pblock

create_pblock pblock_sgd_top_bw_inst2
resize_pblock pblock_sgd_top_bw_inst2 -add SLR2:SLR2
add_cells_to_pblock pblock_sgd_top_bw_inst2 [get_cells network_module_inst]

create_pblock pblock_sgd_top_bw_inst1
resize_pblock pblock_sgd_top_bw_inst1 -add SLR1:SLR1
add_cells_to_pblock pblock_sgd_top_bw_inst1 [get_cells network_stack_inst]
#add_cells_to_pblock pblock_sgd_top_bw_inst1 [get_cells inst_ddr1_driver]
#add_cells_to_pblock pblock_sgd_top_bw_inst1 [get_cells inst_mem_inf_transfer1]


#create_pblock pblock_sgd_top_bw_inst0
#resize_pblock pblock_sgd_top_bw_inst0 -add SLR0:SLR0
#add_cells_to_pblock pblock_sgd_top_bw_inst0 [get_cells dma_interface]
#add_cells_to_pblock pblock_sgd_top_bw_inst0 [get_cells inst_ddr0_driver]
#add_cells_to_pblock pblock_sgd_top_bw_inst0 [get_cells inst_mem_inf_transfer0]
#add_cells_to_pblock pblock_sgd_top_bw_inst0 [get_cells dma_put_data_to_fpga_inst]
#add_cells_to_pblock pblock_sgd_top_bw_inst0 [get_cells dma_get_data_from_fpga_inst]
#add_cells_to_pblock pblock_sgd_top_bw_inst0 [get_cells inst_mpi_reduce_control/mpi_reduce_inst]
