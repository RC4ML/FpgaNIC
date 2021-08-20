`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2020/02/20 21:50:13
// Design Name: 
// Module Name: hbm_driver
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////
`include"example_module.vh"

module mem_choice_off_path( 

    //user clock input
    input wire                  clk,
    input wire                  rstn,

    //control signal
    input wire                  mem_choice, //1:user ddr conn 0:dma ddr conn
	
    //ddr memory streams
    axis_mem_cmd.master    		m_axis_mem_write_cmd[0:1],
    axis_mem_status.slave     	s_axis_mem_write_sts[0:1],
    axi_stream.master   		m_axis_mem_write_data[0:1],
    axis_mem_cmd.master    		m_axis_mem_read_cmd[0:1],
    axis_mem_status.slave     	s_axis_mem_read_sts[0:1],
    axi_stream.slave    		s_axis_mem_read_data[0:1],    

    //dma memory streams
    axis_mem_cmd.slave          s_axis_mem_dma_read_cmd[0:1],
    axis_mem_cmd.slave          s_axis_mem_dma_write_cmd[0:1],
    axis_mem_status.master      m_axis_mem_dma_read_sts[0:1],
    axis_mem_status.master      m_axis_mem_dma_write_sts[0:1],
    axi_stream.master           m_axis_mem_dma_read_data[0:1],
    axi_stream.slave            s_axis_mem_dma_write_data[0:1],   

    //off path user interface streams
    axis_mem_cmd.slave          s_axis_mem_user_read_cmd[0:1],
    axis_mem_cmd.slave          s_axis_mem_user_write_cmd[0:1],
    axis_mem_status.master      m_axis_mem_user_read_sts[0:1],
    axis_mem_status.master      m_axis_mem_user_write_sts[0:1],
    axi_stream.master           m_axis_mem_user_read_data[0:1],
    axi_stream.slave            s_axis_mem_user_write_data[0:1], 

	//control reg
	output wire[31:0][31:0]		status_reg

	
	);

    // memory cmd streams
    axis_mem_cmd        axis_mem_read_cmd[2]();
    axis_mem_cmd        axis_mem_write_cmd[2]();
    // memory sts streams
    axis_mem_status     axis_mem_read_sts[2]();
    axis_mem_status     axis_mem_write_sts[2]();
    // memory data streams
    axi_stream          axis_mem_read_data[2]();
    axi_stream          axis_mem_write_data[2](); 

    // memory cmd streams
    axis_mem_cmd        axis_mem_dma_read_cmd[2]();
    axis_mem_cmd        axis_mem_dma_write_cmd[2]();
    // memory sts streams
    axis_mem_status     axis_mem_dma_read_sts[2]();
    axis_mem_status     axis_mem_dma_write_sts[2]();
    // memory data streams
    axi_stream          axis_mem_dma_read_data[2]();
    axi_stream          axis_mem_dma_write_data[2]();  

    // memory cmd streams
    axis_mem_cmd        axis_mem_user_read_cmd[2]();
    axis_mem_cmd        axis_mem_user_write_cmd[2]();
    // memory sts streams
    axis_mem_status     axis_mem_user_read_sts[2]();
    axis_mem_status     axis_mem_user_write_sts[2]();
    // memory data streams
    axi_stream          axis_mem_user_read_data[2]();
    axi_stream          axis_mem_user_write_data[2]();


    axis_register_slice_96 axis_mem_write_cmd0 (
        .aclk(clk),                    // input wire aclk
        .aresetn(rstn),              // input wire aresetn
        .s_axis_tvalid(axis_mem_write_cmd[0].valid),  // input wire s_axis_tvalid
        .s_axis_tready(axis_mem_write_cmd[0].ready),  // output wire s_axis_tready
        .s_axis_tdata({axis_mem_write_cmd[0].address,axis_mem_write_cmd[0].length}),    // input wire [95 : 0] s_axis_tdata
        .m_axis_tvalid(m_axis_mem_write_cmd[0].valid),  // output wire m_axis_tvalid
        .m_axis_tready(m_axis_mem_write_cmd[0].ready),  // input wire m_axis_tready
        .m_axis_tdata({m_axis_mem_write_cmd[0].address,m_axis_mem_write_cmd[0].length})    // output wire [95 : 0] m_axis_tdata
    );

    axis_register_slice_8 axis_mem_write_sts0 (
        .aclk(clk),                    // input wire aclk
        .aresetn(rstn),              // input wire aresetn
        .s_axis_tvalid(s_axis_mem_write_sts[0].valid),  // input wire s_axis_tvalid
        .s_axis_tready(s_axis_mem_write_sts[0].ready),  // output wire s_axis_tready
        .s_axis_tdata(s_axis_mem_write_sts[0].data),    // input wire [7 : 0] s_axis_tdata
        .m_axis_tvalid(axis_mem_write_sts[0].valid),  // output wire m_axis_tvalid
        .m_axis_tready(axis_mem_write_sts[0].ready),  // input wire m_axis_tready
        .m_axis_tdata(axis_mem_write_sts[0].data)    // output wire [7 : 0] m_axis_tdata
    );

    axis_register_slice_512 axis_mem_write_data0 (
        .aclk(clk),                    // input wire aclk
        .aresetn(rstn),              // input wire aresetn
        .s_axis_tvalid(axis_mem_write_data[0].valid),  // input wire s_axis_tvalid
        .s_axis_tready(axis_mem_write_data[0].ready),  // output wire s_axis_tready
        .s_axis_tdata(axis_mem_write_data[0].data),    // input wire [511 : 0] s_axis_tdata
        .s_axis_tkeep(axis_mem_write_data[0].keep),    // input wire [63 : 0] s_axis_tkeep
        .s_axis_tlast(axis_mem_write_data[0].last),    // input wire s_axis_tlast
        .m_axis_tvalid(m_axis_mem_write_data[0].valid),  // output wire m_axis_tvalid
        .m_axis_tready(m_axis_mem_write_data[0].ready),  // input wire m_axis_tready
        .m_axis_tdata(m_axis_mem_write_data[0].data),    // output wire [511 : 0] m_axis_tdata
        .m_axis_tkeep(m_axis_mem_write_data[0].keep),    // output wire [63 : 0] m_axis_tkeep
        .m_axis_tlast(m_axis_mem_write_data[0].last)    // output wire m_axis_tlast
    );

    axis_register_slice_96 axis_mem_write_cmd1 (
        .aclk(clk),                    // input wire aclk
        .aresetn(rstn),              // input wire aresetn
        .s_axis_tvalid(axis_mem_write_cmd[1].valid),  // input wire s_axis_tvalid
        .s_axis_tready(axis_mem_write_cmd[1].ready),  // output wire s_axis_tready
        .s_axis_tdata({axis_mem_write_cmd[1].address,axis_mem_write_cmd[1].length}),    // input wire [95 : 0] s_axis_tdata
        .m_axis_tvalid(m_axis_mem_write_cmd[1].valid),  // output wire m_axis_tvalid
        .m_axis_tready(m_axis_mem_write_cmd[1].ready),  // input wire m_axis_tready
        .m_axis_tdata({m_axis_mem_write_cmd[1].address,m_axis_mem_write_cmd[1].length})    // output wire [95 : 0] m_axis_tdata
    );

    axis_register_slice_8 axis_mem_write_sts1 (
        .aclk(clk),                    // input wire aclk
        .aresetn(rstn),              // input wire aresetn
        .s_axis_tvalid(s_axis_mem_write_sts[1].valid),  // input wire s_axis_tvalid
        .s_axis_tready(s_axis_mem_write_sts[1].ready),  // output wire s_axis_tready
        .s_axis_tdata(s_axis_mem_write_sts[1].data),    // input wire [7 : 0] s_axis_tdata
        .m_axis_tvalid(axis_mem_write_sts[1].valid),  // output wire m_axis_tvalid
        .m_axis_tready(axis_mem_write_sts[1].ready),  // input wire m_axis_tready
        .m_axis_tdata(axis_mem_write_sts[1].data)    // output wire [7 : 0] m_axis_tdata
    );

    axis_register_slice_512 axis_mem_write_data1 (
        .aclk(clk),                    // input wire aclk
        .aresetn(rstn),              // input wire aresetn
        .s_axis_tvalid(axis_mem_write_data[1].valid),  // input wire s_axis_tvalid
        .s_axis_tready(axis_mem_write_data[1].ready),  // output wire s_axis_tready
        .s_axis_tdata(axis_mem_write_data[1].data),    // input wire [511 : 0] s_axis_tdata
        .s_axis_tkeep(axis_mem_write_data[1].keep),    // input wire [63 : 0] s_axis_tkeep
        .s_axis_tlast(axis_mem_write_data[1].last),    // input wire s_axis_tlast
        .m_axis_tvalid(m_axis_mem_write_data[1].valid),  // output wire m_axis_tvalid
        .m_axis_tready(m_axis_mem_write_data[1].ready),  // input wire m_axis_tready
        .m_axis_tdata(m_axis_mem_write_data[1].data),    // output wire [511 : 0] m_axis_tdata
        .m_axis_tkeep(m_axis_mem_write_data[1].keep),    // output wire [63 : 0] m_axis_tkeep
        .m_axis_tlast(m_axis_mem_write_data[1].last)    // output wire m_axis_tlast
    );    

    axis_register_slice_96 axis_mem_read_cmd0 (
        .aclk(clk),                    // input wire aclk
        .aresetn(rstn),              // input wire aresetn
        .s_axis_tvalid(axis_mem_read_cmd[0].valid),  // input wire s_axis_tvalid
        .s_axis_tready(axis_mem_read_cmd[0].ready),  // output wire s_axis_tready
        .s_axis_tdata({axis_mem_read_cmd[0].address,axis_mem_read_cmd[0].length}),    // input wire [95 : 0] s_axis_tdata
        .m_axis_tvalid(m_axis_mem_read_cmd[0].valid),  // output wire m_axis_tvalid
        .m_axis_tready(m_axis_mem_read_cmd[0].ready),  // input wire m_axis_tready
        .m_axis_tdata({m_axis_mem_read_cmd[0].address,m_axis_mem_read_cmd[0].length})    // output wire [95 : 0] m_axis_tdata
    );

    axis_register_slice_8 axis_mem_read_sts0 (
        .aclk(clk),                    // input wire aclk
        .aresetn(rstn),              // input wire aresetn
        .s_axis_tvalid(s_axis_mem_read_sts[0].valid),  // input wire s_axis_tvalid
        .s_axis_tready(s_axis_mem_read_sts[0].ready),  // output wire s_axis_tready
        .s_axis_tdata(s_axis_mem_read_sts[0].data),    // input wire [7 : 0] s_axis_tdata
        .m_axis_tvalid(axis_mem_read_sts[0].valid),  // output wire m_axis_tvalid
        .m_axis_tready(axis_mem_read_sts[0].ready),  // input wire m_axis_tready
        .m_axis_tdata(axis_mem_read_sts[0].data)    // output wire [7 : 0] m_axis_tdata
    );

    axis_register_slice_512 axis_mem_read_data0 (
        .aclk(clk),                    // input wire aclk
        .aresetn(rstn),              // input wire aresetn
        .s_axis_tvalid(s_axis_mem_read_data[0].valid),  // input wire s_axis_tvalid
        .s_axis_tready(s_axis_mem_read_data[0].ready),  // output wire s_axis_tready
        .s_axis_tdata(s_axis_mem_read_data[0].data),    // input wire [511 : 0] s_axis_tdata
        .s_axis_tkeep(s_axis_mem_read_data[0].keep),    // input wire [63 : 0] s_axis_tkeep
        .s_axis_tlast(s_axis_mem_read_data[0].last),    // input wire s_axis_tlast
        .m_axis_tvalid(axis_mem_read_data[0].valid),  // output wire m_axis_tvalid
        .m_axis_tready(axis_mem_read_data[0].ready),  // input wire m_axis_tready
        .m_axis_tdata(axis_mem_read_data[0].data),    // output wire [511 : 0] m_axis_tdata
        .m_axis_tkeep(axis_mem_read_data[0].keep),    // output wire [63 : 0] m_axis_tkeep
        .m_axis_tlast(axis_mem_read_data[0].last)    // output wire m_axis_tlast
    );

    axis_register_slice_96 axis_mem_read_cmd1 (
        .aclk(clk),                    // input wire aclk
        .aresetn(rstn),              // input wire aresetn
        .s_axis_tvalid(axis_mem_read_cmd[1].valid),  // input wire s_axis_tvalid
        .s_axis_tready(axis_mem_read_cmd[1].ready),  // output wire s_axis_tready
        .s_axis_tdata({axis_mem_read_cmd[1].address,axis_mem_read_cmd[1].length}),    // input wire [95 : 0] s_axis_tdata
        .m_axis_tvalid(m_axis_mem_read_cmd[1].valid),  // output wire m_axis_tvalid
        .m_axis_tready(m_axis_mem_read_cmd[1].ready),  // input wire m_axis_tready
        .m_axis_tdata({m_axis_mem_read_cmd[1].address,m_axis_mem_read_cmd[1].length})    // output wire [95 : 0] m_axis_tdata
    );

    axis_register_slice_8 axis_mem_read_sts1 (
        .aclk(clk),                    // input wire aclk
        .aresetn(rstn),              // input wire aresetn
        .s_axis_tvalid(s_axis_mem_read_sts[1].valid),  // input wire s_axis_tvalid
        .s_axis_tready(s_axis_mem_read_sts[1].ready),  // output wire s_axis_tready
        .s_axis_tdata(s_axis_mem_read_sts[1].data),    // input wire [7 : 0] s_axis_tdata
        .m_axis_tvalid(axis_mem_read_sts[1].valid),  // output wire m_axis_tvalid
        .m_axis_tready(axis_mem_read_sts[1].ready),  // input wire m_axis_tready
        .m_axis_tdata(axis_mem_read_sts[1].data)    // output wire [7 : 0] m_axis_tdata
    );

    axis_register_slice_512 axis_mem_read_data1 (
        .aclk(clk),                    // input wire aclk
        .aresetn(rstn),              // input wire aresetn
        .s_axis_tvalid(s_axis_mem_read_data[1].valid),  // input wire s_axis_tvalid
        .s_axis_tready(s_axis_mem_read_data[1].ready),  // output wire s_axis_tready
        .s_axis_tdata(s_axis_mem_read_data[1].data),    // input wire [511 : 0] s_axis_tdata
        .s_axis_tkeep(s_axis_mem_read_data[1].keep),    // input wire [63 : 0] s_axis_tkeep
        .s_axis_tlast(s_axis_mem_read_data[1].last),    // input wire s_axis_tlast
        .m_axis_tvalid(axis_mem_read_data[1].valid),  // output wire m_axis_tvalid
        .m_axis_tready(axis_mem_read_data[1].ready),  // input wire m_axis_tready
        .m_axis_tdata(axis_mem_read_data[1].data),    // output wire [511 : 0] m_axis_tdata
        .m_axis_tkeep(axis_mem_read_data[1].keep),    // output wire [63 : 0] m_axis_tkeep
        .m_axis_tlast(axis_mem_read_data[1].last)    // output wire m_axis_tlast
    );


    //////////////////////////////////////////////mem regster



    axis_register_slice_96 axis_mem_dma_write_cmd0 (
        .aclk(clk),                    // input wire aclk
        .aresetn(rstn),              // input wire aresetn
        .s_axis_tvalid(s_axis_mem_dma_write_cmd[0].valid),  // input wire s_axis_tvalid
        .s_axis_tready(s_axis_mem_dma_write_cmd[0].ready),  // output wire s_axis_tready
        .s_axis_tdata({s_axis_mem_dma_write_cmd[0].address,s_axis_mem_dma_write_cmd[0].length}),    // input wire [95 : 0] s_axis_tdata
        .m_axis_tvalid(axis_mem_dma_write_cmd[0].valid),  // output wire m_axis_tvalid
        .m_axis_tready(axis_mem_dma_write_cmd[0].ready),  // input wire m_axis_tready
        .m_axis_tdata({axis_mem_dma_write_cmd[0].address,axis_mem_dma_write_cmd[0].length})    // output wire [95 : 0] m_axis_tdata
    );

    axis_register_slice_8 axis_mem_dma_write_sts0 (
        .aclk(clk),                    // input wire aclk
        .aresetn(rstn),              // input wire aresetn
        .s_axis_tvalid(axis_mem_dma_write_sts[0].valid),  // input wire s_axis_tvalid
        .s_axis_tready(axis_mem_dma_write_sts[0].ready),  // output wire s_axis_tready
        .s_axis_tdata(axis_mem_dma_write_sts[0].data),    // input wire [7 : 0] s_axis_tdata
        .m_axis_tvalid(m_axis_mem_dma_write_sts[0].valid),  // output wire m_axis_tvalid
        .m_axis_tready(m_axis_mem_dma_write_sts[0].ready),  // input wire m_axis_tready
        .m_axis_tdata(m_axis_mem_dma_write_sts[0].data)    // output wire [7 : 0] m_axis_tdata
    );

    axis_register_slice_512 axis_mem_dma_write_data0 (
        .aclk(clk),                    // input wire aclk
        .aresetn(rstn),              // input wire aresetn
        .s_axis_tvalid(s_axis_mem_dma_write_data[0].valid),  // input wire s_axis_tvalid
        .s_axis_tready(s_axis_mem_dma_write_data[0].ready),  // output wire s_axis_tready
        .s_axis_tdata(s_axis_mem_dma_write_data[0].data),    // input wire [511 : 0] s_axis_tdata
        .s_axis_tkeep(s_axis_mem_dma_write_data[0].keep),    // input wire [63 : 0] s_axis_tkeep
        .s_axis_tlast(s_axis_mem_dma_write_data[0].last),    // input wire s_axis_tlast
        .m_axis_tvalid(axis_mem_dma_write_data[0].valid),  // output wire m_axis_tvalid
        .m_axis_tready(axis_mem_dma_write_data[0].ready),  // input wire m_axis_tready
        .m_axis_tdata(axis_mem_dma_write_data[0].data),    // output wire [511 : 0] m_axis_tdata
        .m_axis_tkeep(axis_mem_dma_write_data[0].keep),    // output wire [63 : 0] m_axis_tkeep
        .m_axis_tlast(axis_mem_dma_write_data[0].last)    // output wire m_axis_tlast
    );

    axis_register_slice_96 axis_mem_dma_write_cmd1 (
        .aclk(clk),                    // input wire aclk
        .aresetn(rstn),              // input wire aresetn
        .s_axis_tvalid(s_axis_mem_dma_write_cmd[1].valid),  // input wire s_axis_tvalid
        .s_axis_tready(s_axis_mem_dma_write_cmd[1].ready),  // output wire s_axis_tready
        .s_axis_tdata({s_axis_mem_dma_write_cmd[1].address,s_axis_mem_dma_write_cmd[1].length}),    // input wire [95 : 0] s_axis_tdata
        .m_axis_tvalid(axis_mem_dma_write_cmd[1].valid),  // output wire m_axis_tvalid
        .m_axis_tready(axis_mem_dma_write_cmd[1].ready),  // input wire m_axis_tready
        .m_axis_tdata({axis_mem_dma_write_cmd[1].address,axis_mem_dma_write_cmd[1].length})    // output wire [95 : 0] m_axis_tdata
    );

    axis_register_slice_8 axis_mem_dma_write_sts1 (
        .aclk(clk),                    // input wire aclk
        .aresetn(rstn),              // input wire aresetn
        .s_axis_tvalid(axis_mem_dma_write_sts[1].valid),  // input wire s_axis_tvalid
        .s_axis_tready(axis_mem_dma_write_sts[1].ready),  // output wire s_axis_tready
        .s_axis_tdata(axis_mem_dma_write_sts[1].data),    // input wire [7 : 0] s_axis_tdata
        .m_axis_tvalid(m_axis_mem_dma_write_sts[1].valid),  // output wire m_axis_tvalid
        .m_axis_tready(m_axis_mem_dma_write_sts[1].ready),  // input wire m_axis_tready
        .m_axis_tdata(m_axis_mem_dma_write_sts[1].data)    // output wire [7 : 0] m_axis_tdata
    );

    axis_register_slice_512 axis_mem_dma_write_data1 (
        .aclk(clk),                    // input wire aclk
        .aresetn(rstn),              // input wire aresetn
        .s_axis_tvalid(s_axis_mem_dma_write_data[1].valid),  // input wire s_axis_tvalid
        .s_axis_tready(s_axis_mem_dma_write_data[1].ready),  // output wire s_axis_tready
        .s_axis_tdata(s_axis_mem_dma_write_data[1].data),    // input wire [511 : 0] s_axis_tdata
        .s_axis_tkeep(s_axis_mem_dma_write_data[1].keep),    // input wire [63 : 0] s_axis_tkeep
        .s_axis_tlast(s_axis_mem_dma_write_data[1].last),    // input wire s_axis_tlast
        .m_axis_tvalid(axis_mem_dma_write_data[1].valid),  // output wire m_axis_tvalid
        .m_axis_tready(axis_mem_dma_write_data[1].ready),  // input wire m_axis_tready
        .m_axis_tdata(axis_mem_dma_write_data[1].data),    // output wire [511 : 0] m_axis_tdata
        .m_axis_tkeep(axis_mem_dma_write_data[1].keep),    // output wire [63 : 0] m_axis_tkeep
        .m_axis_tlast(axis_mem_dma_write_data[1].last)    // output wire m_axis_tlast
    );    


    axis_register_slice_96 axis_mem_dma_read_cmd0 (
        .aclk(clk),                    // input wire aclk
        .aresetn(rstn),              // input wire aresetn
        .s_axis_tvalid(s_axis_mem_dma_read_cmd[0].valid),  // input wire s_axis_tvalid
        .s_axis_tready(s_axis_mem_dma_read_cmd[0].ready),  // output wire s_axis_tready
        .s_axis_tdata({s_axis_mem_dma_read_cmd[0].address,s_axis_mem_dma_read_cmd[0].length}),    // input wire [95 : 0] s_axis_tdata
        .m_axis_tvalid(axis_mem_dma_read_cmd[0].valid),  // output wire m_axis_tvalid
        .m_axis_tready(axis_mem_dma_read_cmd[0].ready),  // input wire m_axis_tready
        .m_axis_tdata({axis_mem_dma_read_cmd[0].address,axis_mem_dma_read_cmd[0].length})    // output wire [95 : 0] m_axis_tdata
    );

    axis_register_slice_8 axis_mem_dma_read_sts0 (
        .aclk(clk),                    // input wire aclk
        .aresetn(rstn),              // input wire aresetn
        .s_axis_tvalid(axis_mem_dma_read_sts[0].valid),  // input wire s_axis_tvalid
        .s_axis_tready(axis_mem_dma_read_sts[0].ready),  // output wire s_axis_tready
        .s_axis_tdata(axis_mem_dma_read_sts[0].data),    // input wire [7 : 0] s_axis_tdata
        .m_axis_tvalid(m_axis_mem_dma_read_sts[0].valid),  // output wire m_axis_tvalid
        .m_axis_tready(m_axis_mem_dma_read_sts[0].ready),  // input wire m_axis_tready
        .m_axis_tdata(m_axis_mem_dma_read_sts[0].data)    // output wire [7 : 0] m_axis_tdata
    );

    axis_register_slice_512 axis_mem_dma_read_data0 (
        .aclk(clk),                    // input wire aclk
        .aresetn(rstn),              // input wire aresetn
        .s_axis_tvalid(axis_mem_dma_read_data[0].valid),  // input wire s_axis_tvalid
        .s_axis_tready(axis_mem_dma_read_data[0].ready),  // output wire s_axis_tready
        .s_axis_tdata(axis_mem_dma_read_data[0].data),    // input wire [511 : 0] s_axis_tdata
        .s_axis_tkeep(axis_mem_dma_read_data[0].keep),    // input wire [63 : 0] s_axis_tkeep
        .s_axis_tlast(axis_mem_dma_read_data[0].last),    // input wire s_axis_tlast
        .m_axis_tvalid(m_axis_mem_dma_read_data[0].valid),  // output wire m_axis_tvalid
        .m_axis_tready(m_axis_mem_dma_read_data[0].ready),  // input wire m_axis_tready
        .m_axis_tdata(m_axis_mem_dma_read_data[0].data),    // output wire [511 : 0] m_axis_tdata
        .m_axis_tkeep(m_axis_mem_dma_read_data[0].keep),    // output wire [63 : 0] m_axis_tkeep
        .m_axis_tlast(m_axis_mem_dma_read_data[0].last)    // output wire m_axis_tlast
    );

    axis_register_slice_96 axis_mem_dma_read_cmd1 (
        .aclk(clk),                    // input wire aclk
        .aresetn(rstn),              // input wire aresetn
        .s_axis_tvalid(s_axis_mem_dma_read_cmd[1].valid),  // input wire s_axis_tvalid
        .s_axis_tready(s_axis_mem_dma_read_cmd[1].ready),  // output wire s_axis_tready
        .s_axis_tdata({s_axis_mem_dma_read_cmd[1].address,s_axis_mem_dma_read_cmd[1].length}),    // input wire [95 : 0] s_axis_tdata
        .m_axis_tvalid(axis_mem_dma_read_cmd[1].valid),  // output wire m_axis_tvalid
        .m_axis_tready(axis_mem_dma_read_cmd[1].ready),  // input wire m_axis_tready
        .m_axis_tdata({axis_mem_dma_read_cmd[1].address,axis_mem_dma_read_cmd[1].length})    // output wire [95 : 0] m_axis_tdata
    );

    axis_register_slice_8 axis_mem_dma_read_sts1 (
        .aclk(clk),                    // input wire aclk
        .aresetn(rstn),              // input wire aresetn
        .s_axis_tvalid(axis_mem_dma_read_sts[1].valid),  // input wire s_axis_tvalid
        .s_axis_tready(axis_mem_dma_read_sts[1].ready),  // output wire s_axis_tready
        .s_axis_tdata(axis_mem_dma_read_sts[1].data),    // input wire [7 : 0] s_axis_tdata
        .m_axis_tvalid(m_axis_mem_dma_read_sts[1].valid),  // output wire m_axis_tvalid
        .m_axis_tready(m_axis_mem_dma_read_sts[1].ready),  // input wire m_axis_tready
        .m_axis_tdata(m_axis_mem_dma_read_sts[1].data)    // output wire [7 : 0] m_axis_tdata
    );

    axis_register_slice_512 axis_mem_dma_read_data1 (
        .aclk(clk),                    // input wire aclk
        .aresetn(rstn),              // input wire aresetn
        .s_axis_tvalid(axis_mem_dma_read_data[1].valid),  // input wire s_axis_tvalid
        .s_axis_tready(axis_mem_dma_read_data[1].ready),  // output wire s_axis_tready
        .s_axis_tdata(axis_mem_dma_read_data[1].data),    // input wire [511 : 0] s_axis_tdata
        .s_axis_tkeep(axis_mem_dma_read_data[1].keep),    // input wire [63 : 0] s_axis_tkeep
        .s_axis_tlast(axis_mem_dma_read_data[1].last),    // input wire s_axis_tlast
        .m_axis_tvalid(m_axis_mem_dma_read_data[1].valid),  // output wire m_axis_tvalid
        .m_axis_tready(m_axis_mem_dma_read_data[1].ready),  // input wire m_axis_tready
        .m_axis_tdata(m_axis_mem_dma_read_data[1].data),    // output wire [511 : 0] m_axis_tdata
        .m_axis_tkeep(m_axis_mem_dma_read_data[1].keep),    // output wire [63 : 0] m_axis_tkeep
        .m_axis_tlast(m_axis_mem_dma_read_data[1].last)    // output wire m_axis_tlast
    );    

    //////////////////////////////////////////////mem regster



    axis_register_slice_96 axis_mem_user_write_cmd0 (
        .aclk(clk),                    // input wire aclk
        .aresetn(rstn),              // input wire aresetn
        .s_axis_tvalid(s_axis_mem_user_write_cmd[0].valid),  // input wire s_axis_tvalid
        .s_axis_tready(s_axis_mem_user_write_cmd[0].ready),  // output wire s_axis_tready
        .s_axis_tdata({s_axis_mem_user_write_cmd[0].address,s_axis_mem_user_write_cmd[0].length}),    // input wire [95 : 0] s_axis_tdata
        .m_axis_tvalid(axis_mem_user_write_cmd[0].valid),  // output wire m_axis_tvalid
        .m_axis_tready(axis_mem_user_write_cmd[0].ready),  // input wire m_axis_tready
        .m_axis_tdata({axis_mem_user_write_cmd[0].address,axis_mem_user_write_cmd[0].length})    // output wire [95 : 0] m_axis_tdata
    );

    axis_register_slice_8 axis_mem_user_write_sts0 (
        .aclk(clk),                    // input wire aclk
        .aresetn(rstn),              // input wire aresetn
        .s_axis_tvalid(axis_mem_user_write_sts[0].valid),  // input wire s_axis_tvalid
        .s_axis_tready(axis_mem_user_write_sts[0].ready),  // output wire s_axis_tready
        .s_axis_tdata(axis_mem_user_write_sts[0].data),    // input wire [7 : 0] s_axis_tdata
        .m_axis_tvalid(m_axis_mem_user_write_sts[0].valid),  // output wire m_axis_tvalid
        .m_axis_tready(m_axis_mem_user_write_sts[0].ready),  // input wire m_axis_tready
        .m_axis_tdata(m_axis_mem_user_write_sts[0].data)    // output wire [7 : 0] m_axis_tdata
    );

    axis_register_slice_512 axis_mem_user_write_data0 (
        .aclk(clk),                    // input wire aclk
        .aresetn(rstn),              // input wire aresetn
        .s_axis_tvalid(s_axis_mem_user_write_data[0].valid),  // input wire s_axis_tvalid
        .s_axis_tready(s_axis_mem_user_write_data[0].ready),  // output wire s_axis_tready
        .s_axis_tdata(s_axis_mem_user_write_data[0].data),    // input wire [511 : 0] s_axis_tdata
        .s_axis_tkeep(s_axis_mem_user_write_data[0].keep),    // input wire [63 : 0] s_axis_tkeep
        .s_axis_tlast(s_axis_mem_user_write_data[0].last),    // input wire s_axis_tlast
        .m_axis_tvalid(axis_mem_user_write_data[0].valid),  // output wire m_axis_tvalid
        .m_axis_tready(axis_mem_user_write_data[0].ready),  // input wire m_axis_tready
        .m_axis_tdata(axis_mem_user_write_data[0].data),    // output wire [511 : 0] m_axis_tdata
        .m_axis_tkeep(axis_mem_user_write_data[0].keep),    // output wire [63 : 0] m_axis_tkeep
        .m_axis_tlast(axis_mem_user_write_data[0].last)    // output wire m_axis_tlast
    );

    axis_register_slice_96 axis_mem_user_write_cmd1 (
        .aclk(clk),                    // input wire aclk
        .aresetn(rstn),              // input wire aresetn
        .s_axis_tvalid(s_axis_mem_user_write_cmd[1].valid),  // input wire s_axis_tvalid
        .s_axis_tready(s_axis_mem_user_write_cmd[1].ready),  // output wire s_axis_tready
        .s_axis_tdata({s_axis_mem_user_write_cmd[1].address,s_axis_mem_user_write_cmd[1].length}),    // input wire [95 : 0] s_axis_tdata
        .m_axis_tvalid(axis_mem_user_write_cmd[1].valid),  // output wire m_axis_tvalid
        .m_axis_tready(axis_mem_user_write_cmd[1].ready),  // input wire m_axis_tready
        .m_axis_tdata({axis_mem_user_write_cmd[1].address,axis_mem_user_write_cmd[1].length})    // output wire [95 : 0] m_axis_tdata
    );

    axis_register_slice_8 axis_mem_user_write_sts1 (
        .aclk(clk),                    // input wire aclk
        .aresetn(rstn),              // input wire aresetn
        .s_axis_tvalid(axis_mem_user_write_sts[1].valid),  // input wire s_axis_tvalid
        .s_axis_tready(axis_mem_user_write_sts[1].ready),  // output wire s_axis_tready
        .s_axis_tdata(axis_mem_user_write_sts[1].data),    // input wire [7 : 0] s_axis_tdata
        .m_axis_tvalid(m_axis_mem_user_write_sts[1].valid),  // output wire m_axis_tvalid
        .m_axis_tready(m_axis_mem_user_write_sts[1].ready),  // input wire m_axis_tready
        .m_axis_tdata(m_axis_mem_user_write_sts[1].data)    // output wire [7 : 0] m_axis_tdata
    );

    axis_register_slice_512 axis_mem_user_write_data1 (
        .aclk(clk),                    // input wire aclk
        .aresetn(rstn),              // input wire aresetn
        .s_axis_tvalid(s_axis_mem_user_write_data[1].valid),  // input wire s_axis_tvalid
        .s_axis_tready(s_axis_mem_user_write_data[1].ready),  // output wire s_axis_tready
        .s_axis_tdata(s_axis_mem_user_write_data[1].data),    // input wire [511 : 0] s_axis_tdata
        .s_axis_tkeep(s_axis_mem_user_write_data[1].keep),    // input wire [63 : 0] s_axis_tkeep
        .s_axis_tlast(s_axis_mem_user_write_data[1].last),    // input wire s_axis_tlast
        .m_axis_tvalid(axis_mem_user_write_data[1].valid),  // output wire m_axis_tvalid
        .m_axis_tready(axis_mem_user_write_data[1].ready),  // input wire m_axis_tready
        .m_axis_tdata(axis_mem_user_write_data[1].data),    // output wire [511 : 0] m_axis_tdata
        .m_axis_tkeep(axis_mem_user_write_data[1].keep),    // output wire [63 : 0] m_axis_tkeep
        .m_axis_tlast(axis_mem_user_write_data[1].last)    // output wire m_axis_tlast
    );    


    axis_register_slice_96 axis_mem_user_read_cmd0 (
        .aclk(clk),                    // input wire aclk
        .aresetn(rstn),              // input wire aresetn
        .s_axis_tvalid(s_axis_mem_user_read_cmd[0].valid),  // input wire s_axis_tvalid
        .s_axis_tready(s_axis_mem_user_read_cmd[0].ready),  // output wire s_axis_tready
        .s_axis_tdata({s_axis_mem_user_read_cmd[0].address,s_axis_mem_user_read_cmd[0].length}),    // input wire [95 : 0] s_axis_tdata
        .m_axis_tvalid(axis_mem_user_read_cmd[0].valid),  // output wire m_axis_tvalid
        .m_axis_tready(axis_mem_user_read_cmd[0].ready),  // input wire m_axis_tready
        .m_axis_tdata({axis_mem_user_read_cmd[0].address,axis_mem_user_read_cmd[0].length})    // output wire [95 : 0] m_axis_tdata
    );

    axis_register_slice_8 axis_mem_user_read_sts0 (
        .aclk(clk),                    // input wire aclk
        .aresetn(rstn),              // input wire aresetn
        .s_axis_tvalid(axis_mem_user_read_sts[0].valid),  // input wire s_axis_tvalid
        .s_axis_tready(axis_mem_user_read_sts[0].ready),  // output wire s_axis_tready
        .s_axis_tdata(axis_mem_user_read_sts[0].data),    // input wire [7 : 0] s_axis_tdata
        .m_axis_tvalid(m_axis_mem_user_read_sts[0].valid),  // output wire m_axis_tvalid
        .m_axis_tready(m_axis_mem_user_read_sts[0].ready),  // input wire m_axis_tready
        .m_axis_tdata(m_axis_mem_user_read_sts[0].data)    // output wire [7 : 0] m_axis_tdata
    );

    axis_register_slice_512 axis_mem_user_read_data0 (
        .aclk(clk),                    // input wire aclk
        .aresetn(rstn),              // input wire aresetn
        .s_axis_tvalid(axis_mem_user_read_data[0].valid),  // input wire s_axis_tvalid
        .s_axis_tready(axis_mem_user_read_data[0].ready),  // output wire s_axis_tready
        .s_axis_tdata(axis_mem_user_read_data[0].data),    // input wire [511 : 0] s_axis_tdata
        .s_axis_tkeep(axis_mem_user_read_data[0].keep),    // input wire [63 : 0] s_axis_tkeep
        .s_axis_tlast(axis_mem_user_read_data[0].last),    // input wire s_axis_tlast
        .m_axis_tvalid(m_axis_mem_user_read_data[0].valid),  // output wire m_axis_tvalid
        .m_axis_tready(m_axis_mem_user_read_data[0].ready),  // input wire m_axis_tready
        .m_axis_tdata(m_axis_mem_user_read_data[0].data),    // output wire [511 : 0] m_axis_tdata
        .m_axis_tkeep(m_axis_mem_user_read_data[0].keep),    // output wire [63 : 0] m_axis_tkeep
        .m_axis_tlast(m_axis_mem_user_read_data[0].last)    // output wire m_axis_tlast
    );

    axis_register_slice_96 axis_mem_user_read_cmd1 (
        .aclk(clk),                    // input wire aclk
        .aresetn(rstn),              // input wire aresetn
        .s_axis_tvalid(s_axis_mem_user_read_cmd[1].valid),  // input wire s_axis_tvalid
        .s_axis_tready(s_axis_mem_user_read_cmd[1].ready),  // output wire s_axis_tready
        .s_axis_tdata({s_axis_mem_user_read_cmd[1].address,s_axis_mem_user_read_cmd[1].length}),    // input wire [95 : 0] s_axis_tdata
        .m_axis_tvalid(axis_mem_user_read_cmd[1].valid),  // output wire m_axis_tvalid
        .m_axis_tready(axis_mem_user_read_cmd[1].ready),  // input wire m_axis_tready
        .m_axis_tdata({axis_mem_user_read_cmd[1].address,axis_mem_user_read_cmd[1].length})    // output wire [95 : 0] m_axis_tdata
    );

    axis_register_slice_8 axis_mem_user_read_sts1 (
        .aclk(clk),                    // input wire aclk
        .aresetn(rstn),              // input wire aresetn
        .s_axis_tvalid(axis_mem_user_read_sts[1].valid),  // input wire s_axis_tvalid
        .s_axis_tready(axis_mem_user_read_sts[1].ready),  // output wire s_axis_tready
        .s_axis_tdata(axis_mem_user_read_sts[1].data),    // input wire [7 : 0] s_axis_tdata
        .m_axis_tvalid(m_axis_mem_user_read_sts[1].valid),  // output wire m_axis_tvalid
        .m_axis_tready(m_axis_mem_user_read_sts[1].ready),  // input wire m_axis_tready
        .m_axis_tdata(m_axis_mem_user_read_sts[1].data)    // output wire [7 : 0] m_axis_tdata
    );

    axis_register_slice_512 axis_mem_user_read_data1 (
        .aclk(clk),                    // input wire aclk
        .aresetn(rstn),              // input wire aresetn
        .s_axis_tvalid(axis_mem_user_read_data[1].valid),  // input wire s_axis_tvalid
        .s_axis_tready(axis_mem_user_read_data[1].ready),  // output wire s_axis_tready
        .s_axis_tdata(axis_mem_user_read_data[1].data),    // input wire [511 : 0] s_axis_tdata
        .s_axis_tkeep(axis_mem_user_read_data[1].keep),    // input wire [63 : 0] s_axis_tkeep
        .s_axis_tlast(axis_mem_user_read_data[1].last),    // input wire s_axis_tlast
        .m_axis_tvalid(m_axis_mem_user_read_data[1].valid),  // output wire m_axis_tvalid
        .m_axis_tready(m_axis_mem_user_read_data[1].ready),  // input wire m_axis_tready
        .m_axis_tdata(m_axis_mem_user_read_data[1].data),    // output wire [511 : 0] m_axis_tdata
        .m_axis_tkeep(m_axis_mem_user_read_data[1].keep),    // output wire [63 : 0] m_axis_tkeep
        .m_axis_tlast(m_axis_mem_user_read_data[1].last)    // output wire m_axis_tlast
    );  


    // memory cmd streams
    axis_mem_cmd        axis_mem_user_read_cmd[2]();
    axis_mem_cmd        axis_mem_user_write_cmd[2]();
    // memory sts streams
    axis_mem_status     axis_mem_user_read_sts[2]();
    axis_mem_status     axis_mem_user_write_sts[2]();
    // memory data streams
    axi_stream          axis_mem_user_read_data[2]();
    axi_stream          axis_mem_user_write_data[2](); 
    
    assign axis_mem_write_cmd[0].address = mem_choice ? axis_mem_user_write_cmd[0].address : axis_mem_dma_write_cmd[0].address;
    assign axis_mem_write_cmd[0].length = mem_choice ? axis_mem_user_write_cmd[0].length : axis_mem_dma_write_cmd[0].length;
    assign axis_mem_write_cmd[0].valid = mem_choice ? axis_mem_user_write_cmd[0].valid : axis_mem_dma_write_cmd[0].valid;
    assign axis_mem_user_write_cmd[0].ready = mem_choice ? axis_mem_write_cmd[0].ready : 0;
    assign axis_mem_dma_write_cmd[0].ready = mem_choice ? 0 : axis_mem_write_cmd[0].ready;

    assign axis_mem_read_cmd[0].address = mem_choice ? axis_mem_user_read_cmd[0].address : axis_mem_dma_read_cmd[0].address;
    assign axis_mem_read_cmd[0].length = mem_choice ? axis_mem_user_read_cmd[0].length : axis_mem_dma_read_cmd[0].length;
    assign axis_mem_read_cmd[0].valid = mem_choice ? axis_mem_user_read_cmd[0].valid : axis_mem_dma_read_cmd[0].valid;
    assign axis_mem_user_read_cmd[0].ready = mem_choice ? axis_mem_read_cmd[0].ready : 0;
    assign axis_mem_dma_read_cmd[0].ready = mem_choice ? 0 : axis_mem_read_cmd[0].ready;

    assign axis_mem_user_write_sts[0].valid = mem_choice ? axis_mem_write_sts[0].valid : 0;
    assign axis_mem_dma_write_sts[0].valid = mem_choice ? 0 : axis_mem_write_sts[0].valid;    
    assign axis_mem_user_write_sts[0].data = mem_choice ? axis_mem_write_sts[0].data : 0;
    assign axis_mem_dma_write_sts[0].data = mem_choice ? 0 : axis_mem_write_sts[0].data; 
    assign axis_mem_write_sts[0].ready = mem_choice ? axis_mem_user_write_sts[0].ready : axis_mem_dma_write_sts[0].ready;

    assign axis_mem_user_read_sts[0].valid = mem_choice ? axis_mem_read_sts[0].valid : 0;
    assign axis_mem_dma_read_sts[0].valid = mem_choice ? 0 : axis_mem_read_sts[0].valid;    
    assign axis_mem_user_read_sts[0].data = mem_choice ? axis_mem_read_sts[0].data : 0;
    assign axis_mem_dma_read_sts[0].data = mem_choice ? 0 : axis_mem_read_sts[0].data; 
    assign axis_mem_read_sts[0].ready = mem_choice ? axis_mem_user_read_sts[0].ready : axis_mem_dma_read_sts[0].ready;   
    
    assign axis_mem_user_read_data[0].valid = mem_choice ? axis_mem_read_data[0].valid : 0;
    assign axis_mem_dma_read_data[0].valid = mem_choice ? 0 : axis_mem_read_data[0].valid;    
    assign axis_mem_user_read_data[0].data = mem_choice ? axis_mem_read_data[0].data : 0;
    assign axis_mem_dma_read_data[0].data = mem_choice ? 0 : axis_mem_read_data[0].data; 
    assign axis_mem_user_read_data[0].keep = mem_choice ? axis_mem_read_data[0].keep : 0;
    assign axis_mem_dma_read_data[0].keep = mem_choice ? 0 : axis_mem_read_data[0].keep;
    assign axis_mem_user_read_data[0].last = mem_choice ? axis_mem_read_data[0].last : 0;
    assign axis_mem_dma_read_data[0].last = mem_choice ? 0 : axis_mem_read_data[0].last;        
    assign axis_mem_read_data[0].ready = mem_choice ? axis_mem_user_read_data[0].ready : axis_mem_dma_read_data[0].ready;       

    assign axis_mem_write_data[0].data = mem_choice ? axis_mem_user_write_data[0].data : axis_mem_dma_write_data[0].data;
    assign axis_mem_write_data[0].keep = mem_choice ? axis_mem_user_write_data[0].keep : axis_mem_dma_write_data[0].keep;
    assign axis_mem_write_data[0].last = mem_choice ? axis_mem_user_write_data[0].last : axis_mem_dma_write_data[0].last;
    assign axis_mem_write_data[0].valid = mem_choice ? axis_mem_user_write_data[0].valid : axis_mem_dma_write_data[0].valid;
    assign axis_mem_user_write_data[0].ready = mem_choice ? axis_mem_write_data[0].ready : 0;
    assign axis_mem_dma_write_data[0].ready = mem_choice ? 0 : axis_mem_write_data[0].ready;

//
    assign axis_mem_write_cmd[1].address = mem_choice ? axis_mem_user_write_cmd[1].address : axis_mem_dma_write_cmd[1].address;
    assign axis_mem_write_cmd[1].length = mem_choice ? axis_mem_user_write_cmd[1].length : axis_mem_dma_write_cmd[1].length;
    assign axis_mem_write_cmd[1].valid = mem_choice ? axis_mem_user_write_cmd[1].valid : axis_mem_dma_write_cmd[1].valid;
    assign axis_mem_user_write_cmd[1].ready = mem_choice ? axis_mem_write_cmd[1].ready : 0;
    assign axis_mem_dma_write_cmd[1].ready = mem_choice ? 0 : axis_mem_write_cmd[1].ready;

    assign axis_mem_read_cmd[1].address = mem_choice ? axis_mem_user_read_cmd[1].address : axis_mem_dma_read_cmd[1].address;
    assign axis_mem_read_cmd[1].length = mem_choice ? axis_mem_user_read_cmd[1].length : axis_mem_dma_read_cmd[1].length;
    assign axis_mem_read_cmd[1].valid = mem_choice ? axis_mem_user_read_cmd[1].valid : axis_mem_dma_read_cmd[1].valid;
    assign axis_mem_user_read_cmd[1].ready = mem_choice ? axis_mem_read_cmd[1].ready : 0;
    assign axis_mem_dma_read_cmd[1].ready = mem_choice ? 0 : axis_mem_read_cmd[1].ready;

    assign axis_mem_user_write_sts[1].valid = mem_choice ? axis_mem_write_sts[1].valid : 0;
    assign axis_mem_dma_write_sts[1].valid = mem_choice ? 0 : axis_mem_write_sts[1].valid;    
    assign axis_mem_user_write_sts[1].data = mem_choice ? axis_mem_write_sts[1].data : 0;
    assign axis_mem_dma_write_sts[1].data = mem_choice ? 0 : axis_mem_write_sts[1].data; 
    assign axis_mem_write_sts[1].ready = mem_choice ? axis_mem_user_write_sts[1].ready : axis_mem_dma_write_sts[1].ready;

    assign axis_mem_user_read_sts[1].valid = mem_choice ? axis_mem_read_sts[1].valid : 0;
    assign axis_mem_dma_read_sts[1].valid = mem_choice ? 0 : axis_mem_read_sts[1].valid;    
    assign axis_mem_user_read_sts[1].data = mem_choice ? axis_mem_read_sts[1].data : 0;
    assign axis_mem_dma_read_sts[1].data = mem_choice ? 0 : axis_mem_read_sts[1].data; 
    assign axis_mem_read_sts[1].ready = mem_choice ? axis_mem_user_read_sts[1].ready : axis_mem_dma_read_sts[1].ready;   
    
    assign axis_mem_user_read_data[1].valid = mem_choice ? axis_mem_read_data[1].valid : 0;
    assign axis_mem_dma_read_data[1].valid = mem_choice ? 0 : axis_mem_read_data[1].valid;    
    assign axis_mem_user_read_data[1].data = mem_choice ? axis_mem_read_data[1].data : 0;
    assign axis_mem_dma_read_data[1].data = mem_choice ? 0 : axis_mem_read_data[1].data; 
    assign axis_mem_user_read_data[1].keep = mem_choice ? axis_mem_read_data[1].keep : 0;
    assign axis_mem_dma_read_data[1].keep = mem_choice ? 0 : axis_mem_read_data[1].keep;
    assign axis_mem_user_read_data[1].last = mem_choice ? axis_mem_read_data[1].last : 0;
    assign axis_mem_dma_read_data[1].last = mem_choice ? 0 : axis_mem_read_data[1].last;        
    assign axis_mem_read_data[1].ready = mem_choice ? axis_mem_user_read_data[1].ready : axis_mem_dma_read_data[0].ready;       

    assign axis_mem_write_data[1].data = mem_choice ? axis_mem_user_write_data[1].data : axis_mem_dma_write_data[1].data;
    assign axis_mem_write_data[1].keep = mem_choice ? axis_mem_user_write_data[1].keep : axis_mem_dma_write_data[1].keep;
    assign axis_mem_write_data[1].last = mem_choice ? axis_mem_user_write_data[1].last : axis_mem_dma_write_data[1].last;
    assign axis_mem_write_data[1].valid = mem_choice ? axis_mem_user_write_data[1].valid : axis_mem_dma_write_data[1].valid;
    assign axis_mem_user_write_data[1].ready = mem_choice ? axis_mem_write_data[1].ready : 0;
    assign axis_mem_dma_write_data[1].ready = mem_choice ? 0 : axis_mem_write_data[1].ready;  


    //////////////////////////debug//////////////////////



//////////////////////////////////user_counter//////////////////////////////
    reg[31:0]                           user_write_cmd0_counter;

    always@(posedge clk)begin
        if(~rstn)begin
            user_write_cmd0_counter          <= 1'b0;
        end
        else if(axis_mem_user_write_cmd[0].valid && axis_mem_user_write_cmd[0].ready)begin
            user_write_cmd0_counter          <= user_write_cmd0_counter + 1'b1;
        end
        else begin
            user_write_cmd0_counter          <= user_write_cmd0_counter;
        end
    end

    reg[31:0]                           user_write_data0_counter;

    always@(posedge clk)begin
        if(~rstn)begin
            user_write_data0_counter          <= 1'b0;
        end
        else if(axis_mem_user_write_data[0].valid && axis_mem_user_write_data[0].ready)begin
            user_write_data0_counter          <= user_write_data0_counter + 1'b1;
        end
        else begin
            user_write_data0_counter          <= user_write_data0_counter;
        end
    end

    reg[31:0]                           user_read_cmd0_counter;

    always@(posedge clk)begin
        if(~rstn)begin
            user_read_cmd0_counter          <= 1'b0;
        end
        else if(axis_mem_user_read_cmd[0].valid && axis_mem_user_read_cmd[0].ready)begin
            user_read_cmd0_counter          <= user_read_cmd0_counter + 1'b1;
        end
        else begin
            user_read_cmd0_counter          <= user_read_cmd0_counter;
        end
    end

    reg[31:0]                           user_read_data0_counter;

    always@(posedge clk)begin
        if(~rstn)begin
            user_read_data0_counter          <= 1'b0;
        end
        else if(axis_mem_user_read_data[0].valid && axis_mem_user_read_data[0].ready)begin
            user_read_data0_counter          <= user_read_data0_counter + 1'b1;
        end
        else begin
            user_read_data0_counter          <= user_read_data0_counter;
        end
    end

    reg[31:0]                           user_read_cmd1_counter;

    always@(posedge clk)begin
        if(~rstn)begin
            user_read_cmd1_counter          <= 1'b0;
        end
        else if(axis_mem_user_read_cmd[1].valid && axis_mem_user_read_cmd[1].ready)begin
            user_read_cmd1_counter          <= user_read_cmd1_counter + 1'b1;
        end
        else begin
            user_read_cmd1_counter          <= user_read_cmd1_counter;
        end
    end

    reg[31:0]                           user_read_data1_counter;

    always@(posedge clk)begin
        if(~rstn)begin
            user_read_data1_counter          <= 1'b0;
        end
        else if(axis_mem_user_read_data[1].valid && axis_mem_user_read_data[1].ready)begin
            user_read_data1_counter          <= user_read_data1_counter + 1'b1;
        end
        else begin
            user_read_data1_counter          <= user_read_data1_counter;
        end
    end  

//////////////////////////////dma_counter   
     
    
    reg[31:0]                           dma_write_cmd0_counter;

    always@(posedge clk)begin
        if(~rstn)begin
            dma_write_cmd0_counter          <= 1'b0;
        end
        else if(s_axis_mem_dma_write_cmd[0].valid && s_axis_mem_dma_write_cmd[0].ready)begin
            dma_write_cmd0_counter          <= dma_write_cmd0_counter + 1'b1;
        end
        else begin
            dma_write_cmd0_counter          <= dma_write_cmd0_counter;
        end
    end

    reg[31:0]                           dma_write_data0_counter;

    always@(posedge clk)begin
        if(~rstn)begin
            dma_write_data0_counter          <= 1'b0;
        end
        else if(s_axis_mem_dma_write_data[0].valid && s_axis_mem_dma_write_data[0].ready)begin
            dma_write_data0_counter          <= dma_write_data0_counter + 1'b1;
        end
        else begin
            dma_write_data0_counter          <= dma_write_data0_counter;
        end
    end

    reg[31:0]                           dma_read_cmd0_counter;

    always@(posedge clk)begin
        if(~rstn)begin
            dma_read_cmd0_counter          <= 1'b0;
        end
        else if(s_axis_mem_dma_read_cmd[0].valid && s_axis_mem_dma_read_cmd[0].ready)begin
            dma_read_cmd0_counter          <= dma_read_cmd0_counter + 1'b1;
        end
        else begin
            dma_read_cmd0_counter          <= dma_read_cmd0_counter;
        end
    end

    reg[31:0]                           dma_read_data0_counter;

    always@(posedge clk)begin
        if(~rstn)begin
            dma_read_data0_counter          <= 1'b0;
        end
        else if(m_axis_mem_dma_read_data[0].valid && m_axis_mem_dma_read_data[0].ready)begin
            dma_read_data0_counter          <= dma_read_data0_counter + 1'b1;
        end
        else begin
            dma_read_data0_counter          <= dma_read_data0_counter;
        end
    end

    reg[31:0]                           dma_read_cmd1_counter;

    always@(posedge clk)begin
        if(~rstn)begin
            dma_read_cmd1_counter          <= 1'b0;
        end
        else if(s_axis_mem_dma_read_cmd[1].valid && s_axis_mem_dma_read_cmd[1].ready)begin
            dma_read_cmd1_counter          <= dma_read_cmd1_counter + 1'b1;
        end
        else begin
            dma_read_cmd1_counter          <= dma_read_cmd1_counter;
        end
    end

    reg[31:0]                           dma_read_data1_counter;

    always@(posedge clk)begin
        if(~rstn)begin
            dma_read_data1_counter          <= 1'b0;
        end
        else if(m_axis_mem_dma_read_data[1].valid && m_axis_mem_dma_read_data[1].ready)begin
            dma_read_data1_counter          <= dma_read_data1_counter + 1'b1;
        end
        else begin
            dma_read_data1_counter          <= dma_read_data1_counter;
        end
    end  



/////////////////////////////mem_counter    

    reg[31:0]                           mem_write_cmd0_counter;

    always@(posedge clk)begin
        if(~rstn)begin
            mem_write_cmd0_counter          <= 1'b0;
        end
        else if(m_axis_mem_write_cmd[0].valid && m_axis_mem_write_cmd[0].ready)begin
            mem_write_cmd0_counter          <= mem_write_cmd0_counter + 1'b1;
        end
        else begin
            mem_write_cmd0_counter          <= mem_write_cmd0_counter;
        end
    end

    reg[31:0]                           mem_write_data0_counter;

    always@(posedge clk)begin
        if(~rstn)begin
            mem_write_data0_counter          <= 1'b0;
        end
        else if(m_axis_mem_write_data[0].valid && m_axis_mem_write_data[0].ready)begin
            mem_write_data0_counter          <= mem_write_data0_counter + 1'b1;
        end
        else begin
            mem_write_data0_counter          <= mem_write_data0_counter;
        end
    end

    reg[31:0]                           mem_write_cmd1_counter;

    always@(posedge clk)begin
        if(~rstn)begin
            mem_write_cmd1_counter          <= 1'b0;
        end
        else if(m_axis_mem_write_cmd[1].valid && m_axis_mem_write_cmd[1].ready)begin
            mem_write_cmd1_counter          <= mem_write_cmd1_counter + 1'b1;
        end
        else begin
            mem_write_cmd1_counter          <= mem_write_cmd1_counter;
        end
    end

    reg[31:0]                           mem_write_data1_counter;

    always@(posedge clk)begin
        if(~rstn)begin
            mem_write_data1_counter          <= 1'b0;
        end
        else if(m_axis_mem_write_data[1].valid && m_axis_mem_write_data[1].ready)begin
            mem_write_data1_counter          <= mem_write_data1_counter + 1'b1;
        end
        else begin
            mem_write_data1_counter          <= mem_write_data1_counter;
        end
    end


    reg[31:0]                           mem_read_cmd0_counter;

    always@(posedge clk)begin
        if(~rstn)begin
            mem_read_cmd0_counter          <= 1'b0;
        end
        else if(m_axis_mem_read_cmd[0].valid && m_axis_mem_read_cmd[0].ready)begin
            mem_read_cmd0_counter          <= mem_read_cmd0_counter + 1'b1;
        end
        else begin
            mem_read_cmd0_counter          <= mem_read_cmd0_counter;
        end
    end

    reg[31:0]                           mem_read_data0_counter;

    always@(posedge clk)begin
        if(~rstn)begin
            mem_read_data0_counter          <= 1'b0;
        end
        else if(s_axis_mem_read_data[0].valid && s_axis_mem_read_data[0].ready)begin
            mem_read_data0_counter          <= mem_read_data0_counter + 1'b1;
        end
        else begin
            mem_read_data0_counter          <= mem_read_data0_counter;
        end
    end

    reg[31:0]                           mem_read_cmd1_counter;

    always@(posedge clk)begin
        if(~rstn)begin
            mem_read_cmd1_counter          <= 1'b0;
        end
        else if(m_axis_mem_read_cmd[1].valid && m_axis_mem_read_cmd[1].ready)begin
            mem_read_cmd1_counter          <= mem_read_cmd1_counter + 1'b1;
        end
        else begin
            mem_read_cmd1_counter          <= mem_read_cmd1_counter;
        end
    end

    reg[31:0]                           mem_read_data1_counter;

    always@(posedge clk)begin
        if(~rstn)begin
            mem_read_data1_counter          <= 1'b0;
        end
        else if(s_axis_mem_read_data[1].valid && s_axis_mem_read_data[1].ready)begin
            mem_read_data1_counter          <= mem_read_data1_counter + 1'b1;
        end
        else begin
            mem_read_data1_counter          <= mem_read_data1_counter;
        end
    end  


/////////////////////////////////tcp data
    reg[31:0]                           tcp_write_cmd0_counter;

    always@(posedge clk)begin
        if(~rstn)begin
            tcp_write_cmd0_counter          <= 1'b0;
        end
        else if(app_axis_tcp_tx_meta.valid && app_axis_tcp_tx_meta.ready)begin
            tcp_write_cmd0_counter          <= tcp_write_cmd0_counter + 1'b1;
        end
        else begin
            tcp_write_cmd0_counter          <= tcp_write_cmd0_counter;
        end
    end

    reg[31:0]                           tcp_write_data0_counter;

    always@(posedge clk)begin
        if(~rstn)begin
            tcp_write_data0_counter          <= 1'b0;
        end
        else if(app_axis_tcp_tx_data.valid && app_axis_tcp_tx_data.ready)begin
            tcp_write_data0_counter          <= tcp_write_data0_counter + 1'b1;
        end
        else begin
            tcp_write_data0_counter          <= tcp_write_data0_counter;
        end
    end

    reg[31:0]                           tcp_read_cmd0_counter;

    always@(posedge clk)begin
        if(~rstn)begin
            tcp_read_cmd0_counter          <= 1'b0;
        end
        else if(app_axis_tcp_rx_meta.valid && app_axis_tcp_rx_meta.ready)begin
            tcp_read_cmd0_counter          <= tcp_read_cmd0_counter + 1'b1;
        end
        else begin
            tcp_read_cmd0_counter          <= tcp_read_cmd0_counter;
        end
    end

    reg[31:0]                           tcp_read_data0_counter;

    always@(posedge clk)begin
        if(~rstn)begin
            tcp_read_data0_counter          <= 1'b0;
        end
        else if(app_axis_tcp_rx_data.valid && app_axis_tcp_rx_data.ready)begin
            tcp_read_data0_counter          <= tcp_read_data0_counter + 1'b1;
        end
        else begin
            tcp_read_data0_counter          <= tcp_read_data0_counter;
        end
    end

    ////////////////////////////user time

    reg[31:0]                           time_counter;
    reg                                 time_en;

    always@(posedge clk)begin
        if(~rstn)begin
            time_en                     <= 1'b0;
        end
        else if(axis_user_start.valid && axis_user_start.ready)begin
            time_en                     <= 1'b1;
        end
        else if(axis_user_done.valid && axis_user_done.ready)begin
            time_en                     <= 1'b0;
        end        
        else begin
            time_en                     <= time_en;
        end
    end     

    always@(posedge clk)begin
        if(~rstn)begin
            time_counter                <= 1'b0;
        end
        else if(time_en)begin
            time_counter                <= time_counter + 1'b1;
        end
        else begin
            time_counter                <= time_counter;
        end
    end  

    assign status_reg[0] = user_write_cmd0_counter;
    assign status_reg[1] = user_write_data0_counter;
    assign status_reg[2] = user_read_cmd0_counter;
    assign status_reg[3] = user_read_data0_counter;
    assign status_reg[4] = user_read_cmd1_counter;
    assign status_reg[5] = user_read_data1_counter;

    assign status_reg[6] = dma_write_cmd0_counter;
    assign status_reg[7] = dma_write_data0_counter;
    assign status_reg[8] = dma_read_cmd0_counter;
    assign status_reg[9] = dma_read_data0_counter;
    assign status_reg[10] = dma_read_cmd1_counter;
    assign status_reg[11] = dma_read_data1_counter;
    
    assign status_reg[12] = mem_write_cmd0_counter;
    assign status_reg[13] = mem_write_data0_counter;
    assign status_reg[14] = mem_write_cmd1_counter;
    assign status_reg[15] = mem_write_data1_counter;    
    assign status_reg[16] = mem_read_cmd0_counter;
    assign status_reg[17] = mem_read_data0_counter;
    assign status_reg[18] = mem_read_cmd1_counter;
    assign status_reg[19] = mem_read_data1_counter;

    assign status_reg[20] = tcp_write_cmd0_counter;
    assign status_reg[21] = tcp_write_data0_counter;
    assign status_reg[22] = tcp_read_cmd0_counter;
    assign status_reg[23] = tcp_read_data0_counter;



    assign status_reg[24] = time_counter;


endmodule
