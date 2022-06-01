`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2020/02/20 21:27:06
// Design Name: 
// Module Name: dma_inf
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

`timescale 1ns / 1ps
//`default_nettype none

`include "example_module.vh"

module dma_inf_benchmark(
    /*HPY INTERFACE */
    output wire[15 : 0]         pcie_tx_p,
    output wire[15 : 0]         pcie_tx_n,
    input wire[15 : 0]          pcie_rx_p,
    input wire[15 : 0]          pcie_rx_n,

    input wire				    sys_clk_p,
    input wire				    sys_clk_n,
    input wire				    sys_rst_n, 

    /* USER INTERFACE */
    //pcie clock output
    output wire                 pcie_clk,
    output wire                 pcie_aresetn,
    //user clock input
    input wire                  user_clk,
    input wire                  user_aresetn,

    //DMA Commands
    axis_mem_cmd.slave          s_axis_dma_read_cmd[0:3],
    axis_mem_cmd.slave          s_axis_dma_write_cmd[0:3],

    //DMA Data streams      
    axi_stream.slave            s_axis_dma_write_data[0:3],
    axi_stream.master           m_axis_dma_read_data[0:3],

    //Control interface
    //fpga register
    output wire[511:0][31:0]    fpga_control_reg,
    input  wire[511:0][31:0]    fpga_status_reg,
    
    //Bypass interface
    axis_meta.master    axis_tcp_recv_read_cnt,        
    axis_meta.master    bypass_cmd,
    //off path cmd
    axis_meta.master    m_axis_get_data_cmd,
    axis_meta.master    m_axis_put_data_cmd,
    //one side
    axis_meta.master    m_axis_get_data_form_net,
    axis_meta.master    m_axis_put_data_to_net    


// `ifdef XDMA_BYPASS    
//     //Bypass interface
//     // // bypass register
//     ,output wire[31:0][511:0]    bypass_control_reg,
//     input  wire[31:0][511:0]    bypass_status_reg
// `endif

);

//	ila_0 rxinf (
//		.clk(pcie_clk), // input wire clk
	
	
//		.probe0(s_axis_dma_read_cmd[1].valid), // input wire [0:0]  probe0  
//		.probe1(s_axis_dma_read_cmd[1].ready), // input wire [0:0]  probe1 
//		.probe2(s_axis_dma_read_cmd[1].address), // input wire [63:0]  probe2 
//		.probe3(s_axis_dma_read_cmd[1].length), // input wire [31:0]  probe3 
//		.probe4(s_axis_dma_read_cmd[2].valid), // input wire [0:0]  probe4 
//		.probe5(s_axis_dma_read_cmd[2].ready), // input wire [0:0]  probe5 
//		.probe6(0), // input wire [0:0]  probe6 
//		.probe7({0,s_axis_dma_read_cmd[2].address}) // input wire [511:0]  probe7
//	);



    reg[511:0][31:0]            fpga_status_reg_r;
    reg[31:0]                   xdma_axil_rdata;

/*
 * PCIe Signals
 */
 wire pcie_lnk_up;
 wire pcie_ref_clk;
 wire pcie_ref_clk_gt;
 wire perst_n;
 
 // PCIe user clock & reset
// wire pcie_clk;
// wire pcie_aresetn;
 
 
   // Ref clock buffer
   IBUFDS_GTE4 # (.REFCLK_HROW_CK_SEL(2'b00)) refclk_ibuf (.O(pcie_ref_clk_gt), .ODIV2(pcie_ref_clk), .I(sys_clk_p), .CEB(1'b0), .IB(sys_clk_n));
   // Reset buffer
   IBUF   sys_reset_n_ibuf (.O(perst_n), .I(sys_rst_n));


/*
 * DMA Signals
 */
//Axi Lite Control Bus
    axi_lite        axil_control();
    axi_mm#(
        .ADDR_WIDTH (64),
        .DATA_WIDTH (512)
    )axim_control();
    
    wire[3:0]        c2h_dsc_byp_load;
    wire[3:0]        c2h_dsc_byp_ready;
    wire[3:0][63:0]  c2h_dsc_byp_addr;
    wire[3:0][31:0]  c2h_dsc_byp_len;
    
    wire[3:0]        h2c_dsc_byp_load;
    wire[3:0]        h2c_dsc_byp_ready;
    wire[3:0][63:0]  h2c_dsc_byp_addr;
    wire[3:0][31:0]  h2c_dsc_byp_len;
    
    axi_stream  axis_dma_c2h[4]();
    axi_stream  axis_dma_h2c[4]();
    
    wire[3:0][7:0] c2h_sts;
    wire[3:0][7:0] h2c_sts;

/*
 * DMA Driver
 */
 dma_driver dma_driver_inst (
    .sys_clk(pcie_ref_clk),                                       // input wire sys_clk
    .sys_clk_gt(pcie_ref_clk_gt),
    .sys_rst_n(perst_n),                                          // input wire sys_rst_n
    .user_lnk_up(pcie_lnk_up),                                    // output wire user_lnk_up
    .pcie_tx_p(pcie_tx_p),                                        // output wire [15 : 0] pci_exp_txp
    .pcie_tx_n(pcie_tx_n),                                        // output wire [15 : 0] pci_exp_txn
    .pcie_rx_p(pcie_rx_p),                                        // input wire [15 : 0] pci_exp_rxp
    .pcie_rx_n(pcie_rx_n),                                        // input wire [15 : 0] pci_exp_rxn
    .pcie_clk(pcie_clk),                                          // output wire axi_aclk
    .pcie_aresetn(pcie_aresetn),                                  // output wire axi_aresetn
    
   // Axi Lite Control Master interface   
    .m_axil(axil_control),
    // AXI MM Control Interface 
//  `ifdef XDMA_BYPASS  
    .m_axim(axim_control),
//  `endif
    // AXI Stream Interface
    .s_axis_c2h_data(axis_dma_c2h),
    .m_axis_h2c_data(axis_dma_h2c),
  
    // Descriptor Bypass
    .c2h_dsc_byp_ready    (c2h_dsc_byp_ready),
    //.c2h_dsc_byp_src_addr_0 (64'h0),
    .c2h_dsc_byp_addr     (c2h_dsc_byp_addr),
    .c2h_dsc_byp_len      (c2h_dsc_byp_len),
    //.c2h_dsc_byp_ctl_0      (16'h13), //was 16'h3
    .c2h_dsc_byp_load     (c2h_dsc_byp_load),
    
    .h2c_dsc_byp_ready    (h2c_dsc_byp_ready),
    .h2c_dsc_byp_addr     (h2c_dsc_byp_addr),
    //.h2c_dsc_byp_dst_addr_0 (64'h0),
    .h2c_dsc_byp_len      (h2c_dsc_byp_len),
    //.h2c_dsc_byp_ctl_0      (16'h13), //was 16'h3
    .h2c_dsc_byp_load     (h2c_dsc_byp_load),
    
    .c2h_sts(c2h_sts),                                          // output wire [7 : 0] c2h_sts_0
    .h2c_sts(h2c_sts),                                          // output wire [7 : 0] h2c_sts_0


    .s_axil_awaddr          (fpga_control_reg[4]),
    .s_axil_wdata           (fpga_control_reg[5]),
    .s_axil_araddr          (fpga_control_reg[6]),
    .xdma_contrl_wr_rd      (fpga_control_reg[7]),
    .s_axil_rdata           (xdma_axil_rdata)   
  );    


// `ifdef XDMA_BYPASS

    wire                    bypass_en_a;          // output wire bram_en_a
    wire [ 63:0]            bypass_we_a;          // output wire [63 : 0] bram_we_a
    wire [ 19:0]            bypass_addr_a;      // output wire [15 : 0] bram_addr_a
    wire [511:0]            bypass_wrdata_a;  // output wire [511 : 0] bram_wrdata_a
    wire [511:0]            bypass_rddata_a;  // input wire [511 : 0] bram_rddata_a     

///////////////////////////degug//////////////////////////
    reg 									wr_th_en,wr_th_en_r,rd_lat_en;
    reg 									rd_th_en,rd_th_en_r;
    reg [31:0]								wr_th_sum,rd_lat_sum,rd_lat_cnt;
    reg [31:0]								rd_th_sum;
    reg [31:0]								data_cnt,rd_data;
    reg [31:0]                              by_length;

    wire                                    fifo_arlen_empty;
    wire                                    fifo_arlen_full;
    wire                                    fifo_arlen_valid;
    reg                                     fifo_arlen_rd_en;
    wire[7:0]                               fifo_arlen_data;
    reg[7:0]                                by_arlen,by_arcnt;
    reg                                     by_rvalid;
    
    assign axim_control.awready = 1;
    assign axim_control.wready = 1;
    assign axim_control.bid = 0;
    assign axim_control.bresp = 3;
    assign axim_control.bvalid = 1;
    assign axim_control.arready = 1;    
    assign axim_control.rid = 0;  
    assign axim_control.rdata = rd_data;  
    assign axim_control.rresp = 3;  
    assign axim_control.rvalid = 1; 
    assign axim_control.rlast = (1 == by_arcnt) & axim_control.rvalid & axim_control.rready;  

    // always@(posedge pcie_clk)begin
    //     if(~pcie_aresetn)begin
    //         fifo_arlen_rd_en            <= 1'b0;
    //     end
    //     else if ((~fifo_arlen_empty) && (axim_control.rlast || ((by_arcnt == 0) && ~(axim_control.rvalid & axim_control.rready))) ) begin
    //         fifo_arlen_rd_en            <= 1'b1;
    //     end
    //     else begin
    //         fifo_arlen_rd_en            <= 1'b0;
    //     end
    // end

    // always@(posedge pcie_clk)begin
    //     if(~pcie_aresetn)begin
    //         by_rvalid            <= 1'b0;
    //     end
    //     else if(fifo_arlen_valid)begin
    //         by_rvalid            <= 1'b1;
    //     end
    //     else if(axim_control.rlast)begin
    //         by_rvalid            <= 1'b0;
    //     end
    //     else begin
    //         by_rvalid            <= by_rvalid;
    //     end
    // end

    // always@(posedge pcie_clk)begin
    //     if(~pcie_aresetn)begin
    //         by_arlen            <= 1'b0;
    //     end
    //     else if(fifo_arlen_valid)begin
    //         by_arlen            <= fifo_arlen_data;
    //     end
    //     else begin
    //         by_arlen            <= by_arlen;
    //     end
    // end

    always@(posedge pcie_clk)begin
        if(~pcie_aresetn)begin
            by_arcnt            <= 1'b0;
        end
        else if(axim_control.rlast)begin
            by_arcnt            <= 1'b0;
        end
        else if(axim_control.rvalid & axim_control.rready)begin
            by_arcnt            <= by_arcnt + 1;
        end
        else begin
            by_arcnt            <= by_arcnt;
        end
    end

    // fwft_8w fwft_8w (
    //     .clk(pcie_clk),              // input wire clk
    //     .rst(~pcie_aresetn),              // input wire rst
    //     .din(axim_control.arlen),              // input wire [7 : 0] din
    //     .wr_en(axim_control.arready & axim_control.arvalid),          // input wire wr_en
    //     .rd_en(fifo_arlen_rd_en),          // input wire rd_en
    //     .dout(fifo_arlen_data),            // output wire [7 : 0] dout
    //     .full(),            // output wire full
    //     .empty(fifo_arlen_empty),          // output wire empty
    //     .valid(fifo_arlen_valid),          // output wire valid
    //     .prog_full(fifo_arlen_full)  // output wire prog_full
    //   );


//  // AXI stream interface for the CQ forwarding
//    axi_bram_ctrl_1 axi_bram_gen_bypass_inst (
//        .s_axi_aclk         (pcie_clk),
//        .s_axi_aresetn      (pcie_aresetn),
//        .s_axi_awid         (axim_control.awid ),
//        .s_axi_awaddr       (axim_control.awaddr[19:0]),
//        .s_axi_awlen        (axim_control.awlen),
//        .s_axi_awsize       (axim_control.awsize),
//        .s_axi_awburst      (axim_control.awburst),
//        .s_axi_awlock       (1'd0),
//        .s_axi_awcache      (4'd0),
//        .s_axi_awprot       (3'd0),
//        .s_axi_awvalid      (axim_control.awvalid),
//        .s_axi_awready      (axim_control.awready),
//        .s_axi_wdata        (axim_control.wdata),
//        .s_axi_wstrb        (axim_control.wstrb),
//        .s_axi_wlast        (axim_control.wlast),
//        .s_axi_wvalid       (axim_control.wvalid),
//        .s_axi_wready       (axim_control.wready),
//        .s_axi_bid          (axim_control.bid),
//        .s_axi_bresp        (axim_control.bresp),
//        .s_axi_bvalid       (axim_control.bvalid),
//        .s_axi_bready       (axim_control.bready),
//        .s_axi_arid         (axim_control.arid),
//        .s_axi_araddr       (axim_control.araddr[19:0]),
//        .s_axi_arlen        (axim_control.arlen),
//        .s_axi_arsize       (axim_control.arsize),
//        .s_axi_arburst      (axim_control.arburst),
//        .s_axi_arlock       (1'd0),
//        .s_axi_arcache      (4'd0),
//        .s_axi_arprot       (3'd0),
//        .s_axi_arvalid      (axim_control.arvalid),
//        .s_axi_arready      (axim_control.arready),
//        .s_axi_rid          (axim_control.rid),
//        .s_axi_rdata        (axim_control.rdata),
//        .s_axi_rresp        (axim_control.rresp),
//        .s_axi_rlast        (axim_control.rlast),
//        .s_axi_rvalid       (axim_control.rvalid),
//        .s_axi_rready       (axim_control.rready),
//        .bram_rst_a         (),        // output wire bram_rst_a
//        .bram_clk_a         (),        // output wire bram_clk_a
//        .bram_en_a          (bypass_en_a),          // output wire bram_en_a
//        .bram_we_a          (bypass_we_a),          // output wire [63 : 0] bram_we_a
//        .bram_addr_a        (bypass_addr_a),      // output wire [15 : 0] bram_addr_a
//        .bram_wrdata_a      (bypass_wrdata_a),  // output wire [511 : 0] bram_wrdata_a
//        .bram_rddata_a      (rd_data)//bypass_rddata_a)  // input wire [511 : 0] bram_rddata_a        
//    );  

    // dma_bypass_controller inst_dma_bypass_controller(
    // // pcie clk
    //         .pcie_clk           (pcie_clk),
    //         .pcie_aresetn       (pcie_aresetn), 
    // // user clk
    //         .user_clk           (user_clk),
    //         .user_aresetn       (user_aresetn),
        
    //         // Control Interface
    //         .bram_en_a          (bypass_en_a),          // output wire bram_en_a
    //         .bram_we_a          (bypass_we_a[0]),          // output wire [3 : 0] bram_we_a
    //         .bram_addr_a        (bypass_addr_a),      // output wire [15 : 0] bram_addr_a
    //         .bram_wrdata_a      (bypass_wrdata_a),  // output wire [31 : 0] bram_wrdata_a
    //         .bram_rddata_a      (bypass_rddata_a),  // input wire [31 : 0] bram_rddata_a
    //         // // bypass register
    //         .axis_tcp_recv_read_cnt(axis_tcp_recv_read_cnt),             
    //         .bypass_cmd         (bypass_cmd),
    //         //off path cmd
    //         .m_axis_get_data_cmd(m_axis_get_data_cmd),
    //         .m_axis_put_data_cmd(m_axis_put_data_cmd),
    //         //one side
    //         .m_axis_get_data_form_net(m_axis_get_data_form_net),
    //         .m_axis_put_data_to_net(m_axis_put_data_to_net) 
    //         // .bypass_control_reg (bypass_control_reg),
    //         // .bypass_status_reg  (bypass_status_reg)
        
    // );
// ///////////////////////////degug//////////////////////////
//         reg 									wr_th_en,wr_th_en_r,rd_lat_en;
//         reg 									rd_th_en,rd_th_en_r;
//         reg [31:0]								wr_th_sum,rd_lat_sum,rd_lat_cnt;
//         reg [31:0]								rd_th_sum;
//         reg [31:0]								data_cnt,rd_data;
//         reg [31:0]                              by_length;

        always@(posedge pcie_clk)begin
            by_length                           <= fpga_control_reg[40];
            wr_th_en_r                          <= wr_th_en;
            rd_th_en_r                          <= rd_th_en;
        end

        always @(posedge pcie_clk)begin
            if(~pcie_aresetn)begin
                data_cnt 						<= 1'b0;
            end
            else if(data_cnt == by_length)begin
                data_cnt						<= 1'b0;
            end
            else if(axim_control.wready & axim_control.wvalid)begin
                data_cnt						<= data_cnt + 1'b1;
            end
            else begin
                data_cnt						<= data_cnt;
            end		
        end        

        always@(posedge pcie_clk)begin
            if(~pcie_aresetn)begin
                wr_th_en						<= 1'b0;
            end  
            else if(data_cnt == by_length)begin
                wr_th_en						<= 1'b0;
            end
            else if(axim_control.awready & axim_control.awvalid)begin
                wr_th_en						<= 1'b1;
            end		
            else begin
                wr_th_en						<= wr_th_en;
            end
        end
    
        
        always@(posedge pcie_clk)begin
            if(~pcie_aresetn)begin
                wr_th_sum						<= 32'b0;
            end
            else if(wr_th_en & ~wr_th_en_r)begin
                wr_th_sum						<= 32'b0;
            end 
            else if(wr_th_en)begin
                wr_th_sum						<= wr_th_sum + 1'b1;
            end
            else begin
                wr_th_sum						<= wr_th_sum;
            end
        end




        always @(posedge pcie_clk)begin
            if(~pcie_aresetn)begin
                rd_data 						<= 1'b0;
            end
            else if(rd_data == by_length)begin
                rd_data						    <= 1'b0;
            end
            else if(axim_control.rready & axim_control.rvalid)begin
                rd_data						    <= rd_data + 1'b1;
            end
            else begin
                rd_data						    <= rd_data;
            end		
        end        

        always@(posedge pcie_clk)begin
            if(~pcie_aresetn)begin
                rd_th_en						<= 1'b0;
            end  
            else if(rd_data == by_length)begin
                rd_th_en						<= 1'b0;
            end
            else if(axim_control.arready & axim_control.arvalid)begin
                rd_th_en						<= 1'b1;
            end		
            else begin
                rd_th_en						<= rd_th_en;
            end
        end
    
        
        always@(posedge pcie_clk)begin
            if(~pcie_aresetn)begin
                rd_th_sum						<= 32'b0;
            end
            else if(rd_th_en & ~rd_th_en_r)begin
                rd_th_sum						<= 32'b0;
            end 
            else if(rd_th_en)begin
                rd_th_sum						<= rd_th_sum + 1'b1;
            end
            else begin
                rd_th_sum						<= rd_th_sum;
            end
        end





	always@(posedge pcie_clk)begin
		if(~pcie_aresetn)begin
			rd_lat_en						<= 1'b0;
		end	
		else if(axim_control.arready & axim_control.arvalid)begin
			rd_lat_en						<= ~rd_lat_en;
		end  
		else begin
			rd_lat_en						<= rd_lat_en;
		end
	end

	
	always@(posedge pcie_clk)begin
		if(~pcie_aresetn)begin
			rd_lat_cnt						<= 32'b0;
		end
		else if(axim_control.arready & axim_control.arvalid)begin
			rd_lat_cnt						<= 32'b0;
		end 
		else if(rd_lat_en)begin
			rd_lat_cnt						<= rd_lat_cnt + 1'b1;
		end
		else begin
			rd_lat_cnt						<= rd_lat_cnt;
		end
	end

	always@(posedge pcie_clk)begin
		if(~pcie_aresetn)begin
			rd_lat_sum						<= 32'b0;
		end
		else if(axim_control.arready & axim_control.arvalid)begin
			rd_lat_sum						<= rd_lat_cnt;
		end 
		else begin
			rd_lat_sum						<= rd_lat_sum;
		end
	end

//       ila_bypass bypass_ila (
//           .clk(pcie_clk), // input wire clk
        
        
//           .probe0(fifo_arlen_rd_en), // input wire [0:0]  probe0  
//           .probe1(fifo_arlen_empty), // input wire [0:0]  probe1 
//           .probe2(axim_control.awaddr[19:0]), // input wire [19:0]  probe2 
//           .probe3(by_arcnt), // input wire [7:0]  probe3 
//           .probe4(axim_control.rlast), // input wire [0:0]  probe4 
//           .probe5(fifo_arlen_valid), // input wire [0:0]  probe5 
//           .probe6(axim_control.wdata[31:0]), // input wire [511:0]  probe6 
//           .probe7(axim_control.arvalid), // input wire [0:0]  probe7 
//           .probe8(axim_control.arready), // input wire [0:0]  probe8 
//           .probe9(axim_control.araddr[19:0]), // input wire [19:0]  probe9 
//           .probe10(axim_control.arlen), // input wire [7:0]  probe10 
//           .probe11(axim_control.rvalid), // input wire [0:0]  probe11 
//           .probe12(axim_control.rready), // input wire [0:0]  probe12 
//           .probe13(axim_control.rdata[31:0]), // input wire [511:0]  probe13 
//           .probe14(wr_th_en), // input wire [0:0]  probe14 
//           .probe15(data_cnt), // input wire [0:0]  probe15 
//           .probe16(wr_th_sum), // input wire [19:0]  probe16 
//           .probe17(rd_th_en), // input wire [31:0]  probe17 
//           .probe18(rd_data), // input wire [31:0]  probe18
//           .probe19(rd_th_sum) // input wire [31:0]  probe18
//       );




// `endif
///////////axil



    wire bram_en_a;          // output wire bram_en_a
    wire[3:0] bram_we_a;          // output wire [3 : 0] bram_we_a
    wire[11:0] bram_addr_a;      // output wire [15 : 0] bram_addr_a
    wire[31:0] bram_wrdata_a;  // output wire [31 : 0] bram_wrdata_a
    wire[31:0] bram_rddata_a;  // input wire [31 : 0] bram_rddata_a

//----------- Begin Cut here for INSTANTIATION Template ---// INST_TAG
axi_bram_ctrl_0 dma_axil_bram_ctrl (
  .s_axi_aclk(user_clk),        // input wire s_axi_aclk
  .s_axi_aresetn(user_aresetn),  // input wire s_axi_aresetn
  .s_axi_awaddr(axil_control.awaddr),    // input wire [15 : 0] s_axi_awaddr
  .s_axi_awprot(0),    // input wire [2 : 0] s_axi_awprot
  .s_axi_awvalid(axil_control.awvalid),  // input wire s_axi_awvalid
  .s_axi_awready(axil_control.awready),  // output wire s_axi_awready
  .s_axi_wdata(axil_control.wdata),      // input wire [31 : 0] s_axi_wdata
  .s_axi_wstrb(axil_control.wstrb),      // input wire [3 : 0] s_axi_wstrb
  .s_axi_wvalid(axil_control.wvalid),    // input wire s_axi_wvalid
  .s_axi_wready(axil_control.wready),    // output wire s_axi_wready
  .s_axi_bresp(axil_control.bresp),      // output wire [1 : 0] s_axi_bresp
  .s_axi_bvalid(axil_control.bvalid),    // output wire s_axi_bvalid
  .s_axi_bready(axil_control.bready),    // input wire s_axi_bready
  .s_axi_araddr(axil_control.araddr),    // input wire [15 : 0] s_axi_araddr
  .s_axi_arprot(0),    // input wire [2 : 0] s_axi_arprot
  .s_axi_arvalid(axil_control.arvalid),  // input wire s_axi_arvalid
  .s_axi_arready(axil_control.arready),  // output wire s_axi_arready
  .s_axi_rdata(axil_control.rdata),      // output wire [31 : 0] s_axi_rdata
  .s_axi_rresp(axil_control.rresp),      // output wire [1 : 0] s_axi_rresp
  .s_axi_rvalid(axil_control.rvalid),    // output wire s_axi_rvalid
  .s_axi_rready(axil_control.rready),    // input wire s_axi_rready
  .bram_rst_a(),        // output wire bram_rst_a
  .bram_clk_a(),        // output wire bram_clk_a
  .bram_en_a(bram_en_a),          // output wire bram_en_a
  .bram_we_a(bram_we_a),          // output wire [3 : 0] bram_we_a
  .bram_addr_a(bram_addr_a),      // output wire [15 : 0] bram_addr_a
  .bram_wrdata_a(bram_wrdata_a),  // output wire [31 : 0] bram_wrdata_a
  .bram_rddata_a(bram_rddata_a)  // input wire [31 : 0] bram_rddata_a
);


///////////////////////////degug//////////////////////////
reg 									ctrl_lat_en;
reg [31:0]								ctrl_lat_sum;
reg [31:0]                              ctrl_lat_cnt;


always@(posedge pcie_clk)begin
if(~pcie_aresetn)begin
    ctrl_lat_en						<= 1'b0;
end	
else if(axil_control.arready & axil_control.arvalid && (axil_control.araddr[11:2] == 10'd500))begin
    ctrl_lat_en						<= 1'b1;
end  
else if(axil_control.arready & axil_control.arvalid & (axil_control.araddr[11:2] == 0))begin
    ctrl_lat_en						<= 1'b0;
end 
else begin
    ctrl_lat_en						<= ctrl_lat_en;
end
end


always@(posedge pcie_clk)begin
if(~pcie_aresetn)begin
    ctrl_lat_cnt						<= 32'b0;
end
else if(axil_control.arready & axil_control.arvalid && (axil_control.araddr[11:2] == 10'd500))begin
    ctrl_lat_cnt						<= 32'b0;
end 
else if(ctrl_lat_en)begin
    ctrl_lat_cnt						<= ctrl_lat_cnt + 1'b1;
end
else begin
    ctrl_lat_cnt						<= ctrl_lat_cnt;
end
end

// always@(posedge pcie_clk)begin
// if(~pcie_aresetn)begin
//     ctrl_lat_sum						<= 32'b0;
// end
// else if(axil_control.arready & axil_control.arvalid)begin
//     ctrl_lat_sum						<= ctrl_lat_cnt;
// end 
// else begin
//     ctrl_lat_sum						<= ctrl_lat_sum;
// end
// end

// ila_ctrl_lat ila_ctrl_lat_inst (
// 	.clk(pcie_clk), // input wire clk


// 	.probe0(axil_control.arready), // input wire [0:0]  probe0  
// 	.probe1(axil_control.arvalid), // input wire [0:0]  probe1 
// 	.probe2(axil_control.araddr), // input wire [15:0]  probe2 
// 	.probe3(ctrl_lat_en), // input wire [0:0]  probe3 
// 	.probe4(ctrl_lat_cnt) // input wire [31:0]  probe4
// );

// ila_ctrl_lat ila_ctrl_lat_inst (
// 	.clk(pcie_clk), // input wire clk


// 	.probe0(axil_control.arready), // input wire [0:0]  probe0  
// 	.probe1(axil_control.arvalid), // input wire [0:0]  probe1 
// 	.probe2(axil_control.araddr), // input wire [15:0]  probe2 
// 	.probe3(ctrl_lat_en), // input wire [0:0]  probe3 
// 	.probe4(ctrl_lat_cnt), // input wire [31:0]  probe4 
// 	.probe5(axil_control.awvalid), // input wire [0:0]  probe5 
// 	.probe6(axil_control.awready), // input wire [0:0]  probe6 
// 	.probe7(axil_control.awaddr), // input wire [15:0]  probe7 
// 	.probe8(axil_control.wvalid), // input wire [0:0]  probe8 
// 	.probe9(axil_control.wready), // input wire [0:0]  probe9 
// 	.probe10(axil_control.wdata) // input wire [31:0]  probe10
// );
//ila_2 ila_dma_axil_bram_ctrl (
//	.clk(pcie_clk), // input wire clk


//	.probe0(axil_control.awvalid), // input wire [0:0]  probe0  
//	.probe1(axil_control.awready), // input wire [0:0]  probe1 
//	.probe2(axil_control.awaddr), // input wire [31:0]  probe2 
//	.probe3(axil_control.wvalid), // input wire [0:0]  probe3 
//	.probe4(axil_control.wready), // input wire [0:0]  probe4 
//	.probe5(axil_control.wdata), // input wire [31:0]  probe5 
//	.probe6(axil_control.arvalid), // input wire [0:0]  probe6 
//	.probe7(axil_control.arready), // input wire [0:0]  probe7 
//	.probe8(axil_control.araddr), // input wire [31:0]  probe8 
//	.probe9(axil_control.rvalid), // input wire [0:0]  probe9 
//    .probe10(axil_control.rready), // input wire [31:0]  probe10
//    .probe11(axil_control.rdata) // input wire [31:0]  probe10
//);




genvar i;
generate for(i = 0; i < 1; i++) begin


/*
 * TLB wires
 */
wire axis_tlb_interface_valid;
wire axis_tlb_interface_ready;
wire[135:0] axis_tlb_interface_data;
wire axis_pcie_tlb_interface_valid;
wire axis_pcie_tlb_interface_ready;
wire[135:0] axis_pcie_tlb_interface_data;


wire        axis_dma_read_cmd_to_tlb_tvalid;
wire        axis_dma_read_cmd_to_tlb_tready;
wire[95:0]  axis_dma_read_cmd_to_tlb_tdata;
wire        axis_dma_write_cmd_to_tlb_tvalid;
wire        axis_dma_write_cmd_to_tlb_tready;
wire[95:0]  axis_dma_write_cmd_to_tlb_tdata;

wire        axis_dma_read_cmd_to_cc_tvalid;
wire        axis_dma_read_cmd_to_cc_tready;
wire[95:0]  axis_dma_read_cmd_to_cc_tdata;
wire        axis_dma_write_cmd_to_cc_tvalid;
wire        axis_dma_write_cmd_to_cc_tready;
wire[95:0]  axis_dma_write_cmd_to_cc_tdata;

/*
 * DMA wires
 */
 wire        axis_dma_read_cmd_tvalid;
 wire        axis_dma_read_cmd_tready;
 wire[95:0]  axis_dma_read_cmd_tdata;
 
 
 //wire[47:0] axis_dma_read_cmd_addr;
 //assign axis_dma_read_cmd_addr = axis_dma_read_cmd_tdata[47:0];
 
 
 wire        axis_dma_write_cmd_tvalid;
 wire        axis_dma_write_cmd_tready;
 wire[95:0]  axis_dma_write_cmd_tdata;
 
 
 wire[47:0] axis_dma_write_cmd_addr;
 assign axis_dma_write_cmd_addr = axis_dma_write_cmd_tdata[47:0];
 
 
 wire        axis_dma_write_data_tvalid;
 wire        axis_dma_write_data_tready;
 wire[511:0] axis_dma_write_data_tdata;
 wire[63:0]  axis_dma_write_data_tkeep;
 wire        axis_dma_write_data_tlast;
 
 //PCIe clock
 wire        axis_dma_read_data_tvalid;
 wire        axis_dma_read_data_tready;
 wire[511:0] axis_dma_read_data_tdata;
 wire[63:0]  axis_dma_read_data_tkeep;
 wire        axis_dma_read_data_tlast;







/*
 * Memory Page Boundary Checks
 */
//get Base Addr of TLB for page boundary check
reg[47:0] regBaseVaddr;
reg[47:0] regBaseVaddrBoundCheck;
always @(posedge user_clk)
begin 
    if (~user_aresetn) begin
    end
    else begin
        if (axis_tlb_interface_valid /*&& axis_tlb_interface_ready*/ && axis_tlb_interface_data[128]) begin
            regBaseVaddr <= axis_tlb_interface_data[63:0];
            regBaseVaddrBoundCheck <= regBaseVaddr;
        end
    end
end


//TODO Currently supports at max one boundary crossing per command

mem_write_cmd_page_boundary_check_512_ip mem_write_cmd_page_boundary_check_512_inst (
  .s_axis_cmd_V_TVALID(s_axis_dma_write_cmd[i].valid),  // input wire s_axis_cmd_V_TVALID
  .s_axis_cmd_V_TREADY(s_axis_dma_write_cmd[i].ready),  // output wire s_axis_cmd_V_TREADY
  .s_axis_cmd_V_TDATA({s_axis_dma_write_cmd[i].length, s_axis_dma_write_cmd[i].address}),    // input wire [95 : 0] s_axis_cmd_V_TDATA
  .m_axis_cmd_V_TVALID(axis_dma_write_cmd_to_tlb_tvalid),  // output wire m_axis_cmd_V_TVALID
  .m_axis_cmd_V_TREADY(axis_dma_write_cmd_to_tlb_tready),  // input wire m_axis_cmd_V_TREADY
  .m_axis_cmd_V_TDATA(axis_dma_write_cmd_to_tlb_tdata),    // output wire [95 : 0] m_axis_cmd_V_TDATA
  .regBaseVaddr_V(regBaseVaddrBoundCheck),            // input wire [47 : 0] regBaseVaddr_V
  .ap_clk(user_clk),                            // input wire ap_clk
  .ap_rst_n(user_aresetn)                        // input wire ap_rst_n
);



//Boundary check for reads are done in the mem_read_cmd_merger_512
mem_read_cmd_page_boundary_check_512_ip mem_read_cmd_page_boundary_check_512_inst (
    .s_axis_cmd_V_TVALID(s_axis_dma_read_cmd[i].valid),  // input wire s_axis_cmd_V_TVALID
    .s_axis_cmd_V_TREADY(s_axis_dma_read_cmd[i].ready),  // output wire s_axis_cmd_V_TREADY
    .s_axis_cmd_V_TDATA({s_axis_dma_read_cmd[i].length, s_axis_dma_read_cmd[i].address}),    // input wire [95 : 0] s_axis_cmd_V_TDATA
  .m_axis_cmd_V_TVALID(axis_dma_read_cmd_to_tlb_tvalid),  // output wire m_axis_cmd_V_TVALID
  .m_axis_cmd_V_TREADY(axis_dma_read_cmd_to_tlb_tready),  // input wire m_axis_cmd_V_TREADY
  .m_axis_cmd_V_TDATA(axis_dma_read_cmd_to_tlb_tdata),    // output wire [95 : 0] m_axis_cmd_V_TDATA
  .regBaseVaddr_V(regBaseVaddrBoundCheck),            // input wire [47 : 0] regBaseVaddr_V
  .ap_clk(user_clk),                            // input wire ap_clk
  .ap_rst_n(user_aresetn)                        // input wire ap_rst_n
);

/*
 * Clock Conversion Data
 */

//TODO do not use FIFOs?
//axis_clock_converter_512 dma_bench_read_data_cc_inst (
 axis_data_fifo_512_cc dma_bench_read_data_cc_inst (
    .s_axis_aresetn(pcie_aresetn),
    .s_axis_aclk(pcie_clk),
    .s_axis_tvalid(axis_dma_read_data_tvalid),
    .s_axis_tready(axis_dma_read_data_tready),
    .s_axis_tdata(axis_dma_read_data_tdata),
    .s_axis_tkeep(axis_dma_read_data_tkeep),
    .s_axis_tlast(axis_dma_read_data_tlast),
  
    .m_axis_aclk(user_clk),
    .m_axis_tvalid(m_axis_dma_read_data[i].valid),
    .m_axis_tready(m_axis_dma_read_data[i].ready),
    .m_axis_tdata(m_axis_dma_read_data[i].data),
    .m_axis_tkeep(m_axis_dma_read_data[i].keep),
    .m_axis_tlast(m_axis_dma_read_data[i].last)
    
  );
  assign axis_dma_read_data_tvalid = axis_dma_h2c[i].valid;
  assign axis_dma_h2c[i].ready = axis_dma_read_data_tready;
  assign axis_dma_read_data_tdata = axis_dma_h2c[i].data;
  assign axis_dma_read_data_tkeep = axis_dma_h2c[i].keep;
  assign axis_dma_read_data_tlast = axis_dma_h2c[i].last;
  
  //axis_clock_converter_512 dma_bench_write_data_cc_inst (
  axis_data_fifo_512_cc dma_bench_write_data_cc_inst (
    .s_axis_aresetn(user_aresetn),
    .s_axis_aclk(user_clk),
    .s_axis_tvalid(s_axis_dma_write_data[i].valid),
    .s_axis_tready(s_axis_dma_write_data[i].ready),
    .s_axis_tdata(s_axis_dma_write_data[i].data),
    .s_axis_tkeep(s_axis_dma_write_data[i].keep),
    .s_axis_tlast(s_axis_dma_write_data[i].last),
    
    .m_axis_aclk(pcie_clk),
    .m_axis_tvalid(axis_dma_write_data_tvalid),
    .m_axis_tready(axis_dma_write_data_tready),
    .m_axis_tdata(axis_dma_write_data_tdata),
    .m_axis_tkeep(axis_dma_write_data_tkeep),
    .m_axis_tlast(axis_dma_write_data_tlast)
  );
  
  assign axis_dma_c2h[i].valid = axis_dma_write_data_tvalid;
  assign axis_dma_write_data_tready = axis_dma_c2h[i].ready;
  assign axis_dma_c2h[i].data = axis_dma_write_data_tdata;
  assign axis_dma_c2h[i].keep = axis_dma_write_data_tkeep;
  assign axis_dma_c2h[i].last = axis_dma_write_data_tlast;
  
/*
 * TLB
 */
wire tlb_miss_count_valid;
wire[31:0] tlb_miss_count;
wire tlb_page_crossing_count_valid;
wire[31:0] tlb_page_crossing_count;

reg[31:0] tlb_miss_counter;
reg[31:0] tlb_boundary_crossing_counter;
reg[31:0] pcie_tlb_miss_counter;
reg[31:0] pcie_tlb_boundary_crossing_counter;

reg[135:0]  pcie_tlb_data;
reg         pcie_tlb_valid;
reg         tlb_start,tlb_start_r;

always @(posedge user_clk)
begin 
    if (~user_aresetn) begin
        tlb_miss_counter <= 0;
        tlb_boundary_crossing_counter <= 0;
    end
    else begin
        if (tlb_miss_count_valid) begin
            tlb_miss_counter <= tlb_miss_count;
        end
        if (tlb_page_crossing_count_valid) begin
            tlb_boundary_crossing_counter <= tlb_page_crossing_count;
        end
    end
end


always @(posedge pcie_clk) begin 
    pcie_tlb_data                   <= {fpga_control_reg[i*6+12][7:0],fpga_control_reg[i*6+11],fpga_control_reg[i*6+10],fpga_control_reg[i*6+9],fpga_control_reg[i*6+8]};
end

always @(posedge pcie_clk) begin 
    tlb_start                       <= fpga_control_reg[i*6+13][0];
    tlb_start_r                     <= tlb_start;
end

always @(posedge pcie_clk) begin 
    if(~pcie_aresetn) begin
        pcie_tlb_valid              <= 1'b0;
    end
    else if(tlb_start & ~tlb_start_r) begin
        pcie_tlb_valid              <= 1'b1;
    end
    else if(axis_pcie_tlb_interface_valid & axis_pcie_tlb_interface_ready) begin
        pcie_tlb_valid              <= 1'b0;
    end
    else begin
        pcie_tlb_valid              <= pcie_tlb_valid;
    end
end

    assign axis_pcie_tlb_interface_valid    = pcie_tlb_valid;
    assign axis_pcie_tlb_interface_data     = pcie_tlb_data;


axis_clock_converter_136 axis_tlb_if_clock_converter_inst (
   .s_axis_aresetn(pcie_aresetn),  // input wire s_axis_aresetn
   .s_axis_aclk(pcie_clk),        // input wire s_axis_aclk
   
   .s_axis_tvalid(axis_pcie_tlb_interface_valid),    // input wire s_axis_tvalid
   .s_axis_tready(axis_pcie_tlb_interface_ready),    // output wire s_axis_tready
   .s_axis_tdata(axis_pcie_tlb_interface_data),      // input wire [136 : 0] s_axis_tdata
   
   .m_axis_aclk(user_clk),        // input wire m_axis_aclk
   .m_axis_aresetn(user_aresetn),  // input wire m_axis_aresetn
     
   .m_axis_tvalid(axis_tlb_interface_valid),    // output wire m_axis_tvalid
   .m_axis_tready(axis_tlb_interface_ready),    // input wire m_axis_tready
   .m_axis_tdata(axis_tlb_interface_data)      // output wire [136 : 0] m_axis_tdata
);

axis_clock_converter_32 axis_clock_converter_tlb_miss (
   .s_axis_aresetn(user_aresetn),  // input wire s_axis_aresetn
   .s_axis_aclk(user_clk),        // input wire s_axis_aclk
   .s_axis_tvalid(1'b1),    // input wire s_axis_tvalid
   .s_axis_tready(),    // output wire s_axis_tready
   .s_axis_tdata(tlb_miss_counter),
   
   .m_axis_aclk(pcie_clk),        // input wire m_axis_aclk
   .m_axis_aresetn(pcie_aresetn),  // input wire m_axis_aresetn
   .m_axis_tvalid(),    // output wire m_axis_tvalid
   .m_axis_tready(1'b1),    // input wire m_axis_tready
   .m_axis_tdata(pcie_tlb_miss_counter)      // output wire [159 : 0] m_axis_tdata
);

axis_clock_converter_32 axis_clock_converter_tlb_page_crossing (
   .s_axis_aresetn(user_aresetn),  // input wire s_axis_aresetn
   .s_axis_aclk(user_clk),        // input wire s_axis_aclk
   .s_axis_tvalid(1'b1),    // input wire s_axis_tvalid
   .s_axis_tready(),    // output wire s_axis_tready
   .s_axis_tdata(tlb_boundary_crossing_counter),
   
   .m_axis_aclk(pcie_clk),        // input wire m_axis_aclk
   .m_axis_aresetn(pcie_aresetn),  // input wire m_axis_aresetn
   .m_axis_tvalid(),    // output wire m_axis_tvalid
   .m_axis_tready(1'b1),    // input wire m_axis_tready
   .m_axis_tdata(pcie_tlb_boundary_crossing_counter)      // output wire [159 : 0] m_axis_tdata
);




 tlb_ip tlb_inst (
   /*.m_axis_ddr_read_cmd_V_TVALID(axis_ddr_read_cmd_tvalid),    // output wire m_axis_ddr_read_cmd_tvalid
   .m_axis_ddr_read_cmd_V_TREADY(axis_ddr_read_cmd_tready),    // input wire m_axis_ddr_read_cmd_tready
   .m_axis_ddr_read_cmd_V_TDATA(axis_ddr_read_cmd_tdata),      // output wire [71 : 0] m_axis_ddr_read_cmd_tdata
   .m_axis_ddr_write_cmd_V_TVALID(axis_ddr_write_cmd_tvalid),  // output wire m_axis_ddr_write_cmd_tvalid
   .m_axis_ddr_write_cmd_V_TREADY(axis_ddr_write_cmd_tready),  // input wire m_axis_ddr_write_cmd_tready
   .m_axis_ddr_write_cmd_V_TDATA(axis_ddr_write_cmd_tdata),    // output wire [71 : 0] m_axis_ddr_write_cmd_tdata*/
   .m_axis_dma_read_cmd_V_TVALID(axis_dma_read_cmd_to_cc_tvalid),    // output wire m_axis_dma_read_cmd_tvalid
   .m_axis_dma_read_cmd_V_TREADY(axis_dma_read_cmd_to_cc_tready),    // input wire m_axis_dma_read_cmd_tready
   .m_axis_dma_read_cmd_V_TDATA(axis_dma_read_cmd_to_cc_tdata),      // output wire [95 : 0] m_axis_dma_read_cmd_tdata
   .m_axis_dma_write_cmd_V_TVALID(axis_dma_write_cmd_to_cc_tvalid),  // output wire m_axis_dma_write_cmd_tvalid
   .m_axis_dma_write_cmd_V_TREADY(axis_dma_write_cmd_to_cc_tready),  // input wire m_axis_dma_write_cmd_tready
   .m_axis_dma_write_cmd_V_TDATA(axis_dma_write_cmd_to_cc_tdata),    // output wire [95 : 0] m_axis_dma_write_cmd_tdata
   .s_axis_mem_read_cmd_V_TVALID(axis_dma_read_cmd_to_tlb_tvalid),    // input wire s_axis_mem_read_cmd_tvalid
   .s_axis_mem_read_cmd_V_TREADY(axis_dma_read_cmd_to_tlb_tready),    // output wire s_axis_mem_read_cmd_tready
   .s_axis_mem_read_cmd_V_TDATA(axis_dma_read_cmd_to_tlb_tdata),      // input wire [111 : 0] s_axis_mem_read_cmd_tdata
   .s_axis_mem_write_cmd_V_TVALID(axis_dma_write_cmd_to_tlb_tvalid),  // input wire s_axis_mem_write_cmd_tvalid
   .s_axis_mem_write_cmd_V_TREADY(axis_dma_write_cmd_to_tlb_tready),  // output wire s_axis_mem_write_cmd_tready
   .s_axis_mem_write_cmd_V_TDATA(axis_dma_write_cmd_to_tlb_tdata),    // input wire [111 : 0] s_axis_mem_write_cmd_tdata
   .s_axis_tlb_interface_V_TVALID(axis_tlb_interface_valid),  // input wire s_axis_tlb_interface_tvalid
   .s_axis_tlb_interface_V_TREADY(axis_tlb_interface_ready),  // output wire s_axis_tlb_interface_tready
   .s_axis_tlb_interface_V_TDATA(axis_tlb_interface_data),    // input wire [135 : 0] s_axis_tlb_interface_tdata
   .ap_clk(user_clk),                                                // input wire aclk
   .ap_rst_n(user_aresetn),                                          // input wire aresetn
   .regTlbMissCount_V(tlb_miss_count),                      // output wire [31 : 0] regTlbMissCount_V
   .regTlbMissCount_V_ap_vld(tlb_miss_count_valid),
   .regPageCrossingCount_V(tlb_page_crossing_count),                // output wire [31 : 0] regPageCrossingCount_V
   .regPageCrossingCount_V_ap_vld(tlb_page_crossing_count_valid)  // output wire regPageCrossingCount_V_ap_vld
 );

// ila_tlb ila_tlb_inst (
//	.clk(user_clk), // input wire clk


//	.probe0(axis_dma_read_cmd_to_cc_tvalid), // input wire [0:0]  probe0  
//	.probe1(axis_dma_read_cmd_to_cc_tready), // input wire [0:0]  probe1 
//	.probe2(axis_dma_read_cmd_to_cc_tdata), // input wire [95:0]  probe2 
//	.probe3(axis_dma_write_cmd_to_cc_tvalid), // input wire [0:0]  probe3 
//	.probe4(axis_dma_write_cmd_to_cc_tready), // input wire [0:0]  probe4 
//	.probe5(axis_dma_write_cmd_to_cc_tdata), // input wire [95:0]  probe5 
//	.probe6(axis_dma_read_cmd_to_tlb_tvalid), // input wire [0:0]  probe6 
//	.probe7(axis_dma_read_cmd_to_tlb_tready), // input wire [0:0]  probe7 
//	.probe8(axis_dma_read_cmd_to_tlb_tdata), // input wire [95:0]  probe8 
//	.probe9(axis_dma_write_cmd_to_tlb_tvalid), // input wire [0:0]  probe9 
//	.probe10(axis_dma_write_cmd_to_tlb_tready), // input wire [0:0]  probe10 
//	.probe11(axis_dma_write_cmd_to_tlb_tdata), // input wire [95:0]  probe11 
//	.probe12(axis_tlb_interface_valid), // input wire [0:0]  probe12 
//	.probe13(axis_tlb_interface_ready), // input wire [0:0]  probe13 
//	.probe14(axis_tlb_interface_data) // input wire [135:0]  probe14
//);


 /*
  * Clock Conversion Command
  */
axis_clock_converter_96 dma_bench_read_cmd_cc_inst (
  .s_axis_aresetn(user_aresetn),
  .s_axis_aclk(user_clk),
  .s_axis_tvalid(axis_dma_read_cmd_to_cc_tvalid),
  .s_axis_tready(axis_dma_read_cmd_to_cc_tready),
  .s_axis_tdata(axis_dma_read_cmd_to_cc_tdata),
  
  .m_axis_aresetn(pcie_aresetn),
  .m_axis_aclk(pcie_clk),
  .m_axis_tvalid(axis_dma_read_cmd_tvalid),
  .m_axis_tready(axis_dma_read_cmd_tready),
  .m_axis_tdata(axis_dma_read_cmd_tdata)
);

axis_clock_converter_96 dma_bench_write_cmd_cc_inst (
  .s_axis_aresetn(user_aresetn),
  .s_axis_aclk(user_clk),
  .s_axis_tvalid(axis_dma_write_cmd_to_cc_tvalid),
  .s_axis_tready(axis_dma_write_cmd_to_cc_tready),
  .s_axis_tdata(axis_dma_write_cmd_to_cc_tdata),
  
  .m_axis_aresetn(pcie_aresetn),
  .m_axis_aclk(pcie_clk),
  .m_axis_tvalid(axis_dma_write_cmd_tvalid),
  .m_axis_tready(axis_dma_write_cmd_tready),
  .m_axis_tdata(axis_dma_write_cmd_tdata)
);



/*
 * DMA Descriptor bypass
 */
 wire      axis_dma_write_dsc_byp_ready;
 reg       axis_dma_write_dsc_byp_load;
 reg[63:0] axis_dma_write_dsc_byp_addr;
 reg[31:0] axis_dma_write_dsc_byp_len;
 
 wire      axis_dma_read_dsc_byp_ready;
 reg       axis_dma_read_dsc_byp_load;
 reg[63:0] axis_dma_read_dsc_byp_addr;
 reg[31:0] axis_dma_read_dsc_byp_len;
 
 // Write descriptor bypass
 assign axis_dma_write_cmd_tready = axis_dma_write_dsc_byp_ready;
 always @(posedge pcie_clk)
 begin 
     if (~pcie_aresetn) begin
         axis_dma_write_dsc_byp_load <= 1'b0;
     end
     else begin
         axis_dma_write_dsc_byp_load <= 1'b0;
         
         if (axis_dma_write_cmd_tvalid && axis_dma_write_cmd_tready) begin
             axis_dma_write_dsc_byp_load <= 1'b1;
             axis_dma_write_dsc_byp_addr <= axis_dma_write_cmd_tdata[63:0];
             axis_dma_write_dsc_byp_len  <= axis_dma_write_cmd_tdata[95:64];
         end
     end
 end
 
 // Read descriptor bypass
 assign axis_dma_read_cmd_tready = axis_dma_read_dsc_byp_ready;
 always @(posedge pcie_clk)
 begin 
     if (~pcie_aresetn) begin
         axis_dma_read_dsc_byp_load <= 1'b0;
     end
     else begin
         axis_dma_read_dsc_byp_load <= 1'b0;
         
         if (axis_dma_read_cmd_tvalid && axis_dma_read_cmd_tready) begin
             axis_dma_read_dsc_byp_load <= 1'b1;
             axis_dma_read_dsc_byp_addr <= axis_dma_read_cmd_tdata[63:0];
             axis_dma_read_dsc_byp_len  <= axis_dma_read_cmd_tdata[95:64];
         end
     end
 end
 
 //TODO use two engines
 //TODO not necessary
 //Assignments
 assign c2h_dsc_byp_load[i] = axis_dma_write_dsc_byp_load;
 assign axis_dma_write_dsc_byp_ready = c2h_dsc_byp_ready[i];
 assign c2h_dsc_byp_addr[i] = axis_dma_write_dsc_byp_addr;
 assign c2h_dsc_byp_len[i] = axis_dma_write_dsc_byp_len;
 
 
 assign h2c_dsc_byp_load[i] = axis_dma_read_dsc_byp_load;
 assign axis_dma_read_dsc_byp_ready = h2c_dsc_byp_ready[i];
 assign h2c_dsc_byp_addr[i] = axis_dma_read_dsc_byp_addr;
 assign h2c_dsc_byp_len[i] = axis_dma_read_dsc_byp_len;
 
 
 
/*
 * DMA Statistics
 */
reg[31:0] dma_write_cmd_counter;
reg[31:0] dma_write_load_counter;
reg[31:0] dma_write_word_counter;
reg[31:0] dma_write_pkg_counter;
wire reset_dma_write_length_counter;
reg[47:0] dma_write_length_counter;

reg[31:0] dma_read_cmd_counter;
reg[31:0] dma_read_load_counter;
reg[31:0] dma_read_word_counter;
reg[31:0] dma_read_pkg_counter;
wire reset_dma_read_length_counter;
reg[47:0] dma_read_length_counter;
reg dma_reads_flushed;

always @(posedge pcie_clk)begin 
    if (~pcie_aresetn) begin
        dma_write_cmd_counter <= 0;
        dma_write_load_counter <= 0;
        dma_write_word_counter <= 0;
        dma_write_pkg_counter <= 0;

        dma_read_cmd_counter <= 0;
        dma_read_load_counter <= 0;
        dma_read_word_counter <= 0;
        dma_read_pkg_counter <= 0;

        //write_bypass_ready_counter <= 0;
        dma_write_length_counter <= 0;
        dma_read_length_counter <= 0;
        //dma_write_back_pressure_counter <= 0;
        dma_reads_flushed <= 0;
        //invalid_read <= 0;
    end
    else begin
        dma_reads_flushed <= (dma_read_cmd_counter == dma_read_pkg_counter);
        //write
        if (axis_dma_write_cmd_tvalid && axis_dma_write_cmd_tready) begin
            dma_write_cmd_counter <= dma_write_cmd_counter + 1;
            dma_write_length_counter <= dma_write_length_counter + axis_dma_write_cmd_tdata[95:64];
        end
        if (reset_dma_write_length_counter) begin
            dma_write_length_counter <= 0;
        end
        if (axis_dma_write_dsc_byp_load) begin
            dma_write_load_counter <= dma_write_load_counter + 1;
        end
        if (axis_dma_write_data_tvalid && axis_dma_write_data_tready) begin
            dma_write_word_counter <= dma_write_word_counter + 1;
            if (axis_dma_write_data_tlast) begin
                dma_write_pkg_counter <= dma_write_pkg_counter + 1;
            end
        end
        //read
        if (axis_dma_read_cmd_tvalid && axis_dma_read_cmd_tready) begin
            dma_read_cmd_counter <= dma_read_cmd_counter + 1;
            dma_read_length_counter <= dma_read_length_counter + axis_dma_read_cmd_tdata[95:64];
            /*if (axis_dma_read_cmd_tdata[95:64] == 0) begin
                invalid_read <=  1;
            end*/
        end
        if (reset_dma_read_length_counter) begin
            dma_read_length_counter <= 0;
        end
        if (axis_dma_read_dsc_byp_load) begin
            dma_read_load_counter <= dma_read_load_counter + 1;
        end
        if (axis_dma_read_data_tvalid && axis_dma_read_data_tready) begin
            dma_read_word_counter <= dma_read_word_counter + 1;
            if (axis_dma_read_data_tlast) begin
                dma_read_pkg_counter <= dma_read_pkg_counter + 1;
            end
        end
    end
end

always @(posedge pcie_clk)begin 

    fpga_status_reg_r[i*11+8]            <= pcie_tlb_miss_counter;
    fpga_status_reg_r[i*11+9]            <= pcie_tlb_boundary_crossing_counter;
    fpga_status_reg_r[i*11+10]           <= dma_write_cmd_counter;
    fpga_status_reg_r[i*11+11]           <= dma_write_word_counter;
    fpga_status_reg_r[i*11+12]           <= dma_write_pkg_counter;
    fpga_status_reg_r[i*11+13]           <= dma_read_cmd_counter;
    fpga_status_reg_r[i*11+14]           <= dma_read_word_counter;
    fpga_status_reg_r[i*11+15]           <= dma_read_pkg_counter;
    fpga_status_reg_r[i*11+16]           <= dma_write_length_counter;
    fpga_status_reg_r[i*11+17]           <= dma_read_length_counter;
    fpga_status_reg_r[i*11+18]           <= dma_reads_flushed;

end



end
endgenerate




/*
 * DMA Controller
 */
 dma_controller controller_inst(
    .pcie_clk(pcie_clk),
    .pcie_aresetn(pcie_aresetn),
    .user_clk(pcie_clk), //TODO
    .user_aresetn(pcie_aresetn),
    
    .bram_en_a(bram_en_a),          // output wire bram_en_a
    .bram_we_a(bram_we_a),          // output wire [3 : 0] bram_we_a
    .bram_addr_a(bram_addr_a),      // output wire [15 : 0] bram_addr_a
    .bram_wrdata_a(bram_wrdata_a),  // output wire [31 : 0] bram_wrdata_a
    .bram_rddata_a(bram_rddata_a),  // input wire [31 : 0] bram_rddata_a

    .fpga_control_reg                       (fpga_control_reg),
    .fpga_status_reg                        (fpga_status_reg_r)


);



always @(posedge pcie_clk)begin 
    fpga_status_reg_r[0]            <= `FPGA_VERSION;
//`ifdef XDMA_BYPASS    
    fpga_status_reg_r[1]            <= 32'b1;
//`else
//    fpga_status_reg_r[1]            <= 32'b0;
//`endif
    fpga_status_reg_r[3]            <= xdma_axil_rdata;
    fpga_status_reg_r[4]            <= ctrl_lat_cnt;
    fpga_status_reg_r[5]            <= rd_lat_sum;
    fpga_status_reg_r[6]            <= wr_th_sum;
    fpga_status_reg_r[7]            <= rd_th_sum;
    fpga_status_reg_r[511:52]       <= fpga_status_reg[511:52];

end


endmodule
