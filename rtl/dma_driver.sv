`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2020/02/20 20:04:04
// Design Name: 
// Module Name: dma_driver
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
module dma_driver(
    input wire          sys_clk,
    input wire          sys_clk_gt,
    input wire          sys_rst_n,
    output logic        user_lnk_up,

    input wire[15:0]    pcie_rx_p,
    input wire[15:0]    pcie_rx_n,
    output logic[15:0]  pcie_tx_p,
    output logic[15:0]  pcie_tx_n,

    output logic        pcie_clk,
    output reg          pcie_aresetn,
    /*input wire[0:0]     user_irq_req,
    output logic        user_irq_ack,
    output logic        msi_enable,
    output logic[2:0]   msi_vector_width,*/

    // Axi Lite Control interface
    axi_lite.master     m_axil,
`ifdef XDMA_BYPASS  
    // AXI MM Control Interface
    axi_mm.master       m_axim,
`endif
    // AXI Stream Interface
    axi_stream.slave    s_axis_c2h_data[0:3],
    axi_stream.master   m_axis_h2c_data[0:3],

    // Descriptor Bypass
    output logic [3:0]       c2h_dsc_byp_ready,
    //input wire[63:0]    c2h_dsc_byp_src_addr_0,
    input wire[3:0][63:0]    c2h_dsc_byp_addr,
    input wire[3:0][31:0]    c2h_dsc_byp_len,
    //input wire[15:0]    c2h_dsc_byp_ctl_0,
    input wire[3:0]          c2h_dsc_byp_load,
    
    output logic[3:0]        h2c_dsc_byp_ready,
    input wire[3:0][63:0]    h2c_dsc_byp_addr,
    //input wire[63:0]    h2c_dsc_byp_dst_addr_0,
    input wire[3:0][31:0]    h2c_dsc_byp_len,
    //input wire[15:0]    h2c_dsc_byp_ctl_0,
    input wire[3:0]          h2c_dsc_byp_load,
    
    output logic[3:0][7:0]   c2h_sts,
    output logic[3:0][7:0]   h2c_sts,

    input wire[31:0]        s_axil_awaddr,
    input wire[31:0]        s_axil_wdata,
    input wire[31:0]        s_axil_araddr,
    input wire[31:0]        xdma_contrl_wr_rd,
    output reg[31:0]        s_axil_rdata
);

wire    pcie_aresetn_o;
always @(posedge pcie_clk)
    pcie_aresetn                <= pcie_aresetn_o;

//For PCIe 3.0 16x,

axi_stream #(.WIDTH(512))   axis_dma_read_data_to_width[4]();
axi_stream #(.WIDTH(512))   axis_dma_write_data_from_width[4]();

genvar i;
generate for(i = 0; i < 4; i++) begin

assign m_axis_h2c_data[i].valid = axis_dma_read_data_to_width[i].valid;
assign axis_dma_read_data_to_width[i].ready = m_axis_h2c_data[i].ready;
assign m_axis_h2c_data[i].data = axis_dma_read_data_to_width[i].data;
assign m_axis_h2c_data[i].keep = axis_dma_read_data_to_width[i].keep;
assign m_axis_h2c_data[i].last = axis_dma_read_data_to_width[i].last;

assign axis_dma_write_data_from_width[i].valid = s_axis_c2h_data[i].valid;
assign s_axis_c2h_data[i].ready = axis_dma_write_data_from_width[i].ready;
assign axis_dma_write_data_from_width[i].data = s_axis_c2h_data[i].data;
assign axis_dma_write_data_from_width[i].keep = s_axis_c2h_data[i].keep;
assign axis_dma_write_data_from_width[i].last = s_axis_c2h_data[i].last;

end
endgenerate
//////////////////////////test dma/////////////////



// wire[31:0]                  s_axil_awaddr;
// wire[31:0]                  s_axil_wdata;   
// wire[31:0]                  s_axil_araddr; 
// wire[31:0]                  s_axil_rdata; 
// wire                        wr_en,rd_en;
reg                         wren_r,wren_rr,rden_r,rden_rr;
// wire                        s_axil_rvalid;
// wire                        s_axil_awready,s_axil_wready,s_axil_arready; 
reg                         s_axil_awvalid,s_axil_wvalid,s_axil_arvalid;                    

always@(posedge pcie_clk)begin
    wren_r                          <= xdma_contrl_wr_rd[0];
    wren_rr                         <= wren_r;
    rden_r                          <= xdma_contrl_wr_rd[1];
    rden_rr                         <= rden_r;    
end

always@(posedge pcie_clk)begin
    if(~pcie_aresetn)begin
        s_axil_awvalid              <= 1'b0;
    end
    else if(wren_r & ~wren_rr)begin
        s_axil_awvalid              <= 1'b1;
    end
    else if(dma_register_axil.awvalid & dma_register_axil.awready)begin
        s_axil_awvalid              <= 1'b0;
    end
    else begin
        s_axil_awvalid              <= s_axil_awvalid;
    end
end

always@(posedge pcie_clk)begin
    if(~pcie_aresetn)begin
        s_axil_wvalid              <= 1'b0;
    end
    else if(dma_register_axil.awvalid & dma_register_axil.awready)begin
        s_axil_wvalid              <= 1'b1;
    end
    else if(dma_register_axil.wvalid & dma_register_axil.wready)begin
        s_axil_wvalid              <= 1'b0;
    end
    else begin
        s_axil_wvalid              <= s_axil_wvalid;
    end
end

always@(posedge pcie_clk)begin
    if(~pcie_aresetn)begin
        s_axil_arvalid              <= 1'b0;
    end
    else if(rden_r & ~rden_rr)begin
        s_axil_arvalid              <= 1'b1;
    end
    else if(dma_register_axil.arvalid & dma_register_axil.arready)begin
        s_axil_arvalid              <= 1'b0;
    end
    else begin
        s_axil_arvalid              <= s_axil_arvalid;
    end
end

always@(posedge pcie_clk)begin
    if(~pcie_aresetn)begin
        s_axil_rdata                <= 1'b0;
    end  
    else if(dma_register_axil.rvalid & dma_register_axil.rready)begin
        s_axil_rdata                <= dma_register_axil.rdata;
    end      
    else begin
        s_axil_rdata                <= s_axil_rdata;
    end
end

// vio_0 vio_dma_register (
//   .clk(pcie_clk),                // input wire clk
//   .probe_out0(s_axil_awaddr),  // output wire [31 : 0] probe_out0
//   .probe_out1(s_axil_araddr),  // output wire [31 : 0] probe_out1
//   .probe_out2(s_axil_wdata),  // output wire [31 : 0] probe_out2
//   .probe_out3(wr_en),  // output wire [0 : 0] probe_out3
//   .probe_out4(rd_en)  // output wire [0 : 0] probe_out4
// );

//ila_2 ila_dma_register (
//	.clk(pcie_clk), // input wire clk


//	.probe0(dma_register_axil.awvalid), // input wire [0:0]  probe0  
//	.probe1(dma_register_axil.awready), // input wire [0:0]  probe1 
//	.probe2(dma_register_axil.awaddr), // input wire [31:0]  probe2 
//	.probe3(dma_register_axil.wvalid), // input wire [0:0]  probe3 
//	.probe4(dma_register_axil.wready), // input wire [0:0]  probe4 
//	.probe5(dma_register_axil.wdata), // input wire [31:0]  probe5 
//	.probe6(dma_register_axil.arvalid), // input wire [0:0]  probe6 
//	.probe7(dma_register_axil.arready), // input wire [0:0]  probe7 
//	.probe8(dma_register_axil.araddr), // input wire [31:0]  probe8 
//	.probe9(dma_register_axil.rvalid), // input wire [0:0]  probe9 
//	.probe10(dma_register_axil.rdata) // input wire [31:0]  probe10
//);

axi_lite dma_register_axil();
reg[31:0]                   dma_addr,dma_data;
reg[3:0]                    register_num;
reg[3:0]                    config_state;

localparam [3:0]            IDLE = 4'b0001,
                            ADDR = 4'b0010,
                            DATA = 4'b0100,
                            DONE = 4'b1000;


always@(posedge pcie_clk)begin
    if(~pcie_aresetn)begin
        dma_addr                    <= 32'h0000_0104;
        dma_data                    <= 32'h1;        
    end
    else begin
        case(register_num)
            0:begin
                dma_addr            <= 32'h0000_0104;
                dma_data            <= 32'h1;
            end
            1:begin
                dma_addr            <= 32'h0000_1104;
                dma_data            <= 32'h1;
            end
            2:begin
                dma_addr            <= 32'h0000_0204;
                dma_data            <= 32'h1;
            end
            3:begin
                dma_addr            <= 32'h0000_1204;
                dma_data            <= 32'h1;
            end
            4:begin
                dma_addr            <= 32'h0000_0304;
                dma_data            <= 32'h1;
            end   
            5:begin
                dma_addr            <= 32'h0000_1304;
                dma_data            <= 32'h1;
            end 
            6:begin
                dma_addr            <= 32'h0000_0004;
                dma_data            <= 32'h1;
            end
            7:begin
                dma_addr            <= 32'h0000_1004;
                dma_data            <= 32'h1;
            end                        
        endcase                                
    end
end


always@(posedge pcie_clk)begin
    if(~pcie_aresetn)begin
        register_num                <= 4'b0;    
        config_state                <= IDLE;
    end
    else begin
        case(config_state)
            IDLE:begin
                if(register_num == 8)begin
                    config_state    <= DONE;
                end
                else begin
                    config_state    <= ADDR;
                end                
            end
            ADDR:begin
                if(dma_register_axil.awvalid & dma_register_axil.awready)begin
                    config_state    <= DATA;
                end
                else begin
                    config_state    <= ADDR;
                end
            end
            DATA:begin
                if(dma_register_axil.wvalid & dma_register_axil.wready)begin
                    config_state    <= IDLE;
                    register_num    <= register_num + 1'b1;
                end
                else begin
                    config_state    <= DATA;
                end
            end
            DONE:begin
                config_state    <= DONE;              
            end                
        endcase    
    end
end


assign dma_register_axil.awvalid = config_state[3]? s_axil_awvalid : config_state[1];
assign dma_register_axil.wvalid = config_state[3]? s_axil_wvalid : config_state[2];
assign dma_register_axil.awaddr = config_state[3]? s_axil_awaddr : dma_addr;
assign dma_register_axil.wdata = config_state[3]? s_axil_wdata : dma_data;
assign dma_register_axil.arvalid = s_axil_arvalid;
assign dma_register_axil.araddr = s_axil_araddr;
assign dma_register_axil.rready = 1'b1;

// ila_2 ila_dma_register (
// 	.clk(pcie_clk), // input wire clk


// 	.probe0(dma_register_axil.awvalid), // input wire [0:0]  probe0  
// 	.probe1(dma_register_axil.awready), // input wire [0:0]  probe1 
// 	.probe2(dma_addr), // input wire [31:0]  probe2 
// 	.probe3(dma_register_axil.wvalid), // input wire [0:0]  probe3 
// 	.probe4(dma_register_axil.wready), // input wire [0:0]  probe4 
// 	.probe5(dma_data), // input wire [31:0]  probe5 
// 	.probe6(0), // input wire [0:0]  probe6 
// 	.probe7(0), // input wire [0:0]  probe7 
// 	.probe8(register_num), // input wire [31:0]  probe8 
// 	.probe9(0), // input wire [0:0]  probe9 
// 	.probe10(config_state) // input wire [31:0]  probe10
// );



xdma_0 dma_inst (
  .sys_clk(sys_clk),                                              // input wire sys_clk
  .sys_clk_gt(sys_clk_gt),
  .sys_rst_n(sys_rst_n),                                          // input wire sys_rst_n
  .user_lnk_up(user_lnk_up),                                      // output wire user_lnk_up
  .pci_exp_txp(pcie_tx_p),                                      // output wire [7 : 0] pci_exp_txp
  .pci_exp_txn(pcie_tx_n),                                      // output wire [7 : 0] pci_exp_txn
  .pci_exp_rxp(pcie_rx_p),                                      // input wire [7 : 0] pci_exp_rxp
  .pci_exp_rxn(pcie_rx_n),                                      // input wire [7 : 0] pci_exp_rxn
  .axi_aclk(pcie_clk),                                            // output wire axi_aclk
  .axi_aresetn(pcie_aresetn_o),                                      // output wire axi_aresetn
  .usr_irq_req(1'b0),                                      // input wire [0 : 0] usr_irq_req
  .usr_irq_ack(),                                      // output wire [0 : 0] usr_irq_ack
  .msi_enable(),                                        // output wire msi_enable
  .msi_vector_width(),                            // output wire [2 : 0] msi_vector_width
  
  // LITE interface   
  //-- AXI Master Write Address Channel
  .m_axil_awaddr(m_axil.awaddr),              // output wire [31 : 0] m_axil_awaddr
  .m_axil_awprot(),              // output wire [2 : 0] m_axil_awprot
  .m_axil_awvalid(m_axil.awvalid),            // output wire m_axil_awvalid
  .m_axil_awready(m_axil.awready),            // input wire m_axil_awready
  //-- AXI Master Write Data Channel
  .m_axil_wdata(m_axil.wdata),                // output wire [31 : 0] m_axil_wdata
  .m_axil_wstrb(m_axil.wstrb),                // output wire [3 : 0] m_axil_wstrb
  .m_axil_wvalid(m_axil.wvalid),              // output wire m_axil_wvalid
  .m_axil_wready(m_axil.wready),              // input wire m_axil_wready
  //-- AXI Master Write Response Channel
  .m_axil_bvalid(m_axil.bvalid),              // input wire m_axil_bvalid
  .m_axil_bresp(m_axil.bresp),                // input wire [1 : 0] m_axil_bresp
  .m_axil_bready(m_axil.bready),              // output wire m_axil_bready
  //-- AXI Master Read Address Channel
  .m_axil_araddr(m_axil.araddr),              // output wire [31 : 0] m_axil_araddr
  .m_axil_arprot(),              // output wire [2 : 0] m_axil_arprot
  .m_axil_arvalid(m_axil.arvalid),            // output wire m_axil_arvalid
  .m_axil_arready(m_axil.arready),            // input wire m_axil_arready
  .m_axil_rdata(m_axil.rdata),                // input wire [31 : 0] m_axil_rdata
  //-- AXI Master Read Data Channel
  .m_axil_rresp(m_axil.rresp),                // input wire [1 : 0] m_axil_rresp
  .m_axil_rvalid(m_axil.rvalid),              // input wire m_axil_rvalid
  .m_axil_rready(m_axil.rready),              // output wire m_axil_rready
  
  // AXI Stream Interface
  .s_axis_c2h_tvalid_0(axis_dma_write_data_from_width[0].valid),                      // input wire s_axis_c2h_tvalid_0
  .s_axis_c2h_tready_0(axis_dma_write_data_from_width[0].ready),                      // output wire s_axis_c2h_tready_0
  .s_axis_c2h_tdata_0(axis_dma_write_data_from_width[0].data),                        // input wire [255 : 0] s_axis_c2h_tdata_0
  .s_axis_c2h_tkeep_0(axis_dma_write_data_from_width[0].keep),                        // input wire [31 : 0] s_axis_c2h_tkeep_0
  .s_axis_c2h_tlast_0(axis_dma_write_data_from_width[0].last),                        // input wire s_axis_c2h_tlast_0
  .m_axis_h2c_tvalid_0(axis_dma_read_data_to_width[0].valid),                      // output wire m_axis_h2c_tvalid_0
  .m_axis_h2c_tready_0(axis_dma_read_data_to_width[0].ready),                      // input wire m_axis_h2c_tready_0
  .m_axis_h2c_tdata_0(axis_dma_read_data_to_width[0].data),                        // output wire [255 : 0] m_axis_h2c_tdata_0
  .m_axis_h2c_tkeep_0(axis_dma_read_data_to_width[0].keep),                        // output wire [31 : 0] m_axis_h2c_tkeep_0
  .m_axis_h2c_tlast_0(axis_dma_read_data_to_width[0].last),                        // output wire m_axis_h2c_tlast_0

  .s_axis_c2h_tvalid_1(axis_dma_write_data_from_width[1].valid),                      // input wire s_axis_c2h_tvalid_0
  .s_axis_c2h_tready_1(axis_dma_write_data_from_width[1].ready),                      // output wire s_axis_c2h_tready_0
  .s_axis_c2h_tdata_1(axis_dma_write_data_from_width[1].data),                        // input wire [255 : 0] s_axis_c2h_tdata_0
  .s_axis_c2h_tkeep_1(axis_dma_write_data_from_width[1].keep),                        // input wire [31 : 0] s_axis_c2h_tkeep_0
  .s_axis_c2h_tlast_1(axis_dma_write_data_from_width[1].last),                        // input wire s_axis_c2h_tlast_0
  .m_axis_h2c_tvalid_1(axis_dma_read_data_to_width[1].valid),                      // output wire m_axis_h2c_tvalid_0
  .m_axis_h2c_tready_1(axis_dma_read_data_to_width[1].ready),                      // input wire m_axis_h2c_tready_0
  .m_axis_h2c_tdata_1(axis_dma_read_data_to_width[1].data),                        // output wire [255 : 0] m_axis_h2c_tdata_0
  .m_axis_h2c_tkeep_1(axis_dma_read_data_to_width[1].keep),                        // output wire [31 : 0] m_axis_h2c_tkeep_0
  .m_axis_h2c_tlast_1(axis_dma_read_data_to_width[1].last),                        // output wire m_axis_h2c_tlast_0

  .s_axis_c2h_tvalid_2(axis_dma_write_data_from_width[2].valid),                      // input wire s_axis_c2h_tvalid_0
  .s_axis_c2h_tready_2(axis_dma_write_data_from_width[2].ready),                      // output wire s_axis_c2h_tready_0
  .s_axis_c2h_tdata_2(axis_dma_write_data_from_width[2].data),                        // input wire [255 : 0] s_axis_c2h_tdata_0
  .s_axis_c2h_tkeep_2(axis_dma_write_data_from_width[2].keep),                        // input wire [31 : 0] s_axis_c2h_tkeep_0
  .s_axis_c2h_tlast_2(axis_dma_write_data_from_width[2].last),                        // input wire s_axis_c2h_tlast_0
  .m_axis_h2c_tvalid_2(axis_dma_read_data_to_width[2].valid),                      // output wire m_axis_h2c_tvalid_0
  .m_axis_h2c_tready_2(axis_dma_read_data_to_width[2].ready),                      // input wire m_axis_h2c_tready_0
  .m_axis_h2c_tdata_2(axis_dma_read_data_to_width[2].data),                        // output wire [255 : 0] m_axis_h2c_tdata_0
  .m_axis_h2c_tkeep_2(axis_dma_read_data_to_width[2].keep),                        // output wire [31 : 0] m_axis_h2c_tkeep_0
  .m_axis_h2c_tlast_2(axis_dma_read_data_to_width[2].last),                        // output wire m_axis_h2c_tlast_0
  
  .s_axis_c2h_tvalid_3(axis_dma_write_data_from_width[3].valid),                      // input wire s_axis_c2h_tvalid_0
  .s_axis_c2h_tready_3(axis_dma_write_data_from_width[3].ready),                      // output wire s_axis_c2h_tready_0
  .s_axis_c2h_tdata_3(axis_dma_write_data_from_width[3].data),                        // input wire [255 : 0] s_axis_c2h_tdata_0
  .s_axis_c2h_tkeep_3(axis_dma_write_data_from_width[3].keep),                        // input wire [31 : 0] s_axis_c2h_tkeep_0
  .s_axis_c2h_tlast_3(axis_dma_write_data_from_width[3].last),                        // input wire s_axis_c2h_tlast_0
  .m_axis_h2c_tvalid_3(axis_dma_read_data_to_width[3].valid),                      // output wire m_axis_h2c_tvalid_0
  .m_axis_h2c_tready_3(axis_dma_read_data_to_width[3].ready),                      // input wire m_axis_h2c_tready_0
  .m_axis_h2c_tdata_3(axis_dma_read_data_to_width[3].data),                        // output wire [255 : 0] m_axis_h2c_tdata_0
  .m_axis_h2c_tkeep_3(axis_dma_read_data_to_width[3].keep),                        // output wire [31 : 0] m_axis_h2c_tkeep_0
  .m_axis_h2c_tlast_3(axis_dma_read_data_to_width[3].last),                        // output wire m_axis_h2c_tlast_0
  // Descriptor Bypass
  .c2h_dsc_byp_ready_0    (c2h_dsc_byp_ready[0]),
  .c2h_dsc_byp_src_addr_0 (64'h0),
  .c2h_dsc_byp_dst_addr_0 (c2h_dsc_byp_addr[0]),
  .c2h_dsc_byp_len_0      (c2h_dsc_byp_len[0][27:0]),
  .c2h_dsc_byp_ctl_0      (16'h3), //was 16'h3
  .c2h_dsc_byp_load_0     (c2h_dsc_byp_load[0]),
  
  .h2c_dsc_byp_ready_0    (h2c_dsc_byp_ready[0]),
  .h2c_dsc_byp_src_addr_0 (h2c_dsc_byp_addr[0]),
  .h2c_dsc_byp_dst_addr_0 (64'h0),
  .h2c_dsc_byp_len_0      (h2c_dsc_byp_len[0][27:0]),
  .h2c_dsc_byp_ctl_0      (16'h3), //was 16'h3
  .h2c_dsc_byp_load_0     (h2c_dsc_byp_load[0]),
  
  .c2h_sts_0(c2h_sts[0]),                                          // output wire [7 : 0] c2h_sts_0
  .h2c_sts_0(h2c_sts[0]),                                          // output wire [7 : 0] h2c_sts_0


  .c2h_dsc_byp_ready_1    (c2h_dsc_byp_ready[1]),
  .c2h_dsc_byp_src_addr_1 (64'h10000000),
  .c2h_dsc_byp_dst_addr_1 (c2h_dsc_byp_addr[1]),
  .c2h_dsc_byp_len_1      (c2h_dsc_byp_len[1][27:0]),
  .c2h_dsc_byp_ctl_1      (16'h3), //was 16'h3
  .c2h_dsc_byp_load_1     (c2h_dsc_byp_load[1]),
  
  .h2c_dsc_byp_ready_1    (h2c_dsc_byp_ready[1]),
  .h2c_dsc_byp_src_addr_1 (h2c_dsc_byp_addr[1]),
  .h2c_dsc_byp_dst_addr_1 (64'h10000000),
  .h2c_dsc_byp_len_1      (h2c_dsc_byp_len[1][27:0]),
  .h2c_dsc_byp_ctl_1      (16'h3), //was 16'h3
  .h2c_dsc_byp_load_1     (h2c_dsc_byp_load[1]),
  
  .c2h_sts_1(c2h_sts[1]),                                          // output wire [7 : 0] c2h_sts_0
  .h2c_sts_1(h2c_sts[1]),                                          // output wire [7 : 0] h2c_sts_0
  
  
  .c2h_dsc_byp_ready_2    (c2h_dsc_byp_ready[2]),
  .c2h_dsc_byp_src_addr_2 (64'h20000000),
  .c2h_dsc_byp_dst_addr_2 (c2h_dsc_byp_addr[2]),
  .c2h_dsc_byp_len_2      (c2h_dsc_byp_len[2][27:0]),
  .c2h_dsc_byp_ctl_2      (16'h3), //was 16'h3
  .c2h_dsc_byp_load_2     (c2h_dsc_byp_load[2]),
  
  .h2c_dsc_byp_ready_2    (h2c_dsc_byp_ready[2]),
  .h2c_dsc_byp_src_addr_2 (h2c_dsc_byp_addr[2]),
  .h2c_dsc_byp_dst_addr_2 (64'h20000000),
  .h2c_dsc_byp_len_2      (h2c_dsc_byp_len[2][27:0]),
  .h2c_dsc_byp_ctl_2      (16'h3), //was 16'h3
  .h2c_dsc_byp_load_2     (h2c_dsc_byp_load[2]),
  
  .c2h_sts_2(c2h_sts[2]),                                          // output wire [7 : 0] c2h_sts_0
  .h2c_sts_2(h2c_sts[2]),                                          // output wire [7 : 0] h2c_sts_0


  .c2h_dsc_byp_ready_3    (c2h_dsc_byp_ready[3]),
  .c2h_dsc_byp_src_addr_3 (64'h30000000),
  .c2h_dsc_byp_dst_addr_3 (c2h_dsc_byp_addr[3]),
  .c2h_dsc_byp_len_3      (c2h_dsc_byp_len[3][27:0]),
  .c2h_dsc_byp_ctl_3      (16'h3), //was 16'h3
  .c2h_dsc_byp_load_3     (c2h_dsc_byp_load[3]),
  
  .h2c_dsc_byp_ready_3    (h2c_dsc_byp_ready[3]),
  .h2c_dsc_byp_src_addr_3 (h2c_dsc_byp_addr[3]),
  .h2c_dsc_byp_dst_addr_3 (64'h30000000),
  .h2c_dsc_byp_len_3      (h2c_dsc_byp_len[3][27:0]),
  .h2c_dsc_byp_ctl_3      (16'h3), //was 16'h3
  .h2c_dsc_byp_load_3     (h2c_dsc_byp_load[3]),
  
  .c2h_sts_3(c2h_sts[3]),                                          // output wire [7 : 0] c2h_sts_0
  .h2c_sts_3(h2c_sts[3]),                                          // output wire [7 : 0] h2c_sts_0


  .s_axil_awaddr(dma_register_axil.awaddr),                    // input wire [31 : 0] s_axil_awaddr
  .s_axil_awprot(0),                    // input wire [2 : 0] s_axil_awprot
  .s_axil_awvalid(dma_register_axil.awvalid),                  // input wire s_axil_awvalid
  .s_axil_awready(dma_register_axil.awready),                  // output wire s_axil_awready
  .s_axil_wdata(dma_register_axil.wdata),                      // input wire [31 : 0] s_axil_wdata
  .s_axil_wstrb(4'hf),                      // input wire [3 : 0] s_axil_wstrb
  .s_axil_wvalid(dma_register_axil.wvalid),                    // input wire s_axil_wvalid
  .s_axil_wready(dma_register_axil.wready),                    // output wire s_axil_wready
  .s_axil_bvalid(),                    // output wire s_axil_bvalid
  .s_axil_bresp(),                      // output wire [1 : 0] s_axil_bresp
  .s_axil_bready(1),                    // input wire s_axil_bready
  .s_axil_araddr(dma_register_axil.araddr),                    // input wire [31 : 0] s_axil_araddr
  .s_axil_arprot(0),                    // input wire [2 : 0] s_axil_arprot
  .s_axil_arvalid(dma_register_axil.arvalid),                  // input wire s_axil_arvalid
  .s_axil_arready(dma_register_axil.arready),                  // output wire s_axil_arready
  .s_axil_rdata(dma_register_axil.rdata),                      // output wire [31 : 0] s_axil_rdata
  .s_axil_rresp(),                      // output wire [1 : 0] s_axil_rresp
  .s_axil_rvalid(dma_register_axil.rvalid),                    // output wire s_axil_rvalid
  .s_axil_rready(dma_register_axil.rready)                    // input wire s_axil_rready
  

  
  `ifdef XDMA_BYPASS  
  // CQ Bypass ports
  // write address channel 
  ,.m_axib_awid      (m_axim.awid),
  .m_axib_awaddr    (m_axim.awaddr[18:0]),
  .m_axib_awlen     (m_axim.awlen),
  .m_axib_awsize    (m_axim.awsize),
  
  
  
  .m_axib_awburst   (m_axim.awburst),
  .m_axib_awprot    (m_axim.awprot),
  .m_axib_awvalid   (m_axim.awvalid),
  .m_axib_awready   (m_axim.awready),
  .m_axib_awlock    (m_axim.awlock),
  .m_axib_awcache   (m_axim.awcache),
  // write data channel
  .m_axib_wdata     (m_axim.wdata),
  .m_axib_wstrb     (m_axim.wstrb),
  .m_axib_wlast     (m_axim.wlast),
  .m_axib_wvalid    (m_axim.wvalid),
  .m_axib_wready    (m_axim.wready),
  // write response channel
  .m_axib_bid       (m_axim.bid),
  .m_axib_bresp     (m_axim.bresp),
  .m_axib_bvalid    (m_axim.bvalid),
  .m_axib_bready    (m_axim.bready),
  // read address channel 
  .m_axib_arid      (m_axim.arid),
  .m_axib_araddr    (m_axim.araddr[18:0]),
  .m_axib_arlen     (m_axim.arlen),
  .m_axib_arsize    (m_axim.arsize),
  .m_axib_arburst   (m_axim.arburst),
  .m_axib_arprot    (m_axim.arprot),
  .m_axib_arvalid   (m_axim.arvalid),
  .m_axib_arready   (m_axim.arready),
  .m_axib_arlock    (m_axim.arlock),
  .m_axib_arcache   (m_axim.arcache),
  // read data channel 
  .m_axib_rid       (m_axim.rid),
  .m_axib_rdata     (m_axim.rdata),           //256 m_axim.rid m_axim.rdata
  .m_axib_rresp     (m_axim.rresp),
  .m_axib_rlast     (m_axim.rlast),
  .m_axib_rvalid    (m_axim.rvalid),
  .m_axib_rready    (m_axim.rready)
`endif  
);










endmodule
//`default_nettype wire
