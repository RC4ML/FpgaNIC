/*
 * Copyright (c) 2019, Systems Group, ETH Zurich
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
`timescale 1ns / 1ps
`default_nettype none

`define IP_VERSION4
`define POINTER_CHASING

`include "example_module.vh"

module network_stack #(
    parameter NET_BANDWIDTH = 10,
    parameter WIDTH = 512,
    parameter MAC_ADDRESS = 48'hE59D02350A00, // LSB first, 00:0A:35:02:9D:E5
    parameter IPV6_ADDRESS= 128'hE59D_02FF_FF35_0A02_0000_0000_0000_80FE, //LSB first: FE80_0000_0000_0000_020A_35FF_FF02_9DE5,
    parameter IP_SUBNET_MASK = 32'h00FFFFFF,
    parameter IP_DEFAULT_GATEWAY = 32'h00000000,
    parameter DHCP_EN   = 0,
    parameter TCP_EN = 0,
    parameter RX_DDR_BYPASS_EN = 0,
    parameter UDP_EN = 0,
    parameter ROCE_EN = 0
)(  
    /*          gt ports        */
//     input  wire [3:0] gt_rxp_in,
//     input  wire [3:0] gt_rxn_in,
//     output wire [3:0] gt_txp_out,
//     output wire [3:0] gt_txn_out,

// //    input wire          sys_reset_n,
//     input wire        gt_refclk_p,
//     input wire        gt_refclk_n,
    axi_stream.slave    axis_net_rx_data,
    axi_stream.master   axis_net_tx_data,
    /*          clock           */
    // input wire          dclk,
    input wire          user_clk,
    input wire          user_aresetn,

    input wire         net_clk,
    input wire         net_aresetn,

    input wire          mem_clk,

    // /* CONTROL INTERFACE */

    input wire [31:0] set_ip_addr_data,
    input wire [7:0]  set_board_number_data,

    //Application interface streams
    axis_meta.slave     s_axis_listen_port,
    axis_meta.master    m_axis_listen_port_status,
   
    axis_meta.slave     s_axis_open_connection,
    axis_meta.master    m_axis_open_status,
    axis_meta.slave     s_axis_close_connection,

    axis_meta.master    m_axis_notifications,
    axis_meta.slave     s_axis_read_package,
    
    axis_meta.master    m_axis_rx_metadata,
    axi_stream.master   m_axis_rx_data,
    
    axis_meta.slave     s_axis_tx_metadata,
    axi_stream.slave    s_axis_tx_data,
    axis_meta.master    m_axis_tx_status,

    // axi_mm.master       network_tx_mem,

    output wire[15:0][31:0]     status_reg
    
    
 );

 
//     // network interface streams
//  axi_stream #(.WIDTH(WIDTH))       s_axis_net();
//  axi_stream #(.WIDTH(WIDTH))       m_axis_net();


axis_meta #(.WIDTH(32))     axis_tcp_tx_meta();
axi_stream #(.WIDTH(WIDTH))    axis_tcp_tx_data();
axis_meta #(.WIDTH(64))     axis_tcp_tx_status();

// IP Handler Outputs
axi_stream #(.WIDTH(WIDTH))     axis_iph_to_arp_slice();
axi_stream #(.WIDTH(WIDTH))     axis_iph_to_icmp_slice();
axi_stream #(.WIDTH(WIDTH))     axis_iph_to_icmpv6_slice();
axi_stream #(.WIDTH(WIDTH))     axis_iph_to_rocev6_slice();

//Slice connections on RX path
axi_stream #(.WIDTH(WIDTH))     axis_arp_slice_to_arp();
axi_stream #(.WIDTH(64))     axis_icmp_slice_to_icmp();
axi_stream #(.WIDTH(WIDTH))     axis_iph_to_toe();

// MAC-IP Encode Inputs
axi_stream #(.WIDTH(WIDTH))     axis_intercon_to_mie();
axi_stream #(.WIDTH(WIDTH))     axis_mie_to_intercon();
axi_stream #(.WIDTH(WIDTH))     axis_net_tx();

//Slice connections on RX path
axi_stream #(.WIDTH(WIDTH))     axis_arp_to_arp_slice();
axi_stream #(.WIDTH(64))     axis_icmp_to_icmp_slice(); //TODO

//TCP
axi_stream #(.WIDTH(WIDTH))     axis_iph_to_toe_slice();
axi_stream #(.WIDTH(WIDTH))     axis_toe_slice_to_toe();


//UDP
axi_stream #(.WIDTH(WIDTH))     axis_udp_to_udp_slice();
axi_stream #(.WIDTH(WIDTH))     axis_udp_slice_to_merge();
axi_stream #(.WIDTH(WIDTH))     axis_iph_to_udp_slice();
axi_stream #(.WIDTH(WIDTH))     axis_udp_slice_to_udp();

//ROCE
axi_stream #(.WIDTH(WIDTH))     axis_roce_to_roce_slice();
axi_stream #(.WIDTH(WIDTH))     axis_roce_slice_to_merge();
axi_stream #(.WIDTH(WIDTH))     axis_iph_to_roce_slice();
axi_stream #(.WIDTH(WIDTH))     axis_roce_slice_to_roce();


    // memory cmd streams
    axis_mem_cmd    m_axis_read_cmd();
    axis_mem_cmd    m_axis_write_cmd();
    // memory sts streams
    axis_mem_status     s_axis_read_sts();
    axis_mem_status     s_axis_write_sts();
    // memory data streams
    axi_stream    s_axis_read_data();
    axi_stream   m_axis_write_data();



axi_stream #(.WIDTH(WIDTH))     axis_slice_to_ibh();
axi_stream #(.WIDTH(WIDTH))     axis_toe_to_toe_slice();


//ICMPv6
axi_stream #(.WIDTH(WIDTH))   axis_ipv6_to_ethen();
axi_stream #(.WIDTH(WIDTH))   axis_ethencode_to_intercon();

// DHCP Client IP address output //
wire[31:0]  dhcpAddressOut;

//IPV6
axi_stream #(.WIDTH(WIDTH))   axis_icmpv6_to_intercon();
axi_stream #(.WIDTH(WIDTH))   axis_ipv6_to_intercon();

// IPv6 lookup
wire axis_ipv6_res_rsp_TVALID;
wire axis_ipv6_res_rsp_TREADY;
wire [55:0] axis_ipv6_res_rsp_TDATA;

wire axis_ipv6_res_req_TVALID;
wire axis_ipv6_res_req_TREADY;
wire [127:0] axis_ipv6_res_req_TDATA;





wire set_ip_addr_valid;

reg[31:0] local_ip_address;
wire[31:0]ip_address_used;

wire set_board_number_valid;
// wire[7:0] set_board_number_data;
reg[7:0] board_number;


assign set_ip_addr_valid = 1'b1;
//assign set_ip_addr_data = 32'h0b01d4d1;
assign set_board_number_valid = 1;
//assign set_board_number_data =1;

always @(posedge user_clk) begin
    if (~net_aresetn) begin
        local_ip_address <= 32'hD1D4010B;
        board_number <= 0;
    end
    else begin
        if (set_ip_addr_valid) begin
            local_ip_address[7:0] <= set_ip_addr_data[31:24];
            local_ip_address[15:8] <= set_ip_addr_data[23:16];
            local_ip_address[23:16] <= set_ip_addr_data[15:8];
            local_ip_address[31:24] <= set_ip_addr_data[7:0];
        end
        if (set_board_number_valid) begin
            board_number <= set_board_number_data;
        end
    end
end


// Register and distribute ip address
wire[31:0]  dhcp_ip_address;
wire        dhcp_ip_address_en;
reg[47:0]   mie_mac_address;
reg[47:0]   arp_mac_address;
reg[47:0]   ipv6_mac_address;
reg[31:0]   iph_ip_address;
reg[31:0]   arp_ip_address;
reg[31:0]   toe_ip_address;
reg[31:0]   ip_subnet_mask;
reg[31:0]   ip_default_gateway;
reg[127:0] link_local_ipv6_address;



always @(posedge user_clk)
begin
    if (net_aresetn == 0) begin
        mie_mac_address <= 48'h000000000000;
        arp_mac_address <= 48'h000000000000;
        ipv6_mac_address <= 48'h000000000000;
        iph_ip_address <= 32'h00000000;
        arp_ip_address <= 32'h00000000;
        toe_ip_address <= 32'h00000000;
        ip_subnet_mask <= 32'h00000000;
        ip_default_gateway <= 32'h00000000;
        link_local_ipv6_address <= 0;
    end
    else begin
        mie_mac_address <= { (MAC_ADDRESS[47:40]+board_number), MAC_ADDRESS[39:0]};
        arp_mac_address <= { (MAC_ADDRESS[47:40]+board_number), MAC_ADDRESS[39:0]};
        ipv6_mac_address <= { (MAC_ADDRESS[47:40]+board_number), MAC_ADDRESS[39:0]};
        //link_local_ipv6_address[127:80] <= ipv6_mac_address;
        //link_local_ipv6_address[15:0] <= 16'h80fe; // fe80
        //link_local_ipv6_address[79:16] <= 64'h0000_0000_0000_0000;
        link_local_ipv6_address <= {IPV6_ADDRESS[127:120]+board_number, IPV6_ADDRESS[119:0]};
        if (DHCP_EN == 1) begin
            if (dhcp_ip_address_en == 1'b1) begin
                iph_ip_address <= dhcp_ip_address;
                arp_ip_address <= dhcp_ip_address;
                toe_ip_address <= dhcp_ip_address;
            end
        end
        else begin
            iph_ip_address <= local_ip_address;
            arp_ip_address <= local_ip_address;
            toe_ip_address <= local_ip_address;
            ip_subnet_mask <= IP_SUBNET_MASK;
            ip_default_gateway <= {local_ip_address[31:28], 8'h01, local_ip_address[23:0]};
        end
    end
end
// ip address output
assign ip_address_used = iph_ip_address;

//////////mac module/////////////////////////

// wire                            network_init;
// wire                            user_rx_reset,user_tx_reset;  
// axi_stream #(.WIDTH(512))       axis_net_rx_data();
// axi_stream #(.WIDTH(512))       axis_net_tx_data();
// assign net_aresetn              = network_init;


// network_module_100g network_module_inst
// (
//     .dclk (dclk),
//     .user_clk(user_clk),
//     .net_clk(net_clk),
//     .sys_reset (~user_aresetn),
//     .aresetn(net_aresetn),
//     .network_init_done(network_init),
    
//     .gt_refclk_p(gt_refclk_p),
//     .gt_refclk_n(gt_refclk_n),
    
//     .gt_rxp_in(gt_rxp_in),
//     .gt_rxn_in(gt_rxn_in),
//     .gt_txp_out(gt_txp_out),
//     .gt_txn_out(gt_txn_out),
    
//     .user_rx_reset(user_rx_reset),
//     .user_tx_reset(user_tx_reset),
//     .rx_aligned(),
    
//     //master 0
//     .m_axis_net_rx(axis_net_rx_data),
//     .s_axis_net_tx(axis_net_tx_data)

// );

////////////////////////debug mac//////////////////

//    reg start0,start0_r;
//    reg [7:0] axis_net_tx_data0;
//    reg axis_net_tx_data_0_valid;

//     assign axis_net_rx_data.ready = 1;
    
//     assign axis_net_tx_data.keep = 64'hFFFFFFFFFFFFFFFF;
//     assign axis_net_tx_data.last = (axis_net_tx_data0[7:0] == 8'd27);  
//     assign axis_net_tx_data.data = {504'b0,axis_net_tx_data0};
//     assign axis_net_tx_data.valid = axis_net_tx_data_0_valid;  

// always @(posedge user_clk)begin
//     start0_r        <= start0;
// end

// always @(posedge user_clk)begin
//     if(axis_net_tx_data.ready & axis_net_tx_data.valid)
//         axis_net_tx_data0        <= axis_net_tx_data0 + 1'b1;
//     else
//         axis_net_tx_data0        <= axis_net_tx_data0;
// end

// always @(posedge user_clk)begin
//     if(~start0_r & start0)
//         axis_net_tx_data_0_valid        <= 1'b1;
//     else if(axis_net_tx_data0[7:0] == 8'd27)
//         axis_net_tx_data_0_valid        <= 1'b0;
//     else
//         axis_net_tx_data_0_valid        <= axis_net_tx_data_0_valid;
// end




// vio_0 vio_0 (
//   .clk(user_clk),                // input wire clk
//   .probe_out0(start0)  // output wire [0 : 0] probe_out0
// );








/*
 * TCP/IP
 */ 
logic       session_count_valid;
logic[15:0] session_count_data;


tx_data_split inst_tx_data_split(
  .clk(user_clk),
  .rstn(net_aresetn),

  .s_axis_tx_metadata(s_axis_tx_metadata),
  .s_axis_tx_data(s_axis_tx_data),
  .m_axis_tx_status(m_axis_tx_status),
 
  .m_axis_tx_metadata(axis_tcp_tx_meta),
  .m_axis_tx_data(axis_tcp_tx_data),
  .s_axis_tx_status(axis_tcp_tx_status)

  );

 
tcp_stack #(
     .TCP_EN(TCP_EN),
     .WIDTH(WIDTH),
     .RX_DDR_BYPASS_EN(RX_DDR_BYPASS_EN)
 ) tcp_stack_inst(
     .net_clk(user_clk), // input aclk
     .net_aresetn(net_aresetn), // input aresetn
     
     // streams to network
     .s_axis_rx_data(axis_toe_slice_to_toe),
     .m_axis_tx_data(axis_toe_to_toe_slice),
     
     // memory cmd streams
     .m_axis_mem_read_cmd(m_axis_read_cmd),
     .m_axis_mem_write_cmd(m_axis_write_cmd),
     // memory sts streams
     .s_axis_mem_read_sts(s_axis_read_sts),
     .s_axis_mem_write_sts(s_axis_write_sts),
     // memory data streams
     .s_axis_mem_read_data(s_axis_read_data),
     .m_axis_mem_write_data(m_axis_write_data),
     
     //Application
     .s_axis_listen_port(s_axis_listen_port),
     .m_axis_listen_port_status(m_axis_listen_port_status),
     
     .s_axis_open_connection(s_axis_open_connection),
     .m_axis_open_status(m_axis_open_status),
     .s_axis_close_connection(s_axis_close_connection),
     
     .m_axis_notifications(m_axis_notifications),
     .s_axis_read_package(s_axis_read_package),
     
     .m_axis_rx_metadata(m_axis_rx_metadata),
     .m_axis_rx_data(m_axis_rx_data),
     
     .s_axis_tx_metadata(axis_tcp_tx_meta),
     .s_axis_tx_data(axis_tcp_tx_data),
     .m_axis_tx_status(axis_tcp_tx_status),
     
     .local_ip_address(toe_ip_address),
     .session_count_valid(session_count_valid),
     .session_count_data(session_count_data)
);


mem_single_inf mem_single_inf_inst(
    .user_clk               (user_clk),
    .user_aresetn           (net_aresetn),
    .mem_clk                (mem_clk),
    .mem_aresetn            (net_aresetn),
    
    /* USER INTERFACE */    
    //memory access
    //read cmd
    .s_axis_mem_read_cmd    (m_axis_read_cmd),
    //read status
    .m_axis_mem_read_status (s_axis_read_sts),
    //read data stream
    .m_axis_mem_read_data   (s_axis_read_data),
    
    //write cmd
    .s_axis_mem_write_cmd   (m_axis_write_cmd),
    //write status
    .m_axis_mem_write_status(s_axis_write_sts),
    //write data stream
    .s_axis_mem_write_data  (m_axis_write_data),

    // .network_tx_mem         (network_tx_mem),

    .status_reg(status_reg)


    );


/*
 * Test Dropper
 */
 
//`define ENABLE_DROP

`ifdef ENABLE_DROP 
wire        roce_2_drop_valid;
wire        roce_2_drop_ready; 
wire[63:0]  roce_2_drop_data; 
wire[7:0]   roce_2_drop_keep; 
wire        roce_2_drop_last; 
 
test_dropper_ip test_dropper_inst (
   .dropFrequency_V(16'd10),        // input wire [15 : 0] dropFrequency_V
   .m_axis_data_TVALID(axi_udp_to_udp_slice_tvalid),  // output wire m_axis_data_TVALID
   .m_axis_data_TREADY(axi_udp_to_udp_slice_tready),  // input wire m_axis_data_TREADY
   .m_axis_data_TDATA(axi_udp_to_udp_slice_tdata),    // output wire [63 : 0] m_axis_data_TDATA
   .m_axis_data_TKEEP(axi_udp_to_udp_slice_tkeep),    // output wire [7 : 0] m_axis_data_TKEEP
   .m_axis_data_TLAST(axi_udp_to_udp_slice_tlast),    // output wire [0 : 0] m_axis_data_TLAST
   .s_axis_data_TVALID(roce_2_drop_valid),  // input wire s_axis_data_TVALID
   .s_axis_data_TREADY(roce_2_drop_ready),  // output wire s_axis_data_TREADY
   .s_axis_data_TDATA(roce_2_drop_data),    // input wire [63 : 0] s_axis_data_TDATA
   .s_axis_data_TKEEP(roce_2_drop_keep),    // input wire [7 : 0] s_axis_data_TKEEP
   .s_axis_data_TLAST(roce_2_drop_last),    // input wire [0 : 0] s_axis_data_TLAST
   .aclk(user_clk),                              // input wire aclk
   .aresetn(aresetn_reg)                        // input wire aresetn
 );
`endif

/*
 * RoCEv2
 */
//assign s_axis_rxread_sts_TREADY = 1'b1;
//assign s_axis_rxwrite_sts_TREADY = 1'b1;


`ifndef IP_VERSION4
//IPv4
assign axis_iph_to_udp_tready = 1'b1;
assign axis_udp_to_merge_tvalid = 1'b0;
assign axis_udp_to_merge_tdata = 0;
assign axis_udp_to_merge_tkeep = 0;
assign axis_udp_to_merge_tlast = 1'b0;
`else
// IPv6
assign axis_iph_to_rocev6_slice.ready = 1'b1;
assign axis_ipv6_to_intercon.valid = 1'b0;
assign axis_ipv6_to_intercon.data = 0;
assign axis_ipv6_to_intercon.keep = 0;
assign axis_ipv6_to_intercon.last = 1'b0;
`endif



//assign axi_iph_to_toe_slice_tready = 1'b1;

axis_register_slice_512 axis_register_AXI_S (
  .aclk(user_clk),                    // input wire aclk
  .aresetn(net_aresetn),              // input wire aresetn
  .s_axis_tvalid(axis_net_rx_data.valid),  // input wire s_axis_tvalid
  .s_axis_tready(axis_net_rx_data.ready),  // output wire s_axis_tready
  .s_axis_tdata(axis_net_rx_data.data),    // input wire [63 : 0] s_axis_tdata
  .s_axis_tkeep(axis_net_rx_data.keep),    // input wire [7 : 0] s_axis_tkeep
  .s_axis_tlast(axis_net_rx_data.last),    // input wire s_axis_tlast
  .m_axis_tvalid(axis_slice_to_ibh.valid),  // output wire m_axis_tvalid
  .m_axis_tready(axis_slice_to_ibh.ready),  // input wire m_axis_tready
  .m_axis_tdata(axis_slice_to_ibh.data),    // output wire [63 : 0] m_axis_tdata
  .m_axis_tkeep(axis_slice_to_ibh.keep),    // output wire [7 : 0] m_axis_tkeep
  .m_axis_tlast(axis_slice_to_ibh.last)    // output wire m_axis_tlast
);

 
ip_handler_ip ip_handler_inst (
.m_axis_arp_TVALID(axis_iph_to_arp_slice.valid), // output AXI4Stream_M_TVALID
.m_axis_arp_TREADY(axis_iph_to_arp_slice.ready), // input AXI4Stream_M_TREADY
.m_axis_arp_TDATA(axis_iph_to_arp_slice.data), // output [63 : 0] AXI4Stream_M_TDATA
.m_axis_arp_TKEEP(axis_iph_to_arp_slice.keep), // output [7 : 0] AXI4Stream_M_TSTRB
.m_axis_arp_TLAST(axis_iph_to_arp_slice.last), // output [0 : 0] AXI4Stream_M_TLAST

.m_axis_icmp_TVALID(axis_iph_to_icmp_slice.valid), // output AXI4Stream_M_TVALID
.m_axis_icmp_TREADY(axis_iph_to_icmp_slice.ready), // input AXI4Stream_M_TREADY
.m_axis_icmp_TDATA(axis_iph_to_icmp_slice.data), // output [63 : 0] AXI4Stream_M_TDATA
.m_axis_icmp_TKEEP(axis_iph_to_icmp_slice.keep), // output [7 : 0] AXI4Stream_M_TSTRB
.m_axis_icmp_TLAST(axis_iph_to_icmp_slice.last), // output [0 : 0] AXI4Stream_M_TLAST

.m_axis_icmpv6_TVALID(axis_iph_to_icmpv6_slice.valid),
.m_axis_icmpv6_TREADY(1),
.m_axis_icmpv6_TDATA(axis_iph_to_icmpv6_slice.data),
.m_axis_icmpv6_TKEEP(axis_iph_to_icmpv6_slice.keep),
.m_axis_icmpv6_TLAST(axis_iph_to_icmpv6_slice.last),

.m_axis_ipv6udp_TVALID(axis_iph_to_rocev6_slice.valid),
.m_axis_ipv6udp_TREADY(1),
.m_axis_ipv6udp_TDATA(axis_iph_to_rocev6_slice.data), 
.m_axis_ipv6udp_TKEEP(axis_iph_to_rocev6_slice.keep),
.m_axis_ipv6udp_TLAST(axis_iph_to_rocev6_slice.last),

.m_axis_udp_TVALID(axis_iph_to_udp_slice.valid),
.m_axis_udp_TREADY(1),
.m_axis_udp_TDATA(axis_iph_to_udp_slice.data),
.m_axis_udp_TKEEP(axis_iph_to_udp_slice.keep),
.m_axis_udp_TLAST(axis_iph_to_udp_slice.last),

.m_axis_tcp_TVALID(axis_iph_to_toe_slice.valid),
.m_axis_tcp_TREADY(axis_iph_to_toe_slice.ready),
.m_axis_tcp_TDATA(axis_iph_to_toe_slice.data),
.m_axis_tcp_TKEEP(axis_iph_to_toe_slice.keep),
.m_axis_tcp_TLAST(axis_iph_to_toe_slice.last),

.m_axis_roce_TVALID(axis_iph_to_roce_slice.valid),
.m_axis_roce_TREADY(1),
.m_axis_roce_TDATA(axis_iph_to_roce_slice.data),
.m_axis_roce_TKEEP(axis_iph_to_roce_slice.keep),
.m_axis_roce_TLAST(axis_iph_to_roce_slice.last),

.s_axis_raw_TVALID(axis_slice_to_ibh.valid),
.s_axis_raw_TREADY(axis_slice_to_ibh.ready),
.s_axis_raw_TDATA(axis_slice_to_ibh.data),
.s_axis_raw_TKEEP(axis_slice_to_ibh.keep),
.s_axis_raw_TLAST(axis_slice_to_ibh.last),

.myIpAddress_V(iph_ip_address),

.ap_clk(user_clk), // input aclk
.ap_rst_n(net_aresetn) // input aresetn
);

// ARP lookup
wire        axis_arp_lookup_request_TVALID;
wire        axis_arp_lookup_request_TREADY;
wire[31:0]  axis_arp_lookup_request_TDATA;
wire        axis_arp_lookup_reply_TVALID;
wire        axis_arp_lookup_reply_TREADY;
wire[55:0]  axis_arp_lookup_reply_TDATA;

mac_ip_encode_ip mac_ip_encode_inst (
.m_axis_ip_TVALID(axis_mie_to_intercon.valid),
.m_axis_ip_TREADY(axis_mie_to_intercon.ready),
.m_axis_ip_TDATA(axis_mie_to_intercon.data),
.m_axis_ip_TKEEP(axis_mie_to_intercon.keep),
.m_axis_ip_TLAST(axis_mie_to_intercon.last),
.m_axis_arp_lookup_request_V_V_TVALID(axis_arp_lookup_request_TVALID),
.m_axis_arp_lookup_request_V_V_TREADY(axis_arp_lookup_request_TREADY),
.m_axis_arp_lookup_request_V_V_TDATA(axis_arp_lookup_request_TDATA),
.s_axis_ip_TVALID(axis_intercon_to_mie.valid),
.s_axis_ip_TREADY(axis_intercon_to_mie.ready),
.s_axis_ip_TDATA(axis_intercon_to_mie.data),
.s_axis_ip_TKEEP(axis_intercon_to_mie.keep),
.s_axis_ip_TLAST(axis_intercon_to_mie.last),
.s_axis_arp_lookup_reply_V_TVALID(axis_arp_lookup_reply_TVALID),
.s_axis_arp_lookup_reply_V_TREADY(axis_arp_lookup_reply_TREADY),
.s_axis_arp_lookup_reply_V_TDATA(axis_arp_lookup_reply_TDATA),

.myMacAddress_V(mie_mac_address),                                    // input wire [47 : 0] regMacAddress_V
.regSubNetMask_V(ip_subnet_mask),                                    // input wire [31 : 0] regSubNetMask_V
.regDefaultGateway_V(ip_default_gateway),                            // input wire [31 : 0] regDefaultGateway_V
  
.ap_clk(user_clk), // input aclk
.ap_rst_n(net_aresetn) // input aresetn
);



assign axis_ethencode_to_intercon.valid = 1'b0;
assign axis_ethencode_to_intercon.data = 0;
assign axis_ethencode_to_intercon.keep = 0;
assign axis_ethencode_to_intercon.last = 1'b0;




// merges icmp and tcp
axi_stream #(.WIDTH(512))    axis_icmp_slice_to_merge();
axis_64_to_512_converter icmp_out_data_converter (
  .aclk(user_clk),
  .aresetn(net_aresetn),
  .s_axis_tvalid(axis_icmp_to_icmp_slice.valid),
  .s_axis_tready(axis_icmp_to_icmp_slice.ready),
  .s_axis_tdata(axis_icmp_to_icmp_slice.data),
  .s_axis_tkeep(axis_icmp_to_icmp_slice.keep),
  .s_axis_tlast(axis_icmp_to_icmp_slice.last),
  .m_axis_tvalid(axis_icmp_slice_to_merge.valid),
  .m_axis_tready(axis_icmp_slice_to_merge.ready),
  .m_axis_tdata(axis_icmp_slice_to_merge.data),
  .m_axis_tkeep(axis_icmp_slice_to_merge.keep),
  .m_axis_tlast(axis_icmp_slice_to_merge.last)
);
axis_interconnect_512_4to1 ip_merger (
  .ACLK(user_clk),                                  // input wire ACLK
  .ARESETN(net_aresetn),                            // input wire ARESETN
  .S00_AXIS_ACLK(user_clk),                // input wire S00_AXIS_ACLK
  .S01_AXIS_ACLK(user_clk),                // input wire S01_AXIS_ACLK
  .S02_AXIS_ACLK(user_clk),                // input wire S02_AXIS_ACLK
  .S03_AXIS_ACLK(user_clk),                // input wire S03_AXIS_ACLK
  .S00_AXIS_ARESETN(net_aresetn),          // input wire S00_AXIS_ARESETN
  .S01_AXIS_ARESETN(net_aresetn),          // input wire S01_AXIS_ARESETN
  .S02_AXIS_ARESETN(net_aresetn),          // input wire S02_AXIS_ARESETN
  .S03_AXIS_ARESETN(net_aresetn),          // input wire S03_AXIS_ARESETN
  
  .S00_AXIS_TVALID(axis_icmp_slice_to_merge.valid),            // input wire S00_AXIS_TVALID
  .S00_AXIS_TREADY(axis_icmp_slice_to_merge.ready),            // output wire S00_AXIS_TREADY
  .S00_AXIS_TDATA(axis_icmp_slice_to_merge.data),              // input wire [63 : 0] S00_AXIS_TDATA
  .S00_AXIS_TKEEP(axis_icmp_slice_to_merge.keep),              // input wire [7 : 0] S00_AXIS_TKEEP
  .S00_AXIS_TLAST(axis_icmp_slice_to_merge.last),              // input wire S00_AXIS_TLAST

  .S01_AXIS_TVALID(0),            // input wire S01_AXIS_TVALID
  .S01_AXIS_TREADY(axis_udp_slice_to_merge.ready),            // output wire S01_AXIS_TREADY
  .S01_AXIS_TDATA(0),              // input wire [63 : 0] S01_AXIS_TDATA
  .S01_AXIS_TKEEP(0),              // input wire [7 : 0] S01_AXIS_TKEEP
  .S01_AXIS_TLAST(0),              // input wire S01_AXIS_TLAST

  .S02_AXIS_TVALID(axis_toe_to_toe_slice.valid),            // input wire S02_AXIS_TVALID
  .S02_AXIS_TREADY(axis_toe_to_toe_slice.ready),            // output wire S02_AXIS_TREADY
  .S02_AXIS_TDATA(axis_toe_to_toe_slice.data),              // input wire [63 : 0] S02_AXIS_TDATA
  .S02_AXIS_TKEEP(axis_toe_to_toe_slice.keep),              // input wire [7 : 0] S02_AXIS_TKEEP
  .S02_AXIS_TLAST(axis_toe_to_toe_slice.last),              // input wire S02_AXIS_TLAST

  .S03_AXIS_TVALID(0),            // input wire S01_AXIS_TVALID
  .S03_AXIS_TREADY(axis_roce_slice_to_merge.ready),            // output wire S01_AXIS_TREADY
  .S03_AXIS_TDATA(0),              // input wire [63 : 0] S01_AXIS_TDATA
  .S03_AXIS_TKEEP(0),              // input wire [7 : 0] S01_AXIS_TKEEP
  .S03_AXIS_TLAST(0),              // input wire S01_AXIS_TLAST

  .M00_AXIS_ACLK(user_clk),                // input wire M00_AXIS_ACLK
  .M00_AXIS_ARESETN(net_aresetn),          // input wire M00_AXIS_ARESETN
  .M00_AXIS_TVALID(axis_intercon_to_mie.valid),            // output wire M00_AXIS_TVALID
  .M00_AXIS_TREADY(axis_intercon_to_mie.ready),            // input wire M00_AXIS_TREADY
  .M00_AXIS_TDATA(axis_intercon_to_mie.data),              // output wire [63 : 0] M00_AXIS_TDATA
  .M00_AXIS_TKEEP(axis_intercon_to_mie.keep),              // output wire [7 : 0] M00_AXIS_TKEEP
  .M00_AXIS_TLAST(axis_intercon_to_mie.last),              // output wire M00_AXIS_TLAST
  .S00_ARB_REQ_SUPPRESS(1'b0),  // input wire S00_ARB_REQ_SUPPRESS
  .S01_ARB_REQ_SUPPRESS(1'b0),  // input wire S01_ARB_REQ_SUPPRESS
  .S02_ARB_REQ_SUPPRESS(1'b0),  // input wire S02_ARB_REQ_SUPPRESS
  .S03_ARB_REQ_SUPPRESS(1'b0)  // input wire S02_ARB_REQ_SUPPRESS
);

// merges ip and arp
axis_interconnect_512_2to1 mac_merger (
  .ACLK(user_clk), // input ACLK
  .ARESETN(net_aresetn), // input ARESETN
  .S00_AXIS_ACLK(user_clk), // input S00_AXIS_ACLK
  .S01_AXIS_ACLK(user_clk), // input S01_AXIS_ACLK
  //.S02_AXIS_ACLK(user_clk), // input S01_AXIS_ACLK
  .S00_AXIS_ARESETN(net_aresetn), // input S00_AXIS_ARESETN
  .S01_AXIS_ARESETN(net_aresetn), // input S01_AXIS_ARESETN
  //.S02_AXIS_ARESETN(net_aresetn), // input S01_AXIS_ARESETN
  .S00_AXIS_TVALID(axis_arp_to_arp_slice.valid), // input S00_AXIS_TVALID
  .S00_AXIS_TREADY(axis_arp_to_arp_slice.ready), // output S00_AXIS_TREADY
  .S00_AXIS_TDATA(axis_arp_to_arp_slice.data), // input [63 : 0] S00_AXIS_TDATA
  .S00_AXIS_TKEEP(axis_arp_to_arp_slice.keep), // input [7 : 0] S00_AXIS_TKEEP
  .S00_AXIS_TLAST(axis_arp_to_arp_slice.last), // input S00_AXIS_TLAST
  
  .S01_AXIS_TVALID(axis_mie_to_intercon.valid), // input S01_AXIS_TVALID
  .S01_AXIS_TREADY(axis_mie_to_intercon.ready), // output S01_AXIS_TREADY
  .S01_AXIS_TDATA(axis_mie_to_intercon.data), // input [63 : 0] S01_AXIS_TDATA
  .S01_AXIS_TKEEP(axis_mie_to_intercon.keep), // input [7 : 0] S01_AXIS_TKEEP
  .S01_AXIS_TLAST(axis_mie_to_intercon.last), // input S01_AXIS_TLAST
  
  /*.S02_AXIS_TVALID(axis_ethencode_to_intercon.valid), // input S01_AXIS_TVALID
  .S02_AXIS_TREADY(axis_ethencode_to_intercon.ready), // output S01_AXIS_TREADY
  .S02_AXIS_TDATA(axis_ethencode_to_intercon.data), // input [63 : 0] S01_AXIS_TDATA
  .S02_AXIS_TKEEP(axis_ethencode_to_intercon.keep), // input [7 : 0] S01_AXIS_TKEEP
  .S02_AXIS_TLAST(axis_ethencode_to_intercon.last), // input S01_AXIS_TLAST*/
  
  .M00_AXIS_ACLK(user_clk), // input M00_AXIS_ACLK
  .M00_AXIS_ARESETN(net_aresetn), // input M00_AXIS_ARESETN
  .M00_AXIS_TVALID(axis_net_tx.valid), // output M00_AXIS_TVALID
  .M00_AXIS_TREADY(axis_net_tx.ready), // input M00_AXIS_TREADY
  .M00_AXIS_TDATA(axis_net_tx.data), // output [63 : 0] M00_AXIS_TDATA
  .M00_AXIS_TKEEP(axis_net_tx.keep), // output [7 : 0] M00_AXIS_TKEEP
  .M00_AXIS_TLAST(axis_net_tx.last), // output M00_AXIS_TLAST
  .S00_ARB_REQ_SUPPRESS(1'b0), // input S00_ARB_REQ_SUPPRESS
  .S01_ARB_REQ_SUPPRESS(1'b0) // input S01_ARB_REQ_SUPPRESS
  //.S02_ARB_REQ_SUPPRESS(1'b0) // input S01_ARB_REQ_SUPPRESS
);

axis_register_slice_512 axis_register_slice_512_tx (
  .aclk(user_clk),                    // input wire aclk
  .aresetn(1'b1),              // input wire aresetn
  .s_axis_tvalid(axis_net_tx.valid),  // input wire s_axis_tvalid
  .s_axis_tready(axis_net_tx.ready),  // output wire s_axis_tready
  .s_axis_tdata(axis_net_tx.data),    // input wire [511 : 0] s_axis_tdata
  .s_axis_tkeep(axis_net_tx.keep),    // input wire [63 : 0] s_axis_tkeep
  .s_axis_tlast(axis_net_tx.last),    // input wire s_axis_tlast
  .m_axis_tvalid(axis_net_tx_data.valid),  // output wire m_axis_tvalid
  .m_axis_tready(axis_net_tx_data.ready),  // input wire m_axis_tready
  .m_axis_tdata(axis_net_tx_data.data),    // output wire [511 : 0] m_axis_tdata
  .m_axis_tkeep(axis_net_tx_data.keep),    // output wire [63 : 0] m_axis_tkeep
  .m_axis_tlast(axis_net_tx_data.last)    // output wire m_axis_tlast
);


logic[15:0] arp_request_pkg_counter;
logic[15:0] arp_reply_pkg_counter;

arp_server_subnet_ip arp_server_inst(
.m_axis_TVALID(axis_arp_to_arp_slice.valid),
.m_axis_TREADY(axis_arp_to_arp_slice.ready),
.m_axis_TDATA(axis_arp_to_arp_slice.data),
.m_axis_TKEEP(axis_arp_to_arp_slice.keep),
.m_axis_TLAST(axis_arp_to_arp_slice.last),
.m_axis_arp_lookup_reply_V_TVALID(axis_arp_lookup_reply_TVALID),
.m_axis_arp_lookup_reply_V_TREADY(axis_arp_lookup_reply_TREADY),
.m_axis_arp_lookup_reply_V_TDATA(axis_arp_lookup_reply_TDATA),
.m_axis_host_arp_lookup_reply_V_TVALID(axis_host_arp_lookup_reply_TVALID),
.m_axis_host_arp_lookup_reply_V_TREADY(0),
.m_axis_host_arp_lookup_reply_V_TDATA(axis_host_arp_lookup_reply_TDATA),
.s_axis_TVALID(axis_arp_slice_to_arp.valid),
.s_axis_TREADY(axis_arp_slice_to_arp.ready),
.s_axis_TDATA(axis_arp_slice_to_arp.data),
.s_axis_TKEEP(axis_arp_slice_to_arp.keep),
.s_axis_TLAST(axis_arp_slice_to_arp.last),
.s_axis_arp_lookup_request_V_V_TVALID(axis_arp_lookup_request_TVALID),
.s_axis_arp_lookup_request_V_V_TREADY(axis_arp_lookup_request_TREADY),
.s_axis_arp_lookup_request_V_V_TDATA(axis_arp_lookup_request_TDATA),
.s_axis_host_arp_lookup_request_V_V_TVALID(0),
.s_axis_host_arp_lookup_request_V_V_TREADY(axis_host_arp_lookup_request_TREADY),
.s_axis_host_arp_lookup_request_V_V_TDATA(0),

.myMacAddress_V(arp_mac_address),
.myIpAddress_V(arp_ip_address),
.regRequestCount_V(arp_request_pkg_counter),
.regRequestCount_V_ap_vld(),
.regReplyCount_V(arp_reply_pkg_counter),
.regReplyCount_V_ap_vld(),

.ap_clk(user_clk), // input aclk
.ap_rst_n(net_aresetn) // input aresetn
);

// /*assign axis_ttl_to_icmp_tvalid = 0;
// assign axis_ttl_to_icmp_tdata = 0;
// assign axis_ttl_to_icmp_tkeep = 0;
// assign axis_ttl_to_icmp_tlast = 0;*/

 icmp_server_ip icmp_server_inst (
   .s_axis_TVALID(axis_icmp_slice_to_icmp.valid),    // input wire dataIn_TVALID
   .s_axis_TREADY(axis_icmp_slice_to_icmp.ready),    // output wire dataIn_TREADY
   .s_axis_TDATA(axis_icmp_slice_to_icmp.data),      // input wire [63 : 0] dataIn_TDATA
   .s_axis_TKEEP(axis_icmp_slice_to_icmp.keep),      // input wire [7 : 0] dataIn_TKEEP
   .s_axis_TLAST(axis_icmp_slice_to_icmp.last),      // input wire [0 : 0] dataIn_TLAST
   .udpIn_TVALID(1'b0),//(axis_udp_to_icmp_tvalid),           // input wire udpIn_TVALID
   .udpIn_TREADY(),           // output wire udpIn_TREADY
   .udpIn_TDATA(0),//(axis_udp_to_icmp_tdata),             // input wire [63 : 0] udpIn_TDATA
   .udpIn_TKEEP(0),//(axis_udp_to_icmp_tkeep),             // input wire [7 : 0] udpIn_TKEEP
   .udpIn_TLAST(0),//(axis_udp_to_icmp_tlast),             // input wire [0 : 0] udpIn_TLAST
   .ttlIn_TVALID(1'b0),//(axis_ttl_to_icmp_tvalid),           // input wire ttlIn_TVALID
   .ttlIn_TREADY(),           // output wire ttlIn_TREADY
   .ttlIn_TDATA(0),//(axis_ttl_to_icmp_tdata),             // input wire [63 : 0] ttlIn_TDATA
   .ttlIn_TKEEP(0),//(axis_ttl_to_icmp_tkeep),             // input wire [7 : 0] ttlIn_TKEEP
   .ttlIn_TLAST(0),//(axis_ttl_to_icmp_tlast),             // input wire [0 : 0] ttlIn_TLAST
   .m_axis_TVALID(axis_icmp_to_icmp_slice.valid),   // output wire dataOut_TVALID
   .m_axis_TREADY(axis_icmp_to_icmp_slice.ready),   // input wire dataOut_TREADY
   .m_axis_TDATA(axis_icmp_to_icmp_slice.data),     // output wire [63 : 0] dataOut_TDATA
   .m_axis_TKEEP(axis_icmp_to_icmp_slice.keep),     // output wire [7 : 0] dataOut_TKEEP
   .m_axis_TLAST(axis_icmp_to_icmp_slice.last),     // output wire [0 : 0] dataOut_TLAST
   .ap_clk(user_clk),                                    // input wire ap_clk
   .ap_rst_n(net_aresetn)                                // input wire ap_rst_n
 );



assign axis_iph_to_icmpv6_slice.ready = 1'b1;


 /*
  * Slices
  */
  // ARP Input Slice
 register_slice_wrapper #(.WIDTH(WIDTH)) axis_register_arp_in_slice(
  .aclk(user_clk),
  .aresetn(net_aresetn),
  .s_axis(axis_iph_to_arp_slice),
  .m_axis(axis_arp_slice_to_arp)
 );

 // TOE Input Slice
register_slice_wrapper #(.WIDTH(WIDTH)) axis_register_toe_in_slice(
.aclk(user_clk),
.aresetn(net_aresetn),
.s_axis(axis_iph_to_toe_slice),
.m_axis(axis_toe_slice_to_toe)
);

//axis_register_slice_512 axis_register_icmp_in_slice(
axis_512_to_64_converter icmp_in_data_converter (
  .aclk(user_clk),
  .aresetn(net_aresetn),
  .s_axis_tvalid(axis_iph_to_icmp_slice.valid),
  .s_axis_tready(axis_iph_to_icmp_slice.ready),
  .s_axis_tdata(axis_iph_to_icmp_slice.data),
  .s_axis_tkeep(axis_iph_to_icmp_slice.keep),
  .s_axis_tlast(axis_iph_to_icmp_slice.last),
  .m_axis_tvalid(axis_icmp_slice_to_icmp.valid),
  .m_axis_tready(axis_icmp_slice_to_icmp.ready),
  .m_axis_tdata(axis_icmp_slice_to_icmp.data),
  .m_axis_tkeep(axis_icmp_slice_to_icmp.keep),
  .m_axis_tlast(axis_icmp_slice_to_icmp.last)
); 



/*
 * Network Controller
 */

 
 wire        axis_host_arp_lookup_request_TVALID;
 wire        axis_host_arp_lookup_request_TREADY;
 wire[31:0]  axis_host_arp_lookup_request_TDATA;
 wire        axis_host_arp_lookup_reply_TVALID;
 wire        axis_host_arp_lookup_reply_TREADY;
 wire[55:0]  axis_host_arp_lookup_reply_TDATA;
 
wire[31:0]    regCrcDropPkgCount;
wire          regCrcDropPkgCount_valid;
 
wire[31:0]    regInvalidPsnDropCount;
wire          regInvalidPsnDropCount_valid;

// // tx metadata

 



/*
 * Statistics
 */
 /*
logic[31:0] rx_word_counter; 
logic[31:0] rx_pkg_counter; 
logic[31:0] tx_word_counter; 
logic[31:0] tx_pkg_counter;

logic[31:0] tcp_rx_pkg_counter;
logic[31:0] tcp_tx_pkg_counter;
logic[31:0] udp_rx_pkg_counter;
logic[31:0] udp_tx_pkg_counter;
logic[31:0] roce_rx_pkg_counter;
logic[31:0] roce_tx_pkg_counter;

logic[31:0] roce_data_rx_word_counter;
logic[31:0] roce_data_rx_pkg_counter;
logic[31:0] roce_data_tx_role_word_counter;
logic[31:0] roce_data_tx_role_pkg_counter;
logic[31:0] roce_data_tx_host_word_counter;
logic[31:0] roce_data_tx_host_pkg_counter;

logic[31:0] arp_rx_pkg_counter;
logic[31:0] arp_tx_pkg_counter;
logic[31:0] icmp_rx_pkg_counter;
logic[31:0] icmp_tx_pkg_counter;

reg[7:0]  axis_stream_down_counter;
reg axis_stream_down;
reg[7:0]  output_stream_down_counter;
reg output_stream_down;

always @(posedge user_clk) begin
    if (~net_aresetn) begin
        rx_word_counter <= '0;
        rx_pkg_counter <= '0;
        tx_word_counter <= '0;
        tx_pkg_counter <= '0;

        tcp_rx_pkg_counter <= '0;
        tcp_tx_pkg_counter <= '0;

        // roce_data_rx_word_counter <= '0;
        // roce_data_rx_pkg_counter <= '0;
        // roce_data_tx_role_word_counter <= '0;
        // roce_data_tx_role_pkg_counter <= '0;
        // roce_data_tx_host_word_counter <= '0;
        // roce_data_tx_host_pkg_counter <= '0;
        
        // arp_rx_pkg_counter <= '0;
        // arp_tx_pkg_counter <= '0;
        
        // udp_rx_pkg_counter <= '0;
        // udp_tx_pkg_counter <= '0;

        // roce_rx_pkg_counter <= '0;
        // roce_tx_pkg_counter <= '0;

        axis_stream_down_counter <= '0;
        axis_stream_down <= 1'b0;
    end
    else begin
        if (axis_net_rx_data.ready) begin
            axis_stream_down_counter <= '0;
        end
        if (axis_net_rx_data.valid && ~axis_net_rx_data.ready) begin
            axis_stream_down_counter <= axis_stream_down_counter + 1;
        end
        if (axis_stream_down_counter > 2) begin
            axis_stream_down <= 1'b1;
        end
        if (axis_net_rx_data.valid && axis_net_rx_data.ready) begin
            rx_word_counter <= rx_word_counter + 1;
            if (axis_net_rx_data.last) begin
                rx_pkg_counter <= rx_pkg_counter + 1;
            end
        end
        if (axis_net_tx_data.valid && axis_net_tx_data.ready) begin
            tx_word_counter <= tx_word_counter + 1;
            if (axis_net_tx_data.last) begin
                tx_pkg_counter <= tx_pkg_counter + 1;
            end
        end
        // //arp
        // if (axis_arp_slice_to_arp.valid && axis_arp_slice_to_arp.ready) begin
        //     if (axis_arp_slice_to_arp.last) begin
        //         arp_rx_pkg_counter <= arp_rx_pkg_counter + 1;
        //     end
        // end
        // if (axis_arp_to_arp_slice.valid && axis_arp_to_arp_slice.ready) begin
        //     if (axis_arp_to_arp_slice.last) begin
        //         arp_tx_pkg_counter <= arp_tx_pkg_counter + 1;
        //     end
        // end
        // //icmp
        // if (axis_icmp_slice_to_icmp.valid && axis_icmp_slice_to_icmp.ready) begin
        //     if (axis_icmp_slice_to_icmp.last) begin
        //         icmp_rx_pkg_counter <= icmp_rx_pkg_counter + 1;
        //     end
        // end
        // if (axis_icmp_to_icmp_slice.valid && axis_icmp_to_icmp_slice.ready) begin
        //     if (axis_icmp_to_icmp_slice.last) begin
        //         icmp_tx_pkg_counter <= icmp_tx_pkg_counter + 1;
        //     end
        // end
        //tcp
        if (axis_toe_slice_to_toe.valid && axis_toe_slice_to_toe.ready) begin
            if (axis_toe_slice_to_toe.last) begin
                tcp_rx_pkg_counter <= tcp_rx_pkg_counter + 1;
            end
        end
        if (axis_toe_to_toe_slice.valid && axis_toe_to_toe_slice.ready) begin
            if (axis_toe_to_toe_slice.last) begin
                tcp_tx_pkg_counter <= tcp_tx_pkg_counter + 1;
            end
        end
        //udp
        // if (axis_udp_slice_to_udp.valid && axis_udp_slice_to_udp.ready) begin
        //     if (axis_udp_slice_to_udp.last) begin
        //         udp_rx_pkg_counter <= udp_rx_pkg_counter + 1;
        //     end
        // end
        // if (axis_udp_to_udp_slice.valid && axis_udp_to_udp_slice.ready) begin
        //     if (axis_udp_to_udp_slice.last) begin
        //         udp_tx_pkg_counter <= udp_tx_pkg_counter + 1;
        //     end
        // end
        // //roce
        // if (axis_roce_slice_to_roce.valid && axis_roce_slice_to_roce.ready) begin
        //     if (axis_roce_slice_to_roce.last) begin
        //         roce_rx_pkg_counter <= roce_rx_pkg_counter + 1;
        //     end
        // end
        // if (axis_roce_to_roce_slice.valid && axis_roce_to_roce_slice.ready) begin
        //     if (axis_roce_to_roce_slice.last) begin
        //         roce_tx_pkg_counter <= roce_tx_pkg_counter + 1;
        //     end
        // end
        // //roce data
        // if (m_axis_roce_write_data.valid && m_axis_roce_write_data.ready) begin
        //     roce_data_rx_word_counter <= roce_data_rx_word_counter + 1;
        //     if (m_axis_roce_write_data.last) begin
        //         roce_data_rx_pkg_counter <= roce_data_rx_pkg_counter + 1;
        //     end
        // end
        // if (s_axis_roce_read_data.valid && s_axis_roce_read_data.ready) begin
        //     roce_data_tx_host_word_counter <= roce_data_tx_host_word_counter + 1;
        //     if (s_axis_roce_read_data.last) begin
        //         roce_data_tx_host_pkg_counter <= roce_data_tx_host_pkg_counter + 1;
        //     end
        // end
        // if (s_axis_roce_role_tx_data.valid && s_axis_roce_role_tx_data.ready) begin
        //     roce_data_tx_role_word_counter <= roce_data_tx_role_word_counter + 1;
        //     if (s_axis_roce_role_tx_data.last) begin
        //         roce_data_tx_role_pkg_counter <= roce_data_tx_role_pkg_counter + 1;
        //     end
        // end
    end
end
*/
endmodule

`default_nettype wire
