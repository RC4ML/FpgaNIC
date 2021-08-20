`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 07/26/2020 07:43:48 PM
// Design Name: 
// Module Name: cmac_axis_wrapper
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
`timescale 1ns/1ps
`default_nettype none

//`define DEBUG

module cmac_axis_wrapper
(
    input wire                 init_clk,
    input wire                 gt_ref_clk_p,
    input wire                 gt_ref_clk_n,
    input wire [3:0]           gt_rxp_in,
    input wire [3:0]           gt_rxn_in,
    output logic [3:0]          gt_txn_out,
    output logic [3:0]          gt_txp_out,
    input wire                 sys_reset,

    axi_stream.master           m_rx_axis,
    axi_stream.slave            s_tx_axis,
    
    output logic                rx_aligned,
	output logic                usr_tx_clk,
	output logic                tx_rst,
	output logic                rx_rst,
	output logic [3:0]          gt_rxrecclkout       
);


logic           gt_txusrclk2;
logic[11 :0]    gt_loopback_in;

wire         usr_rx_reset_w;
wire         core_tx_reset_w;

wire         gt_rxusrclk2;

reg usr_rx_reset_r;
reg core_tx_reset_r;

always @( posedge gt_rxusrclk2 ) begin //TODO check if this is correct
    usr_rx_reset_r  <= usr_rx_reset_w;
end

always @( posedge gt_txusrclk2 ) begin
    core_tx_reset_r <= core_tx_reset_w;
end

assign rx_aligned = stat_rx_aligned;
assign usr_tx_clk = gt_txusrclk2;

assign rx_rst = usr_rx_reset_r;
assign tx_rst = core_tx_reset_r;


wire [4:0] stat_rx_pcsl_number_0 ;
wire [4:0] stat_rx_pcsl_number_1 ;
wire [4:0] stat_rx_pcsl_number_2 ;
wire [4:0] stat_rx_pcsl_number_3 ;
wire [4:0] stat_rx_pcsl_number_4 ;
wire [4:0] stat_rx_pcsl_number_5 ;
wire [4:0] stat_rx_pcsl_number_6 ;
wire [4:0] stat_rx_pcsl_number_7 ;
wire [4:0] stat_rx_pcsl_number_8 ;
wire [4:0] stat_rx_pcsl_number_9 ;
wire [4:0] stat_rx_pcsl_number_10;
wire [4:0] stat_rx_pcsl_number_11;
wire [4:0] stat_rx_pcsl_number_12;
wire [4:0] stat_rx_pcsl_number_13;
wire [4:0] stat_rx_pcsl_number_14;
wire [4:0] stat_rx_pcsl_number_15;
wire [4:0] stat_rx_pcsl_number_16;
wire [4:0] stat_rx_pcsl_number_17;
wire [4:0] stat_rx_pcsl_number_18;
wire [4:0] stat_rx_pcsl_number_19;


    
    
//RX FSM states
localparam STATE_RX_IDLE             = 0;
localparam STATE_GT_LOCKED           = 1;
localparam STATE_WAIT_RX_ALIGNED     = 2;
localparam STATE_PKT_TRANSFER_INIT   = 3;
localparam STATE_WAIT_FOR_RESTART    = 6;

reg            ctl_rx_enable_r, ctl_rx_force_resync_r; 

////State Registers for RX
reg  [3:0]     rx_prestate;
reg rx_reset_done;




//rx reset handling

reg stat_rx_aligned_1d;
always @(posedge gt_txusrclk2) begin
    if (usr_rx_reset_w) begin
        rx_prestate            <= STATE_RX_IDLE;
        ctl_rx_enable_r        <= 1'b0;
        ctl_rx_force_resync_r  <= 1'b0;
        stat_rx_aligned_1d <= 1'b0;
        rx_reset_done <= 1'b0;
    end
    else begin
        rx_reset_done <= 1'b1;
        stat_rx_aligned_1d <= stat_rx_aligned;
        case (rx_prestate)
            STATE_RX_IDLE: begin
                ctl_rx_enable_r        <= 1'b0;
                ctl_rx_force_resync_r  <= 1'b0;
                if  (rx_reset_done == 1'b1) begin
                    rx_prestate <= STATE_GT_LOCKED;
                end
            end
            STATE_GT_LOCKED: begin
                 ctl_rx_enable_r        <= 1'b1;
                 ctl_rx_force_resync_r  <= 1'b0;
                 rx_prestate <= STATE_WAIT_RX_ALIGNED;
            end
            STATE_WAIT_RX_ALIGNED: begin
                if  (stat_rx_aligned_1d == 1'b1) begin
                    rx_prestate <= STATE_PKT_TRANSFER_INIT;
                end
                else begin
                    rx_prestate <= STATE_WAIT_RX_ALIGNED;
                end
            end
            STATE_PKT_TRANSFER_INIT: begin
                if (stat_rx_aligned_1d == 1'b0) begin
                    rx_prestate <= STATE_RX_IDLE;
                end
            end
        endcase
    end
end
wire ctl_rx_enable;
wire ctl_rx_force_resync;
assign ctl_rx_enable            = ctl_rx_enable_r;
assign ctl_rx_force_resync      = ctl_rx_force_resync_r;




// TX FSM States
localparam STATE_TX_IDLE             = 0;
//localparam STATE_GT_LOCKED           = 1;
//localparam STATE_WAIT_RX_ALIGNED     = 2;
//localparam STATE_PKT_TRANSFER_INIT   = 3;
//localparam STATE_LBUS_TX_ENABLE      = 4;
//localparam STATE_LBUS_TX_HALT        = 5;
//localparam STATE_LBUS_TX_DONE        = 6;
//localparam STATE_WAIT_FOR_RESTART    = 7;
reg  [3:0]     tx_prestate;
reg tx_reset_done;
reg            ctl_tx_enable_r, ctl_tx_send_idle_r, ctl_tx_send_rfi_r, ctl_tx_test_pattern_r;
reg            ctl_tx_send_lfi_r;
reg            tx_rdyout_d, tx_ovfout_d, tx_unfout_d;

wire tx_ovf;//TODO use for debug
wire tx_unf;//TODO use for debug


always @(posedge gt_txusrclk2) begin
    if (core_tx_reset_w) begin
        tx_prestate                       <= STATE_TX_IDLE;
        ctl_tx_enable_r                   <= 1'b0;
        ctl_tx_send_idle_r                <= 1'b0;
        ctl_tx_send_lfi_r                 <= 1'b0;
        ctl_tx_send_rfi_r                 <= 1'b0;
        ctl_tx_test_pattern_r             <= 1'b0;
        tx_reset_done <= 1'b0;
    end
    else begin
        tx_reset_done <= 1'b1;
        tx_ovfout_d                     <= tx_ovf;
        tx_unfout_d                     <= tx_unf;        
        //stat_rx_aligned_1d <= cmac_stat.stat_rx_aligned;
        case (tx_prestate)
            STATE_TX_IDLE: begin
                ctl_tx_enable_r        <= 1'b0;
                ctl_tx_send_idle_r     <= 1'b0;
                ctl_tx_send_lfi_r      <= 1'b0;
                ctl_tx_send_rfi_r      <= 1'b0;
                ctl_tx_test_pattern_r  <= 1'b0;
                if  (tx_reset_done == 1'b1) begin
                    tx_prestate <= STATE_GT_LOCKED;
                end
                /*else begin
                    rx_prestate <= STATE_RX_IDLE;
                end*/
            end
            STATE_GT_LOCKED: begin
                ctl_tx_enable_r        <= 1'b0;
                ctl_tx_send_idle_r     <= 1'b0;
                ctl_tx_send_lfi_r      <= 1'b0;
                ctl_tx_send_rfi_r      <= 1'b1;
                tx_prestate <= STATE_WAIT_RX_ALIGNED;
            end
            STATE_WAIT_RX_ALIGNED: begin //TODO rename?
                if  (stat_rx_aligned_1d == 1'b1) begin
                    tx_prestate <= STATE_PKT_TRANSFER_INIT;
                end
                else begin
                    tx_prestate <= STATE_WAIT_RX_ALIGNED;
                end
            end
            STATE_PKT_TRANSFER_INIT: begin
                ctl_tx_send_idle_r     <= 1'b0;
                ctl_tx_send_lfi_r      <= 1'b0;
                ctl_tx_send_rfi_r      <= 1'b0;
                ctl_tx_enable_r        <= 1'b1;
                if  (stat_rx_aligned_1d == 1'b0) begin
                    tx_prestate <= STATE_TX_IDLE;
                end
            end
        endcase
    end
end
wire ctl_tx_enable;
wire ctl_tx_send_idle;
wire ctl_tx_send_lfi;
wire ctl_tx_send_rfi;
wire ctl_tx_test_pattern;
assign ctl_tx_enable                = ctl_tx_enable_r;
assign ctl_tx_send_idle             = ctl_tx_send_idle_r;
assign ctl_tx_send_lfi              = ctl_tx_send_lfi_r;
assign ctl_tx_send_rfi              = ctl_tx_send_rfi_r;
assign ctl_tx_test_pattern          = ctl_tx_test_pattern_r;



wire            stat_rx_aligned;
wire            stat_rx_aligned_err;
wire [2:0]      stat_rx_bad_code;
wire [2:0]      stat_rx_bad_fcs;
wire            stat_rx_bad_preamble;
wire            stat_rx_bad_sfd;

wire            stat_rx_got_signal_os;
wire            stat_rx_hi_ber;
wire            stat_rx_inrangeerr;
wire            stat_rx_internal_local_fault;
wire            stat_rx_jabber;
wire            stat_rx_local_fault;
wire [19:0]     stat_rx_mf_err;
wire [19:0]     stat_rx_mf_len_err;
wire [19:0]     stat_rx_mf_repeat_err;
wire            stat_rx_misaligned;

wire            stat_rx_received_local_fault;
wire            stat_rx_remote_fault;
wire            stat_rx_status;
wire [2:0]      stat_rx_stomped_fcs;
wire [19:0]     stat_rx_synced;
wire [19:0]     stat_rx_synced_err;
  
//For debug
logic[6:0]  stat_rx_total_bytes;
logic[13:0] stat_rx_good_bytes;
logic       stat_rx_good_packets;
logic[2:0]  stat_rx_total_packets;

logic[5:0]  stat_tx_total_bytes;
logic[13:0] stat_tx_good_bytes;
logic       stat_tx_good_packets;
logic       stat_tx_total_packets;


wire tx_user_rst_i;
assign tx_user_rst_i = sys_reset; //TODO why not 1'b0??









cmac_usplus_axis0 cmac_usplus_axis0_inst (
  .gt_txp_out(gt_txp_out),                                          // output wire [3 : 0] gt_txp_out
  .gt_txn_out(gt_txn_out),                                          // output wire [3 : 0] gt_txn_out
  .gt_rxp_in(gt_rxp_in),                                            // input wire [3 : 0] gt_rxp_in
  .gt_rxn_in(gt_rxn_in),                                            // input wire [3 : 0] gt_rxn_in

  .gt_txusrclk2(gt_txusrclk2),                                      // output wire gt_txusrclk2
  .gt_loopback_in({4{3'b000}}),                                  // input wire [11 : 0] gt_loopback_in
  .gt_ref_clk_out(),                                  // output wire gt_ref_clk_out
  .gt_rxrecclkout(gt_rxrecclkout),                                  // output wire [3 : 0] gt_rxrecclkout
  .gt_powergoodout(),                                // output wire [3 : 0] gt_powergoodout
  .gtwiz_reset_tx_datapath(1'b0),                // input wire gtwiz_reset_tx_datapath
  .gtwiz_reset_rx_datapath(1'b0),                // input wire gtwiz_reset_rx_datapath

  .sys_reset(sys_reset),                                            // input wire sys_reset
  .gt_ref_clk_p(gt_ref_clk_p),                                      // input wire gt_ref_clk_p
  .gt_ref_clk_n(gt_ref_clk_n),                                      // input wire gt_ref_clk_n
  .init_clk(init_clk),                                              // input wire init_clk

  .rx_axis_tvalid(m_rx_axis.valid),                                  // output wire rx_axis_tvalid
  .rx_axis_tdata(m_rx_axis.data),                                    // output wire [511 : 0] rx_axis_tdata
  .rx_axis_tlast(m_rx_axis.last),                                    // output wire rx_axis_tlast
  .rx_axis_tkeep(m_rx_axis.keep),                                    // output wire [63 : 0] rx_axis_tkeep
  .rx_axis_tuser(),                                    // output wire rx_axis_tuser
  .rx_otn_bip8_0(),                                    // output wire [7 : 0] rx_otn_bip8_0
  .rx_otn_bip8_1(),                                    // output wire [7 : 0] rx_otn_bip8_1
  .rx_otn_bip8_2(),                                    // output wire [7 : 0] rx_otn_bip8_2
  .rx_otn_bip8_3(),                                    // output wire [7 : 0] rx_otn_bip8_3
  .rx_otn_bip8_4(),                                    // output wire [7 : 0] rx_otn_bip8_4
  .rx_otn_data_0(),                                    // output wire [65 : 0] rx_otn_data_0
  .rx_otn_data_1(),                                    // output wire [65 : 0] rx_otn_data_1
  .rx_otn_data_2(),                                    // output wire [65 : 0] rx_otn_data_2
  .rx_otn_data_3(),                                    // output wire [65 : 0] rx_otn_data_3
  .rx_otn_data_4(),                                    // output wire [65 : 0] rx_otn_data_4
  .rx_otn_ena(),                                          // output wire rx_otn_ena
  .rx_otn_lane0(),                                      // output wire rx_otn_lane0
  .rx_otn_vlmarker(),                                // output wire rx_otn_vlmarker
  .rx_preambleout(),                                  // output wire [55 : 0] rx_preambleout
  .usr_rx_reset(usr_rx_reset_w),                                      // output wire usr_rx_reset
  .gt_rxusrclk2(gt_rxusrclk2),                                      // output wire gt_rxusrclk2

  .stat_rx_aligned(stat_rx_aligned),                                // output wire stat_rx_aligned
  .stat_rx_aligned_err(stat_rx_aligned_err),                        // output wire stat_rx_aligned_err
  .stat_rx_bad_code(stat_rx_bad_code),                              // output wire [2 : 0] stat_rx_bad_code
  .stat_rx_bad_fcs(stat_rx_bad_fcs),                                // output wire [2 : 0] stat_rx_bad_fcs
  .stat_rx_bad_preamble(stat_rx_bad_preamble),                      // output wire stat_rx_bad_preamble
  .stat_rx_bad_sfd(stat_rx_bad_sfd),                                // output wire stat_rx_bad_sfd
  .stat_rx_bip_err_0(),                            // output wire stat_rx_bip_err_0
  .stat_rx_bip_err_1(),                            // output wire stat_rx_bip_err_1
  .stat_rx_bip_err_10(),                          // output wire stat_rx_bip_err_10
  .stat_rx_bip_err_11(),                          // output wire stat_rx_bip_err_11
  .stat_rx_bip_err_12(),                          // output wire stat_rx_bip_err_12
  .stat_rx_bip_err_13(),                          // output wire stat_rx_bip_err_13
  .stat_rx_bip_err_14(),                          // output wire stat_rx_bip_err_14
  .stat_rx_bip_err_15(),                          // output wire stat_rx_bip_err_15
  .stat_rx_bip_err_16(),                          // output wire stat_rx_bip_err_16
  .stat_rx_bip_err_17(),                          // output wire stat_rx_bip_err_17
  .stat_rx_bip_err_18(),                          // output wire stat_rx_bip_err_18
  .stat_rx_bip_err_19(),                          // output wire stat_rx_bip_err_19
  .stat_rx_bip_err_2(),                            // output wire stat_rx_bip_err_2
  .stat_rx_bip_err_3(),                            // output wire stat_rx_bip_err_3
  .stat_rx_bip_err_4(),                            // output wire stat_rx_bip_err_4
  .stat_rx_bip_err_5(),                            // output wire stat_rx_bip_err_5
  .stat_rx_bip_err_6(),                            // output wire stat_rx_bip_err_6
  .stat_rx_bip_err_7(),                            // output wire stat_rx_bip_err_7
  .stat_rx_bip_err_8(),                            // output wire stat_rx_bip_err_8
  .stat_rx_bip_err_9(),                            // output wire stat_rx_bip_err_9
  .stat_rx_block_lock(),                          // output wire [19 : 0] stat_rx_block_lock
  .stat_rx_broadcast(),                            // output wire stat_rx_broadcast
  .stat_rx_fragment(),                              // output wire [2 : 0] stat_rx_fragment
  .stat_rx_framing_err_0(),                    // output wire [1 : 0] stat_rx_framing_err_0
  .stat_rx_framing_err_1(),                    // output wire [1 : 0] stat_rx_framing_err_1
  .stat_rx_framing_err_10(),                  // output wire [1 : 0] stat_rx_framing_err_10
  .stat_rx_framing_err_11(),                  // output wire [1 : 0] stat_rx_framing_err_11
  .stat_rx_framing_err_12(),                  // output wire [1 : 0] stat_rx_framing_err_12
  .stat_rx_framing_err_13(),                  // output wire [1 : 0] stat_rx_framing_err_13
  .stat_rx_framing_err_14(),                  // output wire [1 : 0] stat_rx_framing_err_14
  .stat_rx_framing_err_15(),                  // output wire [1 : 0] stat_rx_framing_err_15
  .stat_rx_framing_err_16(),                  // output wire [1 : 0] stat_rx_framing_err_16
  .stat_rx_framing_err_17(),                  // output wire [1 : 0] stat_rx_framing_err_17
  .stat_rx_framing_err_18(),                  // output wire [1 : 0] stat_rx_framing_err_18
  .stat_rx_framing_err_19(),                  // output wire [1 : 0] stat_rx_framing_err_19
  .stat_rx_framing_err_2(),                    // output wire [1 : 0] stat_rx_framing_err_2
  .stat_rx_framing_err_3(),                    // output wire [1 : 0] stat_rx_framing_err_3
  .stat_rx_framing_err_4(),                    // output wire [1 : 0] stat_rx_framing_err_4
  .stat_rx_framing_err_5(),                    // output wire [1 : 0] stat_rx_framing_err_5
  .stat_rx_framing_err_6(),                    // output wire [1 : 0] stat_rx_framing_err_6
  .stat_rx_framing_err_7(),                    // output wire [1 : 0] stat_rx_framing_err_7
  .stat_rx_framing_err_8(),                    // output wire [1 : 0] stat_rx_framing_err_8
  .stat_rx_framing_err_9(),                    // output wire [1 : 0] stat_rx_framing_err_9
  .stat_rx_framing_err_valid_0(),        // output wire stat_rx_framing_err_valid_0
  .stat_rx_framing_err_valid_1(),        // output wire stat_rx_framing_err_valid_1
  .stat_rx_framing_err_valid_10(),      // output wire stat_rx_framing_err_valid_10
  .stat_rx_framing_err_valid_11(),      // output wire stat_rx_framing_err_valid_11
  .stat_rx_framing_err_valid_12(),      // output wire stat_rx_framing_err_valid_12
  .stat_rx_framing_err_valid_13(),      // output wire stat_rx_framing_err_valid_13
  .stat_rx_framing_err_valid_14(),      // output wire stat_rx_framing_err_valid_14
  .stat_rx_framing_err_valid_15(),      // output wire stat_rx_framing_err_valid_15
  .stat_rx_framing_err_valid_16(),      // output wire stat_rx_framing_err_valid_16
  .stat_rx_framing_err_valid_17(),      // output wire stat_rx_framing_err_valid_17
  .stat_rx_framing_err_valid_18(),      // output wire stat_rx_framing_err_valid_18
  .stat_rx_framing_err_valid_19(),      // output wire stat_rx_framing_err_valid_19
  .stat_rx_framing_err_valid_2(),        // output wire stat_rx_framing_err_valid_2
  .stat_rx_framing_err_valid_3(),        // output wire stat_rx_framing_err_valid_3
  .stat_rx_framing_err_valid_4(),        // output wire stat_rx_framing_err_valid_4
  .stat_rx_framing_err_valid_5(),        // output wire stat_rx_framing_err_valid_5
  .stat_rx_framing_err_valid_6(),        // output wire stat_rx_framing_err_valid_6
  .stat_rx_framing_err_valid_7(),        // output wire stat_rx_framing_err_valid_7
  .stat_rx_framing_err_valid_8(),        // output wire stat_rx_framing_err_valid_8
  .stat_rx_framing_err_valid_9(),        // output wire stat_rx_framing_err_valid_9
  .stat_rx_got_signal_os(stat_rx_got_signal_os),                    // output wire stat_rx_got_signal_os
  .stat_rx_hi_ber(stat_rx_hi_ber),                                  // output wire stat_rx_hi_ber
  .stat_rx_inrangeerr(stat_rx_inrangeerr),                          // output wire stat_rx_inrangeerr
  .stat_rx_internal_local_fault(stat_rx_internal_local_fault),      // output wire stat_rx_internal_local_fault
  .stat_rx_jabber(stat_rx_jabber),                                  // output wire stat_rx_jabber
  .stat_rx_local_fault(stat_rx_local_fault),                        // output wire stat_rx_local_fault
  .stat_rx_mf_err(stat_rx_mf_err),                                  // output wire [19 : 0] stat_rx_mf_err
  .stat_rx_mf_len_err(stat_rx_mf_len_err),                          // output wire [19 : 0] stat_rx_mf_len_err
  .stat_rx_mf_repeat_err(stat_rx_mf_repeat_err),                    // output wire [19 : 0] stat_rx_mf_repeat_err
  .stat_rx_misaligned(stat_rx_misaligned),                          // output wire stat_rx_misaligned
  .stat_rx_multicast(),                            // output wire stat_rx_multicast
  .stat_rx_oversize(),                              // output wire stat_rx_oversize
  .stat_rx_packet_1024_1518_bytes(),  // output wire stat_rx_packet_1024_1518_bytes
  .stat_rx_packet_128_255_bytes(),      // output wire stat_rx_packet_128_255_bytes
  .stat_rx_packet_1519_1522_bytes(),  // output wire stat_rx_packet_1519_1522_bytes
  .stat_rx_packet_1523_1548_bytes(),  // output wire stat_rx_packet_1523_1548_bytes
  .stat_rx_packet_1549_2047_bytes(),  // output wire stat_rx_packet_1549_2047_bytes
  .stat_rx_packet_2048_4095_bytes(),  // output wire stat_rx_packet_2048_4095_bytes
  .stat_rx_packet_256_511_bytes(),      // output wire stat_rx_packet_256_511_bytes
  .stat_rx_packet_4096_8191_bytes(),  // output wire stat_rx_packet_4096_8191_bytes
  .stat_rx_packet_512_1023_bytes(),    // output wire stat_rx_packet_512_1023_bytes
  .stat_rx_packet_64_bytes(),                // output wire stat_rx_packet_64_bytes
  .stat_rx_packet_65_127_bytes(),        // output wire stat_rx_packet_65_127_bytes
  .stat_rx_packet_8192_9215_bytes(),  // output wire stat_rx_packet_8192_9215_bytes
  .stat_rx_packet_bad_fcs(),                  // output wire stat_rx_packet_bad_fcs
  .stat_rx_packet_large(),                      // output wire stat_rx_packet_large
  .stat_rx_packet_small(),                      // output wire [2 : 0] stat_rx_packet_small

  .ctl_rx_enable(ctl_rx_enable),                                    // input wire ctl_rx_enable
  .ctl_rx_force_resync(ctl_rx_force_resync),                        // input wire ctl_rx_force_resync
  .ctl_rx_test_pattern(1'b0),                        // input wire ctl_rx_test_pattern
  .core_rx_reset(1'b0),                                    // input wire core_rx_reset
  .rx_clk(gt_txusrclk2),                                                  // input wire rx_clk

  .stat_rx_received_local_fault(stat_rx_received_local_fault),      // output wire stat_rx_received_local_fault
  .stat_rx_remote_fault(stat_rx_remote_fault),                      // output wire stat_rx_remote_fault
  .stat_rx_status(stat_rx_status),                                  // output wire stat_rx_status
  .stat_rx_stomped_fcs(stat_rx_stomped_fcs),                        // output wire [2 : 0] stat_rx_stomped_fcs
  .stat_rx_synced(stat_rx_synced),                                  // output wire [19 : 0] stat_rx_synced
  .stat_rx_synced_err(stat_rx_synced_err),                          // output wire [19 : 0] stat_rx_synced_err
  .stat_rx_test_pattern_mismatch(),    // output wire [2 : 0] stat_rx_test_pattern_mismatch
  .stat_rx_toolong(),                                // output wire stat_rx_toolong
  .stat_rx_total_bytes(stat_rx_total_bytes),                        // output wire [6 : 0] stat_rx_total_bytes
  .stat_rx_total_good_bytes(stat_rx_good_bytes),              // output wire [13 : 0] stat_rx_total_good_bytes
  .stat_rx_total_good_packets(stat_rx_good_packets),          // output wire stat_rx_total_good_packets
  .stat_rx_total_packets(stat_rx_total_packets),                    // output wire [2 : 0] stat_rx_total_packets
  .stat_rx_truncated(),                            // output wire stat_rx_truncated
  .stat_rx_undersize(),                            // output wire [2 : 0] stat_rx_undersize
  .stat_rx_unicast(),                                // output wire stat_rx_unicast
  .stat_rx_vlan(),                                      // output wire stat_rx_vlan
  .stat_rx_pcsl_demuxed(),                      // output wire [19 : 0] stat_rx_pcsl_demuxed
  .stat_rx_pcsl_number_0(stat_rx_pcsl_number_0),                    // output wire [4 : 0] stat_rx_pcsl_number_0
  .stat_rx_pcsl_number_1(stat_rx_pcsl_number_1),                    // output wire [4 : 0] stat_rx_pcsl_number_1
  .stat_rx_pcsl_number_10(stat_rx_pcsl_number_10),                  // output wire [4 : 0] stat_rx_pcsl_number_10
  .stat_rx_pcsl_number_11(stat_rx_pcsl_number_11),                  // output wire [4 : 0] stat_rx_pcsl_number_11
  .stat_rx_pcsl_number_12(stat_rx_pcsl_number_12),                  // output wire [4 : 0] stat_rx_pcsl_number_12
  .stat_rx_pcsl_number_13(stat_rx_pcsl_number_13),                  // output wire [4 : 0] stat_rx_pcsl_number_13
  .stat_rx_pcsl_number_14(stat_rx_pcsl_number_14),                  // output wire [4 : 0] stat_rx_pcsl_number_14
  .stat_rx_pcsl_number_15(stat_rx_pcsl_number_15),                  // output wire [4 : 0] stat_rx_pcsl_number_15
  .stat_rx_pcsl_number_16(stat_rx_pcsl_number_16),                  // output wire [4 : 0] stat_rx_pcsl_number_16
  .stat_rx_pcsl_number_17(stat_rx_pcsl_number_17),                  // output wire [4 : 0] stat_rx_pcsl_number_17
  .stat_rx_pcsl_number_18(stat_rx_pcsl_number_18),                  // output wire [4 : 0] stat_rx_pcsl_number_18
  .stat_rx_pcsl_number_19(stat_rx_pcsl_number_19),                  // output wire [4 : 0] stat_rx_pcsl_number_19
  .stat_rx_pcsl_number_2(stat_rx_pcsl_number_2),                    // output wire [4 : 0] stat_rx_pcsl_number_2
  .stat_rx_pcsl_number_3(stat_rx_pcsl_number_3),                    // output wire [4 : 0] stat_rx_pcsl_number_3
  .stat_rx_pcsl_number_4(stat_rx_pcsl_number_4),                    // output wire [4 : 0] stat_rx_pcsl_number_4
  .stat_rx_pcsl_number_5(stat_rx_pcsl_number_5),                    // output wire [4 : 0] stat_rx_pcsl_number_5
  .stat_rx_pcsl_number_6(stat_rx_pcsl_number_6),                    // output wire [4 : 0] stat_rx_pcsl_number_6
  .stat_rx_pcsl_number_7(stat_rx_pcsl_number_7),                    // output wire [4 : 0] stat_rx_pcsl_number_7
  .stat_rx_pcsl_number_8(stat_rx_pcsl_number_8),                    // output wire [4 : 0] stat_rx_pcsl_number_8
  .stat_rx_pcsl_number_9(stat_rx_pcsl_number_9),                    // output wire [4 : 0] stat_rx_pcsl_number_9

  .stat_tx_bad_fcs(),                                // output wire stat_tx_bad_fcs
  .stat_tx_broadcast(),                            // output wire stat_tx_broadcast
  .stat_tx_frame_error(),                        // output wire stat_tx_frame_error
  .stat_tx_local_fault(),                        // output wire stat_tx_local_fault
  .stat_tx_multicast(),                            // output wire stat_tx_multicast
  .stat_tx_packet_1024_1518_bytes(),  // output wire stat_tx_packet_1024_1518_bytes
  .stat_tx_packet_128_255_bytes(),      // output wire stat_tx_packet_128_255_bytes
  .stat_tx_packet_1519_1522_bytes(),  // output wire stat_tx_packet_1519_1522_bytes
  .stat_tx_packet_1523_1548_bytes(),  // output wire stat_tx_packet_1523_1548_bytes
  .stat_tx_packet_1549_2047_bytes(),  // output wire stat_tx_packet_1549_2047_bytes
  .stat_tx_packet_2048_4095_bytes(),  // output wire stat_tx_packet_2048_4095_bytes
  .stat_tx_packet_256_511_bytes(),      // output wire stat_tx_packet_256_511_bytes
  .stat_tx_packet_4096_8191_bytes(),  // output wire stat_tx_packet_4096_8191_bytes
  .stat_tx_packet_512_1023_bytes(),    // output wire stat_tx_packet_512_1023_bytes
  .stat_tx_packet_64_bytes(),                // output wire stat_tx_packet_64_bytes
  .stat_tx_packet_65_127_bytes(),        // output wire stat_tx_packet_65_127_bytes
  .stat_tx_packet_8192_9215_bytes(),  // output wire stat_tx_packet_8192_9215_bytes
  .stat_tx_packet_large(),                      // output wire stat_tx_packet_large
  .stat_tx_packet_small(),                      // output wire stat_tx_packet_small
  .stat_tx_total_bytes(stat_tx_total_bytes),                        // output wire [5 : 0] stat_tx_total_bytes
  .stat_tx_total_good_bytes(stat_tx_good_bytes),              // output wire [13 : 0] stat_tx_total_good_bytes
  .stat_tx_total_good_packets(stat_tx_good_packets),          // output wire stat_tx_total_good_packets
  .stat_tx_total_packets(stat_tx_total_packets),                    // output wire stat_tx_total_packets
  .stat_tx_unicast(),                                // output wire stat_tx_unicast
  .stat_tx_vlan(),                                      // output wire stat_tx_vlan

  .ctl_tx_enable(ctl_tx_enable),                                    // input wire ctl_tx_enable
  .ctl_tx_send_idle(ctl_tx_send_idle),                              // input wire ctl_tx_send_idle
  .ctl_tx_send_rfi(ctl_tx_send_rfi),                                // input wire ctl_tx_send_rfi
  .ctl_tx_send_lfi(ctl_tx_send_lfi),                                // input wire ctl_tx_send_lfi
  .ctl_tx_test_pattern(ctl_tx_test_pattern),                        // input wire ctl_tx_test_pattern
  .core_tx_reset(1'b0),                                    // input wire core_tx_reset

  .tx_axis_tready(s_tx_axis.ready),                                  // output wire tx_axis_tready
  .tx_axis_tvalid(s_tx_axis.valid),                                  // input wire tx_axis_tvalid
  .tx_axis_tdata(s_tx_axis.data),                                    // input wire [511 : 0] tx_axis_tdata
  .tx_axis_tlast(s_tx_axis.last),                                    // input wire tx_axis_tlast
  .tx_axis_tkeep(s_tx_axis.keep),                                    // input wire [63 : 0] tx_axis_tkeep
  .tx_axis_tuser(0),                                    // input wire tx_axis_tuser

  .tx_ovfout(tx_ovf),                                            // output wire tx_ovfout
  .tx_unfout(tx_unf),                                            // output wire tx_unfout
  .tx_preamblein({55{1'b0}}),                                    // input wire [55 : 0] tx_preamblein

  .usr_tx_reset(core_tx_reset_w),                                      // output wire usr_tx_reset
  .core_drp_reset(1'b0),                                  // input wire core_drp_reset
  .drp_clk(1'b0),                                                // input wire drp_clk
  .drp_addr(10'b0),                                              // input wire [9 : 0] drp_addr
  .drp_di(16'b0),                                                  // input wire [15 : 0] drp_di
  .drp_en(1'b0),                                                  // input wire drp_en
  .drp_do(),                                                  // output wire [15 : 0] drp_do
  .drp_rdy(),                                                // output wire drp_rdy
  .drp_we(1'b0)                                                  // input wire drp_we
);







`ifdef DEBUG

logic[31:0] rx_good_packets_count;
logic[31:0] rx_total_packets_count;
logic[31:0] rx_good_bytes_count;
logic[31:0] rx_total_bytes_count;

always @(posedge cmac_drp.gt_txusrclk2) begin
    if (usr_rx_reset_w) begin
        rx_good_packets_count <= '0;
        rx_total_packets_count <= '0;
        rx_good_bytes_count <= '0;
        rx_total_bytes_count <= '0;
    end
    else begin
        rx_good_packets_count <= rx_good_packets_count + stat_rx_good_packets;
        rx_total_packets_count <= rx_total_packets_count + stat_rx_total_packets;
        rx_good_bytes_count <= rx_good_bytes_count + stat_rx_good_bytes;
        rx_total_bytes_count <= rx_total_bytes_count + stat_rx_total_bytes;
    end
end
    
ila_mixed ila_rx (
    .clk(cmac_drp.gt_txusrclk2), // input wire clk


    .probe0(ctl_rx_enable), // input wire [0:0]  probe0
    .probe1(ctl_rx_force_resync), // input wire [0:0]  probe1
    .probe2(0), // input wire [0:0]  probe2
    .probe3(stat_rx_aligned), // input wire [0:0]  probe3
    .probe4(stat_rx_aligned_1d), // input wire [0:0]  probe4
    .probe5(rx_reset_done), // input wire [0:0]  probe5
    .probe6(stat_rx_bad_code[0]), // input wire [0:0]  probe6
    .probe7(stat_rx_bad_code[1]), // input wire [0:0]  probe7
    .probe8({fcs_errors, code_errors, align_errors, rx_prestate}), // input wire [0:0]  probe8
    .probe9(rx_good_packets_count), // input wire [0:0]  probe9
    .probe10(rx_total_packets_count), // input wire [0:0]  probe10
    .probe11(rx_total_bytes_count), // input wire [0:0]  probe11
    .probe12(rx_good_bytes_count), // input wire [0:0]  probe12
    .probe13(stat_rx_synced_err), // input wire [0:0]  probe13
    .probe14({stat_rx_got_signal_os, stat_rx_hi_ber, stat_rx_inrangeerr, stat_rx_internal_local_fault, stat_rx_jabber, stat_rx_local_fault, stat_rx_misaligned}), // input wire [0:0]  probe14
    .probe15({stat_rx_received_local_fault, stat_rx_remote_fault, stat_rx_status, stat_rx_stomped_fcs}) // input wire [0:0]  probe15 
);
    
//counting errors
logic[15:0] align_errors;
logic[15:0] code_errors;
logic[15:0] fcs_errors;
logic[15:0] preamble_errors;
logic[15:0] sfd_errors;

logic[15:0] overflow_count;
logic[15:0] underflow_count;
always @(posedge cmac_drp.gt_txusrclk2) begin
    if (usr_rx_reset_w) begin
        align_errors <= '0;
        code_errors <= '0;
        fcs_errors <= '0;
        preamble_errors <= '0;
        sfd_errors <= '0;
        
        overflow_count <= '0;
        underflow_count <= '0;
    end
    else begin
        if (stat_rx_aligned_err != 0) begin
            align_errors <= align_errors + 1;
        end
        if (stat_rx_bad_code != 0) begin
            code_errors <= code_errors + 1;
        end
        if (stat_rx_bad_fcs != 0) begin
            fcs_errors <= fcs_errors + 1;
        end
        if (stat_rx_bad_preamble != 0) begin
            preamble_errors <= preamble_errors + 1;
        end
        if (stat_rx_bad_sfd != 0) begin
            sfd_errors <= sfd_errors + 1;
        end
        if (cmac_lbus_tx.ovf == 1'b1)  begin
            overflow_count <= overflow_count + 1;
        end
        if (cmac_lbus_tx.unf == 1'b1) begin
            underflow_count <= underflow_count + 1;
        end
    end
end


logic[31:0] tx_good_packets_count;
logic[31:0] tx_total_packets_count;
logic[31:0] tx_good_bytes_count;
logic[31:0] tx_total_bytes_count;

always @(posedge cmac_drp.gt_txusrclk2) begin
    if (core_tx_reset_w) begin
        tx_good_packets_count <= '0;
        tx_total_packets_count <= '0;
        tx_good_bytes_count <= '0;
        tx_total_bytes_count <= '0;
    end
    else begin
        tx_good_packets_count <= tx_good_packets_count + stat_tx_good_packets;
        tx_total_packets_count <= tx_total_packets_count + stat_tx_total_packets;
        tx_good_bytes_count <= tx_good_bytes_count + stat_tx_good_bytes;
        tx_total_bytes_count <= tx_total_bytes_count + stat_tx_total_bytes;
    end
end

ila_mixed ila_tx (
	.clk(cmac_drp.gt_txusrclk2), // input wire clk


	.probe0(ctl_tx_enable), // input wire [0:0]  probe0 
	.probe1(ctl_tx_send_idle), // input wire [0:0]  probe1
	.probe2(ctl_tx_send_lfi), // input wire [0:0]  probe2
	.probe3(ctl_tx_send_rfi), // input wire [0:0]  probe3
	.probe4(ctl_tx_test_pattern), // input wire [0:0]  probe4
	.probe5(cmac_lbus_tx.ovf), // input wire [0:0]  probe5
	.probe6(cmac_lbus_tx.unf), // input wire [0:0]  probe6
	.probe7(cmac_lbus_tx.rdy), // input wire [0:0]  probe7
	.probe8(tx_prestate), // input wire [0:0]  probe8
	.probe9(tx_reset_done), // input wire [0:0]  probe9
	.probe10(tx_good_packets_count), // input wire [0:0]  probe10
	.probe11(tx_total_packets_count), // input wire [0:0]  probe11
	.probe12(tx_good_bytes_count), // input wire [0:0]  probe12
	.probe13(tx_total_bytes_count), // input wire [0:0]  probe13
	.probe14(overflow_count), // input wire [0:0]  probe14
	.probe15(underflow_count) // input wire [0:0]  probe15
);
    

`endif
//ila_2 TX (
//	.clk(usr_tx_clk), // input wire clk


//	.probe0(s_tx_axis.ready), // input wire [0:0]  probe0  
//	.probe1(s_tx_axis.valid), // input wire [0:0]  probe1 
//	.probe2(s_tx_axis.last), // input wire [0:0]  probe2 
//	.probe3(s_tx_axis.data), // input wire [511:0]  probe3
//	.probe4(s_tx_axis.keep), // input wire [63:0]  probe4
//	.probe5(tx_prestate), // input wire [3:0]  probe5
//	.probe6(tx_ovfout_d), // input wire [0:0]  probe6 
//	.probe7(tx_unfout_d) // input wire [0:0]  probe7
//);

//ila_2 RX (
//	.clk(usr_tx_clk), // input wire clk


//	.probe0(m_rx_axis.ready), // input wire [0:0]  probe0  
//	.probe1(m_rx_axis.valid), // input wire [0:0]  probe1 
//	.probe2(m_rx_axis.last), // input wire [0:0]  probe2 
//	.probe3(m_rx_axis.data), // input wire [511:0]  probe3
//	.probe4(m_rx_axis.keep), // input wire [63:0]  probe4
//	.probe5(rx_prestate), // input wire [3:0]  probe5
//	.probe6(stat_rx_received_local_fault), // input wire [0:0]  probe6 
//	.probe7(stat_rx_remote_fault) // input wire [0:0]  probe7
//);
endmodule

`default_nettype wire

