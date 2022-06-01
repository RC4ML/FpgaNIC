/*
 * Copyright 2019 - 2020, RC4ML, Zhejiang University
 *
 * This hardware operator is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
// The objective of lt_engine is to benchmark the read part of memory controller.
//

//`include "sgd_defines.vh"
 module wr_engine #(parameter ENGINE_ID = 0,
    parameter ADDR_WIDTH = 33,                  // 8G-->33 bits
    parameter DATA_WIDTH = 256,                 // parameter bits from PCIe
    parameter PARAMS_BITS = 256,                // parameter bits from PCIe
    parameter ID_WIDTH = 5)
  (input clk,                                   //should be 450MHz, 
    input rst_n,                                //negative reset, 
    input start,
    output reg end_of_exec,
    output reg [63:0] lat_timer_sum,
    input [PARAMS_BITS-1:0] lt_params,
    output m_axi_AWVALID,                       //wr address valid
    output reg [ADDR_WIDTH - 1:0] m_axi_AWADDR, //wr byte address
    output reg [ID_WIDTH - 1:0] m_axi_AWID,    //wr address id
    output reg [7:0] m_axi_AWLEN,               //wr burst = awlen+1, 
    output reg [2:0] m_axi_AWSIZE,              //wr 3'b101, 32B
    output reg [1:0] m_axi_AWBURST,             //wr burst type: 01 (INC), 00 (FIXED)
    output reg [1:0] m_axi_AWLOCK,              //wr no
    output reg [3:0] m_axi_AWCACHE,             //wr no
    output reg [2:0] m_axi_AWPROT,              //wr no
    output reg [3:0] m_axi_AWQOS,               //wr no
    output reg [3:0] m_axi_AWREGION,            //wr no
    input m_axi_AWREADY,                        //wr ready to accept address.
    output m_axi_WVALID,                        //wr data valid
    output reg [DATA_WIDTH - 1:0] m_axi_WDATA,  //wr data
    output reg [DATA_WIDTH/8-1:0] m_axi_WSTRB,  //wr data strob
    output m_axi_WLAST,                         //wr last beat in a burst
    output reg [ID_WIDTH - 1:0] m_axi_WID,      //wr data id
    input m_axi_WREADY,                         //wr ready to accept data
    input m_axi_BVALID,
    input [1:0] m_axi_BRESP,
    input [ID_WIDTH - 1:0] m_axi_BID,
    output m_axi_BREADY);

/////////////////Parameters for arvalid and araddr////////////////
reg [ADDR_WIDTH - 1:0] init_addr      ;
reg             [63:0] num_mem_ops    ;
reg             [15:0] mem_burst_size ;
reg             [31:0] work_group_size;
reg             [31:0] stride         ;
reg                    isRdLatencyTest;
reg             [63:0] mem_op_index   ;
reg [ADDR_WIDTH - 1:0] offset_addr    ;
reg             [31:0] work_group_size_minus_1;
reg             [63:0] num_mem_ops_r, num_mem_ops_minus_1;

reg             [63:0] wr_ops;
reg             [7 :0] burst_inc;
reg                    wr_data_done;

reg                    started, started_r;   //one cycle delay from started...
reg                    is_in_progress;

reg                    guard_AWVALID, guard_WVALID;
always @(posedge clk)
begin
if (~rst_n)
begin
started   <= 1'b0;
started_r <= 1'b0;
end
else
begin
started   <= start;   //1'b0;
started_r <= start;
end
end

reg  [PARAMS_BITS-1:0] lt_params_r; //staged parameters.
always @(posedge clk)
begin
lt_params_r <= lt_params;
end

/////////////////Parameters with fixed values////////////////
always @(posedge clk)
begin
m_axi_AWID  <= {ID_WIDTH{1'b0}};
m_axi_AWLEN <= (mem_burst_size>>($clog2(DATA_WIDTH/8)))-8'b1;
m_axi_AWSIZE      <= (DATA_WIDTH == 256)? 3'b101:3'b110; //just for 256-bit or 512-bit.
m_axi_AWBURST  <= 2'b01;   // INC, not FIXED (00)
m_axi_AWLOCK   <= 2'b00;   // Normal memory operation
m_axi_AWCACHE  <= 4'b0000; //4'b0011; // Normal, non-cacheable, modifiable, bufferable (Xilinx recommends)
m_axi_AWPROT   <= 3'b010; //3'b000;  // Normal, secure, data
m_axi_AWQOS    <= 4'b0000; // Not participating in any Qos schem, a higher value indicates a higher priority transaction
m_axi_AWREGION <= 4'b0000; // Region indicator, default to 0

m_axi_WDATA <= {DATA_WIDTH{1'b0}};        //data port
m_axi_WSTRB <= {(DATA_WIDTH/8){1'b1}};
m_axi_WID   <= {ID_WIDTH{1'b0}};          //maybe play with it.
end


assign m_axi_BREADY = 1'b1;    //always ready to accept data...

assign m_axi_AWVALID = guard_AWVALID; //(mem_op_index < num_mem_ops_r) &

assign m_axi_WLAST  = (burst_inc == m_axi_AWLEN)&guard_WVALID; //wlast is 1 for the last beat.
assign m_axi_WVALID = (wr_ops != num_mem_ops_r)&guard_WVALID; //<


always @(posedge clk)
begin
if (~rst_n)
begin
burst_inc    <= 8'b0;
wr_ops       <= 64'b0;
wr_data_done <= 1'b0;
end
else if (started)
begin
burst_inc    <= 8'b0;
wr_data_done <= 1'b0;
wr_ops       <= 64'b0;
end
else if (is_in_progress)
begin
if (m_axi_WREADY & guard_WVALID)
begin
 burst_inc <= burst_inc + 8'b1;
 if (burst_inc == m_axi_AWLEN)
 begin
     burst_inc <= 8'b0;
     wr_ops    <= wr_ops + 1'b1;
     if (wr_ops == num_mem_ops_minus_1)
     begin
         wr_data_done <= 1'b1;
         //wr_ops     <= 64'b0; //may miss WVALID.
     end
 end
end
end
//all_mem_ops_are_recvd <= lt_params;
end


/////////////////One FSM to decide valid/addr////////////////
reg              [2:0] state;
localparam [2:0]
WR_IDLE = 3'b000, //begining state
WR_STARTED = 3'b001, //receiving the parameters
WR_TH_VALID = 3'b010,
WR_TH_DATA = 3'b011,
WR_END = 3'b100;

always@(posedge clk)
begin
 if (~rst_n)
 begin
     is_in_progress <= 1'b0;
     state          <= WR_IDLE;
 end
 else
 begin
     end_of_exec   <= 1'b0;
     guard_AWVALID <= 1'b0;
     guard_WVALID  <= 1'b0;
     lat_timer_sum <= lat_timer_sum + 64'b1;
     case (state)
         //This state is the beginning of FSM...
         WR_IDLE:
         begin
             lat_timer_sum <= lat_timer_sum;
             if (started)  //stage the parameter when started is 1.
             begin
                 work_group_size <= lt_params_r[31:  0];
                 stride          <= lt_params_r[63: 32];
                 num_mem_ops     <= lt_params_r[127: 64];//[95 : 64];
                 mem_burst_size  <= lt_params_r[159:128];//[127 : 96];
                 init_addr       <= lt_params_r[ADDR_WIDTH+159:160];//[ADDR_WIDTH+127:128]; //ADDR_WIDTH<48.
                 state           <= WR_STARTED;
             end
         end
         
         /* This state is just a stopby state which initilizes the parameters...*/
         WR_STARTED:
         begin
             is_in_progress          <= 1'b1;
             offset_addr             <= {ADDR_WIDTH{1'b0}};
             work_group_size_minus_1 <= work_group_size - 1'b1;
             num_mem_ops_r           <= num_mem_ops; // - 1'b1;
             num_mem_ops_minus_1     <= num_mem_ops - 1'b1;
             mem_op_index            <= 64'b0;
             lat_timer_sum           <= 64'b0;
             state                   <= WR_TH_VALID;
         end
         
         WR_TH_VALID: //For the sample.
         begin
             guard_AWVALID <= 1'b1;
             guard_WVALID  <= 1'b1;
             m_axi_AWADDR  <= init_addr + (offset_addr&work_group_size_minus_1);
             if (m_axi_AWREADY & m_axi_AWVALID)//when ARREADY is 1, increase the address. & guard_AWVALID
             begin
                 offset_addr  <= offset_addr + stride;
                 mem_op_index <= mem_op_index + 1'b1;
                 if ((mem_op_index == num_mem_ops_minus_1))
                 begin
                     guard_AWVALID <= 1'b0;
                     if (wr_data_done)
                         state <= WR_END;
                     else
                         state <= WR_TH_DATA;
                 end
                 else
                     state <= WR_TH_VALID;
             end
         end
         
         ////To start testing throughput////
         WR_TH_DATA:
         begin
             guard_WVALID <= 1'b1;
             if (wr_data_done)
             begin
                 guard_WVALID <= 1'b0;
                 state        <= WR_END;
             end
         end
         
         WR_END:
         begin
             //guard_WVALID <= 1'b0;
             is_in_progress <= 1'b0;
             end_of_exec    <= 1'b1;
             state          <= WR_IDLE; //end of one sample...
         end
         
         default:
         state <= WR_IDLE;
     endcase
     // else kill
 end
end

/*
ila_wr_engine inst_bebug_wr_engine (
.clk (clk),

.probe0  (started),
.probe1  (lt_params),
.probe2  (m_axi_AWREADY),
.probe3  (m_axi_AWVALID),
.probe4  (m_axi_AWADDR),
.probe5  (m_axi_WVALID),
.probe6  (m_axi_WREADY),
.probe7  (state),
.probe8  (m_axi_WLAST),
.probe9  (burst_inc),
.probe10 (m_axi_AWLEN),
.probe11 (wr_ops),
.probe12 (mem_op_index),
.probe13 (num_mem_ops_r),
.probe14 (guard_AWVALID),
.probe15 (guard_WVALID)
);
*/

endmodule
