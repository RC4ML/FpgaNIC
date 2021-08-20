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
 module rd_engine #(
    parameter ENGINE_ID       = 0  ,
    parameter ADDR_WIDTH      = 33 ,  // 8G-->33 bits
    parameter DATA_WIDTH      = 256,  // 512-bit for DDR4
    parameter PARAMS_BITS     = 256,  // parameter bits from PCIe
    parameter ID_WIDTH        = 5     //fixme,
)(
    input                     clk,   //should be 450MHz, 
    input                     rst_n, //negative reset,   
    //---------------------Begin/Stop-----------------------------//
    input                     start,

    output reg                end_of_exec,
    output reg         [63:0] lat_timer_sum,
    output reg                lat_timer_valid, //log down lat_timer when lat_timer_valid is 1. 
    output reg         [15:0] lat_timer,

    //---------------------Parameters-----------------------------//
    input   [PARAMS_BITS-1:0] lt_params,

    //Read Address (Output)  
    output                        m_axi_ARVALID , //rd address valid
    output reg [ADDR_WIDTH - 1:0] m_axi_ARADDR  , //rd byte address
    output reg   [ID_WIDTH - 1:0] m_axi_ARID    , //rd address id
    output reg              [7:0] m_axi_ARLEN   , //rd burst=awlen+1,
    output reg              [2:0] m_axi_ARSIZE  , //rd 3'b101, 32B
    output reg              [1:0] m_axi_ARBURST , //rd burst type: 01 (INC), 00 (FIXED)
    output reg              [1:0] m_axi_ARLOCK  , //rd no
    output reg              [3:0] m_axi_ARCACHE , //rd no
    output reg              [2:0] m_axi_ARPROT  , //rd no
    output reg              [3:0] m_axi_ARQOS   , //rd no
    output reg              [3:0] m_axi_ARREGION, //rd no
    input                         m_axi_ARREADY , //rd ready to accept address.

    //Read Data (input)
    input                      m_axi_RVALID, //rd data valid
    input   [DATA_WIDTH - 1:0] m_axi_RDATA , //rd data 
    input                      m_axi_RLAST , //rd data last
    input     [ID_WIDTH - 1:0] m_axi_RID   , //rd data id
    input                [1:0] m_axi_RRESP , //rd data status. 
    output                     m_axi_RREADY
);
/////////////////Parameters for arvalid and araddr////////////////
reg [ADDR_WIDTH - 1:0] init_addr      ;
reg             [31:0] num_mem_ops    ;
reg             [15:0] mem_burst_size ;
reg             [31:0] work_group_size;
reg             [31:0] stride         ;
reg                    isRdLatencyTest;
reg             [63:0] mem_op_index   ;
reg [ADDR_WIDTH - 1:0] offset_addr    ;
reg             [31:0] work_group_size_minus_1;
reg             [63:0] num_mem_ops_r, num_mem_ops_minus_1;

reg             [63:0] read_resp_lasts;



reg                    started;   //one cycle delay from started...
reg                    guard_ARVALID;
//reg                    is_in_progress;
always @(posedge clk) 
begin
    if(~rst_n)
        started  <= 1'b0;
    else 
        started  <= start;   //1'b0;
end

reg  [PARAMS_BITS-1:0] lt_params_r; //staged parameters.
always @(posedge clk) 
begin
    lt_params_r    <=  lt_params;
end

/////////////////Parameters with fixed values////////////////
always @(posedge clk) 
begin
    m_axi_ARID     <= {ID_WIDTH{1'b0}};
    m_axi_ARLEN    <= (mem_burst_size>>($clog2(DATA_WIDTH/8)))-8'b1;
    m_axi_ARSIZE   <= (DATA_WIDTH==256)? 3'b101:3'b110; //just for 256-bit or 512-bit.
    m_axi_ARBURST  <= 2'b01;   // INC, not FIXED (00)
    m_axi_ARLOCK   <= 2'b00;   // Normal memory operation
    m_axi_ARCACHE  <= 4'b0000; // 4'b0011: Normal, non-cacheable, modifiable, bufferable (Xilinx recommends)
    m_axi_ARPROT   <= 3'b010;  // 3'b000: Normal, secure, data
    m_axi_ARQOS    <= 4'b0000; // Not participating in any Qos schem, a higher value indicates a higher priority transaction
    m_axi_ARREGION <= 4'b0000; // Region indicator, default to 0
end

assign m_axi_RREADY = 1'b1;    //always ready to accept data...

assign m_axi_ARVALID = guard_ARVALID;//(state == RD_TH_VALID)|| (state == RD_LAT_VALID);// //(mem_op_index < num_mem_ops_r) & guard_ARVALID;

always @(posedge clk) 
begin
    if (started)
        read_resp_lasts <= 64'b0;
    else
    begin
        read_resp_lasts <= read_resp_lasts + (m_axi_RLAST&m_axi_RVALID); //(m_axi_RRESP == 2'b0) m_axi_RRESP[1] == 1'b0)&
    end        
    //all_mem_ops_are_recvd    <= lt_params;
end



/////////////////One FSM to decide valid/addr////////////////
reg              [3:0] state;
localparam [3:0]
        RD_IDLE          = 4'b0000, //begining state
        RD_STARTED       = 4'b0001, //receiving the parameters
        RD_LATENCY       = 4'b0010,        
        RD_LAT_VALID     = 4'b0011,
        RD_LAT_RESP      = 4'b0100,
        RD_TH_VALID      = 4'b0101,
        RD_TH_RESP       = 4'b1000,
        RD_END           = 4'b1001;

always@(posedge clk) 
begin
    if(~rst_n) 
        state                             <= RD_IDLE;
    else 
    begin
        end_of_exec                       <= 1'b0;
        guard_ARVALID                     <= 1'b0;
        lat_timer_valid                   <= 1'b0;    
        lat_timer_sum                     <= lat_timer_sum + 1'b1;
        case (state)
            /*This state is the beginning of FSM... wait for started...*/
            RD_IDLE: 
            begin
                lat_timer_sum             <= lat_timer_sum; // freeze the timer. 

                if(started)  //stage the parameter when started is 1.
                begin
                    work_group_size       <= lt_params_r[ 31: 0 ];
                    stride                <= lt_params_r[ 63: 32];
                    num_mem_ops           <= lt_params_r[127: 64];//[ 95 : 64];
                    mem_burst_size        <= lt_params_r[159:128];//[127 : 96];
                    init_addr             <= lt_params_r[ADDR_WIDTH+159:160];//[ADDR_WIDTH+127:128]; //ADDR_WIDTH<48.
                    isRdLatencyTest       <= lt_params_r[224]; //32-bit left for future param.
                    state                 <= RD_STARTED;
                end  
            end

            /* This state initilizes the parameters...*/
            RD_STARTED: 
            begin
                work_group_size_minus_1   <= work_group_size - 1'b1;
                num_mem_ops_r             <= num_mem_ops;
                num_mem_ops_minus_1       <= num_mem_ops - 1'b1;
                mem_op_index              <= 64'b0;
                offset_addr               <= {ADDR_WIDTH{1'b0}};
                lat_timer_sum             <= 1'b0;
                if (isRdLatencyTest)
                    state                 <= RD_LAT_VALID;
                else
                    state                 <= RD_TH_VALID;
            end

            //This state indicates the begginning of each epoch...
        /*    RD_LATENCY: 
            begin
                mem_op_index              <= mem_op_index + 1'b1;
                if (mem_op_index == num_mem_ops_r)
                    state                 <= RD_END; 
                else
                    state                 <= RD_LAT_VALID; 
            end
*/
            RD_LAT_VALID: //wait until the read op is responsed. 
            begin
                offset_addr               <= offset_addr + stride; 
                m_axi_ARADDR              <= init_addr   + (offset_addr&work_group_size_minus_1);    
                guard_ARVALID             <= 1'b1;     //m_axi_ARREADY should be 1.
                if (m_axi_ARREADY & m_axi_ARVALID)begin
                    mem_op_index              <= mem_op_index + 1'b1;
                    //lat_timer_sum             <= lat_timer_sum + 1'b1;
                    lat_timer                 <= 16'b0;    //reset the timer.
                    guard_ARVALID             <= 1'b0;
                    state                     <= RD_LAT_RESP; 
                end
                else begin
                    lat_timer                 <= lat_timer;
                    state                     <= RD_LAT_VALID;
                end
                //end
            end

            //This state indicates the beginning of sample a, but read nothing... 
            RD_LAT_RESP:
            begin
                //lat_timer_sum             <= lat_timer_sum + 1'b1;
                lat_timer                 <= lat_timer + 16'b1; //inc when waiting for resp and last.
                if ( m_axi_RLAST&m_axi_RVALID ) //(m_axi_RRESP == 2'b0) (m_axi_RRESP[1] == 1'b0)&
                begin
                    lat_timer_valid       <= 1'b1;
                    if (mem_op_index == num_mem_ops_r)
                        state             <= RD_END; 
                    else
                        state             <= RD_LAT_VALID; //RD_LATENCY;
                end 
            end

            RD_TH_VALID: //For the sample. 
            begin
                guard_ARVALID             <= 1'b1;
                m_axi_ARADDR              <= init_addr + (offset_addr&work_group_size_minus_1);    
                if (m_axi_ARREADY & m_axi_ARVALID)//when ARREADY is 1, increase the address. US embassy 
                begin
                    offset_addr           <= offset_addr + stride; 
                    mem_op_index          <= mem_op_index + 1'b1;
                    if (mem_op_index >= num_mem_ops_minus_1)begin
                        state             <= RD_TH_RESP; 
                        guard_ARVALID             <= 1'b0;
                    end
                    else
                        state             <= RD_TH_VALID; 
                end
            end
               
            ////To start testing throughput//// 
            RD_TH_RESP:
            begin
                if (read_resp_lasts == num_mem_ops)  //received enough responses.
                begin
                    state                 <= RD_END;           
                end
            end

            RD_END: 
            begin
                //lat_timer_sum             <= lat_timer_sum;
                end_of_exec               <= 1'b1; 
                state                     <= RD_IDLE; //end of one sample...
            end
            
            default:
                state                     <= RD_IDLE;             
        endcase 
         // else kill
    end 
end

//ila_rd_engine inst_bebug_rd_engine (
//   .clk (clk),

//   .probe0  (m_axi_ARVALID   ),
//   .probe1  (m_axi_ARADDR    ),
//   .probe2  (m_axi_ARLEN     ),
//   .probe3  (m_axi_ARSIZE    ),
//   .probe4  (m_axi_ARREADY   ),
//   .probe5  (started         ),
//   .probe6  (m_axi_RVALID    ),
//   .probe7  (m_axi_RLAST     ),
//   .probe8  (num_mem_ops_r   ),
//   .probe9  (stride          ),
//   .probe10 (isRdLatencyTest ),
//   .probe11 (mem_op_index    ),
//   .probe12 (read_resp_lasts ),
//   .probe13 (guard_ARVALID   ),
//   .probe14 (lt_params_r     ),
//   .probe15 (mem_burst_size  ),
//   .probe16 (state           ),
//   .probe17 (m_axi_RRESP     )  
//);


endmodule