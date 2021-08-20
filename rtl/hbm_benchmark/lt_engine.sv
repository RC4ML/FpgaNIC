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
 // The objective of lt_engine is to benchmark DDR/HBM on Xilinx FPGAs.
 // The lt_engine module contains one read and one write modules. 


//`include "sgd_defines.vh"
 module lt_engine #(
    //parameter NUM_CHANNELS    = 32,
    parameter ENGINE_ID       = 0  ,
    parameter ADDR_WIDTH      = 33 ,  //8G-->33 bits
    parameter DATA_WIDTH      = 256, 
    parameter PARAMS_BITS     = 256, 
    parameter ID_WIDTH        = 5    //fixme,
)(
    input                      clk,   //should be 450MHz, 
    input                      rst_n, //negative reset,   
    //--------------------Begin/Stop-----------------------------//
    input                      start_wr,
    input                      start_rd,    
    output                     end_wr,
    output               [63:0]lat_timer_sum_wr,
    output                     end_rd          ,
    output               [63:0]lat_timer_sum_rd,      
    output                     lat_timer_valid , //log down lat_timer when lat_timer_valid is 1. 
    output              [15:0] lat_timer       ,

    //--------------------Parameters-----------------------------//
    input                      ld_params_wr,
    input                      ld_params_rd,
    input  [2*PARAMS_BITS-1:0] lt_params,
//    input                [5:0] engine_id,

    //------------------Output: AXI3 Interface---------------//
    //Write address (output)
    output                     m_axi_AWVALID , //wr address valid
    output  [ADDR_WIDTH - 1:0] m_axi_AWADDR  , //wr byte address
    output  [  ID_WIDTH - 1:0] m_axi_AWID    , //wr address id
    output               [7:0] m_axi_AWLEN   , //wr burst=awlen+1,
    output               [2:0] m_axi_AWSIZE  , //wr 3'b101, 32B
    output               [1:0] m_axi_AWBURST , //wr burst type: 01 (INC), 00 (FIXED)
    output               [1:0] m_axi_AWLOCK  , //wr no
    output               [3:0] m_axi_AWCACHE , //wr no
    output               [2:0] m_axi_AWPROT  , //wr no
    output               [3:0] m_axi_AWQOS   , //wr no
    output               [3:0] m_axi_AWREGION, //wr no
    input                      m_axi_AWREADY , //wr ready to accept address.

    //Write data (output)  
    output                     m_axi_WVALID  , //wr data valid
    output  [DATA_WIDTH - 1:0] m_axi_WDATA   , //wr data
    output  [DATA_WIDTH/8-1:0] m_axi_WSTRB   , //wr data strob
    output                     m_axi_WLAST   , //wr last beat in a burst
    output    [ID_WIDTH - 1:0] m_axi_WID     , //wr data id
    input                      m_axi_WREADY  , //wr ready to accept data

    //Write response (input)  
    input                      m_axi_BVALID  , 
    input                [1:0] m_axi_BRESP   ,
    input     [ID_WIDTH - 1:0] m_axi_BID     ,
    output                     m_axi_BREADY  ,

    //Read Address (Output)  
    output                     m_axi_ARVALID , //rd address valid
    output  [ADDR_WIDTH - 1:0] m_axi_ARADDR  , //rd byte address
    output    [ID_WIDTH - 1:0] m_axi_ARID    , //rd address id
    output               [7:0] m_axi_ARLEN   , //rd burst=awlen+1,
    output               [2:0] m_axi_ARSIZE  , //rd 3'b101, 32B
    output               [1:0] m_axi_ARBURST , //rd burst type: 01 (INC), 00 (FIXED)
    output               [1:0] m_axi_ARLOCK  , //rd no
    output               [3:0] m_axi_ARCACHE , //rd no
    output               [2:0] m_axi_ARPROT  , //rd no
    output               [3:0] m_axi_ARQOS   , //rd no
    output               [3:0] m_axi_ARREGION, //rd no
    input                      m_axi_ARREADY , //rd ready to accept address.

    //Read Data (input)
    input                      m_axi_RVALID  , //rd data valid
    input   [DATA_WIDTH - 1:0] m_axi_RDATA   , //rd data 
    input                      m_axi_RLAST   , //rd data last
    input     [ID_WIDTH - 1:0] m_axi_RID     , //rd data id
    input                [1:0] m_axi_RRESP   , //rd data status. 
    output                     m_axi_RREADY,

    output               [2:0] dbg_state_wr,
    output               [3:0] dbg_state_rd      
);

assign dbg_state_wr = 0;
assign dbg_state_rd = 0;

//reg                     start_wr;
//reg                     start_rd;
reg                     start_wr_reg, start_rd_reg;
reg                     ld_params_wr_reg, ld_params_rd_reg;
reg [2*PARAMS_BITS-1:0] lt_params_reg;
reg [2*PARAMS_BITS-1:0] lt_params_staged;

always @(posedge clk) 
begin
    start_wr_reg    <= start_wr;
    start_rd_reg    <= start_rd;
    ld_params_wr_reg<= ld_params_wr;
    ld_params_rd_reg<= ld_params_rd;    
    lt_params_reg   <= lt_params;

    if ( ld_params_wr_reg & (lt_params_reg[225+:5]==ENGINE_ID) )// (lt_params[233:226] == ENGINE_ID))
        lt_params_staged[PARAMS_BITS-1:0]      <= lt_params_reg[PARAMS_BITS-1:0]; 

    if ( ld_params_rd_reg & (lt_params[(PARAMS_BITS+225)+:5]==ENGINE_ID) )// (lt_params[233:226] == ENGINE_ID))
        lt_params_staged[2*PARAMS_BITS-1:PARAMS_BITS]      <= lt_params_reg[2*PARAMS_BITS-1:PARAMS_BITS];       
end

/*
always @(posedge clk) 
begin
    if ( ld_params_wr & (lt_params[225+:5]==ENGINE_ID) )// (lt_params[233:226] == ENGINE_ID))
        lt_params_staged[PARAMS_BITS-1:0]      <= lt_params[PARAMS_BITS-1:0]; 

    if ( ld_params_rd & (lt_params[(PARAMS_BITS+225)+:5]==ENGINE_ID) )// (lt_params[233:226] == ENGINE_ID))
        lt_params_staged[2*PARAMS_BITS-1:PARAMS_BITS]      <= lt_params[2*PARAMS_BITS-1:PARAMS_BITS];       
end
*/
/*
ila_lt_params inst_ila_lt_params (
    .clk (clk),

    .probe0  (ld_params_wr_reg),
    .probe1  (ld_params_rd_reg),
    .probe2  (lt_params_staged) 
); */
/*
always @(posedge clk) 
begin
    if ( lt_params_valid & (engine_id==ENGINE_ID) )// (lt_params[233:226] == ENGINE_ID))
        lt_params_staged      <= lt_params;        
end

always @(posedge clk) 
begin
    if (~rst_n) 
    begin
        start_wr              <= 1'b0;
        start_rd              <= 1'b0;
    end
    else
    begin
        start_wr              <= start & lt_params_staged[225]; //193-bit is 1--->write
        start_rd              <= start & lt_params_staged[225+PARAMS_BITS];
    end        
end
*/

wr_engine #(
    //.ENGINE_ID        (ENGINE_ID  ),
    .ADDR_WIDTH       (ADDR_WIDTH ),  // 8G-->33 bits
    .DATA_WIDTH       (DATA_WIDTH ),  // 512-bit for DDR4
    .PARAMS_BITS      (PARAMS_BITS),  // parameter bits from PCIe
    .ID_WIDTH         (ID_WIDTH   )   //fixme,
)inst_wr_engine(
    .clk              (clk  ),   //should be 450MHz, 
    .rst_n            (rst_n), //negative reset,   
    //---------------------Begin/Stop-----------------------------//
    .start            (start_wr_reg    ),
    .end_of_exec      (end_wr          ),
    .lat_timer_sum    (lat_timer_sum_wr),

    //---------------------Parameters-----------------------------//
    .lt_params        ( lt_params_staged[PARAMS_BITS-1:0] ),

    .m_axi_AWVALID    (m_axi_AWVALID ), //wr address valid
    .m_axi_AWADDR     (m_axi_AWADDR  ), //wr byte address
    .m_axi_AWID       (m_axi_AWID    ), //wr address id
    .m_axi_AWLEN      (m_axi_AWLEN   ), //wr burst=awlen+1,
    .m_axi_AWSIZE     (m_axi_AWSIZE  ), //wr 3'b101, 32B
    .m_axi_AWBURST    (m_axi_AWBURST ), //wr burst type: 01 (INC), 00 (FIXED)
    .m_axi_AWLOCK     (m_axi_AWLOCK  ), //wr no
    .m_axi_AWCACHE    (m_axi_AWCACHE ), //wr no
    .m_axi_AWPROT     (m_axi_AWPROT  ), //wr no
    .m_axi_AWQOS      (m_axi_AWQOS   ), //wr no
    .m_axi_AWREGION   (m_axi_AWREGION), //wr no
    .m_axi_AWREADY    (m_axi_AWREADY ), //wr ready to accept address.

    //Write data (output)  
    .m_axi_WVALID     (m_axi_WVALID  ), //wr data valid
    .m_axi_WDATA      (m_axi_WDATA   ), //wr data
    .m_axi_WSTRB      (m_axi_WSTRB   ), //wr data strob
    .m_axi_WLAST      (m_axi_WLAST   ), //wr last beat in a burst
    .m_axi_WID        (m_axi_WID     ), //wr data id
    .m_axi_WREADY     (m_axi_WREADY  ), //wr ready to accept data

    //Write response (input)  
    .m_axi_BVALID     (m_axi_BVALID  ), 
    .m_axi_BRESP      (m_axi_BRESP   ),
    .m_axi_BID        (m_axi_BID     ),
    .m_axi_BREADY     (m_axi_BREADY  )
);

rd_engine #(
    //.ENGINE_ID        (ENGINE_ID  ),
    .ADDR_WIDTH       (ADDR_WIDTH ),  // 8G-->33 bits
    .DATA_WIDTH       (DATA_WIDTH ),  // 512-bit for DDR4
    .PARAMS_BITS      (PARAMS_BITS),  // parameter bits from PCIe
    .ID_WIDTH         (ID_WIDTH   )   //fixme,
)inst_rd_engine(
    .clk              (clk            ), //should be 450MHz, 
    .rst_n            (rst_n          ), //negative reset,

    //---------------------Begin/Stop-----------------------------//
    .start            (start_rd_reg    ),
    .end_of_exec      (end_rd          ),
    .lat_timer_sum    (lat_timer_sum_rd),
    .lat_timer_valid  (lat_timer_valid ), //log down lat_timer when lat_timer_valid is 1. 
    .lat_timer        (lat_timer       ),

    //---------------------Parameters-----------------------------//
    .lt_params        ( lt_params_staged[2*PARAMS_BITS-1:PARAMS_BITS] ),

    //Read Address (Output)  
    .m_axi_ARVALID    (m_axi_ARVALID  ), //rd address valid
    .m_axi_ARADDR     (m_axi_ARADDR   ), //rd byte address
    .m_axi_ARID       (m_axi_ARID     ), //rd address id
    .m_axi_ARLEN      (m_axi_ARLEN    ), //rd burst=awlen+1,
    .m_axi_ARSIZE     (m_axi_ARSIZE   ), //rd 3'b101, 32B
    .m_axi_ARBURST    (m_axi_ARBURST  ), //rd burst type: 01 (INC), 00 (FIXED)
    .m_axi_ARLOCK     (m_axi_ARLOCK   ), //rd no
    .m_axi_ARCACHE    (m_axi_ARCACHE  ), //rd no
    .m_axi_ARPROT     (m_axi_ARPROT   ), //rd no
    .m_axi_ARQOS      (m_axi_ARQOS    ), //rd no
    .m_axi_ARREGION   (m_axi_ARREGION ), //rd no
    .m_axi_ARREADY    (m_axi_ARREADY  ), //rd ready to accept address.

    //Read Data (input)
    .m_axi_RVALID     (m_axi_RVALID   ), //rd data valid
    .m_axi_RDATA      (m_axi_RDATA    ), //rd data 
    .m_axi_RLAST      (m_axi_RLAST    ), //rd data last
    .m_axi_RID        (m_axi_RID      ), //rd data id
    .m_axi_RRESP      (m_axi_RRESP    ), //rd data status. 
    .m_axi_RREADY     (m_axi_RREADY   )
);


endmodule