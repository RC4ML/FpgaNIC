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
 module mem_benchmark #(
    parameter N_MEM_INTF      = 32,
    parameter ENGINE_ID       = 0  ,
    parameter ADDR_WIDTH      = 33 ,  //8G-->33 bits
    parameter DATA_WIDTH      = 256, 
    parameter PARAMS_BITS     = 256, 
    parameter ID_WIDTH        = 5    //fixme,
)(
    input                               clk,   //should be 450MHz, 
    input                               rstn, //negative reset,   

    input     		                    hbm_clk,
    input                               hbm_rstn,

    axi_mm.master                       hbm_axi[0:31],    

/////////////////////////
    input[15:0][31:0]                   fpga_control,
    output[15:0][31:0]                  fpga_status    

);


reg  [N_MEM_INTF-1:0]                       start_wr;
reg  [N_MEM_INTF-1:0]                       start_rd; 

reg                                         start,start_r,start_rr;
wire [N_MEM_INTF-1:0]                       end_wr;
reg  [N_MEM_INTF-1:0]                       end_wr_o;
wire [N_MEM_INTF-1:0][63:0]                 lat_timer_sum_wr;
wire [N_MEM_INTF-1:0]                       end_rd          ;
reg  [N_MEM_INTF-1:0]                       end_rd_o;
wire [N_MEM_INTF-1:0][63:0]                 lat_timer_sum_rd;      
wire [N_MEM_INTF-1:0]                       lat_timer_valid ; //log down lat_timer when lat_timer_valid is 1. 
wire [N_MEM_INTF-1:0][15:0]                 lat_timer       ;

wire [31:0]                                 lat_timer_sum_wr_o;
wire [31:0]                                 lat_timer_sum_rd_o;      
reg                                         lat_timer_valid_o ; //log down lat_timer when lat_timer_valid is 1. 
reg  [15:0]                                 lat_timer_o       ;

//--------------------Parameters-----------------------------//
reg  [N_MEM_INTF-1:0]                       ld_params_wr;
reg  [N_MEM_INTF-1:0]                       ld_params_rd;
wire   [N_MEM_INTF-1:0][511:0]               lt_params;
reg  [  7:0]                               hbm_channel;
reg  [N_MEM_INTF-1:0]                      write_enable;
reg  [N_MEM_INTF-1:0]                      read_enable;


assign lat_timer_sum_wr_o = lat_timer_sum_wr[hbm_channel][31:0];
assign lat_timer_sum_rd_o = lat_timer_sum_rd[hbm_channel][31:0];

always @(posedge hbm_clk) begin
    if(~hbm_rstn)begin
        lat_timer_o                 <= 16'b0;
        lat_timer_valid_o           <= 1'b0;
    end
    else begin
        lat_timer_o                 <= lat_timer[hbm_channel] ;
        lat_timer_valid_o           <= lat_timer_valid[hbm_channel];
    end
end




always @(posedge hbm_clk)begin
  write_enable                 <= fpga_control[7];
  read_enable                  <= fpga_control[8];
  hbm_channel                  <= fpga_control[10][7:0];
  start                        <= fpga_control[11][0];
end


assign      fpga_status[0]             = end_wr_o;
assign      fpga_status[1]             = end_rd_o;
assign      fpga_status[2]             = lat_timer_sum_wr_o;
assign      fpga_status[3]             = lat_timer_sum_rd_o;




//generate end generate
genvar i;
// Instantiate engines
generate
for(i = 0; i < N_MEM_INTF; i++) 
begin
    
    always @(posedge hbm_clk) begin
        start_r                         <= start;
        start_rr                        <= start_r;              
    end
    
    
    always @(posedge hbm_clk) begin
        if(~hbm_rstn)begin
            start_wr[i]                 <= 1'b0;
            ld_params_wr[i]             <= 1'b1;
        end
        else if(write_enable[i])begin
            start_wr[i]                 <= (start_r & (~start_rr)) ;
            ld_params_wr[i]             <= 1'b1;
        end
        else begin
            start_wr[i]                 <= start_wr[i];
        end
    end

    always @(posedge hbm_clk) begin
        if(~hbm_rstn)begin
            start_rd[i]                 <= 1'b0;
            ld_params_rd[i]             <= 1'b1;
        end
        else if(read_enable[i])begin
            start_rd[i]                 <= (start_r & (~start_rr)) ;
            ld_params_rd[i]             <= 1'b1;
        end
        else begin
            start_rd[i]                 <= start_rd[i];
        end
    end


    always @(posedge hbm_clk) begin
        if(~hbm_rstn)begin
            end_wr_o[i]                 <= 1'b0;
        end
        else if(start_r & (~start_rr))begin
            end_wr_o[i]                 <= 1'b0;
        end
        else if(end_wr[i])begin
            end_wr_o[i]                 <= 1'b1 ;
        end
        else begin
            end_wr_o[i]                 <= end_wr_o[i];
        end
    end


    always @(posedge hbm_clk) begin
        if(~hbm_rstn)begin
            end_rd_o[i]                 <= 1'b0;
        end
        else if(start_r & (~start_rr))begin
            end_rd_o[i]                 <= 1'b0;
        end
        else if(end_rd[i])begin
            end_rd_o[i]                 <= 1'b1 ;
        end
        else begin
            end_rd_o[i]                 <= end_rd_o[i];
        end
    end
    



    lt_engine #(
        .ENGINE_ID        (i   ),
        .ADDR_WIDTH       (33  ),  // 8G-->33 bits
        .DATA_WIDTH       (256 ),  // 512-bit for DDR4
        .PARAMS_BITS      (256 ),  // parameter bits from PCIe
        .ID_WIDTH         (5   )   //,
    )inst_lt_engine(
    .clk              (hbm_clk),   //should be 450MHz, 
    .rst_n            (hbm_rstn), //negative reset,   
    //---------------------Begin/Stop-----------------------------//
    .start_wr         (start_wr[i]),
    .end_wr           (end_wr[i]),
    .lat_timer_sum_wr (lat_timer_sum_wr[i]),
    .start_rd         (start_rd[i]),
    .end_rd           (end_rd[i]),
    .lat_timer_sum_rd (lat_timer_sum_rd[i]),
    .lat_timer_valid  (lat_timer_valid[i]),
    .lat_timer        (lat_timer[i]),
    //---------------------Parameters-----------------------------//
    .ld_params_wr       (ld_params_wr[i]),
    .ld_params_rd       (ld_params_rd[i]),
    .lt_params        (lt_params[i]),


    .m_axi_AWADDR     (hbm_axi[i].awaddr  ), //wr byte address
    .m_axi_AWBURST    (hbm_axi[i].awburst ), //wr burst type: 01 (INC), 00 (FIXED)
    .m_axi_AWID       (hbm_axi[i].awid    ), //wr address id
    .m_axi_AWLEN      (hbm_axi[i].awlen   ), //wr burst=awlen+1,
    .m_axi_AWSIZE     (hbm_axi[i].awsize  ), //wr 3'b101, 32B
    .m_axi_AWVALID    (hbm_axi[i].awvalid ), //wr address valid
    .m_axi_AWREADY    (hbm_axi[i].awready ), //wr ready to accept address.
    .m_axi_AWLOCK     (), //wr no
    .m_axi_AWCACHE    (), //wr no
    .m_axi_AWPROT     (), //wr no
    .m_axi_AWQOS      (), //wr no
    .m_axi_AWREGION   (), //wr no

    //Write data (output)  
    .m_axi_WDATA      (hbm_axi[i].wdata  ), //wr data
    .m_axi_WLAST      (hbm_axi[i].wlast  ), //wr last beat in a burst
    .m_axi_WSTRB      (hbm_axi[i].wstrb  ), //wr data strob
    .m_axi_WVALID     (hbm_axi[i].wvalid ), //wr data valid
    .m_axi_WREADY     (hbm_axi[i].wready ), //wr ready to accept data
    .m_axi_WID        (), //wr data id

    //Write response (input)  
    .m_axi_BID        (hbm_axi[i].bid    ),
    .m_axi_BRESP      (hbm_axi[i].bresp  ),
    .m_axi_BVALID     (hbm_axi[i].bvalid ), 
    .m_axi_BREADY     (hbm_axi[i].bready ),

    //Read Address (Output)  
    .m_axi_ARADDR     (hbm_axi[i].araddr  ), //rd byte address
    .m_axi_ARBURST    (hbm_axi[i].arburst ), //rd burst type: 01 (INC), 00 (FIXED)
    .m_axi_ARID       (hbm_axi[i].arid    ), //rd address id
    .m_axi_ARLEN      (hbm_axi[i].arlen   ), //rd burst=awlen+1,
    .m_axi_ARSIZE     (hbm_axi[i].arsize  ), //rd 3'b101, 32B
    .m_axi_ARVALID    (hbm_axi[i].arvalid ), //rd address valid
    .m_axi_ARREADY    (hbm_axi[i].arready ), //rd ready to accept address.
    .m_axi_ARLOCK     (), //rd no
    .m_axi_ARCACHE    (), //rd no
    .m_axi_ARPROT     (), //rd no
    .m_axi_ARQOS      (), //rd no
    .m_axi_ARREGION   (), //rd no

    //Read Data (input)
    .m_axi_RDATA      (hbm_axi[i].rdata  ), //rd data 
    .m_axi_RLAST      (hbm_axi[i].rlast  ), //rd data last
    .m_axi_RID        (hbm_axi[i].rid    ), //rd data id
    .m_axi_RRESP      (hbm_axi[i].rresp  ), //rd data status. 
    .m_axi_RVALID     (hbm_axi[i].rvalid ), //rd data valid
    .m_axi_RREADY     (hbm_axi[i].rready )
);
    //assign hbm_axi_clk[i]   = hbm_axi[i].clk;
end
endgenerate

mem_benchmark_ctrl#(
	.N_MEM_INTF			      (32)
)mem_benchmark_ctrl_inst(
    .clk                            (clk),
    .rstn                           (rstn),

    .hbm_clk                        (hbm_clk),
    .hbm_rstn                       (hbm_rstn),

    .lt_params                      (lt_params),     

///////////hbm——test——reg
	.lat_timer_valid                (lat_timer_valid_o) , //log down lat_timer when lat_timer_valid is 1. 
    .lat_timer                      (lat_timer_o),
/////////////////////////
    .fpga_control                   (fpga_control),
    .fpga_status                    (fpga_status[4])

      
    );



endmodule