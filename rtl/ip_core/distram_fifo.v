/*
 * Copyright 2017 - 2018, Zeke Wang, Systems Group, ETH Zurich
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


module  distram_fifo #(
    parameter FIFO_WIDTH                = 64,
    parameter FIFO_DEPTH_BITS           = 5,
    parameter FIFO_ALMOSTFULL_THRESHOLD = 2**FIFO_DEPTH_BITS - 12
) (
    input  wire                         clk,
    input  wire                         reset_n,
    //////////////////////wr////////////////////////////
    input  wire                         we,              // input   write enable
    input  wire      [FIFO_WIDTH - 1:0] din,             // input   write data with configurable width
    output reg                          almostfull,      // output  configurable programmable full/ almost full    
    //////////////////////rd////////////////////////////
    input  wire                         re,              // input   read enable    
    output reg                          valid,           // dout valid
    output wire      [FIFO_WIDTH - 1:0] dout,            // output  read data with configurable width    
    output wire                         empty,           // output  FIFO empty

    output reg  [FIFO_DEPTH_BITS - 1:0] count            // output  FIFOcount
);

    //Warning, when only item is in the fifo, the signal empty is not asserted in the next cycle. I am not sure 
    //whether the read side can get the correct data. Maybe the rd signal should be asserted in the next cycle after identying the deassertion of empty. .
    (* keep = "true" , max_fanout = 200 *)reg  [FIFO_DEPTH_BITS - 1:0]        wr_addr; 
    (* keep = "true" , max_fanout = 200 *)reg  [FIFO_DEPTH_BITS - 1:0]        rd_addr;
    //reg  [FIFO_DEPTH_BITS    :0]        count;

    wire rd_valid = re & (~empty);


    distram_2port  #(.DATA_WIDTH      (FIFO_WIDTH),    
                     .DEPTH_BIT_WIDTH (FIFO_DEPTH_BITS)
    ) inst_distram_fifo (
    .clock     (clk    ),
    .data      (din    ),
    .wraddress (wr_addr),
    .wren      (we     ),
    .rdaddress (rd_addr),
    .q         (dout   )
    );        



    //Output for reading
    assign empty = (wr_addr == rd_addr);
    reg valid_r1, valid_r2;
    always @(posedge clk) //  or negedge reset_n
    begin
        if (~reset_n) 
        begin
            valid_r1             <= 1'b0;
            valid                <= 1'b0;
        end
        else
        begin
            valid_r1             <= rd_valid;
            valid                <= valid_r1;
        end
    end
/////////////////Need to tune the latency here...////////////////////

    //Output for writing
    always @(posedge clk) //  or negedge reset_n
    begin
        if (~reset_n) 
            almostfull           <= 1'b0;
        else
            almostfull           <= (count > FIFO_ALMOSTFULL_THRESHOLD);
    end

    //counter
    always @(posedge clk) //  or negedge reset_n
    begin
        if (~reset_n) 
            count                <= { FIFO_DEPTH_BITS{1'b0} };
        else
        begin
            if (rd_valid & (~we))
                count            <= count - 1'b1;
            else if ( we & (~rd_valid) )
                count            <= count + 1'b1;
            //else do nothing.
        end
    end

    // wr_addr        
    always @(posedge clk)//  or negedge reset_n 
    begin
        if (~reset_n) 
            wr_addr             <= 1'b0;
        else
        begin
            if (we)
                wr_addr         <= wr_addr + 1'b1;
        end
    end

   // rd_addr        
    always @(posedge clk) //  or negedge reset_n
    begin
        if (~reset_n) 
            rd_addr             <= 1'b0;
        else
        begin
            if (rd_valid)
                rd_addr         <= rd_addr + 1'b1;
        end
    end

endmodule


