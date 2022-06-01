/*
 * Copyright (c) 2020, Systems Group, ETH Zurich
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
 
 `include "example_module.vh"
 
 module matrix_multiply (
    input wire         clk,
    input wire         rstn,

    //DMA Commands
    axi_mm.master               matrix0, 
    axi_mm.master               matrix1,

    // control reg
    input wire[15:0][31:0]      control_reg,
    output wire[15:0][31:0]      status_reg     
 
 );
/////////////////Parameters for arvalid and araddr////////////////
    reg                         start,start_r;
    reg [31:0]                  all_counter;

    always@(posedge clk)begin
        all_counter             <= control_reg[6];
        start                   <= control_reg[7][0];
        start_r                 <= start;
    end

 matrix_multiply0 matrix_multiply0_inst( 
    //user clock input
    .clk(clk),
    .rstn(rstn),
    
    //dma memory streams
    .matrix(matrix0),

    //control reg
    .start(start & ~start_r),
    .control_reg(control_reg[7:0]),
    .status_reg()

    
    );

    matrix_multiply1 matrix_multiply1_inst( 
        //user clock input
        .clk(clk),
        .rstn(rstn),
        
        //dma memory streams
        .matrix(matrix1),
    
        //control reg
        .start(start & ~start_r),
        .control_reg(control_reg[15:8]),
        .status_reg()
    
        
        );    

    //////////////////////////debug///////////////////////


//////////////////////////////////matrix_counter//////////////////////////////
    
    
        reg[31:0]                           matrix_read_cmd0_counter;
    
        always@(posedge clk)begin
            if(~rstn)begin
                matrix_read_cmd0_counter          <= 1'b0;
            end
            else if(matrix0.arvalid && matrix0.arready)begin
                matrix_read_cmd0_counter          <= matrix_read_cmd0_counter + 1'b1;
            end
            else begin
                matrix_read_cmd0_counter          <= matrix_read_cmd0_counter;
            end
        end
    
        reg[31:0]                           matrix_read_data0_counter;
    
        always@(posedge clk)begin
            if(~rstn)begin
                matrix_read_data0_counter          <= 1'b0;
            end
            else if(matrix0.rvalid && matrix0.rready)begin
                matrix_read_data0_counter          <= matrix_read_data0_counter + 1'b1;
            end
            else begin
                matrix_read_data0_counter          <= matrix_read_data0_counter;
            end
        end
    
        reg[31:0]                           matrix_read_cmd1_counter;
    
        always@(posedge clk)begin
            if(~rstn)begin
                matrix_read_cmd1_counter          <= 1'b0;
            end
            else if(matrix1.arvalid && matrix1.arready)begin
                matrix_read_cmd1_counter          <= matrix_read_cmd1_counter + 1'b1;
            end
            else begin
                matrix_read_cmd1_counter          <= matrix_read_cmd1_counter;
            end
        end
    
        reg[31:0]                           matrix_read_data1_counter;
    
        always@(posedge clk)begin
            if(~rstn)begin
                matrix_read_data1_counter          <= 1'b0;
            end
            else if(matrix1.rvalid && matrix1.rready)begin
                matrix_read_data1_counter          <= matrix_read_data1_counter + 1'b1;
            end
            else begin
                matrix_read_data1_counter          <= matrix_read_data1_counter;
            end
        end  
    
        ////////////////////////////mpi time
    
        reg[31:0]                           time_counter;
        reg                                 time_en;
    
        always@(posedge clk)begin
            if(~rstn)begin
                time_en                     <= 1'b0;
            end
            else if(start & ~start_r)begin
                time_en                     <= 1'b1;
            end
            else if((all_counter == matrix_read_data0_counter))begin
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
 
        reg[31:0]                           time_counter1;
        reg                                 time_en1;
    
        always@(posedge clk)begin
            if(~rstn)begin
                time_en1                     <= 1'b0;
            end
            else if(start & ~start_r)begin
                time_en1                     <= 1'b1;
            end
            else if((all_counter == matrix_read_data1_counter))begin
                time_en1                     <= 1'b0;
            end        
            else begin
                time_en1                     <= time_en1;
            end
        end     
    
        always@(posedge clk)begin
            if(~rstn)begin
                time_counter1                <= 1'b0;
            end
            else if(time_en1)begin
                time_counter1                <= time_counter1 + 1'b1;
            end
            else begin
                time_counter1                <= time_counter1;
            end
        end 
 
    
        assign status_reg[0] = matrix_read_cmd0_counter;
        assign status_reg[1] = matrix_read_data0_counter;
        assign status_reg[2] = matrix_read_cmd1_counter;
        assign status_reg[3] = matrix_read_data1_counter;    
        assign status_reg[4] = time_counter;
        assign status_reg[5] = time_counter1;









 endmodule
 `default_nettype wire
 