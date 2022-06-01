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
 
 module matrix_multiply1 (
    input wire         clk,
    input wire         rstn,

    //DMA Commands
    axi_mm.master               matrix, 


    // control reg
    input wire                  start,
    input wire[7:0][31:0]      control_reg,
    output wire[15:0][31:0]      status_reg     
 
 );
/////////////////Parameters for arvalid and araddr////////////////
    reg             [31:0] init_addr      ;
    reg             [31:0] initial_addr   ;
    reg             [31:0] num_ones_ops,num_all_ops,num_sec_ops    ;
    reg             [31:0] mem_burst_size,burst,dimension ;
    reg             [31:0] work_group_size;
    reg             [31:0] stride         ;
    reg                    isRdLatencyTest;
    reg             [31:0] one_op_index,sec_op_index,all_op_index   ;
    reg             [31:0] offset_addr    ;
    reg             [31:0] work_group_size_minus_1;
    reg             [31:0] num_mem_ops_r, num_mem_ops_minus_1;
    
    reg             [31:0] read_resp_lasts;
    
    
    
    reg                    started;   //one cycle delay from started...
    reg                    guard_ARVALID;    
    reg             [31:0] m_axi_ARADDR;
    
    reg                     flag;

    always@(posedge clk)begin
        initial_addr                <= control_reg[0];
        num_ones_ops                <= control_reg[1];
        num_sec_ops                 <= control_reg[2];
        num_all_ops                 <= control_reg[3];
        burst                       <= control_reg[4];
        dimension                   <= control_reg[5];
        work_group_size             <= control_reg[6];
        flag                        <= control_reg[7][0];
    end

    always @(posedge clk) 
    begin
        matrix.arid     <= 0;
        matrix.arlen    <= (burst>>5)-8'b1;
        matrix.arsize   <= 3'b101; //just for 256-bit or 512-bit.
        matrix.arburst  <= 2'b01;   // INC, not FIXED (00)
        matrix.arlock   <= 2'b00;   // Normal memory operation
        matrix.arcache  <= 4'b0000; // 4'b0011: Normal, non-cacheable, modifiable, bufferable (Xilinx recommends)
        matrix.arprot   <= 3'b010;  // 3'b000: Normal, secure, data
        matrix.arqos    <= 4'b0000; // Not participating in any Qos schem, a higher value indicates a higher priority transaction
        matrix.arregion <= 4'b0000; // Region indicator, default to 0     

        matrix.araddr <= m_axi_ARADDR;
        matrix.rready <= 1'b1;    //
   
        matrix.arvalid <= guard_ARVALID;//         
    end




    always @(posedge clk) 
    begin
        if(~rstn)
            started  <= 1'b0;
        else 
            started  <= start;   //1'b0;
    end
    
//    assign matrix.araddr = m_axi_ARADDR;
//    assign matrix.rready = 1'b1;    //
    
//    assign matrix.arvalid = guard_ARVALID;//  

    always @(posedge clk) 
    begin
        if (started)
            read_resp_lasts <= 64'b0;
        else
        begin
            read_resp_lasts <= read_resp_lasts + (matrix.rlast&matrix.rvalid); //(m_axi_RRESP == 2'b0) m_axi_RRESP[1] == 1'b0)&
        end        
        //all_mem_ops_are_recvd    <= lt_params;
    end
    
    
    
    /////////////////One FSM to decide valid/addr////////////////
    reg              [3:0] state;
    localparam [3:0]
            RD_IDLE          = 4'b0000, //begining state
            RD_STARTED       = 4'b0001, //receiving the parameters
            RD_NEXTED        = 4'b0010,
            RD_TH_VALID      = 4'b0101,
            RD_TH_RESP       = 4'b1000,
            RD_END           = 4'b1001;
    
    always@(posedge clk) 
    begin
        if(~rstn) 
            state                             <= RD_IDLE;
        else 
        begin
            // end_of_exec                       <= 1'b0;
            guard_ARVALID                     <= 1'b0;
            case (state)
                /*This state is the beginning of FSM... wait for started...*/
                RD_IDLE: 
                begin    
                    if(started)  //stage the parameter when started is 1.
                    begin
                        work_group_size_minus_1   <= work_group_size - 1'b1;
                        stride                <= dimension << 2;
                        all_op_index          <= 32'b0;
                        one_op_index          <= 32'b0;
                        sec_op_index              <= 32'b0;
                        offset_addr           <= 0;
                        mem_burst_size        <= burst;//[127 : 96];
                        init_addr             <= initial_addr;//[ADDR_WIDTH+127:128]; //ADDR_WIDTH<48.
                        state                 <= RD_TH_VALID;
                    end
                    else begin
                        state                 <= RD_IDLE;
                    end  
                end
    
                /* This state initilizes the parameters...*/
                RD_STARTED: 
                begin
                    work_group_size_minus_1   <= work_group_size - 1'b1;
                    one_op_index              <= 32'b0;
                    sec_op_index              <= 32'b0;
                    init_addr                 <= initial_addr;//initial_addr;
                    offset_addr               <= 0;
                    state                     <= RD_TH_VALID;
                end

                RD_NEXTED: 
                begin
                    work_group_size_minus_1   <= work_group_size - 1'b1;
                    one_op_index              <= 32'b0;
                    init_addr                 <= flag ? (init_addr + burst) : initial_addr;
                    offset_addr               <= 0;
                    state                     <= RD_TH_VALID;
                end

    
                RD_TH_VALID: //For the sample. 
                begin
                    guard_ARVALID             <= 1'b1;
                    m_axi_ARADDR              <= init_addr + (offset_addr&work_group_size_minus_1);    
                    if (matrix.arready & matrix.arvalid)//when ARREADY is 1, increase the address. US embassy 
                    begin
                        offset_addr           <= offset_addr + stride; 
                        m_axi_ARADDR          <= init_addr + ((offset_addr + stride)&work_group_size_minus_1); 
                        one_op_index          <= one_op_index + 1'b1;
                        all_op_index          <= all_op_index + 1'b1;
                        sec_op_index          <= sec_op_index + 1'b1;
                        if (one_op_index >= num_ones_ops-1)begin
                            state             <= RD_NEXTED; 
                            guard_ARVALID     <= 1'b0;
                            if(sec_op_index >= num_sec_ops-1)begin
                                state         <= RD_STARTED;
                                if(all_op_index >= num_all_ops-1)begin
                                    state     <= RD_TH_RESP;
                                end
                            end
                        end
                        else
                            state             <= RD_TH_VALID; 
                    end
                end
                   
                ////To start testing throughput//// 
                RD_TH_RESP:
                begin
                    if (read_resp_lasts == num_all_ops)  //received enough responses.
                    begin
                        state                 <= RD_END;           
                    end
                end
    
                RD_END: 
                begin
                    //lat_timer_sum             <= lat_timer_sum;
                    // end_of_exec               <= 1'b1; 
                    state                     <= RD_IDLE; //end of one sample...
                end
                
                default:
                    state                     <= RD_IDLE;             
            endcase 
             // else kill
        end 
    end



    

 endmodule
 `default_nettype wire
 