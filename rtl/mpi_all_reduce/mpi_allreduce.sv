`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2020/02/20 21:50:13
// Design Name: 
// Module Name: hbm_driver
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
`include"example_module.vh"

module mpi_allreduce#(
    parameter   MAX_CLIENT      = 8
)( 

    //user clock input
    input wire                  clk,
    input wire                  rstn,
	
    //ddr memory streams
    axis_mem_cmd.master    		m_axis_dma_write_cmd,
    axi_stream.master   		m_axis_dma_write_data,
    axis_mem_cmd.master    		m_axis_dma_read_cmd,
    axi_stream.slave    		s_axis_dma_read_data,    
    //ddr memory streams
    axi_mm.master    		    m_axis_mem_read,
    axi_mm.master    		    m_axis_mem_write,     
    
    //tcp app interface streams
    axis_meta.master            m_axis_tcp_tx_meta,
    axi_stream.master           m_axis_tcp_tx_data,
    axis_meta.slave             s_axis_tcp_tx_status, 

    axis_meta.slave             s_axis_tcp_rx_meta,
    axi_stream.slave            s_axis_tcp_rx_data,

    //control reg
    input wire[15:0][31:0]      control_reg,
    output wire[3:0][31:0]      status_reg
	
	);
    //control reg
    reg[15:0]                   session_id;
    reg[31:0]                   client_id;
    reg[31:0]                   client_num;
    reg[31:0]                   client_length;
    reg[63:0]                   dma_read_base_addr;
    reg[63:0]                   dma_write_base_addr;
    reg[7:0]                    token_mul,token_divide_num;
    reg                         start,start_r,start_mpi;
    wire[15:0]                  recv_session_id;
    
    wire                        mpi_reduce_done;
    wire                        send_enable_empty;
    wire                        send_enable_rd_en;
    wire                        send_start_empty;
    wire                        send_start_rd_en;
    wire                        send_req_empty;
    wire                        send_req_rd_en;    
    wire                        start_cal_flag;      

    wire                        send_done;  


    always@(posedge clk)begin
        session_id                  <= control_reg[0][15:0];
        client_id                   <= control_reg[1];
        client_num                  <= control_reg[2];
        client_length               <= control_reg[3];
        start                       <= control_reg[4][0];
        start_r                     <= start;
        start_mpi                   <= start & ~start_r; 
        dma_read_base_addr          <= {control_reg[6],control_reg[5]};
        dma_write_base_addr         <= {control_reg[8],control_reg[7]};
        token_divide_num            <= control_reg[9];
        token_mul                   <= control_reg[10];
    end


    mpi_reduce_cal#(
        .MAX_CLIENT                 (8)
    )mpi_reduce_cal( 
    
        //user clock input
        .clk                        (clk),
        .rstn                       (rstn),
        
        //ddr memory streams
        .m_axis_mem_write           (m_axis_mem_write),
        //dma 
        .m_axis_dma_write_cmd       (m_axis_dma_write_cmd),
        .m_axis_dma_write_data      (m_axis_dma_write_data),
        .m_axis_dma_read_cmd        (m_axis_dma_read_cmd),
        .s_axis_dma_read_data       (s_axis_dma_read_data),    
      
        //tcp Commands     
        .s_axis_tcp_rx_meta         (s_axis_tcp_rx_meta),
        .s_axis_tcp_rx_data         (s_axis_tcp_rx_data),
    
        //control reg
        .session_id                 (session_id),
        .client_id                  (client_id),
        .client_num                 (client_num),
        .client_length              (client_length),
        .recv_start                 (start_mpi),
        .dma_read_base_addr         (dma_read_base_addr),
        .dma_write_base_addr        (dma_write_base_addr),
        .recv_session_id            (recv_session_id),
    
        .send_enable_empty          (send_enable_empty),
        .send_enable_rd_en          (send_enable_rd_en),
        .send_start_empty           (send_start_empty),
        .send_start_rd_en           (send_start_rd_en), 
        .send_req_empty             (send_req_empty),
        .send_req_rd_en             (send_req_rd_en),        
        .start_cal_flag             (start_cal_flag),       
        .mpi_reduce_done            (mpi_reduce_done)
        
    );

    mpi_reduce_tx#(
        .MAX_CLIENT                 (8)
    )mpi_reduce_tx( 
    
        //user clock input
        .clk                        (clk),
        .rstn                       (rstn),
        
        //ddr memory streams
        .m_axis_mem_read            (m_axis_mem_read),   
      
        //tcp app interface streams
        .m_axis_tcp_tx_meta         (m_axis_tcp_tx_meta),
        .m_axis_tcp_tx_data         (m_axis_tcp_tx_data),
        .s_axis_tcp_tx_status       (s_axis_tcp_tx_status), 
    
        //control reg
        .session_id                 (session_id),
        .client_id                  (client_id),
        .client_num                 (client_num),
        .client_length              (client_length),
        .token_mul                  (token_mul),
        .token_divide_num           (token_divide_num),        
        .start                      (start_mpi),
        .send_done                  (send_done),
        .recv_session_id            (recv_session_id),
    
        .send_enable_empty          (send_enable_empty),
        .send_enable_rd_en          (send_enable_rd_en),
        .send_req_empty             (send_req_empty),
        .send_req_rd_en             (send_req_rd_en),        
        .send_start_empty           (send_start_empty),
        .send_start_rd_en           (send_start_rd_en)    
        
        );









    assign status_reg[0][0] = send_done;
    assign status_reg[1][0] = mpi_reduce_done;
    assign status_reg[2][0] = start_cal_flag;
    assign status_reg[3][0] = send_enable_rd_en;

endmodule
