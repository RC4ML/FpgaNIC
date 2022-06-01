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

module mpi_reduce_control( 

    //user clock input
    input wire                  clk,
    input wire                  rstn,
	
    //dma memory streams
    axis_mem_cmd.master    		m_axis_dma_write_cmd,
    axi_stream.master   		m_axis_dma_write_data,
    axis_mem_cmd.master    		m_axis_dma_read_cmd,
    axi_stream.slave    		s_axis_dma_read_data,    
    //ddr memory streams
    axi_mm.master    		    m_axis_mem_read0,
    // axi_mm.master    		    m_axis_mem_read1,
    axi_mm.master    		    m_axis_mem_write0,    
    // axi_mm.master    		    m_axis_mem_write1,   
       

    //tcp app interface streams
    axis_meta.master            app_axis_tcp_tx_meta,
    axi_stream.master           app_axis_tcp_tx_data,
    axis_meta.slave             app_axis_tcp_tx_status,    

    axis_meta.slave             app_axis_tcp_rx_meta,
    axi_stream.slave            app_axis_tcp_rx_data,

    //control reg
    input wire[63:0]            dma_base_addr,
	input wire[15:0][31:0]		control_reg,
	output wire[31:0][31:0]		status_reg

	
	);


    
    axi_mm #(.DATA_WIDTH(512),.ADDR_WIDTH(34))       m_axis_mem_read();
    axi_mm #(.DATA_WIDTH(512),.ADDR_WIDTH(34))       m_axis_mem_write();
    reg                                 mpi_change;


///////////////////////////////////hbm/////////////////////////////////
    // assign m_axis_mem_read0.awid        = m_axis_mem_read.awid;
    // assign m_axis_mem_read0.awlen       = m_axis_mem_read.awlen; 
    // assign m_axis_mem_read0.awsize      = m_axis_mem_read.awsize;
    // assign m_axis_mem_read0.awburst     = m_axis_mem_read.awburst;
    // assign m_axis_mem_read0.awcache     = m_axis_mem_read.awcache;
    // assign m_axis_mem_read0.awprot      = m_axis_mem_read.awprot;
    // assign m_axis_mem_read0.awaddr      = m_axis_mem_read.awaddr >> 1;
    // assign m_axis_mem_read0.awlock      = m_axis_mem_read.awlock;
    // assign m_axis_mem_read0.awvalid     = m_axis_mem_read.awvalid & m_axis_mem_read.awready;
    // assign m_axis_mem_read0.awqos       = m_axis_mem_read.awqos;
    // assign m_axis_mem_read0.awregion    = m_axis_mem_read.awregion;                              
    // assign m_axis_mem_read0.wdata       = m_axis_mem_read.wdata[255:0];
    // assign m_axis_mem_read0.wstrb       = m_axis_mem_read.wstrb;
    // assign m_axis_mem_read0.wlast       = m_axis_mem_read.wlast;
    // assign m_axis_mem_read0.wvalid      = m_axis_mem_read.wvalid & m_axis_mem_read.wready;
    // assign m_axis_mem_read0.bready      = m_axis_mem_read.bready;
    // assign m_axis_mem_read0.arid        = m_axis_mem_read.arid;
    // assign m_axis_mem_read0.araddr      = m_axis_mem_read.araddr >> 1;
    // assign m_axis_mem_read0.arlen       = m_axis_mem_read.arlen;
    // assign m_axis_mem_read0.arsize      = m_axis_mem_read.arsize;
    // assign m_axis_mem_read0.arburst     = m_axis_mem_read.arburst;
    // assign m_axis_mem_read0.arcache     = m_axis_mem_read.arcache;
    // assign m_axis_mem_read0.arprot      = m_axis_mem_read.arprot;
    // assign m_axis_mem_read0.arlock      = m_axis_mem_read.arlock;
    // assign m_axis_mem_read0.arvalid     = m_axis_mem_read.arvalid & m_axis_mem_read.arready;
    // assign m_axis_mem_read0.arqos       = m_axis_mem_read.arqos;
    // assign m_axis_mem_read0.arregion    = m_axis_mem_read.arregion;   
    // assign m_axis_mem_read0.rready      = m_axis_mem_read.rready & m_axis_mem_read.rvalid;

    // assign m_axis_mem_read1.awid        = m_axis_mem_read.awid;
    // assign m_axis_mem_read1.awlen       = m_axis_mem_read.awlen; 
    // assign m_axis_mem_read1.awsize      = m_axis_mem_read.awsize;
    // assign m_axis_mem_read1.awburst     = m_axis_mem_read.awburst;
    // assign m_axis_mem_read1.awcache     = m_axis_mem_read.awcache;
    // assign m_axis_mem_read1.awprot      = m_axis_mem_read.awprot;
    // assign m_axis_mem_read1.awaddr      = m_axis_mem_read.awaddr >> 1 + 32'h1000_0000;
    // assign m_axis_mem_read1.awlock      = m_axis_mem_read.awlock;
    // assign m_axis_mem_read1.awvalid     = m_axis_mem_read.awvalid & m_axis_mem_read.awready;
    // assign m_axis_mem_read1.awqos       = m_axis_mem_read.awqos;
    // assign m_axis_mem_read1.awregion    = m_axis_mem_read.awregion;                              
    // assign m_axis_mem_read1.wdata       = m_axis_mem_read.wdata[511:256];
    // assign m_axis_mem_read1.wstrb       = m_axis_mem_read.wstrb;
    // assign m_axis_mem_read1.wlast       = m_axis_mem_read.wlast;
    // assign m_axis_mem_read1.wvalid      = m_axis_mem_read.wvalid & m_axis_mem_read.wready;
    // assign m_axis_mem_read1.bready      = m_axis_mem_read.bready;
    // assign m_axis_mem_read1.arid        = m_axis_mem_read.arid;
    // assign m_axis_mem_read1.araddr      = m_axis_mem_read.araddr >> 1 + 32'h1000_0000;
    // assign m_axis_mem_read1.arlen       = m_axis_mem_read.arlen;
    // assign m_axis_mem_read1.arsize      = m_axis_mem_read.arsize;
    // assign m_axis_mem_read1.arburst     = m_axis_mem_read.arburst;
    // assign m_axis_mem_read1.arcache     = m_axis_mem_read.arcache;
    // assign m_axis_mem_read1.arprot      = m_axis_mem_read.arprot;
    // assign m_axis_mem_read1.arlock      = m_axis_mem_read.arlock;
    // assign m_axis_mem_read1.arvalid     = m_axis_mem_read.arvalid & m_axis_mem_read.arready;
    // assign m_axis_mem_read1.arqos       = m_axis_mem_read.arqos;
    // assign m_axis_mem_read1.arregion    = m_axis_mem_read.arregion;   
    // assign m_axis_mem_read1.rready      = m_axis_mem_read.rready & m_axis_mem_read.rvalid;    
    
    // assign m_axis_mem_read.awready      = m_axis_mem_read0.awready & m_axis_mem_read1.awready;     
    // assign m_axis_mem_read.wready       = m_axis_mem_read0.wready & m_axis_mem_read1.wready;   
    // assign m_axis_mem_read.bid          = m_axis_mem_read0.bid;
    // assign m_axis_mem_read.bresp        = m_axis_mem_read0.bresp;
    // assign m_axis_mem_read.bvalid       = m_axis_mem_read0.bvalid;
    // assign m_axis_mem_read.arready      = m_axis_mem_read0.arready & m_axis_mem_read1.arready;
    // assign m_axis_mem_read.rid          = m_axis_mem_read0.rid;
    // assign m_axis_mem_read.rresp        = m_axis_mem_read0.rresp;
    // assign m_axis_mem_read.rdata        = {m_axis_mem_read1.rdata,m_axis_mem_read0.rdata};                  
    // assign m_axis_mem_read.rvalid       = m_axis_mem_read0.rvalid & m_axis_mem_read1.rvalid;
    // assign m_axis_mem_read.rlast        = m_axis_mem_read0.rlast;    

    // assign m_axis_mem_write0.awid        = m_axis_mem_write.awid;
    // assign m_axis_mem_write0.awlen       = m_axis_mem_write.awlen; 
    // assign m_axis_mem_write0.awsize      = m_axis_mem_write.awsize;
    // assign m_axis_mem_write0.awburst     = m_axis_mem_write.awburst;
    // assign m_axis_mem_write0.awcache     = m_axis_mem_write.awcache;
    // assign m_axis_mem_write0.awprot      = m_axis_mem_write.awprot;
    // assign m_axis_mem_write0.awaddr      = m_axis_mem_write.awaddr >> 1;
    // assign m_axis_mem_write0.awlock      = m_axis_mem_write.awlock;
    // assign m_axis_mem_write0.awvalid     = m_axis_mem_write.awvalid & m_axis_mem_write.awready;
    // assign m_axis_mem_write0.awqos       = m_axis_mem_write.awqos;
    // assign m_axis_mem_write0.awregion    = m_axis_mem_write.awregion;                              
    // assign m_axis_mem_write0.wdata       = m_axis_mem_write.wdata[255:0];
    // assign m_axis_mem_write0.wstrb       = m_axis_mem_write.wstrb;
    // assign m_axis_mem_write0.wlast       = m_axis_mem_write.wlast;
    // assign m_axis_mem_write0.wvalid      = m_axis_mem_write.wvalid & m_axis_mem_write.wready;
    // assign m_axis_mem_write0.bready      = m_axis_mem_write.bready;
    // assign m_axis_mem_write0.arid        = m_axis_mem_write.arid;
    // assign m_axis_mem_write0.araddr      = m_axis_mem_write.araddr >> 1;
    // assign m_axis_mem_write0.arlen       = m_axis_mem_write.arlen;
    // assign m_axis_mem_write0.arsize      = m_axis_mem_write.arsize;
    // assign m_axis_mem_write0.arburst     = m_axis_mem_write.arburst;
    // assign m_axis_mem_write0.arcache     = m_axis_mem_write.arcache;
    // assign m_axis_mem_write0.arprot      = m_axis_mem_write.arprot;
    // assign m_axis_mem_write0.arlock      = m_axis_mem_write.arlock;
    // assign m_axis_mem_write0.arvalid     = m_axis_mem_write.arvalid & m_axis_mem_write.arready;
    // assign m_axis_mem_write0.arqos       = m_axis_mem_write.arqos;
    // assign m_axis_mem_write0.arregion    = m_axis_mem_write.arregion;   
    // assign m_axis_mem_write0.rready      = m_axis_mem_write.rready & m_axis_mem_write.rvalid;

    // assign m_axis_mem_write1.awid        = m_axis_mem_write.awid;
    // assign m_axis_mem_write1.awlen       = m_axis_mem_write.awlen; 
    // assign m_axis_mem_write1.awsize      = m_axis_mem_write.awsize;
    // assign m_axis_mem_write1.awburst     = m_axis_mem_write.awburst;
    // assign m_axis_mem_write1.awcache     = m_axis_mem_write.awcache;
    // assign m_axis_mem_write1.awprot      = m_axis_mem_write.awprot;
    // assign m_axis_mem_write1.awaddr      = m_axis_mem_write.awaddr >> 1 + 32'h1000_0000;
    // assign m_axis_mem_write1.awlock      = m_axis_mem_write.awlock;
    // assign m_axis_mem_write1.awvalid     = m_axis_mem_write.awvalid & m_axis_mem_write.awready;
    // assign m_axis_mem_write1.awqos       = m_axis_mem_write.awqos;
    // assign m_axis_mem_write1.awregion    = m_axis_mem_write.awregion;                              
    // assign m_axis_mem_write1.wdata       = m_axis_mem_write.wdata[511:256];
    // assign m_axis_mem_write1.wstrb       = m_axis_mem_write.wstrb;
    // assign m_axis_mem_write1.wlast       = m_axis_mem_write.wlast;
    // assign m_axis_mem_write1.wvalid      = m_axis_mem_write.wvalid & m_axis_mem_write.wready;
    // assign m_axis_mem_write1.bready      = m_axis_mem_write.bready;
    // assign m_axis_mem_write1.arid        = m_axis_mem_write.arid;
    // assign m_axis_mem_write1.araddr      = m_axis_mem_write.araddr >> 1 + 32'h1000_0000;
    // assign m_axis_mem_write1.arlen       = m_axis_mem_write.arlen;
    // assign m_axis_mem_write1.arsize      = m_axis_mem_write.arsize;
    // assign m_axis_mem_write1.arburst     = m_axis_mem_write.arburst;
    // assign m_axis_mem_write1.arcache     = m_axis_mem_write.arcache;
    // assign m_axis_mem_write1.arprot      = m_axis_mem_write.arprot;
    // assign m_axis_mem_write1.arlock      = m_axis_mem_write.arlock;
    // assign m_axis_mem_write1.arvalid     = m_axis_mem_write.arvalid & m_axis_mem_write.arready;
    // assign m_axis_mem_write1.arqos       = m_axis_mem_write.arqos;
    // assign m_axis_mem_write1.arregion    = m_axis_mem_write.arregion;   
    // assign m_axis_mem_write1.rready      = m_axis_mem_write.rready & m_axis_mem_write.rvalid;    
    
    // assign m_axis_mem_write.awready      = m_axis_mem_write0.awready & m_axis_mem_write1.awready;     
    // assign m_axis_mem_write.wready       = m_axis_mem_write0.wready & m_axis_mem_write1.wready;   
    // assign m_axis_mem_write.bid          = m_axis_mem_write0.bid;
    // assign m_axis_mem_write.bresp        = m_axis_mem_write0.bresp;
    // assign m_axis_mem_write.bvalid       = m_axis_mem_write0.bvalid;
    // assign m_axis_mem_write.arready      = m_axis_mem_write0.arready & m_axis_mem_write1.arready;
    // assign m_axis_mem_write.rid          = m_axis_mem_write0.rid;
    // assign m_axis_mem_write.rresp        = m_axis_mem_write0.rresp;
    // assign m_axis_mem_write.rdata        = {m_axis_mem_write1.rdata,m_axis_mem_write0.rdata};                  
    // assign m_axis_mem_write.rvalid       = m_axis_mem_write0.rvalid & m_axis_mem_write1.rvalid;
    // assign m_axis_mem_write.rlast        = m_axis_mem_write0.rlast; 

///////////////////////////////////////////////////////////////////////
////////////////////////////////ddr//////////////////////////////////
    assign m_axis_mem_read0.awid        = mpi_change ? m_axis_mem_write.awid : m_axis_mem_read.awid;
    assign m_axis_mem_read0.awlen       = mpi_change ? m_axis_mem_write.awlen : m_axis_mem_read.awlen; 
    assign m_axis_mem_read0.awsize      = mpi_change ? m_axis_mem_write.awsize : m_axis_mem_read.awsize;
    assign m_axis_mem_read0.awburst     = mpi_change ? m_axis_mem_write.awburst : m_axis_mem_read.awburst;
    assign m_axis_mem_read0.awcache     = mpi_change ? m_axis_mem_write.awcache : m_axis_mem_read.awcache;
    assign m_axis_mem_read0.awprot      = mpi_change ? m_axis_mem_write.awprot : m_axis_mem_read.awprot;
    assign m_axis_mem_read0.awaddr      = mpi_change ? m_axis_mem_write.awaddr : m_axis_mem_read.awaddr;
    assign m_axis_mem_read0.awlock      = mpi_change ? m_axis_mem_write.awlock : m_axis_mem_read.awlock;
    assign m_axis_mem_read0.awvalid     = mpi_change ? m_axis_mem_write.awvalid : m_axis_mem_read.awvalid;
    assign m_axis_mem_read0.awqos       = mpi_change ? m_axis_mem_write.awqos : m_axis_mem_read.awqos;
    assign m_axis_mem_read0.awregion    = mpi_change ? m_axis_mem_write.awregion : m_axis_mem_read.awregion;                              
    assign m_axis_mem_read0.wdata       = mpi_change ? m_axis_mem_write.wdata : m_axis_mem_read.wdata;
    assign m_axis_mem_read0.wstrb       = mpi_change ? m_axis_mem_write.wstrb : m_axis_mem_read.wstrb;
    assign m_axis_mem_read0.wlast       = mpi_change ? m_axis_mem_write.wlast : m_axis_mem_read.wlast;
    assign m_axis_mem_read0.wvalid      = mpi_change ? m_axis_mem_write.wvalid : m_axis_mem_read.wvalid;
    assign m_axis_mem_read0.bready      = mpi_change ? m_axis_mem_write.bready : m_axis_mem_read.bready;
    assign m_axis_mem_read0.arid        = mpi_change ? m_axis_mem_write.arid : m_axis_mem_read.arid;
    assign m_axis_mem_read0.araddr      = mpi_change ? m_axis_mem_write.araddr : m_axis_mem_read.araddr;
    assign m_axis_mem_read0.arlen       = mpi_change ? m_axis_mem_write.arlen : m_axis_mem_read.arlen;
    assign m_axis_mem_read0.arsize      = mpi_change ? m_axis_mem_write.arsize : m_axis_mem_read.arsize;
    assign m_axis_mem_read0.arburst     = mpi_change ? m_axis_mem_write.arburst : m_axis_mem_read.arburst;
    assign m_axis_mem_read0.arcache     = mpi_change ? m_axis_mem_write.arcache : m_axis_mem_read.arcache;
    assign m_axis_mem_read0.arprot      = mpi_change ? m_axis_mem_write.arprot : m_axis_mem_read.arprot;
    assign m_axis_mem_read0.arlock      = mpi_change ? m_axis_mem_write.arlock : m_axis_mem_read.arlock;
    assign m_axis_mem_read0.arvalid     = mpi_change ? m_axis_mem_write.arvalid : m_axis_mem_read.arvalid;
    assign m_axis_mem_read0.arqos       = mpi_change ? m_axis_mem_write.arqos : m_axis_mem_read.arqos;
    assign m_axis_mem_read0.arregion    = mpi_change ? m_axis_mem_write.arregion : m_axis_mem_read.arregion;   
    assign m_axis_mem_read0.rready      = mpi_change ? m_axis_mem_write.rready : m_axis_mem_read.rready; 

    assign m_axis_mem_read.awready      = mpi_change ? m_axis_mem_write0.awready : m_axis_mem_read0.awready;     
    assign m_axis_mem_read.wready       = mpi_change ? m_axis_mem_write0.wready : m_axis_mem_read0.wready;   
    assign m_axis_mem_read.bid          = mpi_change ? m_axis_mem_write0.bid : m_axis_mem_read0.bid;
    assign m_axis_mem_read.bresp        = mpi_change ? m_axis_mem_write0.bresp : m_axis_mem_read0.bresp;
    assign m_axis_mem_read.bvalid       = mpi_change ? m_axis_mem_write0.bvalid : m_axis_mem_read0.bvalid;
    assign m_axis_mem_read.arready      = mpi_change ? m_axis_mem_write0.arready : m_axis_mem_read0.arready;
    assign m_axis_mem_read.rid          = mpi_change ? m_axis_mem_write0.rid : m_axis_mem_read0.rid;
    assign m_axis_mem_read.rresp        = mpi_change ? m_axis_mem_write0.rresp : m_axis_mem_read0.rresp;
    assign m_axis_mem_read.rdata        = mpi_change ? m_axis_mem_write0.rdata : m_axis_mem_read0.rdata;                  
    assign m_axis_mem_read.rvalid       = mpi_change ? m_axis_mem_write0.rvalid : m_axis_mem_read0.rvalid;
    assign m_axis_mem_read.rlast        = mpi_change ? m_axis_mem_write0.rlast : m_axis_mem_read0.rlast;    

    assign m_axis_mem_write0.awid       = mpi_change ? m_axis_mem_read.awid : m_axis_mem_write.awid;
    assign m_axis_mem_write0.awlen      = mpi_change ? m_axis_mem_read.awlen : m_axis_mem_write.awlen; 
    assign m_axis_mem_write0.awsize     = mpi_change ? m_axis_mem_read.awsize : m_axis_mem_write.awsize;
    assign m_axis_mem_write0.awburst    = mpi_change ? m_axis_mem_read.awburst : m_axis_mem_write.awburst;
    assign m_axis_mem_write0.awcache    = mpi_change ? m_axis_mem_read.awcache : m_axis_mem_write.awcache;
    assign m_axis_mem_write0.awprot     = mpi_change ? m_axis_mem_read.awprot : m_axis_mem_write.awprot;
    assign m_axis_mem_write0.awaddr     = mpi_change ? m_axis_mem_read.awaddr : m_axis_mem_write.awaddr;
    assign m_axis_mem_write0.awlock     = mpi_change ? m_axis_mem_read.awlock : m_axis_mem_write.awlock;
    assign m_axis_mem_write0.awvalid    = mpi_change ? m_axis_mem_read.awvalid : m_axis_mem_write.awvalid;
    assign m_axis_mem_write0.awqos      = mpi_change ? m_axis_mem_read.awqos : m_axis_mem_write.awqos;
    assign m_axis_mem_write0.awregion   = mpi_change ? m_axis_mem_read.awregion : m_axis_mem_write.awregion;                              
    assign m_axis_mem_write0.wdata      = mpi_change ? m_axis_mem_read.wdata : m_axis_mem_write.wdata;
    assign m_axis_mem_write0.wstrb      = mpi_change ? m_axis_mem_read.wstrb : m_axis_mem_write.wstrb;
    assign m_axis_mem_write0.wlast      = mpi_change ? m_axis_mem_read.wlast : m_axis_mem_write.wlast;
    assign m_axis_mem_write0.wvalid     = mpi_change ? m_axis_mem_read.wvalid : m_axis_mem_write.wvalid;
    assign m_axis_mem_write0.bready     = mpi_change ? m_axis_mem_read.bready : m_axis_mem_write.bready;
    assign m_axis_mem_write0.arid       = mpi_change ? m_axis_mem_read.arid : m_axis_mem_write.arid;
    assign m_axis_mem_write0.araddr     = mpi_change ? m_axis_mem_read.araddr : m_axis_mem_write.araddr;
    assign m_axis_mem_write0.arlen      = mpi_change ? m_axis_mem_read.arlen : m_axis_mem_write.arlen;
    assign m_axis_mem_write0.arsize     = mpi_change ? m_axis_mem_read.arsize : m_axis_mem_write.arsize;
    assign m_axis_mem_write0.arburst    = mpi_change ? m_axis_mem_read.arburst : m_axis_mem_write.arburst;
    assign m_axis_mem_write0.arcache    = mpi_change ? m_axis_mem_read.arcache : m_axis_mem_write.arcache;
    assign m_axis_mem_write0.arprot     = mpi_change ? m_axis_mem_read.arprot : m_axis_mem_write.arprot;
    assign m_axis_mem_write0.arlock     = mpi_change ? m_axis_mem_read.arlock : m_axis_mem_write.arlock;
    assign m_axis_mem_write0.arvalid    = mpi_change ? m_axis_mem_read.arvalid : m_axis_mem_write.arvalid;
    assign m_axis_mem_write0.arqos      = mpi_change ? m_axis_mem_read.arqos : m_axis_mem_write.arqos;
    assign m_axis_mem_write0.arregion   = mpi_change ? m_axis_mem_read.arregion : m_axis_mem_write.arregion;   
    assign m_axis_mem_write0.rready     = mpi_change ? m_axis_mem_read.rready : m_axis_mem_write.rready;  

    assign m_axis_mem_write.awready     = mpi_change ? m_axis_mem_read0.awready : m_axis_mem_write0.awready;     
    assign m_axis_mem_write.wready      = mpi_change ? m_axis_mem_read0.wready : m_axis_mem_write0.wready;   
    assign m_axis_mem_write.bid         = mpi_change ? m_axis_mem_read0.bid : m_axis_mem_write0.bid;
    assign m_axis_mem_write.bresp       = mpi_change ? m_axis_mem_read0.bresp : m_axis_mem_write0.bresp;
    assign m_axis_mem_write.bvalid      = mpi_change ? m_axis_mem_read0.bvalid : m_axis_mem_write0.bvalid;
    assign m_axis_mem_write.arready     = mpi_change ? m_axis_mem_read0.arready : m_axis_mem_write0.arready;
    assign m_axis_mem_write.rid         = mpi_change ? m_axis_mem_read0.rid : m_axis_mem_write0.rid;
    assign m_axis_mem_write.rresp       = mpi_change ? m_axis_mem_read0.rresp : m_axis_mem_write0.rresp;
    assign m_axis_mem_write.rdata       = mpi_change ? m_axis_mem_read0.rdata : m_axis_mem_write0.rdata;                  
    assign m_axis_mem_write.rvalid      = mpi_change ? m_axis_mem_read0.rvalid : m_axis_mem_write0.rvalid;
    assign m_axis_mem_write.rlast       = mpi_change ? m_axis_mem_read0.rlast : m_axis_mem_write0.rlast; 
////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////ddr change////////////////////////

    always@(posedge clk)begin
        if(~rstn)begin
            mpi_change              <= 1'b0;
        end
        else if(status_reg[31][0])begin
            mpi_change              <= ~mpi_change;
        end
        else begin
            mpi_change              <= mpi_change;
        end
    end 





/////////////////////////////////////strat done///////////////////////////////////////////////    
    reg                         start,start_r,start_rr;
    reg                         mpi_flag;
    reg[1:0]                    mpi_done;

    always@(posedge clk)begin
        start                   <= control_reg[4][0];
        start_r                 <= start;
        start_rr                <= start_r;
    end

    always@(posedge clk)begin
        if(~rstn)begin
            mpi_done            <= 2'b0;
        end
        else if(status_reg[29][0])begin
            mpi_done[1]         <= 1'b1;
        end
        else if(status_reg[28][0])begin
            mpi_done[0]         <= 1'b1;
        end
        else begin
            mpi_done            <= mpi_done;
        end
    end 


    always@(posedge clk)begin
        if(~rstn)begin
            mpi_flag            <= 1'b0;
        end
        else if(start_r & ~start_rr)begin
            mpi_flag            <= 1'b1;
        end
        else if(mpi_done == 2'b11)begin
            mpi_flag            <= 1'b0;
        end
        else begin
            mpi_flag            <= mpi_flag;
        end
    end 


    mpi_allreduce#(
        .MAX_CLIENT                 (8)
    )mpi_allreduce( 
    
        //user clock input
        .clk                        (clk),
        .rstn                       (rstn),
        
        //dma memory streams
        .m_axis_dma_write_cmd       (m_axis_dma_write_cmd),
        .m_axis_dma_write_data      (m_axis_dma_write_data),
        .m_axis_dma_read_cmd        (m_axis_dma_read_cmd),
        .s_axis_dma_read_data       (s_axis_dma_read_data),    
        //ddr memory streams    
        .m_axis_mem_read            (m_axis_mem_read),
        .m_axis_mem_write           (m_axis_mem_write),     
        
        //tcp app interface streams
        .m_axis_tcp_tx_meta         (app_axis_tcp_tx_meta),
        .m_axis_tcp_tx_data         (app_axis_tcp_tx_data),
        .s_axis_tcp_tx_status       (app_axis_tcp_tx_status), 
    
        .s_axis_tcp_rx_meta         (app_axis_tcp_rx_meta),
        .s_axis_tcp_rx_data         (app_axis_tcp_rx_data),
    
        //control reg
        .control_reg                (control_reg),
        .status_reg                 (status_reg[31:28])
        
        );

    //////////////////////////debug///////////////////////


//////////////////////////////////mpi_counter//////////////////////////////
    reg[31:0]                           mpi_write_cmd0_counter;

    always@(posedge clk)begin
        if(~rstn)begin
            mpi_write_cmd0_counter          <= 1'b0;
        end
        else if(m_axis_mem_write0.awvalid && m_axis_mem_write0.awready)begin
            mpi_write_cmd0_counter          <= mpi_write_cmd0_counter + 1'b1;
        end
        else begin
            mpi_write_cmd0_counter          <= mpi_write_cmd0_counter;
        end
    end

    reg[31:0]                           mpi_write_data0_counter;

    always@(posedge clk)begin
        if(~rstn)begin
            mpi_write_data0_counter          <= 1'b0;
        end
        else if(m_axis_mem_write0.wvalid && m_axis_mem_write0.wready)begin
            mpi_write_data0_counter          <= mpi_write_data0_counter + 1'b1;
        end
        else begin
            mpi_write_data0_counter          <= mpi_write_data0_counter;
        end
    end

    reg[31:0]                           mpi_write_cmd1_counter;

    // always@(posedge clk)begin
    //     if(~rstn)begin
    //         mpi_write_cmd1_counter          <= 1'b0;
    //     end
    //     else if(m_axis_mem_write1.awvalid && m_axis_mem_write1.awready)begin
    //         mpi_write_cmd1_counter          <= mpi_write_cmd1_counter + 1'b1;
    //     end
    //     else begin
    //         mpi_write_cmd1_counter          <= mpi_write_cmd1_counter;
    //     end
    // end

    reg[31:0]                           mpi_write_data1_counter;

    // always@(posedge clk)begin
    //     if(~rstn)begin
    //         mpi_write_data1_counter          <= 1'b0;
    //     end
    //     else if(m_axis_mem_write1.wvalid && m_axis_mem_write1.wready)begin
    //         mpi_write_data1_counter          <= mpi_write_data1_counter + 1'b1;
    //     end
    //     else begin
    //         mpi_write_data1_counter          <= mpi_write_data1_counter;
    //     end
    // end

    reg[31:0]                           mpi_read_cmd0_counter;

    always@(posedge clk)begin
        if(~rstn)begin
            mpi_read_cmd0_counter          <= 1'b0;
        end
        else if(m_axis_mem_read0.arvalid && m_axis_mem_read0.arready)begin
            mpi_read_cmd0_counter          <= mpi_read_cmd0_counter + 1'b1;
        end
        else begin
            mpi_read_cmd0_counter          <= mpi_read_cmd0_counter;
        end
    end

    reg[31:0]                           mpi_read_data0_counter;

    always@(posedge clk)begin
        if(~rstn)begin
            mpi_read_data0_counter          <= 1'b0;
        end
        else if(m_axis_mem_read0.rvalid && m_axis_mem_read0.rready)begin
            mpi_read_data0_counter          <= mpi_read_data0_counter + 1'b1;
        end
        else begin
            mpi_read_data0_counter          <= mpi_read_data0_counter;
        end
    end

    reg[31:0]                           mpi_read_cmd1_counter;

    // always@(posedge clk)begin
    //     if(~rstn)begin
    //         mpi_read_cmd1_counter          <= 1'b0;
    //     end
    //     else if(m_axis_mem_read1.arvalid && m_axis_mem_read1.arready)begin
    //         mpi_read_cmd1_counter          <= mpi_read_cmd1_counter + 1'b1;
    //     end
    //     else begin
    //         mpi_read_cmd1_counter          <= mpi_read_cmd1_counter;
    //     end
    // end

    reg[31:0]                           mpi_read_data1_counter;

    // always@(posedge clk)begin
    //     if(~rstn)begin
    //         mpi_read_data1_counter          <= 1'b0;
    //     end
    //     else if(m_axis_mem_read1.rvalid && m_axis_mem_read1.rready)begin
    //         mpi_read_data1_counter          <= mpi_read_data1_counter + 1'b1;
    //     end
    //     else begin
    //         mpi_read_data1_counter          <= mpi_read_data1_counter;
    //     end
    // end  

//////////////////////////////dma_counter   
     
    
    reg[31:0]                           dma_write_cmd0_counter;

    always@(posedge clk)begin
        if(~rstn)begin
            dma_write_cmd0_counter          <= 1'b0;
        end
        else if(m_axis_dma_write_cmd.valid && m_axis_dma_write_cmd.ready)begin
            dma_write_cmd0_counter          <= dma_write_cmd0_counter + 1'b1;
        end
        else begin
            dma_write_cmd0_counter          <= dma_write_cmd0_counter;
        end
    end

    reg[31:0]                           dma_write_data0_counter;

    always@(posedge clk)begin
        if(~rstn)begin
            dma_write_data0_counter          <= 1'b0;
        end
        else if(m_axis_dma_write_data.valid && m_axis_dma_write_data.ready)begin
            dma_write_data0_counter          <= dma_write_data0_counter + 1'b1;
        end
        else begin
            dma_write_data0_counter          <= dma_write_data0_counter;
        end
    end

    reg[31:0]                           dma_read_cmd0_counter;

    always@(posedge clk)begin
        if(~rstn)begin
            dma_read_cmd0_counter          <= 1'b0;
        end
        else if(m_axis_dma_read_cmd.valid && m_axis_dma_read_cmd.ready)begin
            dma_read_cmd0_counter          <= dma_read_cmd0_counter + 1'b1;
        end
        else begin
            dma_read_cmd0_counter          <= dma_read_cmd0_counter;
        end
    end

    reg[31:0]                           dma_read_data0_counter;

    always@(posedge clk)begin
        if(~rstn)begin
            dma_read_data0_counter          <= 1'b0;
        end
        else if(s_axis_dma_read_data.valid && s_axis_dma_read_data.ready)begin
            dma_read_data0_counter          <= dma_read_data0_counter + 1'b1;
        end
        else begin
            dma_read_data0_counter          <= dma_read_data0_counter;
        end
    end

/////////////////////////////mem_counter    

    reg[31:0]                           mem_write_cmd0_counter;

    always@(posedge clk)begin
        if(~rstn)begin
            mem_write_cmd0_counter          <= 1'b0;
        end
        else if(m_axis_mem_write.awvalid && m_axis_mem_write.awready)begin
            mem_write_cmd0_counter          <= mem_write_cmd0_counter + 1'b1;
        end
        else begin
            mem_write_cmd0_counter          <= mem_write_cmd0_counter;
        end
    end

    reg[31:0]                           mem_write_data0_counter;

    always@(posedge clk)begin
        if(~rstn)begin
            mem_write_data0_counter          <= 1'b0;
        end
        else if(m_axis_mem_write.wvalid && m_axis_mem_write.wready)begin
            mem_write_data0_counter          <= mem_write_data0_counter + 1'b1;
        end
        else begin
            mem_write_data0_counter          <= mem_write_data0_counter;
        end
    end


    reg[31:0]                           mem_read_cmd0_counter;

    always@(posedge clk)begin
        if(~rstn)begin
            mem_read_cmd0_counter          <= 1'b0;
        end
        else if(m_axis_mem_read.arvalid && m_axis_mem_read.arready)begin
            mem_read_cmd0_counter          <= mem_read_cmd0_counter + 1'b1;
        end
        else begin
            mem_read_cmd0_counter          <= mem_read_cmd0_counter;
        end
    end

    reg[31:0]                           mem_read_data0_counter;

    always@(posedge clk)begin
        if(~rstn)begin
            mem_read_data0_counter          <= 1'b0;
        end
        else if(m_axis_mem_read.rvalid && m_axis_mem_read.rready)begin
            mem_read_data0_counter          <= mem_read_data0_counter + 1'b1;
        end
        else begin
            mem_read_data0_counter          <= mem_read_data0_counter;
        end
    end


/////////////////////////////////tcp data
    reg[31:0]                           tcp_write_cmd0_counter;

    always@(posedge clk)begin
        if(~rstn)begin
            tcp_write_cmd0_counter          <= 1'b0;
        end
        else if(app_axis_tcp_tx_meta.valid && app_axis_tcp_tx_meta.ready)begin
            tcp_write_cmd0_counter          <= tcp_write_cmd0_counter + 1'b1;
        end
        else begin
            tcp_write_cmd0_counter          <= tcp_write_cmd0_counter;
        end
    end

    reg[31:0]                           tcp_write_data0_counter;

    always@(posedge clk)begin
        if(~rstn)begin
            tcp_write_data0_counter          <= 1'b0;
        end
        else if(app_axis_tcp_tx_data.valid && app_axis_tcp_tx_data.ready)begin
            tcp_write_data0_counter          <= tcp_write_data0_counter + 1'b1;
        end
        else begin
            tcp_write_data0_counter          <= tcp_write_data0_counter;
        end
    end

    reg[31:0]                           tcp_read_cmd0_counter;

    always@(posedge clk)begin
        if(~rstn)begin
            tcp_read_cmd0_counter          <= 1'b0;
        end
        else if(app_axis_tcp_rx_meta.valid && app_axis_tcp_rx_meta.ready)begin
            tcp_read_cmd0_counter          <= tcp_read_cmd0_counter + 1'b1;
        end
        else begin
            tcp_read_cmd0_counter          <= tcp_read_cmd0_counter;
        end
    end

    reg[31:0]                           tcp_read_data0_counter;

    always@(posedge clk)begin
        if(~rstn)begin
            tcp_read_data0_counter          <= 1'b0;
        end
        else if(app_axis_tcp_rx_data.valid && app_axis_tcp_rx_data.ready)begin
            tcp_read_data0_counter          <= tcp_read_data0_counter + 1'b1;
        end
        else begin
            tcp_read_data0_counter          <= tcp_read_data0_counter;
        end
    end

    ////////////////////////////mpi time

    reg[31:0]                           time_counter;
    reg                                 time_en;

    always@(posedge clk)begin
        if(~rstn)begin
            time_en                     <= 1'b0;
        end
        else if(status_reg[30][0])begin
            time_en                     <= 1'b1;
        end
        else if(mpi_done == 2'b11)begin
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

    assign status_reg[0] = mpi_write_cmd0_counter;
    assign status_reg[1] = mpi_write_data0_counter;
    assign status_reg[2] = mpi_write_cmd1_counter;
    assign status_reg[3] = mpi_write_data1_counter;    
    assign status_reg[4] = mpi_read_cmd0_counter;
    assign status_reg[5] = mpi_read_data0_counter;
    assign status_reg[6] = mpi_read_cmd1_counter;
    assign status_reg[7] = mpi_read_data1_counter;

    assign status_reg[8] = dma_write_cmd0_counter;
    assign status_reg[9] = dma_write_data0_counter;
    assign status_reg[10] = dma_read_cmd0_counter;
    assign status_reg[11] = dma_read_data0_counter;

    
    assign status_reg[12] = mem_write_cmd0_counter;
    assign status_reg[13] = mem_write_data0_counter;
 
    assign status_reg[16] = mem_read_cmd0_counter;
    assign status_reg[17] = mem_read_data0_counter;


    assign status_reg[20] = tcp_write_cmd0_counter;
    assign status_reg[21] = tcp_write_data0_counter;
    assign status_reg[22] = tcp_read_cmd0_counter;
    assign status_reg[23] = tcp_read_data0_counter;



    assign status_reg[24] = time_counter;


endmodule
