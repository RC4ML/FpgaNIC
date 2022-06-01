/*
 * Copyright (c) 2019, Systems Group, ETH Zurich
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
`ifndef EXAMPLE_TYPES_SVH_
`define EXAMPLE_TYPES_SVH_
//`default_nettype none



interface axis_mem_cmd;
    logic valid;
    logic ready;
    logic[63:0]     address;
    logic[31:0]     length;
    //logic           dest;

    modport master (output valid,
                    input ready,
                    output address,
                    output length);

    modport slave ( input valid,
                    output ready,
                    input address,
                    input length);

    /*modport rmaster(output valid,
                    input ready,
                    output address,
                    output length,
                    output dest);*/

endinterface

interface axi_stream #(
    parameter WIDTH = 512
);
    logic valid;
    logic ready;
    logic[WIDTH-1:0]    data;
    logic[(WIDTH/8)-1:0] keep;
    logic           last;
    logic           dest;

    modport master (output valid,
                    input ready,
                    output data,
                    output keep,
                    output last);

    modport slave ( input valid,
                    output ready,
                    input data,
                    input keep,
                    input last);

    modport rmaster(output valid,
                    input ready,
                    output data,
                    output keep,
                    output last,
                    output dest);

    /*modport rslave (input valid,
                    output ready,
                    input data,
                    input keep,
                    input last,
                    input dest);*/

endinterface

interface axis_mem_status;
    logic valid;
    logic ready;
    logic[7:0]     data;

    modport master (output valid,
                    input  ready,
                    output data);

    modport slave ( input  valid,
                    output ready,
                    input  data);
endinterface

interface axis_meta #(
    parameter WIDTH = 96,
    parameter DEST_WIDTH = 1
);
    logic valid;
    logic ready;
    logic[WIDTH-1:0]        data;
    logic[DEST_WIDTH-1:0]   dest;

    modport master (output valid,
                    input ready,
                    output data);

    modport slave ( input valid,
                    output ready,
                    input data);

    modport rmaster(output valid,
                    input ready,
                    output data,
                    output dest);

endinterface


interface axi_lite;
    //write address
    logic [31: 0]   awaddr;
    logic           awvalid;
    logic           awready;
 
    //write data
    logic [31: 0]   wdata;
    logic [3: 0]    wstrb;
    logic           wvalid;
    logic           wready;
 
    //write response (handhake)
    logic [1:0]     bresp;
    logic           bvalid;
    logic           bready;
 
    //read address
    logic [31: 0]   araddr;
    logic           arvalid;
    logic           arready;
 
    //read data
    logic [31: 0]   rdata;
    logic [1:0]     rresp;
    logic           rvalid;
    logic           rready;

    modport master (output  awvalid,
                    input   awready,
                    output  awaddr,
                    output  wdata,
                    output  wstrb,
                    output  wvalid,
                    input   wready,
                    input   bvalid,
                    output  bready,
                    input   bresp,
                    output  arvalid,
                    input   arready,
                    output  araddr,
                    input   rvalid,
                    output  rready,
                    input   rdata,
                    input   rresp);

    modport slave ( input   awvalid,
                    output  awready,
                    input   awaddr,
                    input   wdata,
                    input   wstrb,
                    input   wvalid,
                    output  wready,
                    output  bvalid,
                    input   bready,
                    output  bresp,
                    input   arvalid,
                    output  arready,
                    input   araddr,
                    output  rvalid,
                    input   rready,
                    output  rdata,
                    output  rresp);

endinterface

interface axi_mm #(
    parameter ADDR_WIDTH = 33,
    parameter DATA_WIDTH = 256
);
    //write address
    logic [5:0]     awid;
    logic [7:0]     awlen;
    logic [2:0]     awsize;
    logic [1:0]     awburst;    
    logic [3:0]     awcache;
    logic [2:0]     awprot;
    logic [ADDR_WIDTH-1:0]    awaddr;
    logic           awlock;
    logic           awvalid;
    logic           awready;
    logic[3:0]      awqos;
    logic[3:0]      awregion;
    logic[4:0]      awuser;


    //write data
    logic [DATA_WIDTH-1:0]   wdata;
    logic [DATA_WIDTH/8-1:0]    wstrb;
    logic           wlast;
    logic           wvalid;
    logic           wready;
    logic[4:0]      wuser;

    //write response (handhake)
    logic [5:0]     bid;
    logic [1:0]     bresp;
    logic           bvalid;
    logic           bready;
    logic[4:0]      buser;
 
    //read address
    logic [5:0]     arid;
    logic [ADDR_WIDTH-1: 0]   araddr;
    logic [7:0]     arlen;
    logic [2:0]     arsize;
    logic [1:0]     arburst;
    logic [3:0]     arcache;
    logic [2:0]     arprot;
    logic           arlock;
    logic           arvalid;
    logic           arready;
    logic[3:0]      arqos;
    logic[3:0]      arregion;
    logic[4:0]      aruser;


    //read data
    logic [5:0]     rid;
    logic [1:0]     rresp; 
    logic [DATA_WIDTH-1:0]   rdata;
    logic           rvalid;
    logic           rlast;
    logic           rready;
    logic[4:0]      ruser;

    modport master (output  awid,
                    output  awlen, 
                    output  awsize,
                    output  awburst,
                    output  awcache,
                    output  awprot,
                    output  awaddr,
                    output  awlock,
                    output  awvalid,
                    output  awqos,
                    output  awregion,
                    output  awuser,
                    input   awready,
                                
                    output  wdata,
                    output  wstrb,
                    output  wlast,
                    output  wvalid,
                    output  wuser,
                    input   wready,                    

                    input   bid,
                    input   bresp,
                    input   bvalid,
                    input   buser,
                    output  bready,

                    output  arid,
                    output  araddr,
                    output  arlen,
                    output  arsize,
                    output  arburst,
                    output  arcache,
                    output  arprot,
                    output  arlock,
                    output  arvalid,
                    output  arqos,
                    output  arregion,
                    output  aruser,
                    input   arready,

                    input   rid,
                    input   rresp,
                    input   rdata,                  
                    input   rvalid,
                    input   rlast,
                    input   ruser,
                    output  rready);


     modport slave (input   awid,
                    input   awlen, 
                    input   awsize,
                    input   awburst,
                    input   awcache,
                    input   awprot,
                    input   awaddr,
                    input   awlock,
                    input   awvalid,
                    input   awqos,
                    input   awregion,
                    input   awuser,
                    output  awready,
                                  
                    input   wdata,
                    input   wstrb,
                    input   wlast,
                    input   wvalid,
                    input   wuser,
                    output  wready,                    

                    output  bid,
                    output  bresp,
                    output  bvalid,
                    output  buser,
                    input   bready,

                    input   arid,
                    input   araddr,
                    input   arlen,
                    input   arsize,
                    input   arburst,
                    input   arcache,
                    input   arprot,
                    input   arlock,
                    input   arvalid,
                    input   arqos,
                    input   arregion,
                    input   aruser,
                    output  arready,

                    output  rid,
                    output  rresp,
                    output  rdata,                   
                    output  rvalid,
                    output  rlast,
                    output  ruser,
                    input   rready);
// Tie off unused master signals
task tie_off_m ();
    araddr    = 0;
    arburst   = 2'b01;
    arcache   = 4'b0;
    arid      = 0;
    arlen     = 8'b0;   
    arlock    = 1'b0;   
    arprot    = 3'b0;   
    arqos     = 4'b0;   
    arregion  = 4'b0;   
    arsize    = 3'b0;   
    arvalid   = 1'b0;   
    aruser    = 0;
    awaddr    = 0;  
    awburst   = 2'b01;
    awcache   = 4'b0;   
    awid      = 0;
    awlen     = 8'b0;   
    awlock    = 1'b0;   
    awprot    = 3'b0;   
    awqos     = 4'b0;   
    awregion  = 4'b0;   
    awsize    = 3'b0;   
    awvalid   = 1'b0;
    awuser    = 0;
    bready    = 1'b0;    
    rready    = 1'b0;   
    wdata     = 0;  
    wlast     = 1'b0;
    wstrb     = 0;  
    wvalid    = 1'b0;   
    wuser     = 0;
endtask

// Tie off unused slave signals
task tie_off_s ();
    arready  = 1'b0;     
    awready  = 1'b0;
    bresp    = 2'b0;
    bvalid   = 1'b0;
    bid      = 0;   
    buser    = 0;
    rdata    = 0;
    rid      = 0;
    rlast    = 1'b0;
    rresp    = 2'b0;
    rvalid   = 1'b0;
    ruser    = 0;
    wready   = 1'b0;
endtask                    
                    
endinterface

interface tcp_notification;
    logic valid;
    logic ready;
    logic[15:0]     session;
    logic[31:0]     ipaddr;
    logic[15:0]     port;
    logic[15:0]     length;
    logic[7:0]      closeflag;

    //logic           dest;

    modport master (output valid,
                    input ready,
                    output session,
                    output ipaddr,
                    output port,
                    output length,
                    output closeflag);

    modport slave ( input valid,
                    output ready,
                    input session,
                    input ipaddr,
                    input port,
                    input length,
                    input closeflag);



endinterface

`define FPGA_VERSION 20200610

`define TLB_NUM 8

`endif
