/*
 * Copyright 2020-2021 , HHJ, RC4ML , ZJU
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


 module  mmcm_clk #(
    parameter MMCM_DIVCLK_DIVIDE            = 16,
    parameter MMCM_CLKFBOUT_MULT_F          = 2,
    parameter MMCM_CLKOUT0_DIVIDE_F         = 2,
    parameter MMCM_CLKOUT1_DIVIDE_F         = 2,
    parameter MMCM_CLKOUT2_DIVIDE_F         = 2,
    parameter MMCM_CLKOUT3_DIVIDE_F         = 2,
    parameter MMCM_CLKOUT4_DIVIDE_F         = 2,
    parameter MMCM_CLKOUT5_DIVIDE_F         = 2,
    parameter MMCM_CLKOUT6_DIVIDE_F         = 2,
    parameter MMCM_CLKIN1_PERIOD            = 10.000
) (
    input  wire                         clk_in_p,
    input  wire                         clk_in_n,
    input  wire                         rst_in,
    //////////////////////clkout////////////////////////////
    output wire                         mmcm_lock,                  
    output wire                         clk_out0,           
    output wire                         clk_out1,              
    output wire                         clk_out2, 
    output wire                         clk_out3,
    output wire                         clk_out4,
    output wire                         clk_out5,
    output wire                         clk_out6        
);

    wire                                clk_in;
    wire                                clk_in1_buf; 
    wire                                AXI_ACLK0_st0;
    wire                                AXI_ACLK1_st0;
    wire                                AXI_ACLK2_st0;
    wire                                AXI_ACLK3_st0;
    wire                                AXI_ACLK4_st0;
    wire                                AXI_ACLK5_st0;
    wire                                AXI_ACLK6_st0;   
    wire                                mmcm_lock_in;       


     IBUFDS #(
       .IBUF_LOW_PWR("TRUE")     // Low power="TRUE", Highest performance="FALSE" 
    ) IBUFDS0_inst (
       .O(clk_in),  // Buffer output
       .I(clk_in_p),  // Diff_p buffer input (connect directly to top-level port)
       .IB(clk_in_n) // Diff_n buffer input (connect directly to top-level port)
    );
 
   
      BUFG BUFG0_inst (
       .O(clk_in1_buf), // 1-bit output: Clock output
       .I(clk_in)  // 1-bit input: Clock input
    );







MMCME4_ADV
  #(.BANDWIDTH            ("OPTIMIZED"),
    .CLKOUT4_CASCADE      ("FALSE"),
    .COMPENSATION         ("INTERNAL"),
    .STARTUP_WAIT         ("FALSE"),
    .DIVCLK_DIVIDE        (MMCM_DIVCLK_DIVIDE),
    .CLKFBOUT_MULT_F      (MMCM_CLKFBOUT_MULT_F),
    .CLKFBOUT_PHASE       (0.000),
    .CLKFBOUT_USE_FINE_PS ("FALSE"),
    .CLKOUT0_DIVIDE_F     (MMCM_CLKOUT0_DIVIDE_F),
    .CLKOUT0_PHASE        (0.000),
    .CLKOUT0_DUTY_CYCLE   (0.500),
    .CLKOUT0_USE_FINE_PS  ("FALSE"),
    .CLKOUT1_DIVIDE       (MMCM_CLKOUT1_DIVIDE_F),
    .CLKOUT2_DIVIDE       (MMCM_CLKOUT2_DIVIDE_F),
    .CLKOUT3_DIVIDE       (MMCM_CLKOUT3_DIVIDE_F),
    .CLKOUT4_DIVIDE       (MMCM_CLKOUT4_DIVIDE_F),
    .CLKOUT5_DIVIDE       (MMCM_CLKOUT5_DIVIDE_F),
    .CLKOUT6_DIVIDE       (MMCM_CLKOUT6_DIVIDE_F),
    .CLKIN1_PERIOD        (MMCM_CLKIN1_PERIOD),
    .REF_JITTER1          (0.010))
  u_mmcm_0
    // Output clocks
   (
    .CLKFBOUT            (),
    .CLKFBOUTB           (),
    .CLKOUT0             (AXI_ACLK0_st0),

    .CLKOUT0B            (),
    .CLKOUT1             (AXI_ACLK1_st0),
    .CLKOUT1B            (),
    .CLKOUT2             (AXI_ACLK2_st0),
    .CLKOUT2B            (),
    .CLKOUT3             (AXI_ACLK3_st0),
    .CLKOUT3B            (),
    .CLKOUT4             (AXI_ACLK4_st0),
    .CLKOUT5             (AXI_ACLK5_st0),
    .CLKOUT6             (AXI_ACLK6_st0),
     // Input clock control
    .CLKFBIN             (), //mmcm_fb
    .CLKIN1              (clk_in1_buf),
    .CLKIN2              (1'b0),
    // Other control and status signals
    .LOCKED              (mmcm_lock_in),
    .PWRDWN              (1'b0),
    .RST                 (rst_in),
  
    .CDDCDONE            (),
    .CLKFBSTOPPED        (),
    .CLKINSTOPPED        (),
    .DO                  (),
    .DRDY                (),
    .PSDONE              (),
    .CDDCREQ             (1'b0),
    .CLKINSEL            (1'b1),
    .DADDR               (7'b0),
    .DCLK                (1'b0),
    .DEN                 (1'b0),
    .DI                  (16'b0),
    .DWE                 (1'b0),
    .PSCLK               (1'b0),
    .PSEN                (1'b0),
    .PSINCDEC            (1'b0)
  );

BUFG u_AXI_ACLK0_st0  (
  .I (AXI_ACLK0_st0),
  .O (clk_out0)
);

BUFG u_AXI_ACLK1_st0  (
  .I (AXI_ACLK1_st0),
  .O (clk_out1)
);

BUFG u_AXI_ACLK2_st0  (
  .I (AXI_ACLK2_st0),
  .O (clk_out2)
);

BUFG u_AXI_ACLK3_st0  (
  .I (AXI_ACLK3_st0),
  .O (clk_out3)
);

BUFG u_AXI_ACLK4_st0  (
  .I (AXI_ACLK4_st0),
  .O (clk_out4)
);

BUFG u_AXI_ACLK5_st0  (
  .I (AXI_ACLK5_st0),
  .O (clk_out5)
);

BUFG u_AXI_ACLK6_st0  (
  .I (AXI_ACLK6_st0),
  .O (clk_out6)
);   

BUFG u_MMCM_LOCK  (
  .I (mmcm_lock_in),
  .O (mmcm_lock)
);  

endmodule


