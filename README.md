# FPGANIC
FpgaNIC is an FPGA-based, GPU-centric, versatile SmartNIC that enables direct PCIe P2P communication with local GPUs using GPU virtual address, and that provides reliable 100Gb network access to remote GPUs.
FpgaNIC allows to offload various complex compute tasks to a customized data-path accelerator for line-rate in-network computing on the FPGA, thereby complementing the processing at the GPU. 



## Cite this work
If you use it in your paper, please cite our work ([full version](https://www.usenix.org/conference/atc22/presentation/wang-zeke)).
```
@inproceedings{wang_atc22,
  title={FpgaNIC: An FPGA-based Versatile 100Gb SmartNIC for GPUs},
  author={Zeke Wang and Hongjing Huang and Jie Zhang and Fei Wu and Gustavo Alonso},
  year={2020},
  booktitle={2022 USENIX Annual Technical Conference (ATC)},
}
```



## Build example project

## Rrerequisites

```
* GPU and FPGA are under the same PCIe switch.
* Linux 4.15.0-20-generic;  Nvidia Driver Version: 450.51.05;  CUDA Version: 11.0
```

## Getting Started
```
$ git clone https://github.com/RC4ML/FpgaNIC.git
$ git submodule update --init --recursive
```

## Build FPGA Project 

It takes about 1-2 hours to generate a bitstream, you can also skip this step and directly use the generated bitstream file in the bitstream folder.

### Prerequisites

- Xilinx Vivado 2020.1
- Ubuntu (not sure whether other Linux OS works or not)

Supported boards 
- Xilinx Alveo U280

### Steps for Building an FPGA Bitstream for the direct mood

#### 1. Create build directory
```
$ mkdir build
$ cd build
```

#### 2. Configure xdma project build
```
$ cmake ..

```

#### 3. Make HLS IP Core
```
$ make installip
```

#### 4. Create vivado project (You can choose one project to create)

##### a. Create direct project(Figure 6)
```
$ make direct
```
##### b. Create pcie_benchmark project(Figure 3 4)
```
$ make pcie_benchmark
```
##### c. Create tcp_latency project(Figure 5a)
```
$ make tcp_latency
```
##### d. Create tcp_benchmark project(Figure 5b)
```
$ make tcp_benchmark
```
##### e. Create allreduce project(Figure 7 8)
```
$ make allreduce
```
##### f. Create hyperloglog project(Figure 9)
```
$ make hyperloglog
```



#### 5. Generate bitstream

-open the project by Vivado2020.1 and generate bitstream

## Download the bitstream to FPGAs

### 1.Connect the download server
```
$ ssh -p 6000 atc_bitstream@101.37.28.229
```

### 2.Open the vivado

We need to open the GUI of Vivado to download the bitstream, so we need a terminal that supports X11 forwarding, such as MobaXterm.

```
$ vivado
```
### 3. Open hardware manage

As shown, click it.

![image](https://github.com/RC4ML/FpgaNIC/blob/gpu_hll/img/openhw.jpg)

### 4. Open target

Click "Open target" and "Open New Target..""

![image](https://github.com/RC4ML/FpgaNIC/blob/gpu_hll/img/opentar.jpg)

Click "Next"

![image](https://github.com/RC4ML/FpgaNIC/blob/gpu_hll/img/opentar1.jpg)

Choose "Remote server", the "Host name" is 192.168.189.23, "Port" is 3121, click "Next"

![image](https://github.com/RC4ML/FpgaNIC/blob/gpu_hll/img/opentar2.jpg)

Click "next", then "Finish"

![image](https://github.com/RC4ML/FpgaNIC/blob/gpu_hll/img/opentar3.jpg)

### 5. Download the bitstream

Right click "xilinx_tcf/Xilinx/221770205K038A" (server act_m4) and click "Open Target"

![image](https://github.com/RC4ML/FpgaNIC/blob/gpu_hll/img/downbit1.jpg)

Right click "xcu280_u55_0" and click "Program Device.."

![image](https://github.com/RC4ML/FpgaNIC/blob/gpu_hll/img/downbit2.jpg)

Click "..." to choose the bitstream

![image](https://github.com/RC4ML/FpgaNIC/blob/gpu_hll/img/downbit3.jpg)

Choose the bitstream for the experiments, such as pcie_benchmark.bit for Figure 3. the dictionary of bitstream is /home/atc_bitstream/bitstream

![image](https://github.com/RC4ML/FpgaNIC/blob/gpu_hll/img/opentar5.jpg)

Click "Program" to download the bitstrem to the FPGA

![image](https://github.com/RC4ML/FpgaNIC/blob/gpu_hll/img/downbit4.jpg)

### 6.Download the bitstream to the other machine

Open another terminal and repeat steps 1-5. In step 5, right click "xilinx_tcf/Xilinx/221770202700VA" (server act_m7). Then, download the same bitstream as server act_m4.

![image](https://github.com/RC4ML/FpgaNIC/blob/gpu_hll/img/opentar6.jpg)

### 7.Reboot server atc_m4 and atc_m7.

After the bitstreams are completely downloaded to the servers, open a terminal in atc_m4 and atc_m7 respectively, and reboot the server
```
$ sudo reboot
```

The driver will be loaded automatically, and the application can be executed in the ./sw or ./sw_dev dictionary.


## Build XDMA Driver
```
$  cd driver/
```

According to driver/README.md, build the driver and insmod the driver

## Build Software Project
```
$  cd sw/
```

According to sw/README.md, build the software project and run the application. 

Note: Some of the experimental programs are in the ./sw_dev folder! But the experimental instructions are all in sw/README.md.
```
$ cd sw_dev/
```

