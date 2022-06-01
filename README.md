# FPGANIC

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

#### 3. make HLS IP Core
```
$ make installip
```

#### 4. Create vivado project (You can choose one project to create)

##### a. create direct project(Figure 6)
```
$ make direct
```
##### b. create pcie_benchmark project(Figure 3 4)
```
$ make pcie_benchmark
```
##### c. create tcp_latency project(Figure 5a)
```
$ make tcp_latency
```
##### d. create tcp_benchmark project(Figure 5b)
```
$ make tcp_benchmark
```
##### e. create allreduce project(Figure 7 8)
```
$ make allreduce
```
##### f. create hyperloglog project(Figure 9)
```
$ make hyperloglog
```



#### 5. Generate bitstream

-open the project by Vivado2020.1 and generate bitstream

## Build XDMA Driver
```
$  cd driver/
```

According to driver/README.md, build the driver and insmod the driver

## Build Software Project
```
$  cd sw/
```

According to sw/README.md, build the software project and run the application