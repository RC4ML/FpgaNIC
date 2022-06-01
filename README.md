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

#### 4. Create vivado project
```
$ make project
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