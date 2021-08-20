# How to create example module xdma 

## Build example project

## Rrerequisites

```
* GPU and FPGA are under the same PCIe switch.
* Linux 4.15.0-20-generic;  Nvidia Driver Version: 450.51.05;  CUDA Version: 11.0
```

### 1. Create build directory
```
$ mkdir build
$ cd build
```

#### 2. Configure xdma project build
```
$ cmake .. -DBYPASS=disable

```

All options:
| Name                  | Values                       | Desription                                           |
|-----------------------|------------------------------|------------------------------------------------------|
| BYPASS                | <enable,disable>             | enable/disable xdma bypass interface                 |

### 3. make HLS IP Core
```
$ make installip
```

### 4. Create vivado project
```
$ make project
```


# How to add xdma module in your project

## 1.build HLS IP Core

### 1). Create build directory
```
$ mkdir build
$ cd build
```

### 2). Configure xdma
```
$ cmake .. 

```

### 3). make HLS IP Core

```
$ make installip 

```

## 2.Copy files to your project

### 1). copy the folders and files to your project 
copy ```/rtl, /iprepo, /constraints ``` to the path: $prj_dir/$project_name.srcs/sources_1/

you can use these tcl cmd to get the path
```
set prj_dir [file normalize [get_property DIRECTORY [current_project]]]
set project_name [get_property NAME [current_project]]
```

### 2). copy tcl files to your project
copy ```scripts/add_xdma_module.tcl, scripts/add_xdma_bypass_module.tcl``` to the path:$prj_dir/

## 3.Run TCL file in your project

### a. If you want create xdma module without bypass interface, run the cmd in vivado Tcl Console
```
source ./add_xdma_module.tcl
```

### b. If you want create xdma module with bypass interface, run the cmd in vivado Tcl Console

```
source ./add_xdma_bypass_module.tcl
```

## 4.add BYPASS define in ```example_module.vh```

if you want create xdma module with bypass interface, add these codes in ```example_module.vh``` file's end
```
    `ifndef XDMA_BYPASS
    `define XDMA_BYPASS
    `endif
```


# How to use xdma module 

## Hardware Part of XDMA
### 1.Instantiate module ```dma_inf``` for XDMA

all PHY INTERFACE connect to IO ports

All  USER INTERFACE:
| Name                  | in/out                       | Desription                                           |
|-----------------------|------------------------------|------------------------------------------------------|
| pcie_clk              | output                       | pcie output clock, 250M Fre. for internal pcie user interface                     |
| pcie_aresetn          | output                       | pcie output rstn, for internal pcie user interface                      |
| user_clk              | input                        | input user clock, for all xdma user interface         |
| user_aresetn          | input                        | input user rstn, for all xdma user interface                         |
| s_axis_dma_read_cmd   | axis_mem_cmd                 | xdma read cmd interface                        |
| s_axis_dma_write_cmd  | axis_mem_cmd                 | xdma write cmd interface                        |
| m_axis_dma_read_data  | axi_stream                   | xdma read data interface               |
| s_axis_dma_write_data | axi_stream                   | xdma write data interface               |
| fpga_control_reg      | output                       | xdma control registers ,CPU send massages to FPGA by these regs               |
| fpga_status_reg       | input                        | xdma status registers ,FPGA send massages to CPU by these regs               |
| bypass_control_reg    | output                       | bypass control registers ,CPU send massages to FPGA by these regs               |
| bypass_status_reg     | input                        | bypass status registers ,FPGA send massages to CPU by these regs               |

### 2.read data from XDMA

FPGA send a read command to XDMA module ,including starting address and data length.
Then FPGA read data from the AXI-Stream interface : m_axis_dma_read_data.

### 3.write data to XDMA

FPGA send a write command to XDMA module ,including starting address and data length.
Then FPGA write data from the AXI-Stream interface : s_axis_dma_write_data.

## Software Part of XDMA

### 1. Prerequisites
#### a. Install the following package (cmake) for Ubuntu:
```
$ sudo apt install libboost-program-options-dev cmake
```
#### b. Make sure you have sudo priority, which is required when installing PCIe kernel and running related application code. 


### 2. Setup Hugepages
#### a. Create a group for users of hugepages, and retrieve its GID (in this example, 1001) then add yourself to the group.
```
$ sudo groupadd hugetlbfs

$ sudo getent group hugetlbfs

$ sudo adduser dasidler hugetlbfs
```

#### b. Edit `/etc/sysctl.conf` and add this text to specify the number of pages you want to reserve (see page-size)
```
# Allocate 8192*2MiB for HugePageTables
vm.nr_hugepages = 8192

# Members of group hugetlbfs(1001) can allocate "huge" shared memory segments
vm.hugetlb_shm_group = 1001
```
#### c. Create a mount point for the file system
```
$ mkdir /media/huge
```

#### d. Add this line in `/etc/fstab` (The mode 1770 allows anyone in the group to create files but not unlink or rename each other's files.)
```
# hugetlbfs
hugetlbfs /media/huge hugetlbfs mode=1770,gid=1001 0 0
```

#### e. Reboot

#### f. Add the following line to `/etc/security/limits.conf` to configure the amount of memory a user can lock, so an application can't crash your operating system by locking all the memory. 
```
@hugetlbfs	hard	memlock		1048576
```



### 3. Kernel Part
Loading PCIe kernel module if not loaded yet. 
```
$ cd driver
$ make clean
$ make
$ sudo insmod xdma_driver.ko
```
Please make sure your kernel module is successfully installed for Ubuntu.


### 4. Application Part
#### a. Compile application code
```
$ cd ..
$ mkdir build && cd build
$ cmake ../src
$ make
```
#### b.use xdma 

```
fpga::XDMAController* controller = fpga::XDMA::getController(); //initial xdma memory
uint64_t* dmaBuffer =  (uint64_t*) fpga::XDMA::allocate(1024*1024*2*256); //allocate dma buffer. read/write dma by the buffer
```   

#### c.use xdma register
```
controller ->writeReg(unsigned int ID,unsigned int data); //write message to fpga_control_reg

controller ->readReg(unsigned int ID);      //read message to fpga_control_reg & fpga_status_reg
```

#### d.use xdma bypass register
```
controller ->writeBypassReg(unsigned int ID,uint64* data);//write message to bypass_control_reg
controller ->readBypassReg(unsigned int ID,uint64* data);//read message to bypass_control_reg & bypass_status_reg
```