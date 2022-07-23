# FpgaNIC
FpgaNIC is an FPGA-based, GPU-centric, versatile SmartNIC 

1, that enables direct PCIe P2P communication with local GPUs using GPU virtual address, 

2, that allows GPUs to directly manipulate FpganIC without CPU intervention, 

3, that provides reliable 100Gb network access to remote GPUs, and 

4, that allows to offload various complex compute tasks to a customized data-path accelerator for line-rate in-network computing on the FPGA, thereby complementing the processing at the GPU. 

Besides, 


## Check-list
1. At least two nodes, each has a GPU that supports NVIDIA GPUDirect and the Xilinx U280 or U50 card.

2. Each FPGA card is connected to a 100Gbps Ethernet switch.

3. FPGA card and GPU are connected to the same PCIe switch.

4. Host OS: Linux 4.15.0-20-generic 

5. Nvidia Driver Version: 450.51.05 

6. CUDA Version: 11.0

7, Make sure that each server has enabled Hugepages. 

## How to run Experiment: Three steps.
There are three steps to run each experiment. Before running FpgaNIC, please clone the source code:

$ git clone https://github.com/RC4ML/FpgaNIC

### Hardware: FPGA Bitstream
1. $ mkdir build && cd build 

2.  $ cmake ..

3. Make HLS IP core
    $ make installip
    
4. Create vivado project, add the hardware project option after make.
     $ make pcie_benchmark
     
5. Now the hardware project is produced, generate bitstream using vivado and flush it to every FPGA card.

6. Every time you download the bitstream to the FPGA, you have to reboot the machine, do not forget to reinstall
xdma driver and GDR driver.


### Software: Driver Installation
1. $ cd FpgaNIC/driver

2. $ make && sudo insmod xdma_driver.ko

3. $ cd FpgaNIC/gdrcopy

4. $ sudo ./insmod.sh

5. Note that you need to reinstall xdma driver and gdr driver every time you reboot your machine.

### Software: Running Application Code
1. $ cd FpgaNIC/sw && mkdir build && cd build

2. $ cmake ../src

3. $ make

4. $ sudo ./dma-example -b 0

5. $ Above command would report GPU read CPU memory latency, for more details, please refer to sw/README.md




## Cite this work
If you use it in your paper, please cite our work
```
@inproceedings{wang_atc22,
  title={FpgaNIC: An FPGA-based Versatile 100Gb SmartNIC for GPUs},
  author={Zeke Wang and Hongjing Huang and Jie Zhang and Fei Wu and Gustavo Alonso},
  year={2022},
  booktitle={2022 USENIX Annual Technical Conference (ATC)},
}
```

