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




## Cite this work
If you use it in your paper, please cite our work ([full version](https://www.usenix.org/conference/atc22/presentation/wang-zeke)).
```
@inproceedings{wang_atc22,
  title={FpgaNIC: An FPGA-based Versatile 100Gb SmartNIC for GPUs},
  author={Zeke Wang and Hongjing Huang and Jie Zhang and Fei Wu and Gustavo Alonso},
  year={2022},
  booktitle={2022 USENIX Annual Technical Conference (ATC)},
}
```

