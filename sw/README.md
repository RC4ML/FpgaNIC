# Reproduce results in the paper

## Figure 3

Downloaded the ~/bitstream/pcie_benchmark.bit to the board of atc_m4 machine, reboot.

### Compile the code if it has not been done.
```
$ cd ~/xdma/sw/build
$ cmake ../src/
$ make

And then run the commands in ~/xdma/sw/build
```


1. To see the result of A100 read CPU memory latency, which is around 1.6us.
	```
	$ sudo ./dma-example -b 0
	```
	The expected result is as follows:
	```
	...
	A100 read CPU latency: 1.7539 us
	A100 read CPU latency: 1.60993 us
	A100 read CPU latency: 1.62908 us
	A100 read CPU latency: 1.63617 us
	...
	```
	Each line represents each read operation.

2. To see the result of A100 read FPGA latency, which is around 0.85us.
	```
	$ sudo ./dma-example -b 1 -g 1
	```
	The expected result is as follows:
	```
	...
	ignore it:0
	ignore it:0
	A100 read FPGA latency : 0.848227 us
	ignore it:0
	ignore it:0
	A100 read FPGA latency : 0.862411 us
	...
	```
	Ignore the line starts with ignore it, each line starts with 'A100 read FPGA latency' represents each read operation

3. To see the result of CPU read FPGA latency, which is around 0.9us.
	```
	$ sudo ./dma-example -b 1 -g 0
	```
	The expected result is as follows:
	```
	...
	CPU read FPGA latency: 0.856 us
	CPU read FPGA latency: 0.892 us
	CPU read FPGA latency: 0.88 us
	...
	```
	Each line represents each read operation.


## Figure 4

Downloaded the ~/bitstream/pcie_benchmark.bit to the board of atc_m4 machine, reboot.

### Compile the code if it has not been done.
```
$ cd ~/xdma/sw/build
$ cmake ../src/
$ make

And then run the commands in ~/xdma/sw/build
```

1. FPGA read/write A100 memory throughput:
	```
	$ sudo ./dma-example -b 2
	```
	This would take a few minutes to run with different bursts. The expected result is as follows:
	```
	...
	########### FPGA start reading memory
	busrt:32768 Bytes, ops:10000 mode:2
	FPGA read memory speed : 12.5862 GB/s
	########### end of this batch!


	########### FPGA start writing memory
	busrt:64 Bytes, ops:10000 mode:1
	FPGA write memory speed : 0.197554 GB/s
	########### end of this batch!
	...
	```
	Each block represents a batch of FPGA read or write A100 memory operations with different burst.


2. FPGA read/write CPU memory throughput:
	```
	$ sudo ./dma-example -b 2 -g 0
	```

## Figure 5(a)

Downloaded the ~/bitstream/tcp_latency.bit to the board of atc_m4 and atc_m7 machine, reboot them.
### Compile the code if it has not been done.
```
$ cd ~/xdma/sw_dev/sw/build
$ cmake ../src/
$ make

And then run the commands in ~/xdma/sw_dev/sw/build
```

Figure 5a is the TCP throughput on different data size.
```
On atc_m4:
sudo ./tcp_latency -n 4 -m 7

On atc_m7:
sudo ./tcp_latency -n 7 -m 4
```

## Figure 5(b)

Downloaded the ~/bitstream/tcp_benchmark.bit to the board of atc_m4 and atc_m7 machine, reboot them.
### Compile the code if it has not been done.
```
$ cd ~/xdma/sw_dev/sw/build
$ cmake ../src/
$ make

And then run the commands in ~/xdma/sw_dev/sw/build
```

Figure 5b is the TCP throughput on different packet size.

As shown in the figure below, modify the TCP packet size in vio of Vivado Hardware Manage on machines. The value is 64/128/256/512/1024/1408
![image](https://github.com/RC4ML/FpgaNIC/blob/gpu_hll/img/tcp_benchmark.jpg)
Then, run the test.
```
On atc_m4:
sudo ./tcp_benchmark -n 4 -m 7 -t 1

On atc_m7:
sudo ./tcp_benchmark -n 7 -m 4 -t 1 
```
On atc_m4:

After atc_m7 print conn success, press any key on atc_m4 to show the result，the result is as follows:
```
wait the client conn success, then press any key to show the result:
1
1
 rd_speed: 9.14285 Gbps
```
parameter after -t means session num, it can be 1/10/100/1000






## Figure 6


Downloaded the ~/bitstream/direct.bit to the board of atc_m4 and atc_m7 machine, reboot.

### Compile the code if it has not been done.
```
$ cd ~/xdma/sw/build
$ cmake ../src/
$ make

And then run the commands in ~/xdma/sw/build
```


This experiment require a different bitstream to be downloaded on both of atc_m4's and atc_m7's FPGA.


After reboot two servers, run command on both machine's '~/xdma/sw/build'

To run a network throughput test:
```
On atc_m7:
$ sudo ./dma-example -t server "several parameters"

On atc_m4:
sudo ./dma-example -t client "several parameters (must be the same with the ones used on atc_m7)"
```
If atc_m4 prints that there is no connection, maybe the server(atc_m7) is not ready when atc_m4 tried to connect it. Just press ctrl+C on atc_m4 and run the command again.

You can see following lines in atc_m7:
```
data recv done! 332929 speed: 8.271752
data recv done! 332828 speed: 8.274263
data recv done! 332208 speed: 8.289705
data recv done! 332770 speed: 8.275705
```
The overall throughput is the slowest one of one, instead of the sum of them. Press ctrl+C to exit in both machine.

1. Figure 6a
	Figure 6a is the throughput on different slot size.
	```
	On atc_m7:
	sudo ./dma-example -t server -b 4 -g 1 -s 128 -m 128

	On atc_m4:
	sudo ./dma-example -t client -b 4 -g 1 -s 128 -m 128

	parameter after -m (max slot size, in kilo bytes) can be 16/32/64/128. If you want to try '-m 4' and '-m 8', you must change '-s' too. '-s 64 -m 8' and '-s 32 -m 4'
	```
	

2. Figure 6b
	Figure 6b is the throughput with or without control panel offloading on different slot size.
	
	With control panel offloading:
	```
	On atc_m7:
	sudo ./dma-example -t server -b 4 -g 1 -s 128 -m 1024

	On atc_m4:
	sudo ./dma-example -t client -b 4 -g 1 -s 128 -m 1024

	parameter after -m can be 64/128/256/512/1024
	```

	Without control panel offloading:
	```
	On atc_m7:
	sudo ./dma-example -t server -b 5 -g 1 -s 128 -m 1024

	On atc_m4:
	sudo ./dma-example -t client -b 5 -g 1 -s 128 -m 1024

	parameter after -m can be 64/128/256/512/1024
	```

3. Figure 6c
	Figure 6c is the throughput with or without control panel offloading on different transfer size.
	With control panel offloading:
	```
	On atc_m7:
	sudo ./dma-example -t server -b 4 -g 1 -m 64 -s 512

	On atc_m4:
	sudo ./dma-example -t client -b 4 -g 1 -m 64 -s 512

	parameter after -m can be 64/128/256/512/1024
	```

	Without control panel offloading:
	```
	On atc_m7:
	sudo ./dma-example -t server -b 5 -g 1 -m 64 -s 512

	On atc_m4:
	sudo ./dma-example -t client -b 5 -g 1 -m 64 -s 512

	parameter after -s (transfer size, in megabytes) can be 2/4/8/16/32/64/128/256/512
	```

## Figure 8

Downloaded the ~/bitstream/allreduce.bit to the board of atc_m4 and atc_m7 machine, reboot them.
### Compile the code if it has not been done.
```
$ cd ~/xdma/sw_dev/sw/build
$ cmake ../src/
$ make

And then run the commands in ~/xdma/sw_dev/sw/build
```

Figure 8 is the AllReduce throughput on different packet size.

```
On atc_m4:
sudo ./allreduce -n 0 -m 2

On atc_m7:
sudo ./allreduce -n 1 -m 2
```
On atc_m4,atc_m7:

After all the node prepare, press any key to continue, the result is as follows:
```
 data size: 16777216
After all the node prepare, press any key to continue:
1
 allreduce_throughput: 10.8451 GB/s
```


## Part of Figure 9

Downloaded the ~/bitstream/direct.bit to the board of atc_m4 and atc_m7 machine, reboot.

### Compile the code if it has not been done.
```
$ cd ~/xdma/sw/build
$ cmake ../src/
$ make

And then run the commands in ~/xdma/sw/build
```

A100 needs 8 SMs to consume 100 Gbps HLL data stream.
```
using 4 SMs
$ sudo ./dma-example -b 3 -h 4

using 8 SMs
$ sudo ./dma-example -b 3 -h 8
```
The throughput result is as follows (After folloing lines show up, press ctrl+C to exit the program, it will not exit itself): 
```
speed: 8.375580 GB/s
speed: 8.373021 GB/s
speed: 8.344674 GB/s
speed: 8.072473 GB/s
```
Each line represent the speed calculated by the total length and duration calculated by each SM. So the overall speed is the slowest speed among them instead of the sum of them. We can see 4 SMs can only consume 8GB/s (64Gb/s) HLL data stream. Such that we need 8 SMs. 

## Figure 9

Downloaded the ~/bitstream/hll.bit to the board of atc_m4 and atc_m7 machine, reboot them.
### Compile the code if it has not been done.
```
$ cd ~/xdma/sw_dev/sw/build
$ cmake ../src/
$ make

And then run the commands in ~/xdma/sw_dev/sw/build
```

Figure 9 is the HLL throughput on different packet size.
```
On atc_m4:
sudo ./hyperloglog -n 4 -m 7 -p 64

On atc_m7:
sudo ./hyperloglog -n 7 -m 4 -p 64 
```
On atc_m4:

After atc_m7 print conn success, press any key on atc_m4 to show the result，the result is as follows:
```
wait the client conn success, then press any key to show the result:
1
--------ATC FIGURE 9: the payloadsize is 512 -------
, write_throughput: 84.7399 Gbps
, write+hll_throughput: 84.9599 Gbps

```
parameter after -p means packet size, it can be 64/128/256/512/1024/1408