/*
 * Copyright 2019 - 2020, RC4ML, Zhejiang University
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

#include "main.h"
#include "tool/log.hpp"
#include "common.hpp"
// #include "cuda/inference.cuh"
#include "cuda/interface.cuh"
#include "cuda/util.cuh"
#include "cuda/test.cuh"
#include "cuda/app.cuh"
#include "cuda/hll.cuh"
#include "cuda/kvs.cuh"
// #include "cuda/cu_torch.cuh"


#include "tool/test.hpp"
#include "tool/input.hpp"
#include <fstream>
#include <iostream>

extern "C"

void useCUDA();
void write(void* addr);
using namespace std;
using namespace gdrcopy::test;


#define SHOWINFO 0
size_t gpu_mem_size = size_t(4)*1024*1024*1024;//bytes  220*1024*1024  size_t(1)*1024*1024*1024
int dev_id = 0;//gpu device
CUdevice dev;
CUcontext dev_ctx;
CUdeviceptr d_A;
uint32_t *init_buf;
gdr_t g;
gdr_mh_t mh;
uint32_t *buf_ptr;
void *map_d_ptr  = NULL;
int app_type = -1;
int node_index = -1;
int benchmark_type = 0;
int is_gpu_tlb = 1;
size_t max_block_size_kilobyte = 64;
size_t transfer_megabyte = 2;
int hll_sm_num = 4;

void get_opt(int argc, char *argv[]){
	int o;  // getopt() 的返回值
    const char *optstring = "t:n:m:b:g:s:h:"; // 设置短参数类型及是否需要参数

     while ((o = getopt(argc, argv, optstring)) != -1) {
        switch (o) {
            case 't':
				if(optarg==string("server")){
					cjdebug("app_type:server\n");
					app_type = 1;
				}else if(optarg==string("client")){
					cjdebug("app_type:client\n");
					app_type = 0;
				}else{
					cjerror("Error app_type!\n");
				}
                break;
			case 'n':
				node_index = atoi(optarg);
				cjdebug("node_index:%d\n",node_index);
				break;
			case 'b':
				benchmark_type = atoi(optarg);
				break;
			case 'g':
				is_gpu_tlb = atoi(optarg);
				break;
			case 'm':
				max_block_size_kilobyte = atoi(optarg);
				break;
			case 's':
				transfer_megabyte = atoi(optarg);
				break;
			case 'h':
				hll_sm_num = atoi(optarg);
				break;
            case '?':
                cjerror("error optopt: %c\n", optopt);
                cjerror("error opterr: %d\n", opterr);
                break;
        }
    }
}

int main(int argc, char *argv[]) {

	// printf("fre:%f\n",get_fre());
	// return 0;
	get_opt(argc,argv);
	cjdebug("is_gpu_tlb:%d\n",is_gpu_tlb);
	cjdebug("benchmark_type:%d\n",benchmark_type);
	cjdebug("transfer_megabyte:%ld\n",transfer_megabyte);
	cjdebug("max_block_size_kilobyte:%ld\n",max_block_size_kilobyte);
	cjdebug("hll_sm_num:%d\n",hll_sm_num);
	if(is_gpu_tlb){
		set_page_table();
		for(unsigned int i=0;i<m_page_table.page_entries-1;i++){
			size_t t = m_page_table.pages[i+1]-m_page_table.pages[i];
			if(t!=65536){
				cout<<t<<"###############################error!\n";
			}
		}
	}

	
	param_test_t param;
	param.controller = fpga::XDMA::getController();
	if(is_gpu_tlb){
		param.map_d_ptr = (void *)d_A;
		param.tlb_start_addr = (unsigned int *)d_A;
		param.d_mem_cpu = (void *)buf_ptr;
	}else{
		uint64_t* dmaBuffer =  (uint64_t*) fpga::XDMA::allocate(2*1024*1024);//1024*1024*480
		param.map_d_ptr = (void *)0;
		param.tlb_start_addr = (unsigned int *)dmaBuffer;
	}
	
	// {
	// 	kvs_benchmark(param);
	// 	sleep(3);
	// 	start_cmd_control(param.controller);
	// }
	// {
		// hll_simple_dma_benchmark(param);
		// sleep(3);
		// start_cmd_control(param.controller);
	// }
	// {
	// 	test_2080(param);
	// 	sleep(3);
	// 	start_cmd_control(param.controller);
	// }
	
	if(benchmark_type == 0){ //test latency and throughput between gpu and cpu
		printf("ATC::Figure 3, A100 read CPU latenct test\n");
		int stride=32*1024*1024;
		for(int i=0;i<1;i++){
			test_simple(stride);
		}
	}

	if(benchmark_type == 1){//test fpga latency with gpu or cpu
		printf("ATC::Figure 3, A100 read FPGA and CPU read FPGA latenct test\n");
		if(is_gpu_tlb){
			cjinfo("test gpu-fpga latency\n");
			for(int i=0;i<10;i++){
				test_latency_fpga_gpu(param);
			}
		}else{
			cjinfo("test cpu-fpga latency\n");
			for(int i=0;i<10;i++){
				test_latency_fpga_cpu(param);
			}
		}
	}
		

	{//gpu mem throughput test on block nums and threads
		//test_gpu_throughput(param);
	}

	{
		//cj_debug(param);
	}

	if(benchmark_type == 4){ //smart nic, direct send/recv
		printf("ATC::Figure 6a, A100 send to A100 speed, offload control panel\n");
		socket_send_test(param);
		sleep(3);
		start_cmd_control(param.controller);
	}

	if(benchmark_type == 5){ //do not offload control panel
		printf("ATC::Figure 6bc, A100 send to A100 speed, do not offload control panel\n");
		socket_send_test_offload_control(param);
		sleep(3);
		start_cmd_control(param.controller);
	}
	
	{
		// mpi_allreduce(param);
		// sleep(3);
		// start_cmd_control(param.controller);
	}

	{
		// inference_sample(param);
		// sleep(3);
		// start_cmd_control(param.controller);
	}


	if(benchmark_type == 2){
		printf("ATC::Figure 4, FPGA read memory and FPGA write memory throughput test\n");
		ofstream outfile;
		int burst = 64;
		int ops =10000;
		for(int i=0;i<10;i++){
			pressure_test(param,burst,ops,2);
			burst *= 2;
		}
		outfile.open("data.txt", ios::out |ios::app );
		outfile<<endl;
		outfile.close();

		burst = 64;
		for(int i=0;i<10;i++){
			pressure_test(param,burst,ops,1);
			burst *= 2;
		}
		outfile.open("data.txt", ios::out |ios::app );
		outfile<<endl;
		outfile.close();
	}
	if(benchmark_type == 3){
		printf("ATC:: test hll throughput with %d SMs, each with 512 threads\n",hll_sm_num);
		hll_sample(param);
		sleep(3);
		start_cmd_control(param.controller);
	}
	
	// uint64_t r_addr = controller ->getBypassAddr(0);
	// cout<<"addr:"<<r_addr<<endl;
	// uint64_t* a = (uint64_t*)r_addr;
	// for(int i=0;i<8;i++){
	// 	a[i]=i;
	// }
    // fpga::XDMA::clear();
    // close_device();
	return 0;
}

void set_page_table(){
	size_t copy_size = gpu_mem_size;
	size_t size = (gpu_mem_size + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;
	size_t copy_offset = 0;
	if(SHOWINFO){
		cout << "rounded size: " << size << endl;
	}
	ASSERTDRV(cuInit(0));
	int n_devices = 0;
    ASSERTDRV(cuDeviceGetCount(&n_devices));
	for (int n=0; n<n_devices; ++n) {
        char dev_name[256];
        int dev_pci_domain_id;
        int dev_pci_bus_id;
        int dev_pci_device_id;
        ASSERTDRV(cuDeviceGet(&dev, n));
        ASSERTDRV(cuDeviceGetName(dev_name, sizeof(dev_name) / sizeof(dev_name[0]), dev));
        ASSERTDRV(cuDeviceGetAttribute(&dev_pci_domain_id, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, dev));
        ASSERTDRV(cuDeviceGetAttribute(&dev_pci_bus_id, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, dev));
        ASSERTDRV(cuDeviceGetAttribute(&dev_pci_device_id, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, dev));
		if(SHOWINFO){
			cout << "GPU id:" << n << "; name: " << dev_name 
            << "; Bus id: "
            << std::hex 
            << std::setfill('0') << std::setw(4) << dev_pci_domain_id
            << ":" << std::setfill('0') << std::setw(2) << dev_pci_bus_id
            << ":" << std::setfill('0') << std::setw(2) << dev_pci_device_id
            << std::dec
            << endl;
		}
        
    }//output device info
	if(SHOWINFO){
		cout << "selecting device " << dev_id << endl;
	}
	ASSERTDRV(cuDeviceGet(&dev, dev_id));
	ASSERTDRV(cuDevicePrimaryCtxRetain(&dev_ctx, dev));
    ASSERTDRV(cuCtxSetCurrent(dev_ctx));
	ASSERTDRV(gpuMemAlloc(&d_A, size));
	if(SHOWINFO){
		cout<<"device malloc done!\n";
	}
	init_buf = (uint32_t *)malloc(size);
	ASSERT_NEQ(init_buf, (void*)0);
	if(SHOWINFO){
		cout<<"initbuf malloc done!\n";
	}
	g = gdr_open();
	if(SHOWINFO){
		cout<<"gdr opened!\n";
	}
	ASSERT_NEQ(g, (void*)0);

	do{
		BREAK_IF_NEQ(gdr_pin_buffer(g, d_A, size, 0, 0, &mh), 0);
		if(SHOWINFO){
			cjdebug("page entries:%lu\n",m_page_table.page_entries);
			for(int i =0;i<30;i++){
				cout<<i<<":"<<hex<<m_page_table.pages[i]<<endl;
			}
		}
		
		ASSERT_NEQ(mh, null_mh);

		ASSERT_EQ(gdr_map(g, mh, &map_d_ptr, size), 0);

		gdr_info_t info;
        ASSERT_EQ(gdr_get_info(g, mh, &info), 0);
		if(SHOWINFO){
			cout <<	"d_A: " << hex << d_A << dec << endl;
			cout <<	"map_d_ptr: " << hex << map_d_ptr << dec << endl;
			cout << "info.va: " << hex << info.va << dec << endl;
			cout << "info.mapped_size: " << info.mapped_size << endl;
			cout << "info.page_size: " << info.page_size << endl;
			cout << "info.mapped: " << info.mapped << endl;
			cout << "info.wc_mapping: " << info.wc_mapping << endl;
		}
        

		int off = info.va - d_A;
		if(SHOWINFO){
			cout<<"offset:"<<off<<endl;
		}
		buf_ptr = (uint32_t *)((char *)map_d_ptr + off);
		gdr_copy_to_mapping(mh, buf_ptr + copy_offset/4, init_buf, copy_size);
		if(SHOWINFO){
			cout<<"buf_ptr:"<<buf_ptr<<endl;
			cout<<"write to gpu mem done!\n";
		}
	}while(0);
	
}
void copy_to_cpu(){
	size_t copy_offset = 0;
	size_t copy_size = gpu_mem_size;
	gdr_copy_from_mapping(mh, init_buf, buf_ptr + copy_offset/4, copy_size);
}
void close_device(){
	size_t size = (gpu_mem_size + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;
	cjinfo("unmapping buffer\n" );

	ASSERT_EQ(gdr_unmap(g, mh, map_d_ptr, size), 0);
	ASSERT_EQ(gdr_unpin_buffer(g, mh), 0);
	cjinfo("closing gdrdrv\n" );
	ASSERT_EQ(gdr_close(g), 0);
	ASSERTDRV(gpuMemFree(d_A));
	ASSERTDRV(cuDevicePrimaryCtxRelease(dev));
}