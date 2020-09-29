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
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include<string>
#include <fstream>
#include <iomanip>
#include <bitset>
#include <cuda.h>
#include <iostream>
#include <memory.h>
#include <gdrapi.h>
#include "main.h"
#include "common.hpp"
#include "cuda/interface.cuh"
#include "tool/test.hpp"
extern "C"
void useCUDA();
void write(void* addr);
using namespace std;
using namespace gdrcopy::test;


#define SHOWINFO 0
size_t gpu_mem_size = size_t(8)*1024*1024*1024;//bytes  220*1024*1024  size_t(8)*1024*1024*1024
int dev_id = 0;//gpu device
CUdevice dev;
CUcontext dev_ctx;
CUdeviceptr d_A;
uint32_t *init_buf;
gdr_t g;
gdr_mh_t mh;
uint32_t *buf_ptr;
void *map_d_ptr  = NULL;


int main(int argc, char *argv[]) {
	set_page_table();
	if(SHOWINFO){
		cout<<"m_page_table.page_entries:"<<m_page_table.page_entries<<endl;
	}
	for(int i=0;i<m_page_table.page_entries-1;i++){
		size_t t = m_page_table.pages[i+1]-m_page_table.pages[i];
		if(t!=65536){
			cout<<t<<"###############################error!\n";
		}
	}
	param_test_t param;
	param.controller = fpga::XDMA::getController();
	uint64_t* dmaBuffer =  (uint64_t*) fpga::XDMA::allocate(1024);//1024*1024*480
	param.addr = (uint64_t)dmaBuffer;
	param.cpu_buf = init_buf;
	param.map_d_ptr = (void *)d_A;
	param.mem_size = gpu_mem_size;
	param.tlb_start_addr = (unsigned int *)dmaBuffer;

	// param.controller->writeReg(160,(unsigned int)param.addr);
	// param.controller->writeReg(161,(unsigned int)(param.addr>>32));

	//stream_transfer(param);
	socket_send_test(param);
	sleep(1000);
	// uint64_t r_addr = controller ->getBypassAddr(0);
	// cout<<"addr:"<<r_addr<<endl;
	// uint64_t* a = (uint64_t*)r_addr;
	// for(int i=0;i<8;i++){
	// 	a[i]=i;
	// }


    fpga::XDMA::clear();
    close_device();
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
			printf("page entries:%lu\n",m_page_table.page_entries);
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
	if(SHOWINFO){
		cout << "unmapping buffer" << endl;
	}
	ASSERT_EQ(gdr_unmap(g, mh, map_d_ptr, size), 0);
	ASSERT_EQ(gdr_unpin_buffer(g, mh), 0);
	if(SHOWINFO){
		cout << "closing gdrdrv" << endl;
	}
	ASSERT_EQ(gdr_close(g), 0);
	ASSERTDRV(gpuMemFree(d_A));
	ASSERTDRV(cuDevicePrimaryCtxRelease(dev));
}