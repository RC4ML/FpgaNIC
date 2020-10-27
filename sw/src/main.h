#ifndef __MAIN_H__
#define __MAIN_H__

#include "fpga/XDMA.h"
#include "fpga/XDMAController.h"

void set_page_table();
void copy_to_cpu();
void close_device();

extern int app_type;

typedef struct param_test{
	fpga::XDMAController* controller;
	uint64_t addr;
	uint32_t *cpu_buf;
	void *map_d_ptr;
	uint32_t mem_size;
	unsigned int* tlb_start_addr;
}param_test_t;

typedef struct param_mover{
	void* write_count_addr0;
	void* write_count_addr1;
	void* write_count_addr2;
	void* write_count_addr3;

	void* read_count_addr0;
	void* read_count_addr1;
	void* read_count_addr2;
	void* read_count_addr3;

	unsigned int* dev_addr0;
	unsigned int* dev_addr1;
	unsigned int* dev_addr2;
	unsigned int* dev_addr3;

	uint64_t data_length;
	uint64_t buffer_pages;
	uint32_t offset;
}param_mover_t;

typedef struct param_interface_socket{
	unsigned int* buffer_addr;
	unsigned int mac,ip,port;
	unsigned int* tlb_start_addr;
	fpga::XDMAController* controller;
}param_interface_socket_t;
#endif