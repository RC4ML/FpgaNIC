/*
 * Copyright (c) 2014-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in 
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <stdlib.h>
#include <memory.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <cuda.h>

using namespace std;

#include <gdrapi.h>
#include "common.hpp"

using namespace gdrcopy::test;



int main(int argc, char *argv[])
{
    size_t _size = 2*1024*1024;
    size_t copy_size = _size;
    size_t copy_offset = 0;
    int dev_id = 0;
    size_t size = (_size + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;

    ASSERTDRV(cuInit(0));

    int n_devices = 0;
    ASSERTDRV(cuDeviceGetCount(&n_devices));

    CUdevice dev;
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

        cout << "GPU id:" << n << "; name: " << dev_name 
            << "; Bus id: "
            << std::hex 
            << std::setfill('0') << std::setw(4) << dev_pci_domain_id
            << ":" << std::setfill('0') << std::setw(2) << dev_pci_bus_id
            << ":" << std::setfill('0') << std::setw(2) << dev_pci_device_id
            << std::dec
            << endl;
    }
    cout << "selecting device " << dev_id << endl;
    ASSERTDRV(cuDeviceGet(&dev, dev_id));

    CUcontext dev_ctx;
    ASSERTDRV(cuDevicePrimaryCtxRetain(&dev_ctx, dev));
    ASSERTDRV(cuCtxSetCurrent(dev_ctx));
    cout << "rounded size: " << size << endl;

    CUdeviceptr d_A;
    ASSERTDRV(gpuMemAlloc(&d_A, size));
    cout << "device ptr: " << hex << d_A << dec << endl;

    uint32_t *init_buf = NULL;
    init_buf = (uint32_t *)malloc(size);
    ASSERT_NEQ(init_buf, (void*)0);
    init_hbuf_walking_bit(init_buf, size);
	for(int i=0;i<160;i++){
		init_buf[i]=i;
	}
    gdr_t g = gdr_open();
    ASSERT_NEQ(g, (void*)0);


    gdr_mh_t mh;
    BEGIN_CHECK {

        BREAK_IF_NEQ(gdr_pin_buffer(g, d_A, size, 0, 0, &mh), 0);
		printf("page entries:%lu\n",m_page_table.page_entries);
		for(int i =0;i<30;i++){
			cout<<i<<":"<<hex<<m_page_table.pages[i]<<endl;
		}
        ASSERT_NEQ(mh, null_mh);

        void *map_d_ptr  = NULL;
        ASSERT_EQ(gdr_map(g, mh, &map_d_ptr, size), 0);
        cout << "map_d_ptr: " << map_d_ptr << endl;

        gdr_info_t info;
        ASSERT_EQ(gdr_get_info(g, mh, &info), 0);
        cout << "info.va: " << hex << info.va << dec << endl;
        cout << "info.mapped_size: " << info.mapped_size << endl;
        cout << "info.page_size: " << info.page_size << endl;
        cout << "info.mapped: " << info.mapped << endl;
        cout << "info.wc_mapping: " << info.wc_mapping << endl;

        // remember that mappings start on a 64KB boundary, so let's
        // calculate the offset from the head of the mapping to the
        // beginning of the buffer
        int off = info.va - d_A;
        cout << "page offset: " << off << endl;

        uint32_t *buf_ptr = (uint32_t *)((char *)map_d_ptr + off);
        // copy to GPU benchmark
        cout << "writing test, size=" << copy_size << " offset=" << copy_offset << " num_iters=" << endl;
        struct timespec beg, end;
        clock_gettime(MYCLOCK, &beg);
        gdr_copy_to_mapping(mh, buf_ptr + copy_offset/4, init_buf, copy_size);
        clock_gettime(MYCLOCK, &end);
		cout<<"1 data:"<<buf_ptr[1]<<endl;
        double woMBps;
        {
            double byte_count = (double) copy_size;
            double dt_ms = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
            double Bps = byte_count / dt_ms * 1e3;
            woMBps = Bps / 1024.0 / 1024.0;
            cout << "write BW: " << woMBps << "MB/s" << endl;
        }
		
		for(int i=0;i<16;i++){
			cout<<hex<<init_buf[i]<<endl;
		}
		int a;
		cout<<"start waiting input:\n";
		cin>>a;
		cout<<a<<endl;

        compare_buf(init_buf, buf_ptr + copy_offset/4, copy_size);

        // copy from GPU benchmark
        clock_gettime(MYCLOCK, &beg);
        gdr_copy_from_mapping(mh, init_buf, buf_ptr + copy_offset/4, copy_size);
        clock_gettime(MYCLOCK, &end);
		for(int i=0;i<16;i++){
			cout<<hex<<init_buf[16*i]<<endl;
		}
        double roMBps;
        {
            double byte_count = (double) copy_size;
            double dt_ms = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
            double Bps = byte_count / dt_ms * 1e3;
            roMBps = Bps / 1024.0 / 1024.0;
            cout << "read BW: " << roMBps << "MB/s" << endl;
        }

        cout << "unmapping buffer" << endl;
        ASSERT_EQ(gdr_unmap(g, mh, map_d_ptr, size), 0);

        cout << "unpinning buffer" << endl;
        ASSERT_EQ(gdr_unpin_buffer(g, mh), 0);
    } END_CHECK;

    cout << "closing gdrdrv" << endl;
    ASSERT_EQ(gdr_close(g), 0);

    ASSERTDRV(gpuMemFree(d_A));

    ASSERTDRV(cuDevicePrimaryCtxRelease(dev));

    return 0;
}