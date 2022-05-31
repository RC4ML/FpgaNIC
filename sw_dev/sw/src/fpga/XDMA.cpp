/*
 * Copyright (c) 2018, Systems Group, ETH Zurich
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "XDMA.h"
#include <sys/mman.h>
#include <sys/ioctl.h>
#include "../../../driver/xdma_ioctl.h"

#include <fpga/Configuration.h>



namespace fpga {

XDMAController* XDMA::controller = nullptr;
MemoryManager* XDMA::mm = nullptr;
int XDMA::nodeId;
int XDMA::fd;
int XDMA::byfd;
int XDMA::hfd;
void* XDMA::huge_base = nullptr;


void XDMA::setNodeId(int _nodeId) {
   nodeId = _nodeId;
}

void XDMA::initializeMemory() {
   //Open huge pages device
   if ((hfd = open("/media/huge/abc", O_CREAT | O_RDWR | O_SYNC, 0755)) == -1) {
      std::cerr << "[ERROR] on open /media/huge/abc, maybe you need to add 'sudo'";
      exit(1);
   }
   printf("huge device /media/huge/abc opened.\n"); fflush(stdout);

   //Map huge pages
   huge_base = mmap(0, fpga::Configuration::DMA_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, hfd, 0);
   if (huge_base == (void*) -1) {
      std::cerr << "[ERROR] on mmap of huge_base";
      exit(1);
   }
   printf("huge device mapped at %p\n", huge_base);

   //Open xdma_control device
   if ((fd = open("/dev/xdma_control", O_RDWR | O_SYNC)) == -1) {
      std::cerr << "[ERROR] on open /dev/xdma_control";
      exit(1);
   }

   //Open xdma_bypass device
   if ((byfd = open("/dev/xdma_bypass", O_RDWR | O_SYNC)) == -1) {
      std::cerr << "[ERROR] on open /dev/xdma_bypass";
      exit(1);
   }

   printf("dma device /dev/xdma_control opened.\n"); fflush(stdout);
   printf("bypass device /dev/xdma_bypass opened.\n"); fflush(stdout);
   if  (controller == nullptr) {
      controller = new XDMAController(fd, byfd);
   }

   printf("huge_base: %p\n", huge_base);
   //ioctl to get mapping
   struct xdma_huge huge;
   huge.addr = (unsigned long) huge_base;
   huge.size = (unsigned long) fpga::Configuration::DMA_SIZE;
   printf("IOCTL_XDMA_BUFFER_Set\n"); fflush(stdout);

   if (ioctl(fd, IOCTL_XDMA_BUFFER_SET, &huge) == -1) {
      printf("IOCTL SET failed.\n");
      //return -1;
   }

   //Get mapping from driver
   struct xdma_huge_mapping map;
   map.npages = huge.size / (2 * 1024 * 1024);
   map.dma_addr = (unsigned long*) calloc(map.npages, sizeof(unsigned long*));
   printf("IOCTL_XDMA_MAPPING_GET\n"); fflush(stdout);

   if (ioctl(fd, IOCTL_XDMA_MAPPING_GET, &map) == -1) {
      printf("IOCTL GET failed.\n");
      //return -1;
   }

   //Insert TLB entries
   printf("npages: %lu, write TLB\n", map.npages); fflush(stdout);

   unsigned long vaddr = (unsigned long) huge_base;

#if TLB_PAGE_SIZE == 64
   unsigned int offset = 0;
   for (int i = 0; i < map.npages; i++) { 
      for(int j=0; j< 2048/64; j++){
         controller->writeTlb(vaddr, map.dma_addr[i]+offset  , (i == 0)&&(j==0));
         vaddr += (64*1024);
         offset += (64*1024);
      }
      offset = 0;
   }

#else

   for (int i = 0; i < map.npages; i++) { 
      controller->writeTlb(vaddr, map.dma_addr[i], (i == 0));
      vaddr += (2*1024*1024);
   }

#endif
   printf("TLB done\n"); fflush(stdout);
   //free(map.dma_addr);
   
   mm = new MemoryManager((void*) huge.addr, huge.size);

   //return 0;
}

void XDMA::clear() {
   delete controller;
   delete mm;
   munmap(huge_base, fpga::Configuration::DMA_SIZE); //Check return???
   close(fd);
   close(byfd);
   close(hfd);
}


void* XDMA::allocate(uint64_t size) {
   return mm->allocate(size);
}

void XDMA::free(void* ptr) {
   mm->free(ptr);
}


} /* namespace fpga */
