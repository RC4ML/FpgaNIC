#include "XDMAController.h"

#include <cstring>
#include <thread>
#include <chrono>
              
#include <fstream>
#include <iomanip>
#include <immintrin.h>
//#define PRINT_DEBUG

using namespace std::chrono_literals;
using namespace std;

namespace fpga {

std::mutex XDMAController::ctrl_mutex;
std::mutex XDMAController::btree_mutex;
std::atomic_uint XDMAController::cmdCounter = ATOMIC_VAR_INIT(0);
uint64_t XDMAController::mmTestValue;

XDMAController::XDMAController(int fd, int byfd)
{
   //open control device
   m_base = mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
   //open bypass device
   by_base =  mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, byfd, 0);
}

XDMAController::~XDMAController()
{
   if (munmap(m_base, MAP_SIZE) == -1)
   {
      std::cerr << "Error on unmap of control device" << std::endl;
   }

   if (munmap(by_base, MAP_SIZE) == -1)
   {
      std::cerr << "unmap of bypass device" << std::endl;
   }
}



void XDMAController::writeTlb(unsigned long vaddr, unsigned long paddr, bool isBase)
{ 
   std::lock_guard<std::mutex> guard(ctrl_mutex);
   // cout << hex << "vaddr: " << vaddr << "  paddr: " << paddr << "  isBase" << isBase << endl;
#ifdef PRINT_DEBUG
   printf("Writing tlb mapping\n");fflush(stdout);
#endif
   writeReg(14, (uint32_t) 0);
   writeReg(8, (uint32_t) vaddr);
   writeReg(9, (uint32_t) (vaddr >> 32));
   writeReg(10, (uint32_t) paddr);
   writeReg(11, (uint32_t) (paddr >> 32));
   writeReg(12, (uint32_t) isBase);
   writeReg(13, (uint32_t) 0);
   writeReg(13, (uint32_t) 1);

   // writeReg(14, (uint32_t) vaddr);
   // writeReg(15, (uint32_t) (vaddr >> 32));
   // writeReg(16, (uint32_t) paddr);
   // writeReg(17, (uint32_t) (paddr >> 32));
   // writeReg(18, (uint32_t) isBase);
   // writeReg(19, (uint32_t) 0);
   // writeReg(19, (uint32_t) 1);


   // writeReg(20, (uint32_t) vaddr);
   // writeReg(21, (uint32_t) (vaddr >> 32));
   // writeReg(22, (uint32_t) paddr);
   // writeReg(23, (uint32_t) (paddr >> 32));
   // writeReg(24, (uint32_t) isBase);
   // writeReg(25, (uint32_t) 0);
   // writeReg(25, (uint32_t) 1);

   // writeReg(26, (uint32_t) vaddr);
   // writeReg(27, (uint32_t) (vaddr >> 32));
   // writeReg(28, (uint32_t) paddr);
   // writeReg(29, (uint32_t) (paddr >> 32));
   // writeReg(30, (uint32_t) isBase);
   // writeReg(31, (uint32_t) 0);
   // writeReg(31, (uint32_t) 1);   


#ifdef PRINT_DEBUG
   printf("done\n");fflush(stdout);
#endif
}

// void XDMAController::writeTlb(unsigned long vaddr, unsigned long paddr, bool isBase)
// { 
//    std::lock_guard<std::mutex> guard(ctrl_mutex);
// #ifdef PRINT_DEBUG
//    printf("Writing tlb mapping\n");fflush(stdout);
// #endif
//    writeReg(14, (uint32_t) vaddr);
//    writeReg(15, (uint32_t) (vaddr >> 32));
//    writeReg(16, (uint32_t) paddr);
//    writeReg(17, (uint32_t) (paddr >> 32));
//    writeReg(18, (uint32_t) isBase);
//    writeReg(19, (uint32_t) 0);
//    writeReg(19, (uint32_t) 1);

// #ifdef PRINT_DEBUG
//    printf("done\n");fflush(stdout);
// #endif
// }





bool XDMAController::checkBypass(){
   return readReg(513);//1 means bypass enable
}


void XDMAController::writeReg(uint32_t addr, uint32_t value)
{
   volatile uint32_t* wPtr = (uint32_t*) (((uint64_t) m_base) + (uint64_t) ((uint32_t) addr << 2));
   uint32_t writeVal = htols(value);
   *wPtr = writeVal;
}


void XDMAController::writeBypassReg(uint32_t addr, uint64_t* value)
{
   if(checkBypass() == 1){
      volatile __m512i* wPtr = (__m512i*) (((uint64_t) by_base) + (uint64_t) ((uint32_t) addr << 6));
      // cout<< "bypass: " << by_base << " wPtr: " <<(uint64_t) wPtr <<endl;
      // *wPtr = _mm512_set_epi32 (value[15], value[14], value[13], value[12], value[11], value[10], value[9], value[8], value[7],value[6],value[5],value[4],value[3],value[2],value[1],value[0]);
      *wPtr = _mm512_set_epi64 (value[7],value[6],value[5],value[4],value[3],value[2],value[1],value[0]);
   }else{
      cout<<"bypass disabled, write failed!\n";
   }
}
   



uint32_t XDMAController::readReg(uint32_t addr)
{
   volatile uint32_t* rPtr = (uint32_t*) (((uint64_t) m_base)  + (uint64_t) ((uint32_t) addr << 2));
  return htols(*rPtr);
}

void XDMAController::readBypassReg(uint32_t addr,uint64_t* res)
{
   if(checkBypass() == 1){
      volatile __m512i* rPtr = (__m512i*) (((uint64_t) by_base)  + (uint64_t) ((uint32_t) addr << 6));
      for(int i=0;i<8;i++){
         res[i] = rPtr[0][i];
      }
   }else{
      cout<<"bypass disabled, read failed!\n";
   }
//   return htols(*rPtr);
}






} /* namespace fpga */
