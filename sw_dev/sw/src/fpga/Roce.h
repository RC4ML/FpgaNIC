#ifndef SRC_HASHJOIN_COMMUNICATION_HARDROCECOMMUNICATOR_H_
#define SRC_HASHJOIN_COMMUNICATION_HARDROCECOMMUNICATOR_H_

#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <atomic>


#include "Configuration.h"
//#include <communication/HardRoceWindow.h>
#include "XDMAController.h"

#define MAX_WIN 8

namespace fpga {
   class XDMAController;
}

namespace RDMA {

struct IbQueue {
   uint32_t   qpn;      
   uint32_t   psn;   
   uint32_t   rkey;
   uint64_t   vaddr;     //base vaddr of this queue pair
   uint32_t   size;     //total length of this queue pair
   uint32_t   ipaddr;
};

struct QueuePair {
   IbQueue   local;
   IbQueue   remote;
};


struct RoceWinMeta {
   void*    base;
   uint64_t size;
};

struct RoceWin {
   RoceWinMeta windows[MAX_WIN];
};


class Roce {

public:

	Roce(fpga::XDMAController* fpga, uint32_t nodeId, uint32_t numberOfNodes, const char* ipAddress);
	~Roce();


public:


   void write(const void* originAddr, uint64_t originLength, uint64_t originOffset, int targetProcess, uint64_t targetOffset, RDMA::RoceWin *win);
   void read(const void* originAddr, uint64_t originLength, uint64_t originOffset, int targetProcess, uint64_t targetOffset, RDMA::RoceWin *win);

protected:

   std::string encode(IbQueue q);
   RDMA::IbQueue decode(char* buf, size_t len);

   void initializeLocalQueues();
   int masterExchangeQueues();
   int clientExchangeQueues();

   int masterExchangeWindow(RoceWin* win);
   int slaveExchangeWindow(RoceWin* win);

public:
   void exchangeWindow(void* base, uint64_t size, RoceWin* win);

protected:
   fpga::XDMAController* fpga;
	const char* masterIpAddress;
   uint32_t nodeId;
   uint32_t numberOfNodes;
   int*  connections;
   uint16_t port;
   RDMA::QueuePair*   pairs;
   std::atomic<uint64_t> pushedLength;

   static uint64_t*   reducedInput;
   static uint64_t*   reducedOutput;



};

} /* namespace communication */

#endif 