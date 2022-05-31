/*
 * HardRoceCommunicator.cpp
 *
 *  Created on: Nov 13, 2017
 *      Author: dasidler
 */

#include "Roce.h"

#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <cstring>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <random>
#include <chrono>
#include <thread>
#include <limits>
#include <assert.h>

//#include <hashjoin/core/Configuration.h>
//#include <hashjoin/utils/Debug.h>

//#include <communication/HardRoceWindow.h>

//namespace hashjoin {
namespace RDMA {


//TODO maybe pass FPGA object instead
Roce::Roce(fpga::XDMAController* fpga, uint32_t nodeId, uint32_t numberOfNodes, const char* masterIpAddress) {
   
   port = 18515;
   this->fpga = fpga;
	this->nodeId = nodeId;
	this->numberOfNodes = numberOfNodes;
	this->masterIpAddress = masterIpAddress;

   this->connections = new int[numberOfNodes];
   this->pairs = new Roce::QueuePair[numberOfNodes];
   this->pushedLength = 0;

   initializeLocalQueues();


   uint32_t baseIpAddr = fpga::Configuration::BASE_IP_ADDR;

   fpga->writeReg(128, (uint32_t)nodeId);                //set mac address
   fpga->writeReg(129, (uint32_t)(baseIpAddr + nodeId)); //set ip address

   
   int ret = 1;
   if (nodeId == 0) {
      ret = masterExchangeQueues();
   } else {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      ret = clientExchangeQueues();
   }
   if (ret) {
      std::cerr << "Exchange failed";
      //return 1;
   }
   std::cout << "exchange done." << std::endl;

   //load context & connections to FPGA
   for (int i = 0; i < numberOfNodes; i++) {
      if (i == nodeId) {
         continue;
      }

      controller -> writeReg(143,(uint32_t)pairs[i].local.vaddr);
      controller -> writeReg(144,(uint32_t)(pairs[i].local.vaddr>>32));
      controller -> writeReg(147,pairs[i].local.qpn);
      controller -> writeReg(148,pairs[i].remote.qpn);
      controller -> writeReg(149,pairs[i].local.psn);
      controller -> writeReg(150,pairs[i].remote.psn);  
      controller -> writeReg(151,pairs[i].remote.ipaddr);   
      controller -> writeReg(152,port); 
      controller -> writeReg(154,0);     //qp_state
      controller -> writeReg(156,pairs[i].remote.rkey);        

      controller -> writeReg(157,0);//qp_start
      controller -> writeReg(157,1);
      controller -> writeReg(158,0);//qp_conn_start
      controller -> writeReg(158,1);



   }

   for (int i = 0; i < numberOfNodes; i++) {
      if (i == nodeId) {
         continue;
      }
      controller -> writeReg(141,(baseIpAddr + i));
      controller -> writeReg(142,0);
      controller -> writeReg(142,1);


   }
}

Roce::~Roce() {

   for (int i = 0; i < numberOfNodes; i++) {
      if (i == nodeId) {
         continue;
      }
      close(connections[i]);
   }

   delete[] connections;
   delete[] pairs;
}

std::string Roce::encode(IbQueue q) {
   std::uint32_t lid = 0;
   std::ostringstream msgStream;
   msgStream << std::setfill('0') << std::setw(4) << std::hex << lid << ":";
   msgStream << std::setfill('0') << std::setw(6) << std::hex << q.qpn << ":";
   msgStream << std::setfill('0') << std::setw(6) << std::hex << (q.psn & 0xFFFFFF) << ":";   
   msgStream << std::setfill('0') << std::setw(8) << std::hex << q.rkey << ":";
   msgStream << std::setfill('0') << std::setw(16) << std::hex << q.vaddr << ":";
   msgStream << q.ipaddr;

   std::string msg = msgStream.str();
   return msg;
}

RDMA::IbQueue Roce::decode(char* buf, size_t len) {
   RDMA::IbQueue q;
   if (len < 45) {
      std::cerr << "unexpected length " << len << " in decode ib connection\n";
      return;
   }
   buf[4] = ' ';
   buf[11] = ' ';
   buf[18] = ' ';
   buf[27] = ' ';
   buf[44] = ' ';
   std::uint32_t lid = 0;
	//std::cout << "buf " << buf << std::endl;
	std::string recvMsg(buf, len);
	 //std::cout << "string " << recvMsg << ", length: " << recvMsg.length() << std::endl;
	std::istringstream recvStream(recvMsg);
	recvStream >> std::hex >> lid >> q.qpn >> q.psn;
	recvStream >> std::hex >> q.rkey >> q.vaddr >> q.ipaddr;
   return q;
}



void Roce::exchangeWindow(void* base, uint64_t size, RoceWin* win) {

   win->windows[nodeId].base = base;
   win->windows[nodeId].size = size;
   //Exchange windows
   if (this->nodeId == 0) {
       masterExchangeWindow(win);
   } else {
       slaveExchangeWindow(win);
   }

}

void Roce::put(const void* originAddr, uint64_t originLength, uint64_t originOffset, int targetProcess, uint64_t targetOffset, RoceWin* win) {
   void* localAddr = (void*) (((char*) originAddr) + originOffset);
   void* remotAddr = (void*) (((char*) win->windows[targetProcess].base) + targetOffset);
   //Check if local Put
   if (targetProcess == nodeId) {
      memcpy(remotAddr, localAddr, originLength);
   } else {
      pushedLength += originLength;
      while (originLength > std::numeric_limits<uint32_t>::max()) {
         fpga->postWrite(&(pairs[targetProcess]), localAddr, std::numeric_limits<uint32_t>::max(), remotAddr);
         localAddr = (void*) (((char*) localAddr) + std::numeric_limits<uint32_t>::max());
         originLength -= std::numeric_limits<uint32_t>::max();
      }
      fpga->postWrite(&(pairs[targetProcess]), localAddr, (uint32_t) originLength, remotAddr);
   }
}

void Roce::get(const void* originAddr, uint64_t originLength, uint64_t originOffset, int targetProcess, uint64_t targetOffset, RoceWin* win) {
   void* localAddr = (void*) (((char*) originAddr) + originOffset);
   void* remotAddr = (void*) (((char*) win->windows[targetProcess].base) + targetOffset);
   //Check if local Get
   if (targetProcess == nodeId) {
      memcpy(localAddr, remotAddr, originLength);
   } else {
      while (originLength > std::numeric_limits<uint32_t>::max()) {
         fpga->postRead(&(pairs[targetProcess]), localAddr, std::numeric_limits<uint32_t>::max(), remotAddr);
         localAddr = (void*) (((char*) localAddr) + std::numeric_limits<uint32_t>::max());
         originLength -= std::numeric_limits<uint32_t>::max();
      }
      fpga->postRead(&(pairs[targetProcess]), localAddr, (uint32_t) originLength, remotAddr);
   }
}

static unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

void Roce::initializeLocalQueues() {
   std::default_random_engine rand_gen(seed);
   std::uniform_int_distribution<int> distr(0, std::numeric_limits<std::uint32_t>::max());

   //Assume IPv4
   uint32_t ipAddr = fpga::Configuration::BASE_IP_ADDR;
   ipAddr += nodeId;

   for (int i = 0; i < numberOfNodes; ++i) {
      pairs[i].local.ipaddr = ipAddr;
      pairs[i].local.qpn = 0x3 + i;
      pairs[i].local.psn = distr(rand_gen);
      pairs[i].local.rkey = 0;
      pairs[i].local.vaddr = 0; //TODO remove
   }

 
}


int Roce::masterExchangeQueues() {
   char *service;
   char recv_buf[100];
   int n;
   int sockfd = -1, connfd;
   struct sockaddr_in server;
   memset(recv_buf, 0, 100);

   std::cout << "server exchange" << std::endl;

   /*if (asprintf(&service, "%d", data->port) < 0)
      return 1;*/
   sockfd = ::socket(AF_INET, SOCK_STREAM, 0);
   if (sockfd == -1) {
      std::cerr << "Could not create socket" << std::endl;
      return 1;
   }

   //n = getaddrinfo(NULL, service, &hints, 
   //Prepare the sockaddr_in structure
   server.sin_family = AF_INET;
   server.sin_addr.s_addr = INADDR_ANY;
   server.sin_port = htons( port);

   if (::bind(sockfd, (struct sockaddr*)&server, sizeof(server)) < 0) {
      std::cerr << "Could not bind socket" << std::endl;
      return 1;
   }


   if (sockfd < 0 ) {
      std::cerr << "Could not listen to port " << port << std::endl;
      return 1;
   }

   RDMA::IbQueue* allQueues = new RDMA::IbQueue[numberOfNodes*numberOfNodes];

   listen(sockfd, numberOfNodes);
   int i = 1;
   while (i < numberOfNodes) {
      //std::cout << "count: " << count << std::endl;
      connfd = ::accept(sockfd, NULL, 0);
      if (connfd < 0) {
         std::cerr << "Accep() failed" << std::endl;
         return 1;
      }

      //generate local string, this is just for getting the length...
      std::string msgString = encode(pairs[0].local);
      size_t msgLen = msgString.length();

      for (int j = 0; j < numberOfNodes; j++) {
         if (j == i) {
            continue;
         }
         //read msg
         n = ::read(connfd, recv_buf, msgLen);
         if (n != msgLen) {
            std::cerr << "Could not read remote address" << std::endl;
            close(connfd);
            return 1;
         }

         //parse remote connection
         allQueues[i*numberOfNodes+j] = decode(recv_buf, msgLen);
         printf("%s:  LID 0x%04x, QPN 0x%06x, PSN 0x%06x\n",
               "remote: ", 0, allQueues[i*numberOfNodes+j].qpn, allQueues[i*numberOfNodes+j].psn);
         printf("RKey %#08x VAddr %016lx\n", allQueues[i*numberOfNodes+j].rkey, allQueues[i*numberOfNodes+j].vaddr);

         //Check if it is for master
         if (j == 0) {
            pairs[i].remote = decode(recv_buf, msgLen);
         }
      }//for
      connections[i] = connfd;
      i++;
   }//while

   //send queues
   i = 1;
   while(i < numberOfNodes) {


      for (int j = 0; j < numberOfNodes; j++) {
         if (i == j) {
            continue;
         }
         std::string msgString;
         if (j == 0) {
            msgString = encode(pairs[i].local);
         } else {
            msgString = encode(allQueues[j*fpga::Configuration::MAX_NODES+i]);
         }
         size_t msgLen = msgString.length();

         //write msg
         if (::write(connections[i], msgString.c_str(), msgLen) != msgLen)  {
            std::cerr << "Could not send local address" << std::endl;
            ::close(connections[i]);
            return 1;
         }
      }//for
      i++;
   }//while

   ::close(sockfd);
   delete[] allQueues;
   return 0;
}

int Roce::clientExchangeQueues() {
   struct addrinfo *res, *t;
   struct addrinfo hints = {};
   hints.ai_family = AF_INET;
   hints.ai_socktype = SOCK_STREAM;

   char* service;
   char recv_buf[100];
   int n = 0;
   int sockfd = -1;
   memset(recv_buf, 0, 100);

   std::cout << "client exchange" << std::endl;

   if (asprintf(&service, "%d", port) < 0) {
      std::cerr << "service failed" << std::endl;
      return 1;
   }

   n = getaddrinfo(masterIpAddress, service, &hints, &res);
   if (n < 0) {
      std::cerr << "[ERROR] getaddrinfo";
      free(service);
      return 1;
   }

   for (t = res; t; t = t->ai_next) {
      sockfd = ::socket(t->ai_family, t->ai_socktype, t->ai_protocol);
      if (sockfd >= 0) {
         if (!::connect(sockfd, t->ai_addr, t->ai_addrlen)) {
            break;
         }
         ::close(sockfd);
         sockfd = -1;
      }
   }

   if (sockfd < 0) {
      std::cerr << "Could not connect to master: " << masterIpAddress << ":" << port << std::endl;
      return 1;
   }

   for (int i = 0; i < numberOfNodes; i++) {
      if (i == nodeId) {
         continue;
      }
      std::cout << "build msg" << std::endl;
      std::string msgString = encode(pairs[i].local);
      std::cout << "local : " << msgString << std::endl;

      size_t msgLen = msgString.length();
      if (write(sockfd, msgString.c_str(), msgLen) != msgLen) {
         std::cerr << "could not send local address" << std::endl;
         close(sockfd);
         return 1;
      }

      if ((n = ::read(sockfd, recv_buf, msgLen)) != msgLen) {
         std::cout << "n: " << n << ", instread of " << msgLen << std::endl; 
         std::cout << "received msg: " << recv_buf << std::endl;
         std::cerr << "Could not read remote address" << std::endl;
         ::close(sockfd);
         return 1;
      }

      pairs[i].remote = decode(recv_buf, msgLen);
      printf("%s:  LID 0x%04x, QPN 0x%06x, PSN 0x%06x\n",
           "remote: ", 0, pairs[i].remote.qpn, pairs[i].remote.psn);
      printf("RKey %#08x VAddr %016lx\n", pairs[i].remote.rkey, pairs[i].remote.vaddr);


   }//for

   //keep connection around
   connections[0] = sockfd;

   if (res) 
      freeaddrinfo(res);
   free(service);

   return 0;
}


int Roce::masterExchangeWindow(RoceWin* win) {
   int n = 0;
   int recvCount = 0;

   for (uint32_t i = 1; i < numberOfNodes; ++i) {
      char* recvPtr = (char*) &(win->windows[i]);
      recvCount = 0;
      //read window
      do {
         n = ::read(connections[i], recvPtr, sizeof(RoceWinMeta)-recvCount);
         if (n < 0) {
            std::cerr << "Could not read window: " << n << std::endl;
            ::close(connections[i]);
            return 1;
         }
         recvPtr += n;
         recvCount += n;
      } while(recvCount < sizeof(RoceWinMeta));
   }//for

   //send accumulated values
   for (uint32_t i = 1; i < numberOfNodes; ++i) {
      //write windows
      for (int j  = 0; j < numberOfNodes; j++) {
         if (i == j) {
            continue;
         }
         if ((n = ::write(connections[i], &(win->windows[j]), sizeof(RoceWinMeta))) != sizeof(RoceWinMeta))  {
            std::cerr << "Could not send" << std::endl;
            ::close(connections[i]);
            return 1;
         }
      }//for
   }//for

   //delete[] basetempbuf;
   return 0;
}


int Roce::slaveExchangeWindow(RoceWin* win) {
   int n = 0;
   int recvCount = 0;

   //write window
   if ((n = ::write(connections[0], &(win->windows[nodeId]), sizeof(RoceWinMeta))) != sizeof(RoceWinMeta))  {
            std::cerr << "Could not send window" << std::endl;
            ::close(connections[0]);
            return 1;
   }
   //read windows
   for (uint32_t i = 0; i < numberOfNodes; ++i) {
      if (nodeId == i) {
         continue;
      }
      char* recvPtr = (char*) &(win->windows[i]);
      recvCount = 0;
      //read window
      do {
         n = ::read(connections[0], recvPtr, sizeof(RoceWinMeta)-recvCount);
         if (n < 0) {
            std::cerr << "Could not read window: " << n << std::endl;
            ::close(connections[0]);
            return 1;
         }
         recvPtr += n;
         recvCount += n;
      } while(recvCount < sizeof(RoceWinMeta));
   }//for

   return 0;
}

} /* namespace communication */
//} /* namespace hashjoin */

