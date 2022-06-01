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

#include "fpga/XDMA.h"
#include "fpga/XDMAController.h"
#include <fstream>
#include <iomanip>
#include <bitset>


using namespace std;
int node_index;
int remote_node;

void get_opt(int argc, char *argv[])
{
   int o;                        // getopt() 的返回值
   const char *optstring = "n:m:"; // 设置短参数类型及是否需要参数

   while ((o = getopt(argc, argv, optstring)) != -1)
   {
      switch (o)
      {
      case 'n':
         node_index = atoi(optarg);
         printf("node_index:%d\n", node_index);
         break;   
      case 'm':
         remote_node = atoi(optarg);
         printf("remote_node:%d\n", remote_node);
         break;               
      case '?':
         printf("error optopt: %c\n", optopt);
         printf("error opterr: %d\n", opterr);
         break;
      }
   }
}



int main(int argc, char *argv[]) {


   

   fpga::XDMAController* controller = fpga::XDMA::getController();
   uint64_t* dmaBuffer =  (uint64_t*) fpga::XDMA::allocate(1024*1024*480);
   uint64_t addr0,addr1,addr2,addr3;

   for(int i=0;i<1024*1024*60;i++){
      for(int ii=0;ii<7;ii++){
         dmaBuffer[8*i+ii]=0;
      }
      dmaBuffer[8*i+7]=i;
   }

   //  cout << "addr:" << (uint64_t)dmaBuffer << endl;
   addr0 = (uint64_t)dmaBuffer;
   addr1 = (uint64_t)(&dmaBuffer[1024*1024*20]);
   addr2 = (uint64_t)(&dmaBuffer[1024*1024*400]);
   addr3 = (uint64_t)(&dmaBuffer[1024*1024*600]);



   int length,rd_sum,wr_sum;
   int burst = 1024 * 1024;
   int start = 2; //bit 0 write start  bit 1 read start
   float rd_speed,wr_speed;
   int rd_lat;

   get_opt(argc, argv);



// for(int i=0;i<20;i++){

   int mac = node_index;
   int ip_addr = 0xc0a8bd00 + node_index;
   int listen_port = 1235;
   int conn_ip = 0xc0a8bd00 + remote_node;
   int conn_port = 1235;
   int session_id;
   

   controller ->writeReg(0,0);
   controller ->writeReg(0,1);
   sleep(1);
   // controller ->writeReg(180,(controller->readReg(658)));
   controller ->writeReg(128,(uint32_t)mac);              
   controller ->writeReg(129,(uint32_t)ip_addr);
   controller ->writeReg(130,(uint32_t)listen_port);

   controller ->writeReg(132,(uint32_t)conn_ip);
   controller ->writeReg(133,(uint32_t)conn_port);   


   controller ->writeReg(131,(uint32_t)0);
   controller ->writeReg(131,(uint32_t)1);

   if(node_index==4){
      cout << "socket listen " << endl;
      while(((controller->readReg(640)) >> 1)==0 ){
         sleep(1);
         // cout << "listen status: " << controller->readReg(640) << endl;
      };
      // cout << "listen status: " << controller->readReg(640) << endl;     

   int tcp_length = 1024;
   int mode = 0;
   int ops = 10000;
   int offset = 2;


   // controller ->writeReg(48,(uint32_t)session_id);
   controller ->writeReg(49,(uint32_t)mode);
   controller ->writeReg(51,(uint32_t)1024);

   sleep(10);

   }



   if(node_index==7){

      controller ->writeReg(134,(uint32_t)0);
      controller ->writeReg(134,(uint32_t)1);


      cout << "start conn: " << endl;
      while(((controller->readReg(641)) >> 16)==0 ){
         // cout << "conn status: " << controller->readReg(641) << endl;
         sleep(1);
      };
      session_id = controller->readReg(641) & 0x0000ffff;
      cout << "session_id: " << session_id << endl;
      cout << "conn success: " << endl;
      sleep(1);
      controller ->writeReg(134,(uint32_t)0);
   int mode = 1;


   int offset = 2;



   for(int tcp_length = 128;tcp_length <= 1024*32; tcp_length = tcp_length*2){
 
   // int tcp_length = 1024;
   // int ops = 10000;
   int offset = 2;


   controller ->writeReg(0,0);
   controller ->writeReg(0,1);
   sleep(1);

   controller ->writeReg(48,(uint32_t)session_id);
   controller ->writeReg(49,(uint32_t)mode);
   controller ->writeReg(51,(uint32_t)tcp_length);
   controller ->writeReg(50,(uint32_t)0);
   controller ->writeReg(50,(uint32_t)1);
   




   sleep(3);

	//dma speed latency

   wr_sum = controller ->readReg(608);
   // cout << " wr_sum: " << wr_sum <<endl; 
   std::cout << " ATC FIGURE 5(a) tcp_length: " << tcp_length <<  std::endl;
 
      // std::cout << " wr_sum: " << controller ->readReg(608) <<  std::endl; 

   wr_speed = 1.0*wr_sum*4/1000;
   std::cout <<  std::dec << " latency: " << wr_speed << " us" << std::endl;  



   }



   }



   fpga::XDMA::clear();

	return 0;

}
