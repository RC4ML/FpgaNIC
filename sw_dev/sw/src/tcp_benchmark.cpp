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
int session_num;

void get_opt(int argc, char *argv[])
{
   int o;                        // getopt() 的返回值
   const char *optstring = "n:m:t:"; // 设置短参数类型及是否需要参数

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
      case 't':
         session_num = atoi(optarg);
         printf("session_num:%d\n", session_num);
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

   get_opt(argc, argv);
   int length,rd_sum,wr_sum;
   int burst = 1024 * 1024;
   int start = 2; //bit 0 write start  bit 1 read start
   float rd_speed,wr_speed;
   int rd_lat;

   int mac = node_index;
   int ip_addr = 0xc0a8bd00 + node_index;
   int listen_port = 1235;
   int conn_ip = 0xc0a8bd00 + remote_node;
   int conn_port = 1235;
   int session_id[1000];
   

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



      controller ->writeReg(0,0);
      controller ->writeReg(0,1);
      sleep(1);

      int tcp_length = 1024*1024*64;
      int ops = 1;
      int offset = 2;


      // controller ->writeReg(48,(uint32_t)session_id);
      controller ->writeReg(49,(uint32_t)tcp_length);
      controller ->writeReg(50,(uint32_t)ops);
      controller ->writeReg(51,(uint32_t)offset);
      // controller ->writeReg(55,(uint32_t)0);
      // controller ->writeReg(55,(uint32_t)1);
      controller ->writeReg(56,0);


      cout << " ATC FIGURE 5(b) connection num: "<< session_num << endl;
      //dma speed latency

      // cout << " wr_sum: " << wr_sum <<endl; 
      // cout << " rd_sum: " << rd_sum <<endl;

      cout <<"wait the client conn success, then press any key to show the result:"<<endl;

      int a;
      cin>>a;
      cout <<a<<endl;

      rd_sum = controller ->readReg(616);
      wr_sum = controller ->readReg(608);

         // std::cout << " tcp_tx_meta_counter: " << controller ->readReg(609) <<  std::endl;  
         // std::cout << " tcp_tx_word_counter: " << controller ->readReg(610) << std::endl; 

         // std::cout << " error_cnt: " << controller ->readReg(617) <<  std::endl; 
         // std::cout << " error_index: " << controller ->readReg(618) <<  std::endl; 
         // std::cout << " tcp_rx_word_counter: " << controller ->readReg(619) <<  std::endl; 



         // std::cout << " rd_sum: " << controller ->readReg(616) <<  std::endl; 
         // std::cout << " wr_sum: " << controller ->readReg(608) <<  std::endl; 

      // wr_speed = 1.0*tcp_length*ops*250/wr_sum/1000;
      // std::cout <<  std::dec << ", wr_speed: " << wr_speed << " GB/s" << std::endl;
      // file1<<wr_speed<<endl;  




      rd_speed = 1.0*tcp_length*ops*250*8/rd_sum/1000;
      std::cout <<  std::dec << " rd_speed: " << rd_speed << " Gbps" << std::endl;   
      // file1<<rd_speed<<endl;  


   }


   if(node_index==7){

   controller ->writeReg(134,(uint32_t)0);
   controller ->writeReg(134,(uint32_t)1);      
      while(((controller->readReg(641)) >> 16)==0 ){
         sleep(1);
      };
      // session_id[0] = controller->readReg(641) & 0x0000ffff;

   for(int i = 1;i < session_num; i++){

   sleep(0.5);
   // cout << "conn status: " << controller->readReg(641) << endl;
      while(((controller->readReg(641)) >> 16)==0 ){
         // cout << "conn status: " << controller->readReg(641) << endl;
         if(controller->readReg(641) == 0){
            // i--;
            break;         
         }

      };
      if(controller->readReg(641) == 0){continue;}
      else{
         session_id[i] = controller->readReg(641) & 0x0000ffff;
      }
      // cout << "conn status: " << controller->readReg(641) << endl;
      sleep(1);
      controller ->writeReg(134,(uint32_t)0);
   }
      // cout << "session_id: " << session_id << endl;
      cout << "conn success: " << endl;


      int tcp_length = 1024*1024*64;
      int ops = 1;
      int offset = 2; 


      controller ->writeReg(48,(uint32_t)session_id[0]);
      controller ->writeReg(49,(uint32_t)tcp_length);
      controller ->writeReg(50,(uint32_t)ops);
      controller ->writeReg(51,(uint32_t)offset);
      controller ->writeReg(52,(uint32_t)session_num);
      controller ->writeReg(55,(uint32_t)0);
      controller ->writeReg(55,(uint32_t)2);

         // while((controller->readReg(610)) < (1024*8*(i+1)) ){
         //    // std::cout << i << " : "<< 1024*8*(i+1) << "----------"<< controller ->readReg(610) << std::endl; 
         //    // sleep(1);
         // };


   }


   fpga::XDMA::clear();

	return 0;

}
