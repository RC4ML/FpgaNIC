

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
#include <string>

#include "fpga/XDMA.h"
#include "fpga/XDMAController.h"
#include <fstream>
#include <iomanip>
#include <bitset>

using namespace std;
int node_index;
int node_num;

void get_opt(int argc, char *argv[])
{
   int o;                        // getopt() 的返回�?   
   const char *optstring = "n:m:"; // 设置短参数类型及是否需要参�?
   while ((o = getopt(argc, argv, optstring)) != -1)
   {
      switch (o)
      {
      case 'n':
         node_index = atoi(optarg);
         printf("node_index:%d\n", node_index);
         break;
      case 'm':
         node_num = atoi(optarg)-1;
         printf("node_index:%d\n", node_num+1);
         break;         
      case '?':
         printf("error optopt: %c\n", optopt);
         printf("error opterr: %d\n", opterr);
         break;
      }
   }
}


void send_allreduce_cmd(fpga::XDMAController *controller, uint64_t addr0, uint32_t client_length, u_int32_t client_num, u_int32_t head_length, u_int32_t tail_length, u_int32_t int_flag ){
   uint64_t data[8];
   u_int32_t total_length = client_length*(client_num+1);
   data[0] = addr0;
   data[1] = client_length + ((uint64_t)total_length<<32);
   data[2] = tail_length + ((uint64_t)head_length<<32);
   data[3] = int_flag;

   cout << "After all the node prepare, press any key to continue:" <<endl;
   int b;
   cin>>b;

   controller ->writeBypassReg(8,data);

   sleep(3);


   // cout << "tcp_split_overflow: " << controller->readReg(687) << endl;

   // cout << "write_cmd0_counter: " << controller->readReg(780) << endl;
   // cout << "write_data0_counter: " << controller->readReg(781) << endl;

   // cout << "read_cmd0_counter: " << controller->readReg(784) << endl;
   // cout << "read_data0_counter: " << controller->readReg(785) << endl;


   // cout << "dma_wr_cmd0_counter: " << controller->readReg(776) << endl;
   // cout << "dma_wr_data0_counter: " << controller->readReg(777) << endl;
   // cout << "dma_rd_cmd1_counter: " << controller->readReg(778) << endl;
   // cout << "dma_rd_data1_counter: " << controller->readReg(779) << endl;


   // cout << "tcp_tx_cmd0_counter: " << controller->readReg(788) << endl;
   // cout << "tcp_tx_data0_counter: " << controller->readReg(789) << endl;
   // cout << "tcp_rx_cmd0_counter: " << controller->readReg(790) << endl;
   // cout << "tcp_rx_data0_counter: " << controller->readReg(791) << endl;

   // cout << "rx meta over flow cnt: " << controller->readReg(648) << endl;
   // cout << "rx data over flow cnt: " << controller->readReg(649) << endl;

   // cout << "sed_req_fifo_cnt: " << controller->readReg(800) << endl;
   // cout << "session_data_fifo_cnt: " << controller->readReg(801) << endl;
   // cout << "dma_read_fifo_cnt: " << controller->readReg(802) << endl;
   // cout << "cal_read_fifo_cnt: " << controller->readReg(803) << endl;
   // cout << "send_enable_fifo_cnt: " << controller->readReg(804) << endl;
   // cout << "send_start_fifo_cnt: " << controller->readReg(805) << endl;
   // cout << "wr_dma_fifo_cnt: " << controller->readReg(806) << endl;
   // cout << "state: " << controller->readReg(807) << endl;
   // cout << "start_state: " << controller->readReg(808) << endl;

   // cout << "mem_read_fifo_cnt: " << controller->readReg(816) << endl;
   // cout << "tx_state: " << controller->readReg(817) << endl;

   float th;
   int time_cnt = controller->readReg(792);
   // cout << "time_counter: " << controller->readReg(792) << endl;
   // cout << "time_counter: " << controller->readReg(792) << endl;
   
   th = 2.0*client_length*(client_num+1)*500/time_cnt/1000;
   std::cout <<  std::dec << " allreduce_throughput: " << th << " GB/s" << std::endl;  
   // cout << "resp: " << controller->readReg(797) << endl;
}


int main(int argc, char *argv[])
{

   // boost::program_options::options_description programDescription("Allowed options");
   // programDescription.add_options()("workGroupSize,m", boost::program_options::value<unsigned long>(), "Size of the memory region")
   //                                  ("readEnable,m",boost::program_options::value<unsigned long>(),"enable signal");

   // boost::program_options::variables_map commandLineArgs;
   // boost::program_options::store(boost::program_options::parse_command_line(argc, argv, programDescription), commandLineArgs);
   // boost::program_options::notify(commandLineArgs);
   // if(commandLineArgs.count("readEnable") > 0){
   //    read_enable = commandLineArgs["readEnable"].as<unsigned long>();
   //    cout<<bitset<sizeof(int)*8>(read_enable)<<endl;
   // }

   fpga::XDMAController *controller = fpga::XDMA::getController();
   uint64_t *dmaBuffer = (uint64_t *)fpga::XDMA::allocate(1024 * 1024 * 512);
   uint64_t addr0, addr1, addr2, addr3;

   // for(int i=0;i<1024*1024*60;i++){
   //    for(int ii=0;ii<7;ii++){
   //       dmaBuffer[8*i+ii]=0;
   //    }
   //    dmaBuffer[8*i+7]=i;
   // }

   float* p = (float*)dmaBuffer;

   for (int i = 0; i < 1024 * 1024 * 32; i++)
   {
      p[i] = 1.0*i;
   }

   // for (int i = 0; i < 1024 * 1024 * 2; i++)
   // {
   //    dmaBuffer[i] = i;
   // }

   //  cout << "addr:" << (uint64_t)dmaBuffer << endl;
   addr0 = (uint64_t)dmaBuffer;
   addr1 = (uint64_t)(&dmaBuffer[1024 * 200]);
   addr2 = (uint64_t)(&dmaBuffer[1024 * 400]);
   addr3 = (uint64_t)(&dmaBuffer[1024 * 600]);

   // int a;
   // cin >> a;
   get_opt(argc, argv);



   // cout << "tcp_split_overflow: " << controller->readReg(687) << endl;

   // cout << "write_cmd0_counter: " << controller->readReg(780) << endl;
   // cout << "write_data0_counter: " << controller->readReg(781) << endl;

   // cout << "read_cmd0_counter: " << controller->readReg(784) << endl;
   // cout << "read_data0_counter: " << controller->readReg(785) << endl;


   // cout << "dma_wr_cmd0_counter: " << controller->readReg(776) << endl;
   // cout << "dma_wr_data0_counter: " << controller->readReg(777) << endl;
   // cout << "dma_rd_cmd1_counter: " << controller->readReg(778) << endl;
   // cout << "dma_rd_data1_counter: " << controller->readReg(779) << endl;


   // cout << "tcp_tx_cmd0_counter: " << controller->readReg(788) << endl;
   // cout << "tcp_tx_data0_counter: " << controller->readReg(789) << endl;
   // cout << "tcp_rx_cmd0_counter: " << controller->readReg(790) << endl;
   // cout << "tcp_rx_data0_counter: " << controller->readReg(791) << endl;

   // cout << "rx meta over flow cnt: " << controller->readReg(648) << endl;
   // cout << "rx data over flow cnt: " << controller->readReg(649) << endl;

   // cout << "sed_req_fifo_cnt: " << controller->readReg(800) << endl;
   // cout << "session_data_fifo_cnt: " << controller->readReg(801) << endl;
   // cout << "dma_read_fifo_cnt: " << controller->readReg(802) << endl;
   // cout << "cal_read_fifo_cnt: " << controller->readReg(803) << endl;
   // cout << "send_enable_fifo_cnt: " << controller->readReg(804) << endl;
   // cout << "send_start_fifo_cnt: " << controller->readReg(805) << endl;
   // cout << "wr_dma_fifo_cnt: " << controller->readReg(806) << endl;
   // cout << "state: " << controller->readReg(807) << endl;
   // cout << "start_state: " << controller->readReg(808) << endl;
   // cout << "cal_error: " << controller->readReg(809) << endl;

   // cout << "mem_read_fifo_cnt: " << controller->readReg(816) << endl;
   // cout << "tx_state: " << controller->readReg(817) << endl;

   // cout << "time_counter: " << controller->readReg(792) << endl;
   // cout << "time_counter: " << controller->readReg(792) << endl;

   // cout << "resp: " << controller->readReg(797) << endl;


   // cout << "tlb_miss0: " << controller->readReg(520) << endl;
   // cout << "tlb_cross0: " << controller->readReg(521) << endl;

   // cout << "tlb_miss1: " << controller->readReg(531) << endl;
   // cout << "tlb_cross1: " << controller->readReg(532) << endl;

   // cout << "tlb_miss2: " << controller->readReg(542) << endl;
   // cout << "tlb_cross2: " << controller->readReg(543) << endl;

   // cout << "tlb_miss3: " << controller->readReg(553) << endl;
   // cout << "tlb_cross3: " << controller->readReg(554) << endl;

   // cout << "tlb_miss4: " << controller->readReg(564) << endl;
   // cout << "tlb_cross4: " << controller->readReg(565) << endl;

   // cout << "tlb_miss5: " << controller->readReg(575) << endl;
   // cout << "tlb_cross5: " << controller->readReg(576) << endl;

   // cout << "tlb_miss6: " << controller->readReg(586) << endl;
   // cout << "tlb_cross6: " << controller->readReg(587) << endl;

   // cout << "tlb_miss7: " << controller->readReg(597) << endl;
   // cout << "tlb_cross7: " << controller->readReg(598) << endl;

   // return 0;



   int mac = node_index;
   int ip_addr = 0xc0a9bd00 + node_index;
   int listen_port = 1235;
   int conn_ip;
   if(node_index == node_num){
      conn_ip = 0xc0a9bd00;
   }
   else{
      conn_ip = 0xc0a9bd01 + node_index;
   }
   
   // cout << "mac" << node_index<< endl;
   // cout << "ip_addr" << ip_addr<< endl;
   // cout << "conn_ip" << conn_ip<< endl;

   int conn_port = 1235;
   int session_id;

   controller->writeReg(0, 0);
   controller->writeReg(0, 1);
   controller->writeReg(0, 0);
   sleep(1);
   // controller ->writeReg(180,(controller->readReg(658)));
   controller->writeReg(128, (uint32_t)mac);
   controller->writeReg(129, (uint32_t)ip_addr);
   controller->writeReg(130, (uint32_t)listen_port);

   controller->writeReg(132, (uint32_t)conn_ip);
   controller->writeReg(133, (uint32_t)conn_port);

   controller->writeReg(131, (uint32_t)0);
   controller->writeReg(131, (uint32_t)1);

   // cout << "listen status: " << controller->readReg(640) << endl;
   while (((controller->readReg(640)) >> 1) == 0)
   {
      sleep(1);
      // cout << "listen status: " << controller->readReg(640) << endl;
   };
   // cout << "listen status: " << controller->readReg(640) << endl;
   sleep(2);
   int a;
   // cin >> a;
   // cout << a <<endl;

   controller->writeReg(134, (uint32_t)0);
   controller->writeReg(134, (uint32_t)1);

   // cout << "conn status: " << controller->readReg(641) << endl;
   while (((controller->readReg(641)) >> 16) == 0)
   {
      // cout << "conn status: " << controller->readReg(641) << endl;
      sleep(1);
   };
   session_id = controller->readReg(641) & 0x0000ffff;
   cout << "session_id: " << session_id << endl;
   // cout << "conn status: " << controller->readReg(641) << endl;
   sleep(1);
   controller->writeReg(134, (uint32_t)0);



   uint32_t client_id = node_index;
   uint32_t client_num = node_num;
   uint32_t large_length = 16*1024;
   uint32_t client_length = 2880;
   uint32_t total_length = client_length*(client_num+1);
   uint32_t token_divide = 11;
   uint32_t token_mul=3;
   uint32_t head_length=0;
   uint32_t tail_length=0;
   uint32_t int_flag=0;   //int: 1 float: 0

   

   controller->writeReg(256, (uint32_t)session_id);

   controller->writeReg(257, (uint32_t)client_id);
   controller->writeReg(258, (uint32_t)client_num);
   controller->writeReg(259, (uint32_t)large_length);
   // controller->writeReg(261, (uint32_t)addr0);
   // controller->writeReg(262, (uint32_t)(addr0 >> 32));
   // controller->writeReg(263, (uint32_t)20);
   // controller->writeReg(264, (uint32_t)16);
   controller->writeReg(265, (uint32_t)token_divide);
   controller->writeReg(266, (uint32_t)token_mul);
   controller->writeReg(267, (uint32_t)0);


   // int b;
   // cin >> b;
   // cout << b << endl;





   // controller->writeReg(260, (uint32_t)0);
   // controller->writeReg(260, (uint32_t)1);

    cout << "--------------ATC FIGURE 7-------------" << endl;
   for(client_length = 16*1024 ;client_length <= 8*1024*1024; client_length = client_length*2){
   sleep(1);
   cout << " data size: " << client_length*2  <<endl;
   send_allreduce_cmd(controller, addr0, client_length,  client_num,  head_length,  tail_length,  int_flag );
   
   }


   //    for(int i=0;i<128;i++){
   //       for(int j=0;j<16;j++){
   //          cout<<" "<< p[i*16+j];
   //       }
   //       cout << endl;
   //    }

   // client_length = 8202240;
   // send_allreduce_cmd(controller, addr0, client_length,  client_num,  head_length,  tail_length,  int_flag );

   //    for(int i=0;i<128;i++){
   //       for(int j=0;j<16;j++){
   //          cout<<" "<< p[i*16+j];
   //       }
   //       cout << endl;
   //    }

   // client_length = 205521920;
   // send_allreduce_cmd(controller, addr0, client_length,  client_num,  head_length,  tail_length,  int_flag );
   // controller->writeReg(135, (uint32_t)session_id);
   // controller->writeReg(136, (uint32_t)0);
   // controller->writeReg(136, (uint32_t)1);
   // controller ->writeBypassReg(4,data);

   // sleep(1);

   // for (int i = 0; i < 1024; i++)
   // {
   //    cout << dmaBuffer[1024 * 1024 * 200 + i] << endl;
   // }

   // cout << "write_cmd0_counter: " << controller->readReg(768) << endl;
   // cout << "write_data0_counter: " << controller->readReg(769) << endl;

   // cout << "read_cmd0_counter: " << controller->readReg(772) << endl;
   // cout << "read_data0_counter: " << controller->readReg(773) << endl;


   // cout << "dma_wr_cmd0_counter: " << controller->readReg(776) << endl;
   // cout << "dma_wr_data0_counter: " << controller->readReg(777) << endl;
   // cout << "dma_rd_cmd1_counter: " << controller->readReg(778) << endl;
   // cout << "dma_rd_data1_counter: " << controller->readReg(779) << endl;


   // cout << "tcp_tx_cmd0_counter: " << controller->readReg(788) << endl;
   // cout << "tcp_tx_data0_counter: " << controller->readReg(789) << endl;
   // cout << "tcp_rx_cmd0_counter: " << controller->readReg(790) << endl;
   // cout << "tcp_rx_data0_counter: " << controller->readReg(791) << endl;

   // cout << "rx meta over flow cnt: " << controller->readReg(648) << endl;
   // cout << "rx data over flow cnt: " << controller->readReg(649) << endl;

   // cout << "time_counter: " << controller->readReg(792) << endl;
   // cout << "time_counter: " << controller->readReg(792) << endl;

   // cout << "resp: " << controller->readReg(797) << endl;

      // for(int i=0;i<128;i++){
      //    for(int j=0;j<16;j++){
      //       cout<<" "<< p[i*16+j];
      //    }
      //    cout << endl;
      // }

      // for(int i=0;i<64;i++){
      //    for(int j=0;j<16;j++){
      //       cout<<" "<< dmaBuffer[i*16+j];
      //    }
      //    cout << endl;
      // }

   //  for (int i = 8; i < 1024 * 1024 * 2; i++)
   // {
   //    if(p[i]!=2.0*i){
   //       cout<<"i "<<i<<" -- "<< p[i]<<endl;
   //       break;
   //    }
   // }     


   fpga::XDMA::clear();

   return 0;
}
