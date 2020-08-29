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
#include <boost/program_options.hpp>

#include "fpga/XDMA.h"
#include "fpga/XDMAController.h"
#include <fstream>
#include <iomanip>
#include <bitset>


using namespace std;



int main(int argc, char *argv[]) {

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

   fpga::XDMAController* controller = fpga::XDMA::getController();
   uint64_t* dmaBuffer =  (uint64_t*) fpga::XDMA::allocate(1024*1024*480);

   for(int i=0;i<1024*1024*60;i++){
      for(int ii=0;ii<7;ii++){
         dmaBuffer[8*i+ii]=0;
      }
      dmaBuffer[8*i+7]=i;
   }

   cout << "addr:" << (uint64_t)dmaBuffer << endl;
   int a;
   cin >>  a ;
   cout <<a<<endl;

   uint32_t read_enable         = 0x0000; //0x80000000 
   controller ->writeReg(511,123);
   cout << controller ->readReg(128) << endl;
   cout << controller ->readReg(129) << endl;
   uint64_t data[16]={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
   uint64_t res[8];
   controller ->writeBypassReg(31,data);
   controller ->readBypassReg(31,res);
   for(int i=0;i<8;i++){
      cout<<res[i]<<endl;
   }
   fpga::XDMA::clear();

	return 0;

}
