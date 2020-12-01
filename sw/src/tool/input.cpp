#include"input.hpp"
#include "tool/log.hpp"

void start_cmd_control(fpga::XDMAController* controller){
	string cmd;
	int reg_id, value;
	while(true){
		cjdebug("Enter CMD:\n");
		cin>>cmd;
		if(cmd==string("rd")){
			cin>>reg_id;
			uint res = controller->readReg(reg_id);
			cout<<"read "<<reg_id<<" with hex value: "<<hex<<res<<dec<<endl;
			
		}else if(cmd==string("wr")){
			cin>>reg_id>>hex>>value>>dec;
			cout<<"write "<<reg_id<<" with hex value: "<<hex<<(unsigned int)value<<dec<<" "<<endl;
			controller->writeReg(reg_id,(unsigned int)value);
		}else if(cmd==string("rd_by")){
			cin>>reg_id;
			uint64_t res[8];
			controller->readBypassReg(reg_id,res);
			cout<<"read bypass: "<<reg_id<<" with hex value: ";
			for(int i=0;i<8;i++){
				cout<<hex<<res[i]<<" ";
			}
			cout<<dec<<endl;
			
		}else{
			cout<<"Undefined CMD:"<<cmd<<endl;
		}
		
		
	}
		
}