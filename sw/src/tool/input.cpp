#include"input.hpp"

void start_cmd_control(fpga::XDMAController* controller){
	string cmd;
	int reg_id, value;
	while(true){
		cout<<"Enter cmd:";
		cin>>cmd;
		if(cmd==string("rd")){
			cin>>reg_id;
			uint res = controller->readReg(reg_id);
			cout<<"read "<<reg_id<<" with hex value:"<<hex<<res<<dec<<endl;
			
		}else if(cmd==string("wr")){
			cin>>reg_id>>hex>>value>>dec;
			cout<<"write "<<reg_id<<" with hex value:"<<hex<<(unsigned int)value<<dec<<" "<<endl;
			controller->writeReg(reg_id,(unsigned int)value);
		}else{
			cout<<"Undefined CMD:"<<cmd<<endl;
		}
		
		
	}
		
}