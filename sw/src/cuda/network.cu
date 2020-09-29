#include "network.cuh"
#include <assert.h>



__global__ void create_socket(socket_context_t* ctx,int* socket){
	BEGIN_SINGLE_THREAD_DO
		*(socket) = atomicAdd(&(ctx->socket_num),1);
		ctx->socket_info[*(socket)].type	=	0;//0 for client
		printf("create socket success with socket_id:%d\n",*(socket));
	END_SINGLE_THREAD_DO
}
__global__ void socket_listen(socket_context_t* ctx,int socket,int port){
	BEGIN_SINGLE_THREAD_DO
		printf("enter listen function!\n");
		if(false==check_socket_validation(ctx,socket)){
			printf("socket %d does not exists!\n",socket);
			return;
		}

		*(ctx->listen_port)		=	port;
		*(ctx->listen_start)	=	1;

		int res = wait_done(ctx->listen_status,1);
		*(ctx->listen_start)	=	0;

		if(res==-1){//timeout
			printf("socket listen timeout\n");
			return;
		}else{
			if(is_zero(ctx->listen_status,0)){
				printf("socket listen failed\n");
				return;
			}else{
				printf("socket listen success!\n");
			}
		}
		ctx->socket_info[socket].type	=	1;//0 for client
		ctx->socket_info[socket].port	=	port;
	END_SINGLE_THREAD_DO
}


__device__ int connect(socket_context_t* ctx,int socket,sock_addr_t addr){
		printf("dst_ip:%x dst_port:%d\n",addr.ip,addr.port);
		*(ctx->conn_ip)		=	addr.ip;
		*(ctx->conn_port)	=	addr.port;
		*(ctx->conn_start)	=	1;
		
		volatile int res = wait_done(ctx->con_session_status,17);
		*(ctx->conn_start)	=	0;

		if(res==-1){//timeout
			printf("socket connect timeout\n");
			return -1;
		}else{
			if(is_zero((unsigned int *)&res,16)){
				printf("socket connect failed\n");
				printf("waited reg:%x\n",res);
				return -1;
			}else{
				int session_id = (res)&0xFFFF;
				printf("socket:%d connect success with session_id:%d\n",socket,session_id);
				return session_id;
			}
		}
}

__device__ int get_session(socket_context_t* ctx,int socket,sock_addr_t dst_addr){
		printf("get_session called!\n");
		int p = 0;
		while(ctx->session_tbl[socket][p].valid == 1){
			if(ctx->session_tbl[socket][p].ip == dst_addr.ip && ctx->session_tbl[socket][p].port == dst_addr.port){
				return ctx->session_tbl[socket][p].session_id;
			}
			p+=1;
		}
		printf("session not found, try to connect a new one\n");
		int session_id = connect(ctx,socket,dst_addr);//using int to has -1 which means failed
		
		if(session_id==-1){//connect failed
			printf("connect faild!\n");
		}else{
			ctx->session_tbl[socket][p].valid		=	1;
			ctx->session_tbl[socket][p].port		=	dst_addr.port;
			ctx->session_tbl[socket][p].ip			=	dst_addr.ip;
			ctx->session_tbl[socket][p].session_id	=	session_id;
		}
		return session_id;
}

__device__ int get_session_first(socket_context_t* ctx,int socket){
	BEGIN_SINGLE_THREAD_DO
		printf("get_session_first called!\n");
		if(ctx->session_tbl[socket][0].valid == 1){
			return ctx->session_tbl[socket][0].session_id;
		}else{
			return -1;
		}
	END_SINGLE_THREAD_DO
}



socket_context_t* get_socket_context(unsigned int *dev_buffer,unsigned int *tlb_start_addr,fpga::XDMAController* controller){
	printf("socket get context function called!\n");
	cudaStream_t stream_send,stream_recv;
	cudaStreamCreate(&stream_send);
	cudaStreamCreate(&stream_recv);

	socket_context_t* ctx;
	cudaMalloc(&ctx,sizeof(socket_context_t));

	//ip
	unsigned int ip = get_ip();
	controller->writeReg(129,(unsigned int)ip);

	//mac
	controller->writeReg(128,ip>>24);
	// printf("mac:%d\n",ip>>24);


	size_t length = 100*1024*1024;
	//send buffer
	unsigned long tlb_start_addr_value = (unsigned long)tlb_start_addr;
	controller->writeReg(160,(unsigned int)tlb_start_addr_value);
	controller->writeReg(161,(unsigned int)(tlb_start_addr_value>>32));
	controller->writeReg(162,(unsigned int)length);
	controller->writeReg(163,(unsigned int)(length>>32));

	//recv buffer
	unsigned int* recv_tlb_start_addr = tlb_start_addr+int(100*1024*1024/sizeof(int));
	unsigned long recv_tlb_start_addr_value = (unsigned long)recv_tlb_start_addr;
	controller->writeReg(176,(unsigned int)recv_tlb_start_addr_value);
	controller->writeReg(177,(unsigned int)(recv_tlb_start_addr_value>>32));
	controller->writeReg(178,(unsigned int)length);
	controller->writeReg(179,(unsigned int)(length>>32));

	// std::cout << "listen status: " << controller->readReg(641) << std::endl;
	controller ->writeReg(0,0);
	controller ->writeReg(0,1);
	sleep(1);
	printf("read reg send buffer:%x %x\n",controller->readReg(160),controller->readReg(161));
	fpga_registers_t registers;
	registers.read_count					=	map_reg_4(47,controller);//useless

	registers.con_session_status			=	map_reg_4(641,controller);
	registers.send_write_count				=	map_reg_4(169,controller);
	registers.send_read_count				=	map_reg_4(656,controller);
	registers.recv_read_count				=	map_reg_4(180,controller);
	registers.recv_write_count				=	map_reg_4(658,controller);

	registers.listen_status					=	map_reg_4(640,controller);
	registers.listen_port					=	map_reg_4(130,controller);
	registers.listen_start					=	map_reg_4(131,controller);

	registers.conn_ip						=	map_reg_4(132,controller);
	registers.conn_port						=	map_reg_4(133,controller);
	registers.conn_start					=	map_reg_4(134,controller);

	registers.send_info_session_id			=	map_reg_4(164,controller);
	registers.send_info_addr_offset			=	map_reg_4(165,controller);
	registers.send_info_length				=	map_reg_4(167,controller);
	registers.send_info_start				=	map_reg_4(168,controller);
	
	printf("debug:%x\n",controller->readReg(641));
	send_kernel<<<1,1,0,stream_send>>>(ctx,dev_buffer,registers);
	recv_kernel<<<1,1024,0,stream_recv>>>(ctx,dev_buffer+int(100*1024*1024/sizeof(int)),registers);
	return ctx;
}


//util function
unsigned int* map_reg_4(int reg,fpga::XDMAController* controller){
	cudaError_t err;
	void * addr = (void*)(controller->getRegAddr(reg));
	unsigned int * dev_addr;
	err = cudaHostRegister(addr,4,cudaHostRegisterIoMemory);
	ErrCheck(err);
	cudaHostGetDevicePointer((void **) &(dev_addr), addr, 0);
	return dev_addr;
}

__device__ unsigned int get_info_tbl_index(socket_context_t* ctx){
	unsigned int res = ctx->send_info_tbl_index;
	ctx->send_info_tbl_index		=	(res+1)%MAX_CMD;
	return res;
}

__device__ void move_data(socket_context_t* ctx,int block_length,int *data_addr,int addr_offset){
	int index = blockIdx.x*blockDim.x+threadIdx.x;
	int op_num = int(block_length/sizeof(int));
	int total_threads = blockDim.x;
	int iter_num = int(op_num/total_threads);
	addr_offset = int(addr_offset/sizeof(int));

	BEGIN_SINGLE_THREAD_DO
		printf("block_length:%d  total_threads:%d  iter_num:%d  addr_offset:%d\n",block_length,total_threads,iter_num,addr_offset);
		printf("ctx->send buffer addr:%lx\n",ctx->send_buffer);
	END_SINGLE_THREAD_DO

	for(int i=0;i<iter_num;i++){
		ctx->send_buffer[addr_offset+total_threads*i+index]		=	data_addr[total_threads*i+index];
		//printf("%d\n",ctx->send_buffer[addr_offset+total_threads*i+index]);
	}
	if(op_num%total_threads!=0){
		printf("data length does not align!\n");
	}
}

__device__ void move_data_recv(socket_context_t* ctx,int block_length,int *data_addr,int addr_offset){
	int index = blockIdx.x*blockDim.x+threadIdx.x;
	int op_num = int(block_length/sizeof(int));
	int total_threads = blockDim.x;
	int iter_num = int(op_num/total_threads);
	addr_offset = int(addr_offset/sizeof(int));

	for(int i=0;i<iter_num;i++){
		data_addr[total_threads*i+index]	=	ctx->recv_buffer[addr_offset+total_threads*i+index];
	}
	if(op_num%total_threads!=0){
		printf("data length does not align!\n");
	}
}

__device__ bool check_socket_validation(socket_context_t* ctx,int socket){
	return socket>=0 && socket<ctx->socket_num;
}

__device__ recv_info_t read_info(socket_context_t* ctx,int index){
	index%=MAX_INFO_NUM;
	index*=16;

	recv_info_t res;

	//session_id,length,src_ip,src_port,session_close,addr_offset,dst_port
	res.session_id		= ctx->info_buffer[index+0];
	res.length 			= ctx->info_buffer[index+1];
	res.ip				= ctx->info_buffer[index+2];
	res.src_port		= ctx->info_buffer[index+3];
	res.session_close	= ctx->info_buffer[index+4];
	res.addr_offset		= ctx->info_buffer[index+5];
	res.dst_port		= ctx->info_buffer[index+6];
	return res;
}

__device__ int enroll(socket_context_t* ctx,int socket_id,int *data_addr,size_t length){//todo atomic
	int cur_index = ctx->enroll_list_pointer;
	ctx->enroll_list[cur_index].socket_id		=	socket_id;
	ctx->enroll_list[cur_index].type			=	ctx->socket_info[socket_id].type;
	ctx->enroll_list[cur_index].done			=	0;
	ctx->enroll_list[cur_index].data_addr		=	data_addr;
	ctx->enroll_list[cur_index].length			=	length;
	ctx->enroll_list[cur_index].cur_length		=	0;
	if(ctx->socket_info[socket_id].type==0){
		int session_id 	=	get_session_first(ctx,socket_id);
		if(session_id	==	-1){
			printf("enroll failed, client socket does not have session!\n");
			return -1;
		}
		ctx->enroll_list[cur_index].session_id	=	session_id;
	}else{
		ctx->enroll_list[cur_index].port			=	ctx->socket_info[socket_id].port;
	}
	ctx->enroll_list_pointer		=	cur_index+1;
	return cur_index;
}