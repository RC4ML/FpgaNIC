#include "network.cuh"
#include <assert.h>



__global__ void create_socket(socket_context_t* ctx,int* socket){
	BEGIN_SINGLE_THREAD_DO
		*(socket) = atomicAdd(&(ctx->socket_num),1);
		printf("create socket success with socket_id:%d\n",*(socket));
	END_SINGLE_THREAD_DO
}
__global__ void socket_listen(socket_context_t* ctx,int *socket,int port){
	BEGIN_SINGLE_THREAD_DO
		int socket_id = *socket;
		printf("enter listen function!\n");
		if(false==check_socket_validation(ctx,socket_id)){
			printf("socket_id %d does not exists!\n",socket_id);
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
				ctx->socket_info[socket_id].is_listening	=	1;
				ctx->socket_info[socket_id].port			=	port;
				printf("socket listen success!\n");
			}
		}
	END_SINGLE_THREAD_DO
}

__global__ void accept(socket_context_t* ctx,int socket_id,connection_t* connection){
	BEGIN_SINGLE_THREAD_DO
		if(ctx->socket_info[socket_id].is_listening == 0){
			printf("accept failed, no listen port set!\n");
			return;
		}else{
			int accept_list_index = accept_list_enroll(ctx,socket_id);
			if(accept_list_index == -1){
				printf("accept_list_enroll failed!\n");
			}else{
				while(ctx->accept_list[accept_list_index].done==0){
					cu_sleep(4);
					printf("accepting!\n");
				}
				//todo
				int buffer_id=0;
				connection->session_id 		= ctx->accept_list[accept_list_index].session_id;
				connection->src_ip			= ctx->accept_list[accept_list_index].src_ip;
				connection->src_port		= ctx->accept_list[accept_list_index].src_port;
				connection->buffer_id		= ctx->accept_list[accept_list_index].buffer_id;
				buffer_id					= connection->buffer_id;
				
				ctx->buffer_info[buffer_id].valid		=	1;
				ctx->buffer_info[buffer_id].type		=	1;
				ctx->buffer_info[buffer_id].connection	=	connection;

			}
		}
	END_SINGLE_THREAD_DO
}

__global__ void connect(socket_context_t* ctx,int socket_id,sock_addr_t addr){
	BEGIN_SINGLE_THREAD_DO
		if(false==check_socket_validation(ctx,socket_id)){
			printf("socket %d does not exists!\n",socket_id);
			return;
		}
		printf("connecting, dst_ip:%x dst_port:%d\n",addr.ip,addr.port);
		if(ctx->buffer_used == MAX_BUFFER_NUM){
			printf("buffer runs out, can't connect!\n");
			return;
		}

		*(ctx->conn_ip)		=	addr.ip;
		*(ctx->conn_port)	=	addr.port;
		//buffer_id todo
		*(ctx->conn_start)	=	1;
		
		volatile int res = wait_done(ctx->con_session_status,17);
		*(ctx->conn_start)	=	0;

		if(res==-1){//timeout
			printf("socket connect timeout\n");
			return;
		}else{
			if(is_zero((unsigned int *)&res,16)){
				printf("socket connect failed\n");
				printf("waited reg:%x\n",res);
				return;
			}else{
				int buffer_id;
				buffer_id = get_empty_buffer(ctx);
				if(buffer_id==-1){
					printf("connect get empty buffer error!\n");
					return;
				}
				for(int i=0;i<MAX_BUFFER_NUM;i++){
					if(ctx->buffer_valid[i] == 0){
						buffer_id = i;
						break;
					}
				}
				int session_id = (res)&0xFFFF;
				ctx->buffer_valid[buffer_id] 				=	1;
				ctx->socket_info[socket_id].valid			=	1;//todo
				ctx->socket_info[socket_id].buffer_id 		=	buffer_id;
				ctx->buffer_info[buffer_id].type			=	0;
				ctx->buffer_info[buffer_id].valid			=	1;
				ctx->buffer_info[buffer_id].session_id		=	session_id;
				ctx->buffer_info[buffer_id].socket_id		=	socket_id;
				
				ctx->send_read_count[buffer_id]		=	0;
				ctx->send_write_count[buffer_id]	=	0;
				ctx->recv_read_count[buffer_id]		=	0;
				printf("socket:%d connect success with session_id:%d\n",socket_id,session_id);
				return;
			}
		}
	END_SINGLE_THREAD_DO
}

// __device__ connection_t get_session(socket_context_t* ctx,int socket,sock_addr_t dst_addr){
// 		printf("get_session called!\n");
// 		int p = 0;
// 		connection_t session;
// 		while(ctx->connection_tbl[socket][p].valid == 1){
// 			if(ctx->connection_tbl[socket][p].ip == dst_addr.ip && ctx->connection_tbl[socket][p].port == dst_addr.port){
// 				return ctx->connection_tbl[socket][p];
// 			}
// 			p+=1;
// 		}
// 		printf("session not found, try to connect a new one\n");
// 		if(ctx->buffer_used == MAX_BUFFER_NUM){
// 			printf("buffer runs out, can't connect!\n");
// 			session.session_id = -1;
// 			return session;
// 		}
// 		int session_id = connect(ctx,socket,dst_addr);//using int to has -1 which means failed
		
// 		if(session_id==-1){//connect failed
// 			printf("connect faild!\n");
// 			session.session_id = -1;
// 			return session;
// 		}else{
// 			ctx->connection_tbl[socket][p].valid		=	1;
// 			ctx->connection_tbl[socket][p].port		=	dst_addr.port;
// 			ctx->connection_tbl[socket][p].ip			=	dst_addr.ip;
// 			ctx->connection_tbl[socket][p].session_id	=	session_id;
// 			int buffer_id;
// 			for(int i=0;i<MAX_BUFFER_NUM;i++){
// 				if(ctx->buffer_valid[i] == 0){
// 					ctx->connection_tbl[socket][p].buffer_id = i;
// 					buffer_id = i;
// 					break;
// 				}
// 			}
// 			ctx->send_read_count[buffer_id]		=	0;
// 			ctx->send_write_count[buffer_id]	=	0;
// 			ctx->recv_read_count[buffer_id]		=	0;

// 		}
// 		return ctx->connection_tbl[socket][p];
// }

// __device__ int get_session_first(socket_context_t* ctx,int socket){
// 	BEGIN_SINGLE_THREAD_DO
// 		printf("get_session_first called!\n");
// 		if(ctx->connection_tbl[socket][0].valid == 1){
// 			return ctx->connection_tbl[socket][0].session_id;
// 		}else{
// 			return -1;
// 		}
// 	END_SINGLE_THREAD_DO
// }



socket_context_t* get_socket_context(unsigned int *dev_buffer,unsigned int *tlb_start_addr,fpga::XDMAController* controller){
	//dev_buffer corresponds to tlb_start_addr
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


	//send buffer
	unsigned long tlb_start_addr_value = (unsigned long)tlb_start_addr;
	controller->writeReg(160,(unsigned int)tlb_start_addr_value);
	controller->writeReg(161,(unsigned int)(tlb_start_addr_value>>32));
	controller->writeReg(162,SINGLE_BUFFER_LENGTH);
	
	controller->writeReg(164,MAX_BUFFER_NUM);

	//recv buffer

	unsigned int* recv_tlb_start_addr = tlb_start_addr+int((TOTAL_BUFFER_LENGTH)/sizeof(int));//102M
	unsigned long recv_tlb_start_addr_value = (unsigned long)recv_tlb_start_addr;
	controller->writeReg(176,(unsigned int)recv_tlb_start_addr_value);
	controller->writeReg(177,(unsigned int)(recv_tlb_start_addr_value>>32));
	controller->writeReg(178,SINGLE_BUFFER_LENGTH);
	controller->writeReg(179,INFO_BUFFER_LENGTH);
	controller->writeReg(180,ALMOST_FULL_LENGTH);
	controller->writeReg(181,MAX_BUFFER_NUM);

	// std::cout << "listen status: " << controller->readReg(641) << std::endl;
	controller ->writeReg(0,0);
	controller ->writeReg(0,1);
	sleep(1);
	printf("read reg send buffer:%x %x\n",controller->readReg(160),controller->readReg(161));
	fpga_registers_t registers;
	registers.read_count					=	map_reg_4(47,controller);//useless

	registers.con_session_status			=	map_reg_4(641,controller);

	registers.listen_status					=	map_reg_4(640,controller);
	registers.listen_port					=	map_reg_4(130,controller);
	registers.listen_start					=	map_reg_4(131,controller);

	registers.conn_ip						=	map_reg_4(132,controller);
	registers.conn_port						=	map_reg_4(133,controller);
	registers.conn_start					=	map_reg_4(134,controller);

	registers.conn_response					=	map_reg_4(140,controller);//todo

	// registers.send_info_session_id			=	map_reg_4(164,controller);
	// registers.send_info_addr_offset			=	map_reg_4(165,controller);
	// registers.send_info_length				=	map_reg_4(167,controller);
	// registers.send_info_start				=	map_reg_4(168,controller);

	registers.send_data_cmd_bypass_reg		=	map_reg_64(1,controller);
	registers.recv_read_count_bypass_reg		=	map_reg_64(2,controller);

	printf("debug:%x\n",controller->readReg(641));
	//attention 
	send_kernel<<<1,1,0,stream_send>>>(ctx,dev_buffer,registers);//0-2 info  2-102 send buffer
	recv_kernel<<<1,1024,0,stream_recv>>>(ctx,dev_buffer+int((INFO_BUFFER_LENGTH+TOTAL_BUFFER_LENGTH)/sizeof(int)),registers);
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

unsigned int* map_reg_64(int reg,fpga::XDMAController* controller){
	cudaError_t err;
	void * addr = (void*)(controller->getBypassAddr(reg));
	unsigned int * dev_addr;
	err = cudaHostRegister(addr,64,cudaHostRegisterIoMemory);
	ErrCheck(err);
	cudaHostGetDevicePointer((void **) &(dev_addr), addr, 0);
	return dev_addr;
}



__device__ void move_data_to_send_buffer(socket_context_t* ctx,int buffer_id,int block_length,int *data_addr){
	int index = blockIdx.x*blockDim.x+threadIdx.x;
	int op_num = int(block_length/sizeof(int));
	int total_threads = blockDim.x;
	int iter_num = int(op_num/total_threads);
	int addr_base_offset = SINGLE_BUFFER_LENGTH*buffer_id+ctx->send_buffer_offset[buffer_id];//base offset
	addr_base_offset = int(addr_base_offset/sizeof(int));

	BEGIN_SINGLE_THREAD_DO
		printf("block_length:%d  total_threads:%d  iter_num:%d  addr_base_offset:%d\n",block_length,total_threads,iter_num,addr_base_offset);
		if(op_num%total_threads!=0){
			printf("data length does not align!\n");
		}
	END_SINGLE_THREAD_DO

	for(int i=0;i<iter_num;i++){
		ctx->send_buffer[addr_base_offset+total_threads*i+index]		=	data_addr[total_threads*i+index];
		//printf("%d\n",ctx->send_buffer[addr_offset+total_threads*i+index]);
	}
	
}

__device__ void move_data_from_recv_buffer(socket_context_t* ctx,int buffer_id,int block_length,int *data_addr){
	int index = blockIdx.x*blockDim.x+threadIdx.x;
	int op_num = int(block_length/sizeof(int));
	int total_threads = blockDim.x;
	int iter_num = int(op_num/total_threads);
	int addr_base_offset = SINGLE_BUFFER_LENGTH*buffer_id+ctx->recv_read_count[buffer_id];//base offset
	addr_base_offset = int(addr_base_offset/sizeof(int));

	BEGIN_SINGLE_THREAD_DO
		printf("block_length:%d  total_threads:%d  iter_num:%d  addr_base_offset:%d\n",block_length,total_threads,iter_num,addr_base_offset);
		if(op_num%total_threads!=0){
			printf("data length does not align!\n");
		}
	END_SINGLE_THREAD_DO
	for(int i=0;i<iter_num;i++){
		data_addr[total_threads*i+index]	=	ctx->recv_buffer[total_threads*i+index];
	}
	
}

__device__ bool check_socket_validation(socket_context_t* ctx,int socket){
	return socket>=0 && socket<ctx->socket_num;
}

// __device__ recv_info_t read_info(socket_context_t* ctx,int index){
// 	index%=MAX_INFO_NUM;
// 	index*=16;

// 	recv_info_t res;

// 	//session_id,length,src_ip,src_port,session_close,addr_offset,dst_port
// 	res.session_id		= ctx->info_buffer[index+0];
// 	res.length 			= ctx->info_buffer[index+1];
// 	res.ip				= ctx->info_buffer[index+2];
// 	res.src_port		= ctx->info_buffer[index+3];
// 	res.session_close	= ctx->info_buffer[index+4];
// 	res.addr_offset		= ctx->info_buffer[index+5];
// 	res.dst_port		= ctx->info_buffer[index+6];
// 	return res;
// }

// __device__ int enroll(socket_context_t* ctx,int socket_id,int *data_addr,size_t length){//todo atomic
// 	int cur_index = ctx->enroll_list_pointer;
// 	ctx->enroll_list[cur_index].socket_id		=	socket_id;
// 	ctx->enroll_list[cur_index].type			=	ctx->socket_info[socket_id].type;
// 	ctx->enroll_list[cur_index].done			=	0;
// 	ctx->enroll_list[cur_index].data_addr		=	data_addr;
// 	ctx->enroll_list[cur_index].length			=	length;
// 	ctx->enroll_list[cur_index].cur_length		=	0;
// 	if(ctx->socket_info[socket_id].type==0){
// 		int session_id 	=	get_session_first(ctx,socket_id);
// 		if(session_id	==	-1){
// 			printf("enroll failed, client socket does not have session!\n");
// 			return -1;
// 		}
// 		ctx->enroll_list[cur_index].session_id	=	session_id;
// 	}else{
// 		ctx->enroll_list[cur_index].port			=	ctx->socket_info[socket_id].port;
// 	}
// 	ctx->enroll_list_pointer		=	cur_index+1;
// 	return cur_index;
// }

__device__ void write_bypass(volatile unsigned int *dev_addr,unsigned int *data){
	int index = blockIdx.x*blockDim.x+threadIdx.x;
	__syncthreads();
	if(index<16){
		dev_addr[index] = data[index];	
	}
	__syncthreads();
}

__device__ void read_info(socket_context_t* ctx){
	int offset = ctx->info_offset;
	if(ctx->info_buffer[offset+0] > ctx->info_count){
		ctx->info_count++;
		int info_type = ctx->info_buffer[offset+1];
		if(info_type==2){//update read_count
			int buffer_id = ctx->info_buffer[offset+2];//todo
			int session_id = ctx->info_buffer[offset+3];
			unsigned long read_count=(ctx->info_buffer[offset+6]);
			read_count = read_count<<32;
			read_count+=ctx->info_buffer[offset+7];
			ctx->send_read_count[buffer_id] = read_count;
		}else if(info_type==0){//open todo
			if(ctx->buffer_used<MAX_BUFFER_NUM){
				int dst_port = ctx->info_buffer[offset+6];
				int list_index = get_accept_list_index(ctx,dst_port);
				if(list_index==-1){
					printf("accept failed, can't find object in accept list waiting port:%d\n",dst_port);
					*(ctx->conn_response)	=	0x10000000;
				}else{
					int buffer_id = get_empty_buffer(ctx);
					int session_id = ctx->info_buffer[offset+3];
					int src_ip = ctx->info_buffer[offset+4];
					int src_port = ctx->info_buffer[offset+5];
					ctx->accept_list[list_index].src_ip			= src_ip;
					ctx->accept_list[list_index].src_port		= src_port;
					ctx->accept_list[list_index].session_id		= session_id;
					ctx->accept_list[list_index].buffer_id		= buffer_id;
					ctx->accept_list[list_index].done			= 1;

					*(ctx->conn_response)	=	0x20000000+buffer_id;//todo
				}

			}else{
				printf("accepting failed, no more buffer\n");
				*(ctx->conn_response)	=	0x10000000;
			}
			
		}else if(info_type==1){//close

		}
		
		ctx->info_offset+=16;//++512 bit
	}
}

__device__ int get_empty_buffer(socket_context_t* ctx){
	if(ctx->buffer_used == MAX_BUFFER_NUM){
		printf("buffer runs out, can't find!\n");
		return -1;
	}else{
		for(int i=0;i<MAX_BUFFER_NUM;i++){
			if(ctx->buffer_valid[i] == 0){
				return i;
			}
		}
		return -1;
	}
}

__device__ int accept_list_enroll(socket_context_t* ctx,int socket_id){
	if(ctx->accept_num==MAX_ACCEPT_LIST_LENGTH){
		return -1;
	}else{
		for(int i=0;i<MAX_ACCEPT_LIST_LENGTH;i++){
			if(ctx->accept_list[i].valid==0){
				ctx->accept_list[i].valid=1;
				ctx->accept_list[i].done=0;
				ctx->accept_list[i].listening_port=ctx->socket_info[socket_id].port;
				ctx->accept_num+=1;
				return i;
			}
		}
		return -1;
	}
}

__device__ int get_accept_list_index(socket_context_t* ctx,int port){
	int check_num=0;
	for(int i=0;i<MAX_ACCEPT_LIST_LENGTH;i++){
		if(ctx->accept_list[i].valid==1){
			if(ctx->accept_list[i].listening_port == port){
				return i;
			}
			check_num++;
		}
		if(check_num == ctx->accept_num){
			break;
		}
	}
	return -1;
}

__device__ int fetch_head(socket_context_t* ctx,int buffer_id){
	int offset = int(ctx->recv_read_count[buffer_id]/sizeof(int));
	while(ctx->recv_buffer[offset+0]==0){//todo
		cu_sleep(4);
		printf("waiting data head!\n");
	}
	int length = ctx->recv_buffer[offset+1];
	ctx->recv_read_count[buffer_id]+=64;
	return length;
}