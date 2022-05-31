#include "network.cuh"
#include <assert.h>
#include <iostream>
#include "tool/log.hpp"

__global__ void create_socket(socket_context_t* ctx,int* socket){
	BEGIN_SINGLE_THREAD_DO
		*(socket) = atomicAdd(&(ctx->socket_num),1);
		cjprint("create socket success with socket_id:%d\n",*(socket));
	END_SINGLE_THREAD_DO
}
__global__ void socket_listen(socket_context_t* ctx,int *socket,int port){
	BEGIN_SINGLE_THREAD_DO
		int socket_id = *socket;
		cjinfo("enter listen function!\n");
		if(false==check_socket_validation(ctx,socket_id)){
			cjerror("socket_id %d does not exists!\n",socket_id);
			return;
		}

		*(ctx->listen_port)		=	port;
		*(ctx->listen_start)	=	1;
		cjprint("try listen,socket:%d\n",socket_id);
		int res = wait_done(ctx->listen_status,1);
		*(ctx->listen_start)	=	0;

		if(res==-1){//timeout
			cjerror("socket listen timeout\n");
			return;
		}else{
			cjerror("socket listen  with reg:%x\n",res);//hhj has a bug to fix
			ctx->server_socket_id						=	socket_id;
			ctx->server_port							=	port;
			ctx->is_listening							=	1;

			// if(is_zero((unsigned int *)&res,0)){//todo
			// 	cjerror("socket listen failed with reg:%x\n",res);
			// 	return;
			// }else{
			// 	ctx->server_socket_id						=	socket_id;
			// 	ctx->server_port							=	port;
			// 	ctx->is_listening							=	1;
			// 	cjprint("socket %d listen success!\n",socket_id);
			// }
		}
	END_SINGLE_THREAD_DO
}

__global__ void accept(socket_context_t* ctx,int* socket,connection_t* connection){
	__shared__ int socket_id;
	BEGIN_SINGLE_THREAD_DO
		socket_id	=	*socket;
		if(ctx->is_listening == 0 || socket_id!=ctx->server_socket_id){
			cjerror("accept failed, not listening!\n");
			return;
		}else if(ctx->is_accepting==1){
			cjerror("accept failed,is accepting!\n");
			return;
		}else{
			ctx->accepted=0;
			ctx->connection_builder	=	connection;
			ctx->is_accepting		=	1;
			cjprint("accepting!\n");
			while(ctx->accepted==0){
				// #if SLOW_DEBUG
				// 	cjdebug("accepting!\n");
				// 	cu_sleep(1);
				// #endif
			}
			//todo
			int buffer_id							=	connection->buffer_id;
			ctx->buffer_info[buffer_id].session_id	=	connection->session_id;
			ctx->buffer_info[buffer_id].valid		=	1;
			ctx->buffer_info[buffer_id].type		=	1;
			ctx->buffer_info[buffer_id].connection	=	connection;
			ctx->accepted							=	0;
			ctx->is_accepting						=	0;

			ctx->recv_package_count[buffer_id]		=	0;
			cjprint("accepted a connection!\n");
			
		}
	END_SINGLE_THREAD_DO
}

__global__ void connect(socket_context_t* ctx,int *socket,sock_addr_t addr){
	__shared__ int socket_id;
	__shared__ int buffer_id;
	BEGIN_SINGLE_THREAD_DO
		socket_id = *socket;
		if(false==check_socket_validation(ctx,socket_id)){
			cjerror("socket %d does not exists!\n",socket_id);
			return;
		}
		cjprint("connecting, dst_ip:%x dst_port:%d\n",addr.ip,addr.port);
		if(ctx->buffer_used >= MAX_BUFFER_NUM){
			cjerror("buffer runs out, can't connect!\n");
			return;
		}
		
		buffer_id = get_empty_buffer(ctx);
		if(buffer_id==-1){
			cjerror("connect get empty buffer error!\n");
			return;
		}
		*(ctx->conn_ip)			=	addr.ip;
		*(ctx->conn_port)		=	addr.port;
		*(ctx->conn_buffer_id)	=	buffer_id;
		*(ctx->conn_start)	=	1;
		volatile int res = wait_done(ctx->con_session_status,17);
		//cjinfo("connect response reg :%x\n",res);
		*(ctx->conn_start)	=	0;

		if(res==-1){//timeout
			cjerror("socket connect timeout\n");
			return;
		}else{
			if(is_zero((unsigned int *)&res,16)){
				cjerror("socket connect failed\n");
				return;
			}else{

				int session_id = (res)&0xFFFF;
				ctx->buffer_valid[buffer_id] 				=	1;
				ctx->socket_info[socket_id].valid			=	1;
				ctx->socket_info[socket_id].buffer_id 		=	buffer_id;

				ctx->buffer_info[buffer_id].type			=	0;
				ctx->buffer_info[buffer_id].valid			=	1;
				ctx->buffer_info[buffer_id].session_id		=	session_id;
				ctx->buffer_info[buffer_id].socket_id		=	socket_id;
				
				ctx->send_read_count[buffer_id]		=	0;
				ctx->send_write_count[buffer_id]	=	0;
				ctx->recv_read_count[buffer_id]		=	0;
				ctx->recv_package_count[buffer_id]	=	0;
				cu_sleep(2);
				cjprint("socket:%d connect success with session_id:%d\n",socket_id,session_id);
				return;
			}
		}
	END_SINGLE_THREAD_DO
}



socket_context_t* get_socket_context(unsigned int *dev_buffer,unsigned int *tlb_start_addr,fpga::XDMAController* controller,int node_type){
	controller ->writeReg(0,0);
	controller ->writeReg(0,1);
	sleep(1);//reset

	
	//dev_buffer corresponds to tlb_start_addr
	cjinfo("socket get context function called!\n");
	cudaStream_t stream_send;
	cudaStreamCreate(&stream_send);

	socket_context_t* ctx;
	cudaMalloc(&ctx,sizeof(socket_context_t));

	//ip
	unsigned int ip = get_ip();
	controller->writeReg(129,(unsigned int)ip);

	//mac
	unsigned int mac = ip;
	controller->writeReg(128,mac);
	cjinfo("mac:%d\n",ip&0xff);


	//send buffer
	unsigned long tlb_start_addr_value = (unsigned long)tlb_start_addr;
	cjinfo("send buffer addr start:%ld\n",(unsigned long)tlb_start_addr);
	controller->writeReg(160,(unsigned int)tlb_start_addr_value);
	controller->writeReg(161,(unsigned int)(tlb_start_addr_value>>32));
	controller->writeReg(162,SINGLE_BUFFER_LENGTH);
	
	controller->writeReg(164,MAX_BUFFER_NUM);

	controller->writeReg(166,TOKEN_SPEED);

	//recv buffer

	unsigned int* recv_tlb_start_addr = tlb_start_addr+int((TOTAL_BUFFER_LENGTH)/sizeof(int));//102M
	unsigned long recv_tlb_start_addr_value = (unsigned long)recv_tlb_start_addr;
	controller->writeReg(176,(unsigned int)recv_tlb_start_addr_value);
	controller->writeReg(177,(unsigned int)(recv_tlb_start_addr_value>>32));
	controller->writeReg(178,SINGLE_BUFFER_LENGTH);
	controller->writeReg(179,INFO_BUFFER_LENGTH);
	controller->writeReg(180,MAX_PACKAGE_LENGTH);
	controller->writeReg(181,MAX_BUFFER_NUM);
	controller->writeReg(182,PACKAGE_LENGTH_512);

	fpga_registers_t registers;
	registers.con_session_status			=	map_reg_4(641,controller);

	registers.listen_status					=	map_reg_4(640,controller);
	registers.listen_port					=	map_reg_4(130,controller);
	registers.listen_start					=	map_reg_4(131,controller);

	registers.conn_ip						=	map_reg_4(132,controller);
	registers.conn_port						=	map_reg_4(133,controller);
	registers.conn_buffer_id				=	map_reg_4(134,controller);
	registers.conn_start					=	map_reg_4(135,controller);

	registers.conn_re_session_id			=	map_reg_4(138,controller);
	registers.conn_response					=	map_reg_4(139,controller);
	registers.conn_res_start				=	map_reg_4(140,controller);

	registers.tcp_conn_close_session		=	map_reg_4(136,controller);
	registers.tcp_conn_close_start			=	map_reg_4(137,controller);

	registers.send_cmd_fifo_count			=	map_reg_4(659,controller);
	

	registers.send_data_cmd_bypass_reg		=	map_reg_64(3,controller);
	registers.recv_read_count_bypass_reg		=	map_reg_64(2,controller);

	//attention 
	send_kernel<<<1,1,0,stream_send>>>(ctx,dev_buffer,registers,node_type);//0-2 info  2-102 send buffer
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

unsigned int* map_reg_cj(int reg,fpga::XDMAController* controller,size_t length){
	cudaError_t err;
	void * addr = (void*)(controller->getBypassAddr(reg));
	unsigned int * dev_addr;
	err = cudaHostRegister(addr,length,cudaHostRegisterIoMemory);
	ErrCheck(err);
	cudaHostGetDevicePointer((void **) &(dev_addr), addr, 0);
	return dev_addr;
}



__device__ void move_data_to_send_buffer(socket_context_t* ctx,int buffer_id,int block_length,int *data_addr){
	int index = blockIdx.x*blockDim.x+threadIdx.x;
	int op_num = int(block_length/sizeof(int));
	int total_threads = blockDim.x;
	int iter_num = int(op_num/total_threads);
	int addr_base_offset = SINGLE_BUFFER_LENGTH*buffer_id+ctx->send_buffer_offset[buffer_id];//base offset in bytes
	addr_base_offset = int(addr_base_offset/sizeof(int));

	// BEGIN_SINGLE_THREAD_DO
	// 	cjdebug("sending, block_length:%x  buffer_id:%d addr_base_offset:%x\n",block_length,buffer_id,addr_base_offset);
	// 	if(op_num%total_threads!=0){
	// 		cjerror("data length does not align!\n");
	// 		return;
	// 	}
	// END_SINGLE_THREAD_DO
	{//move code
		for(int i=0;i<iter_num;i++){
			ctx->send_buffer[addr_base_offset+total_threads*i+index]		=	data_addr[total_threads*i+index];
		}
	}
	

	{//ptx
		// for(int i=0;i<iter_num;i++){
		// 	volatile uint64_t  addr = (uint64_t)((ctx->send_buffer)+addr_base_offset+total_threads*i+index);
		// 	volatile unsigned int value=(unsigned int)data_addr[total_threads*i+index];
		// 	asm volatile(
		// 		"st.u32.wt [%0],%1;\n\t"
		// 		:"+l"(addr):"r"(value):"memory"
		// 	);
		// }
	}

		//st addr
	// 	for(int i=0;i<iter_num;i++){
	// 		if((addr_base_offset+total_threads*i+index)%16==0){
	// 			ctx->send_buffer[addr_base_offset+total_threads*i+index] = ((unsigned long)(ctx->send_buffer+addr_base_offset+total_threads*i+index));
	// 		}else{
	// 			ctx->send_buffer[addr_base_offset+total_threads*i+index]=ctx->send_buffer[addr_base_offset+total_threads*i+index];
	// 		}
	// 	}
	// __threadfence();
	
}

__device__ void move_data_from_recv_buffer(socket_context_t* ctx,int buffer_id,int block_length,int *data_addr){
	int index = blockIdx.x*blockDim.x+threadIdx.x;
	int op_num = int(block_length/sizeof(int));
	int total_threads = blockDim.x;
	int iter_num = int(op_num/total_threads);
	int rest_num = op_num%total_threads;
	int addr_base_offset = SINGLE_BUFFER_LENGTH*buffer_id+ctx->recv_buffer_offset[buffer_id];//base offset
	addr_base_offset = int(addr_base_offset/sizeof(int));

	BEGIN_SINGLE_THREAD_DO
		//ptmark
		// cjdebug("buffer:%d recv_read_count:%ld rdcount_record:%d\n",buffer_id,ctx->recv_read_count[buffer_id],ctx->buffer_read_count_record[buffer_id]);
		// if(op_num%total_threads!=0){
		//  	cjerror("data length does not align, op_num:%d\n",op_num);
		// }
	END_SINGLE_THREAD_DO
	
	{//move mark
		for(int i=0;i<iter_num;i++){
			data_addr[total_threads*i+index]	=	ctx->recv_buffer[addr_base_offset+total_threads*i+index];
			ctx->recv_buffer[addr_base_offset+total_threads*i+index]	=	0;
		}
		if(index<rest_num){//process tail data
			data_addr[total_threads*iter_num+index]	=	ctx->recv_buffer[addr_base_offset+total_threads*iter_num+index];
			ctx->recv_buffer[addr_base_offset+total_threads*iter_num+index]	=	0;
		}
	}
	
	
}

__device__ bool check_socket_validation(socket_context_t* ctx,int socket){
	return socket>=0 && socket<ctx->socket_num;
}


__device__ void write_bypass(volatile unsigned int *dev_addr,unsigned int *data){
	int index = threadIdx.x;
	__syncthreads();
	if(index<16){
		dev_addr[index] = data[index];	
	}
	__syncthreads();
}

__device__ int read_info(socket_context_t* ctx){
	int offset = ctx->info_offset;
	if(ctx->info_buffer[offset+0] > ctx->info_count){
		ctx->info_offset+=16;//++512 bit
		ctx->info_count++;
		int info_type = ctx->info_buffer[offset+1];
		// cjinfo("read a info, type:%d\n",info_type);//cjmark
		// for(int i=0;i<16;i++){//cjmark
		// 	cjdebug("%x ",ctx->info_buffer[offset+i]);
		// }
		// cjdebug("\n");
		if(info_type==2){//update read_count
			int buffer_id = ctx->info_buffer[offset+2];//todo seq done
			unsigned long read_count=(ctx->info_buffer[offset+7]);
			read_count = read_count<<32;
			read_count+=ctx->info_buffer[offset+6];
			ctx->send_read_count[buffer_id] = read_count;
			//cjdebug("---update rd count buffer:%d with %lx\n",buffer_id,read_count);//cjmark
		}else if(info_type==0){//open todo
			if(ctx->is_accepting == 0){
				cjerror("nobody is accepting\n");
				*(ctx->conn_response)	=	0x00000000;
			}else if(ctx->buffer_used<MAX_BUFFER_NUM){
				
				int buffer_id								=	get_empty_buffer(ctx);
				int session_id 								=	ctx->info_buffer[offset+3];
				int src_ip 									=	ctx->info_buffer[offset+4];
				int src_port								=	ctx->info_buffer[offset+5];
				ctx->connection_builder->session_id			=	session_id;
				ctx->connection_builder->src_ip 			=	src_ip;
				ctx->connection_builder->src_port			=	src_port;
				ctx->connection_builder->buffer_id			=	buffer_id;
				ctx->connection_builder->valid				=	1;

				*(ctx->conn_re_session_id)					=	session_id;
				*(ctx->conn_response)						=	0xf0000000+buffer_id;
				ctx->accepted								=	1;
				ctx->buffer_read_count_record[buffer_id]	=	1;//flow control
				//todo
				

			}else if(ctx->buffer_used>=MAX_BUFFER_NUM){
				cjerror("accepting failed, no more buffer\n");
				*(ctx->conn_response)	=	0x00000000;
			}
			*(ctx->conn_res_start)	=	1;
			*(ctx->conn_res_start)	=	0;
		}else if(info_type==1){//close

		}
		return 1;
	}else{
		return 0;
	}
}


__device__ int fetch_head(socket_context_t* ctx,int buffer_id){
	int offset = int(ctx->recv_buffer_offset[buffer_id]/sizeof(int));
	while(ctx->recv_buffer[offset+1]==0){
		// ptmark
		// #if SLOW_DEBUG
		// 	cu_sleep(1);
		// 	cjdebug("waiting data head!\n");
		// #endif
	}
	int length = ctx->recv_buffer[offset+1];
	int tail_offset = int(length/sizeof(int)) - int(64/sizeof(int));
	while(ctx->recv_buffer[offset+1+tail_offset] != length){
		// ptmark
		// #if SLOW_DEBUG
		// 	cu_sleep(1);
		// 	cjdebug("waiting data tail!\n");
		// 	for(int i=0;i<16;i++){
		// 		cjdebug("%d ",ctx->recv_buffer[offset+i]);
		// 	}
		// 	cjdebug("\n");
		// #endif
	}
	ctx->recv_buffer[offset+1] = 0;
	ctx->recv_buffer[offset+1+tail_offset] = 0;

	ctx->recv_buffer_offset[buffer_id]+=64;
	return length-128;
}

__device__ int get_empty_buffer(socket_context_t* ctx){
	
	if(ctx->buffer_used == MAX_BUFFER_NUM){
		cjerror("buffer runs out, can't find!\n");
		return -1;
	}else{
		for(int i=0;i<MAX_BUFFER_NUM;i++){
			if(ctx->buffer_valid[i] == 0){
				ctx->buffer_valid[i] = 1;//to do seq
				cjdebug("get empty buffer %d\n",i);
				return i;
			}
		}
		cjerror("error,buffer runs out, buffer_used is wrong!\n");
		return -1;
	}
}