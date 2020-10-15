#include "network_kernel.cuh"

__device__ void _socket_send(socket_context_t* ctx,int buffer_id,int * data_addr,size_t length){
	__shared__ unsigned int block_length;
	__shared__ int session_id;

	BEGIN_SINGLE_THREAD_DO
		session_id = ctx->buffer_info[buffer_id].session_id;
	END_SINGLE_THREAD_DO
	for(size_t i=0;i<length;i+=MAX_BLOCK_SIZE){
		BEGIN_SINGLE_THREAD_DO
			block_length = min((unsigned long)MAX_BLOCK_SIZE,length-i);
			while(ctx->send_write_count[buffer_id] > ctx->send_read_count[buffer_id]+(SINGLE_BUFFER_LENGTH-ALMOST_FULL_LENGTH)){
				cu_sleep(4);
			}//stuck if space not enough
		END_SINGLE_THREAD_DO
		
		// parallel move data code
		move_data_to_send_buffer(ctx,buffer_id,block_length,data_addr);
		data_addr+=int(block_length/sizeof(int));
		
		//inform code
		__shared__ unsigned int data[16];
		BEGIN_SINGLE_THREAD_DO
			printf("move data done! offset:%d  \n",ctx->send_buffer_offset[buffer_id]);
			data[0] = block_length;//todo
			data[1] = ctx->send_buffer_offset[buffer_id];
			data[2] = session_id;
			data[3] = buffer_id;

			ctx->send_write_count[buffer_id]	+=	block_length;
			ctx->send_buffer_offset[buffer_id]		+=	block_length;
			if(ctx->send_buffer_offset[buffer_id]>(SINGLE_BUFFER_LENGTH-ALMOST_FULL_LENGTH)){
				ctx->send_buffer_offset[buffer_id]	=	0;
			}
		END_SINGLE_THREAD_DO
		write_bypass(ctx->send_data_cmd_bypass_reg,data);
	}

}
__global__ void socket_send(socket_context_t* ctx,int* socket,int * data_addr,size_t length){
	__shared__ int buffer_id;
	//verify socket
	BEGIN_SINGLE_THREAD_DO
		printf("function socket_send called!\n");
		int socket_id = *socket;
		if(false==check_socket_validation(ctx,socket_id)){
			printf("socket %d does not exists!\n",socket_id);
			return;
		}
		
		if(ctx->socket_info[socket_id].valid == 0 ){
			printf("socket %d does not have connections!\n",socket_id);
			return;
		}

		buffer_id = ctx->socket_info[socket_id].buffer_id;
		printf("send data with socket:%d  buffer_id:%d	data_addr:%lx	length:%ld\n",(*socket),buffer_id,data_addr,length);
	END_SINGLE_THREAD_DO
	_socket_send(ctx,buffer_id,data_addr,length);
}


__global__ void socket_send(socket_context_t* ctx,connection_t* connection,int * data_addr,size_t length){
	__shared__ int buffer_id;
	//verify connection
	BEGIN_SINGLE_THREAD_DO
		printf("function socket_send connection type called!\n");
		if(connection->valid==0){
			printf("connection is not valid!\n");
			return;
		}
		buffer_id = connection->buffer_id;
	END_SINGLE_THREAD_DO
	_socket_send(ctx,buffer_id,data_addr,length);
}

__device__ void _socket_recv(socket_context_t* ctx,int buffer_id,int * data_addr,size_t length){
	__shared__ int socket_id;
	__shared__ int session_id;
	__shared__ size_t cur_length;
	__shared__ size_t flow_control_flag;
	__shared__ unsigned int data[16];
	BEGIN_SINGLE_THREAD_DO
		cur_length=0;
		flow_control_flag=0;
		session_id = ctx->buffer_info[buffer_id].session_id;
	END_SINGLE_THREAD_DO
	__shared__ unsigned int block_length;
	while(1){
		BEGIN_SINGLE_THREAD_DO
			block_length = fetch_head(ctx,buffer_id);
		END_SINGLE_THREAD_DO
		move_data_from_recv_buffer(ctx,buffer_id,block_length,data_addr);
		data_addr+=int(block_length/sizeof(int));
		BEGIN_SINGLE_THREAD_DO
			ctx->recv_read_count[buffer_id]+=block_length;
			cur_length+=block_length;
			unsigned long threshold = (unsigned long)(SINGLE_BUFFER_LENGTH*FLOW_CONTROL_RATIO*(ctx->buffer_read_count_record[buffer_id]));
			
			if(cur_length==length){
				printf("recv done!\n");
				break;
			}else if(cur_length>length){
				printf("recv done,but data is a little bit more!\n");
				break;
			};

			if(ctx->recv_read_count[buffer_id]>threshold){//flow control
				ctx->buffer_read_count_record[buffer_id]+=1;
				flow_control_flag=1;
				data[0] = (ctx->recv_read_count[buffer_id])>>32;
				data[1] = (ctx->recv_read_count[buffer_id]);
				data[2] = session_id;
			}
		END_SINGLE_THREAD_DO
		if(flow_control_flag==1){
			write_bypass(ctx->recv_read_count_bypass_reg,data);
			flow_control_flag=0;
		}
	}
}
__global__ void socket_recv(socket_context_t* ctx,int* socket,int * data_addr,size_t length){
	__shared__ int socket_id;
	__shared__ int buffer_id;
	BEGIN_SINGLE_THREAD_DO
		printf("enter recv function!\n");
		if(false==check_socket_validation(ctx,*socket)){
			printf("socket %d does not exists!\n",*socket);
			return;
		}
		socket_id = *socket;
		if(ctx->socket_info[socket_id].valid == 0 ){
			printf("socket %d does not have connections!\n",socket_id);
			return;
		}
		buffer_id = ctx->socket_info[socket_id].buffer_id;
		printf("recv data with socket:%d  buffer_id:%d	data_addr:%lx	length:%ld\n",socket_id,buffer_id,data_addr,length);
	END_SINGLE_THREAD_DO
	_socket_recv(ctx,buffer_id,data_addr,length);
}

__global__ void socket_recv(socket_context_t* ctx,connection_t* connection,int * data_addr,size_t length){
	__shared__ int buffer_id;
	//verify connection
	BEGIN_SINGLE_THREAD_DO
		printf("enter recv function!\n");
		if(connection->valid==0){
			printf("connection is not valid!\n");
			return;
		}
		buffer_id = connection->buffer_id;
	END_SINGLE_THREAD_DO
	_socket_recv(ctx,buffer_id,data_addr,length);
}

__global__ void send_kernel(socket_context_t* ctx,unsigned int *dev_buffer,fpga_registers_t registers){

	BEGIN_SINGLE_THREAD_DO
		// printf("send kernel start!\n");
		ctx->info_buffer				=	dev_buffer;
		ctx->send_buffer				=	dev_buffer+int(2*1024*1024/sizeof(int));
		ctx->socket_num					=	0;
		ctx->mutex						=	0;
		ctx->read_count					=	registers.read_count;
		ctx->con_session_status			=	registers.con_session_status;
		
		ctx->listen_status				=	registers.listen_status;
		ctx->listen_port				=	registers.listen_port;
		ctx->listen_start				=	registers.listen_start;

		ctx->conn_ip					=	registers.conn_ip;
		ctx->conn_port					=	registers.conn_port;
		ctx->conn_start					=	registers.conn_start;

		//bypass
		ctx->send_data_cmd_bypass_reg	=	registers.send_data_cmd_bypass_reg;
		//printf("read_count:%d\n",*(ctx->read_count));
	END_SINGLE_THREAD_DO
		while(1){
			BEGIN_SINGLE_THREAD_DO
				read_info(ctx);
				cu_sleep(8);
				printf("loop send kernel\n");
			END_SINGLE_THREAD_DO
		}
	
}

__global__ void recv_kernel(socket_context_t* ctx,unsigned int *dev_buffer,fpga_registers_t registers){
	__shared__ int read_count;
	__shared__ int index;
	__shared__ int found;
	__shared__ int type;
	BEGIN_SINGLE_THREAD_DO
		// printf("recv kernel start!\n");
		ctx->recv_buffer				=	dev_buffer;
		ctx->recv_read_count_bypass_reg		=	registers.recv_read_count_bypass_reg;
		read_count = *(ctx->recv_read_count);
	END_SINGLE_THREAD_DO
	while(1){
		// BEGIN_SINGLE_THREAD_DO
		// 	found=0;
		// 	while(*(ctx->recv_write_count)<=read_count){//data not read		
		// 		cu_sleep(8);
		// 		printf("loop recv kernel\n");
		// 	}
		// 	res = read_info(ctx,read_count);
		// 	for(int i=0;i<ctx->enroll_list_pointer;i++){
		// 		if(ctx->enroll_list[i].type==0   &&  ctx->enroll_list[i].session_id == res.session_id){
		// 			found=1;
		// 			index=i;
		// 			type=0;
		// 			break;
		// 		}else if(ctx->enroll_list[i].type==1   &&  ctx->enroll_list[i].port == res.dst_port){
		// 			found=1;
		// 			index=i;
		// 			type=1;
		// 			break;
		// 		}
		// 	}
		// 	if(0==found){
		// 		printf("package has no coressponding recver!\n");
		// 		read_count+=1;
		// 		*(ctx->recv_read_count) = read_count;
		// 	}
		// END_SINGLE_THREAD_DO
			// if(1==found){
			// 	int * data_addr = ctx->enroll_list[index].data_addr+int(ctx->enroll_list[index].cur_length/sizeof(int));
			// 	move_data_from_recv_buffer(ctx,res.length,data_addr,res.addr_offset);

			// 	BEGIN_SINGLE_THREAD_DO
			// 		ctx->enroll_list[index].cur_length+=res.length;
			// 		read_count+=1;
			// 		*(ctx->recv_read_count) = read_count;
			// 		if(ctx->enroll_list[index].cur_length == ctx->enroll_list[index].length){
			// 			ctx->enroll_list[index].done = 1;
			// 		}
			// 	END_SINGLE_THREAD_DO
			// }
	}
	
}