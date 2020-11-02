#include "network_kernel.cuh"

__device__ void _socket_close(socket_context_t* ctx,int session_id){
	BEGIN_SINGLE_THREAD_DO
		*(ctx->tcp_conn_close_session)		=	(unsigned int)session_id;
		*(ctx->tcp_conn_close_start)		=	1;
		*(ctx->tcp_conn_close_start)		=	0;
	END_SINGLE_THREAD_DO
}

__global__ void socket_close(socket_context_t* ctx,connection_t* connection){
	int session_id;
	BEGIN_SINGLE_THREAD_DO
		session_id = connection->session_id;
		printf("close session_id:%d\n",session_id);
	END_SINGLE_THREAD_DO
	_socket_close(ctx,session_id);
}
__device__ void _socket_send(socket_context_t* ctx,int buffer_id,int * data_addr,size_t length){
	__shared__ unsigned int block_length;
	__shared__ int session_id;

	BEGIN_SINGLE_THREAD_DO
		session_id = ctx->buffer_info[buffer_id].session_id;
	END_SINGLE_THREAD_DO
	for(size_t i=0;i<length;i+=MAX_BLOCK_SIZE){
		BEGIN_SINGLE_THREAD_DO
			block_length = min((unsigned long)MAX_BLOCK_SIZE,length-i);
			while(ctx->send_write_count[buffer_id] > ctx->send_read_count[buffer_id]+(SINGLE_BUFFER_LENGTH-MAX_PACKAGE_LENGTH-OVERHEAD)){
				cu_sleep(1);
				printf("wait flow ctrl\n");
			}//stuck if space not enough
			{
				printf("before wr:");
				for(int i =0;i<16;i++){
					printf("%d ",ctx->send_buffer[i]);
				}
				printf("\n");
			}
		END_SINGLE_THREAD_DO
		// parallel move data code
		move_data_to_send_buffer(ctx,buffer_id,block_length,data_addr);
		data_addr+=int(block_length/sizeof(int));
		
		//inform code
		__shared__ unsigned int data[16];
		BEGIN_SINGLE_THREAD_DO
			printf("move a block data done! offset:%x buffer:%d \n",ctx->send_buffer_offset[buffer_id],buffer_id);
			{//ptx test
				printf("ptx:");
				for(int i=0;i<16;i++){
					volatile uint64_t  addr = (uint64_t)((ctx->send_buffer)+i);
					volatile unsigned int y=0;
					asm volatile(
						"ld.u32.cv %0,[%1];\n\t"//没有就是7 有的话16   ca=16  cg=16   cs=16 lu=16  cv=16
						:"=r"(y):"l"(addr):"memory"
					);
					printf("%u ",y);
				}
			}
			printf("\n");
			printf("normal:");
			for(int i =0;i<16;i++){
				printf("%x ",ctx->send_buffer[i]);
			}
			printf("\n");
			data[0] = block_length;//todo seq done
			data[1] = ctx->send_buffer_offset[buffer_id];
			data[2] = session_id;
			data[3] = buffer_id;

			ctx->send_write_count[buffer_id]	+=	block_length;
			ctx->send_buffer_offset[buffer_id]	+=	block_length;
			if(ctx->send_buffer_offset[buffer_id]>(SINGLE_BUFFER_LENGTH-MAX_PACKAGE_LENGTH)){
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
		// printf("function socket_send called!\n");
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
		// printf("send data,socket:%d  buffer_id:%d data_addr:%lx length:%ld\n",(*socket),buffer_id,data_addr,length);
	END_SINGLE_THREAD_DO
	_socket_send(ctx,buffer_id,data_addr,length);
}


__global__ void socket_send(socket_context_t* ctx,connection_t* connection,int * data_addr,size_t length){
	__shared__ int buffer_id;
	//verify connection
	BEGIN_SINGLE_THREAD_DO
		// printf("function socket_send connection type called!\n");
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
			ctx->recv_buffer_offset[buffer_id]+=(block_length+64);
			if(ctx->recv_buffer_offset[buffer_id]>(SINGLE_BUFFER_LENGTH-MAX_PACKAGE_LENGTH)){
				ctx->recv_buffer_offset[buffer_id] = 0;
			}
			cur_length+=block_length;
			unsigned long threshold = (unsigned long)(SINGLE_BUFFER_LENGTH*FLOW_CONTROL_RATIO*(ctx->buffer_read_count_record[buffer_id]));
			
			if(ctx->recv_read_count[buffer_id]>threshold){//flow control
				ctx->buffer_read_count_record[buffer_id] = ctx->buffer_read_count_record[buffer_id]+1;
				flow_control_flag=1;
				printf("---send flow ctl buffer:%d %lx\n",buffer_id,ctx->recv_read_count[buffer_id]);
				data[0] = (ctx->recv_read_count[buffer_id]);//todo seq
				data[1] = (ctx->recv_read_count[buffer_id])>>32;
				data[2] = session_id;
			}
		END_SINGLE_THREAD_DO
		if(flow_control_flag==1){
			write_bypass(ctx->recv_read_count_bypass_reg,data);
			flow_control_flag=0;
		}
		if(cur_length==length){
			BEGIN_SINGLE_THREAD_DO
				printf("recv done!\n");
			END_SINGLE_THREAD_DO
			break;
		}else if(cur_length>length){
			BEGIN_SINGLE_THREAD_DO
				printf("recv done,but data is a little bit more!\n");
			END_SINGLE_THREAD_DO
			break;
		};
	}
}
__global__ void socket_recv(socket_context_t* ctx,int* socket,int * data_addr,size_t length){
	__shared__ int socket_id;
	__shared__ int buffer_id;
	BEGIN_SINGLE_THREAD_DO
		// printf("enter recv function!\n");
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
		// printf("enter recv function!\n");
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
		ctx->info_buffer				=	dev_buffer+int(100*1024*1024/sizeof(int));
		ctx->send_buffer				=	dev_buffer;
		ctx->socket_num					=	0;
		ctx->mutex						=	0;
		ctx->con_session_status			=	registers.con_session_status;
		
		ctx->listen_status				=	registers.listen_status;
		ctx->listen_port				=	registers.listen_port;
		ctx->listen_start				=	registers.listen_start;

		ctx->conn_ip					=	registers.conn_ip;
		ctx->conn_port					=	registers.conn_port;
		ctx->conn_buffer_id				=	registers.conn_buffer_id;
		ctx->conn_start					=	registers.conn_start;

		ctx->conn_response				=	registers.conn_response;
		ctx->conn_res_start				=	registers.conn_res_start;
		ctx->conn_re_session_id			=	registers.conn_re_session_id;

		ctx->tcp_conn_close_session		=	registers.tcp_conn_close_session;
		ctx->tcp_conn_close_start		=	registers.tcp_conn_close_start;


		//bypass
		ctx->send_data_cmd_bypass_reg	=	registers.send_data_cmd_bypass_reg;
		ctx->recv_read_count_bypass_reg	=	registers.recv_read_count_bypass_reg;
		ctx->recv_buffer				=	dev_buffer+int(102*1024*1024/sizeof(int));
		
		//printf("read_count:%d\n",*(ctx->read_count));
	END_SINGLE_THREAD_DO
		while(1){
			BEGIN_SINGLE_THREAD_DO
				read_info(ctx);
				cu_sleep(1);
				printf("kernel pooling\n");
			END_SINGLE_THREAD_DO
		}
	
}
