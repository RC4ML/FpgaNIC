#include "network_kernel.cuh"
#include "tool/log.hpp"

__device__ void _socket_close(socket_context_t* ctx,int session_id){
	BEGIN_SINGLE_THREAD_DO
		*(ctx->tcp_conn_close_session)		=	(unsigned int)session_id;
		*(ctx->tcp_conn_close_start)		=	1;
		*(ctx->tcp_conn_close_start)		=	0;
	END_SINGLE_THREAD_DO
}

__global__ void socket_close(socket_context_t* ctx,int* socket){
	int session_id,buffer_id,socket_id;
	BEGIN_SINGLE_THREAD_DO
		socket_id = *socket;
		buffer_id = ctx->socket_info[socket_id].buffer_id;
		session_id = ctx->buffer_info[buffer_id].session_id;
		cjprint("close session_id:%d\n",session_id);
	END_SINGLE_THREAD_DO
	_socket_close(ctx,session_id);
}

__global__ void socket_close(socket_context_t* ctx,connection_t* connection){
	int session_id;
	BEGIN_SINGLE_THREAD_DO
		session_id = connection->session_id;
		cjprint("close session_id:%d\n",session_id);
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
				#if SLOW_DEBUG
					cu_sleep(1);
					cjdebug("wait flow ctrl\n");
				#endif
			}//stuck if space not enough

		END_SINGLE_THREAD_DO
		// parallel move data code
		move_data_to_send_buffer(ctx,buffer_id,block_length,data_addr);
		data_addr+=int(block_length/sizeof(int));
		
		//inform code
		__shared__ unsigned int data[16];
		BEGIN_SINGLE_THREAD_DO
			cjdebug("move a block data done! offset:%x buffer:%d \n",ctx->send_buffer_offset[buffer_id],buffer_id);//cjmark
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
		cjinfo("function socket_send called!\n");
		int socket_id = *socket;
		if(false==check_socket_validation(ctx,socket_id)){
			cjerror("socket %d does not exists!\n",socket_id);
			return;
		}
		
		if(ctx->socket_info[socket_id].valid == 0 ){
			cjerror("socket %d does not have connections!\n",socket_id);
			return;
		}

		buffer_id = ctx->socket_info[socket_id].buffer_id;
		cjdebug("send data,socket:%d  buffer_id:%d data_addr:%lx length:%ld\n",(*socket),buffer_id,data_addr,length);
	END_SINGLE_THREAD_DO
	_socket_send(ctx,buffer_id,data_addr,length);
}


__global__ void socket_send(socket_context_t* ctx,connection_t* connection,int * data_addr,size_t length){
	__shared__ int buffer_id;
	//verify connection
	BEGIN_SINGLE_THREAD_DO
		cjdebug("function socket_send connection type called!\n");
		if(connection->valid==0){
			cjerror("connection is not valid!\n");
			return;
		}
		buffer_id = connection->buffer_id;
	END_SINGLE_THREAD_DO
	_socket_send(ctx,buffer_id,data_addr,length);
}

__device__ void _socket_recv(socket_context_t* ctx,int buffer_id,int * data_addr,size_t length,size_t* swap_data){
	int index = blockIdx.x*blockDim.x+threadIdx.x;
	int total_threads = blockDim.x*gridDim.x-32;

	__shared__ int socket_id;
	__shared__ int session_id;
	__shared__ size_t cur_length;
	__shared__ size_t flow_control_flag;
	__shared__ unsigned int data[16];
	__shared__ unsigned long threshold,portion;
	__shared__ unsigned int block_length;
	clock_t s,e;
	int clock_flag;
	BEGIN_SINGLE_THREAD_DO
		clock_flag=1;
		cur_length=0;
		flow_control_flag=0;
		session_id = ctx->buffer_info[buffer_id].session_id;
		threshold = (unsigned long)(SINGLE_BUFFER_LENGTH*FLOW_CONTROL_RATIO*(ctx->buffer_read_count_record[buffer_id]));
		portion =  (unsigned long)(SINGLE_BUFFER_LENGTH*FLOW_CONTROL_RATIO);
	END_SINGLE_THREAD_DO

	clock_t s1[16],e1[16];
	clock_t s2[16],e2[16];
	clock_t s3[16],e3[16];
	int index1=0,index2=0,index3=0;
	
	while(1){
		BEGIN_SINGLE_THREAD_DO
			s1[index1]=clock64();
			block_length = fetch_head(ctx,buffer_id);
			e1[index1]=clock64();
			if(index1<15){
				index1++;
			}
			
			if(clock_flag==1){
				clock_flag=0;
				s=clock64();
			}
		END_SINGLE_THREAD_DO

		s2[index2]=clock64();
		if(index<992)
		{//move code
			int op_num = int(block_length/sizeof(int));
			int iter_num = int(op_num/total_threads);
			int rest_num = op_num%total_threads;
			int addr_base_offset = SINGLE_BUFFER_LENGTH*buffer_id+ctx->recv_buffer_offset[buffer_id];
			addr_base_offset = int(addr_base_offset/sizeof(int));
			for(int i=0;i<iter_num;i++){
				data_addr[total_threads*i+index]	=	ctx->recv_buffer[addr_base_offset+total_threads*i+index];
				ctx->recv_buffer[addr_base_offset+total_threads*i+index]	=	0;
			}
			if(index<rest_num){//process tail data
				data_addr[total_threads*iter_num+index]	=	ctx->recv_buffer[addr_base_offset+total_threads*iter_num+index];
				ctx->recv_buffer[addr_base_offset+total_threads*iter_num+index]	=	0;
			}
			data_addr+=int(block_length/sizeof(int));
		}		
		e2[index2]=clock64();
		if(index2<15){
			index2++;
		}

		s3[index3]=clock64();
		BEGIN_SINGLE_THREAD_DO
			
			ctx->recv_read_count[buffer_id]+=block_length;
			ctx->recv_buffer_offset[buffer_id]+=(block_length+64);
			if(ctx->recv_buffer_offset[buffer_id]>(SINGLE_BUFFER_LENGTH-MAX_PACKAGE_LENGTH)){
				ctx->recv_buffer_offset[buffer_id] = 0;
			}
			cur_length+=block_length;
			
			
			if(ctx->recv_read_count[buffer_id]>threshold){//flow control
				ctx->buffer_read_count_record[buffer_id] += 1;
				threshold += portion;
				flow_control_flag=1;
				//ptmark
				//cjdebug("send flow ctl buffer:%d %lx\n",buffer_id,ctx->recv_read_count[buffer_id]);//cjmark
				data[0] = (ctx->recv_read_count[buffer_id]);//todo seq
				data[1] = (ctx->recv_read_count[buffer_id])>>32;
				data[2] = session_id;
			}
		END_SINGLE_THREAD_DO
		if(flow_control_flag==1){
			write_bypass(ctx->recv_read_count_bypass_reg,data);
			flow_control_flag=0;
		}
		if(cur_length>=length){
			break;
		}
		e3[index3]=clock64();
		if(index3<15){
			index3++;
		}
	}
	BEGIN_SINGLE_THREAD_DO
		e = clock64();
		float time = (e-s)/1.41/1e9;//time fre attention  rtx8000 1.77  a100 1.41
		if(cur_length==length){
			cjprint("recv done!\n");
		}else if(cur_length>length){
			cjerror("recv done,but data is a little bit more!\n");
			
		};
		cjprint("time=%f speed=%f\n",time,length/time/1024/1024/1024);
		for(int i=0;i<15;i++){
			printf("%ld %ld %ld\n",e1[i]-s1[i],e2[i]-s2[i],e3[i]-s3[i]);
			swap_data[i] = e1[i]-s1[i];
			swap_data[i+50] = e2[i]-s2[i];
			swap_data[i+100] = e3[i]-s3[i];
		}
	END_SINGLE_THREAD_DO
}
__global__ void socket_recv(socket_context_t* ctx,int* socket,int * data_addr,size_t length,size_t* swap_data){
	__shared__ int socket_id;
	__shared__ int buffer_id;
	BEGIN_SINGLE_THREAD_DO
		cjdebug("enter recv function!\n");
		if(false==check_socket_validation(ctx,*socket)){
			cjerror("socket %d does not exists!\n",*socket);
			return;
		}
		socket_id = *socket;
		if(ctx->socket_info[socket_id].valid == 0 ){
			cjerror("socket %d does not have connections!\n",socket_id);
			return;
		}
		buffer_id = ctx->socket_info[socket_id].buffer_id;
		cjdebug("recv data with socket:%d  buffer_id:%d	data_addr:%lx	length:%ld\n",socket_id,buffer_id,data_addr,length);
	END_SINGLE_THREAD_DO
	_socket_recv(ctx,buffer_id,data_addr,length,swap_data);
}

__global__ void socket_recv(socket_context_t* ctx,connection_t* connection,int * data_addr,size_t length,size_t* swap_data){
	__shared__ int buffer_id;
	//verify connection
	BEGIN_SINGLE_THREAD_DO
		cjinfo("enter recv function!\n");
		if(connection->valid==0){
			cjerror("connection is not valid!\n");
			return;
		}
		buffer_id = connection->buffer_id;
	END_SINGLE_THREAD_DO
	_socket_recv(ctx,buffer_id,data_addr,length,swap_data);
}

__global__ void send_kernel(socket_context_t* ctx,unsigned int *dev_buffer,fpga_registers_t registers){

	BEGIN_SINGLE_THREAD_DO
		cjinfo("send kernel start!\n");
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
		
	END_SINGLE_THREAD_DO
		while(1){
			BEGIN_SINGLE_THREAD_DO
				read_info(ctx);
				#if SLOW_DEBUG
					cu_sleep(1);
					cjdebug("kernel pooling\n");
				#endif
			END_SINGLE_THREAD_DO
		}
	
}
