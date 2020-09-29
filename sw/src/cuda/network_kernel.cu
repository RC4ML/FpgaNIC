#include "network_kernel.cuh"


__global__ void socket_send(socket_context_t* ctx,int* socket,int * data_addr,size_t length,sock_addr_t dst_addr){
	__shared__ int session_id;
	//verify socket
	BEGIN_SINGLE_THREAD_DO
		printf("function socket_send called!\n");
		int socket_id = *socket;
		if(false==check_socket_validation(ctx,socket_id)){
				printf("socket %d does not exists!\n",socket_id);
			return;
		}
		session_id	=	get_session(ctx,*socket,dst_addr);
		if(session_id	==	-1){
			printf("send data failed!\n");
			return;//does not have session and connect failed
		}
		printf("send data with socket:%d	session_id:%d	data_addr:%lx	length:%ld\n",(*socket),session_id,data_addr,length);
	END_SINGLE_THREAD_DO

	__shared__ int write_count;
	__shared__ unsigned int addr_offset;
	__shared__ unsigned int block_length;
	__shared__ int send_info_tbl_index;
	
	for(size_t i=0;i<length;i+=MAX_BLOCK_SIZE){
		BEGIN_SINGLE_THREAD_DO
			block_length = min((unsigned long)MAX_BLOCK_SIZE,length-i);
			lock(&(ctx->mutex));
			write_count = *(ctx->send_write_count);
			while(write_count-*(ctx->send_read_count)>=(MAX_CMD-1)){//todo
			}
			
			if(ctx->current_send_addr_offset>=MAX_BLOCK_SIZE*(MAX_CMD-1)){
				ctx->current_send_addr_offset=0;
			}
			addr_offset				=	ctx->current_send_addr_offset;
			send_info_tbl_index		=	get_info_tbl_index(ctx);
			ctx->current_send_addr_offset	=	ctx->current_send_addr_offset+block_length;
			unlock(&(ctx->mutex));
		END_SINGLE_THREAD_DO
		
		// parallel move data code
		move_data(ctx,block_length,data_addr,addr_offset);
		data_addr+=int(block_length/sizeof(int));
		
		//inform code
		BEGIN_SINGLE_THREAD_DO
			printf("move data done! offset:%d  \n",addr_offset);
			ctx->send_info_tbl[send_info_tbl_index].addr_offset	=	addr_offset;
			ctx->send_info_tbl[send_info_tbl_index].length		=	block_length;
			ctx->send_info_tbl[send_info_tbl_index].session_id	=	session_id;
			ctx->send_info_tbl[send_info_tbl_index].valid		=	1;
		END_SINGLE_THREAD_DO

	}
}

__global__ void socket_recv(socket_context_t* ctx,int* socket,int * data_addr,size_t length){
	BEGIN_SINGLE_THREAD_DO
		printf("enter recv function!\n");
		if(false==check_socket_validation(ctx,*socket)){
			printf("socket %d does not exists!\n",*socket);
			return;
		}
		int enroll_num = enroll(ctx,*socket,data_addr,length);
		if(enroll_num==-1){
			printf("enroll failed!\n");
			return;
		}else{
			while(ctx->enroll_list[enroll_num].done != 1){
				cu_sleep(8);
				printf("loop recving\n");
			}
		}
		//waiting done
	END_SINGLE_THREAD_DO
}

__global__ void send_kernel(socket_context_t* ctx,unsigned int *dev_buffer,fpga_registers_t registers){

	BEGIN_SINGLE_THREAD_DO
		// printf("send kernel start!\n");
		ctx->send_buffer				=	dev_buffer;
		ctx->socket_num					=	0;
		ctx->mutex						=	0;
		ctx->current_send_addr_offset	=	0;
		ctx->send_info_tbl_index		=	0;
		ctx->send_info_tbl_pointer		=	0;
		ctx->read_count					=	registers.read_count;
		ctx->con_session_status			=	registers.con_session_status;
		ctx->send_write_count			=	registers.send_write_count;
		ctx->send_read_count			=	registers.send_read_count;
		
		ctx->listen_status				=	registers.listen_status;
		ctx->listen_port				=	registers.listen_port;
		ctx->listen_start				=	registers.listen_start;

		ctx->conn_ip					=	registers.conn_ip;
		ctx->conn_port					=	registers.conn_port;
		ctx->conn_start					=	registers.conn_start;

		ctx->send_info_session_id		=	registers.send_info_session_id;
		ctx->send_info_addr_offset		=	registers.send_info_addr_offset;
		ctx->send_info_length			=	registers.send_info_length;
		ctx->send_info_start			=	registers.send_info_start;
		//printf("read_count:%d\n",*(ctx->read_count));

		while(1){
			int index	=	(ctx->send_info_tbl_pointer);
			if(	0	==	ctx->send_info_tbl[index].valid){
				
			}
			else{
				ctx->send_info_tbl[index].valid		=	0;

				*(ctx->send_info_session_id)		=	ctx->send_info_tbl[index].session_id;
				*(ctx->send_info_addr_offset)		=	ctx->send_info_tbl[index].addr_offset;
				*(ctx->send_info_length)			=	ctx->send_info_tbl[index].length;
				*(ctx->send_info_start)				=	0;
				*(ctx->send_info_start)				=	1;

				*(ctx->send_write_count)	=	*(ctx->send_write_count)+1;
				ctx->send_info_tbl_pointer	=	ctx->send_info_tbl_pointer	+	1;
				printf("send kernel find sth to send!  session_id:%d  addr_offset:%d   length:%d\n",*(ctx->send_info_session_id),*(ctx->send_info_addr_offset),*(ctx->send_info_length));
			}
			
			cu_sleep(8);
			printf("loop send kernel\n");
		}
	END_SINGLE_THREAD_DO
}

__global__ void recv_kernel(socket_context_t* ctx,unsigned int *dev_buffer,fpga_registers_t registers){
	__shared__ int read_count;
	__shared__ int index;
	__shared__ int found;
	__shared__ int type;
	BEGIN_SINGLE_THREAD_DO
		// printf("recv kernel start!\n");
		ctx->info_buffer				=	dev_buffer;
		ctx->recv_buffer				=	dev_buffer+int(2*1024*1024/sizeof(int));
		ctx->recv_read_count			=	registers.recv_read_count;
		ctx->recv_write_count			=	registers.recv_write_count;
		ctx->enroll_list_pointer		=	0;
		read_count = *(ctx->recv_read_count);
	END_SINGLE_THREAD_DO
	__shared__ recv_info_t res;
	while(1){
		BEGIN_SINGLE_THREAD_DO
			found=0;
			while(*(ctx->recv_write_count)<=read_count){//data not read		
				cu_sleep(8);
				printf("loop recv kernel\n");
			}
			res = read_info(ctx,read_count);
			for(int i=0;i<ctx->enroll_list_pointer;i++){
				if(ctx->enroll_list[i].type==0   &&  ctx->enroll_list[i].session_id == res.session_id){
					found=1;
					index=i;
					type=0;
					break;
				}else if(ctx->enroll_list[i].type==1   &&  ctx->enroll_list[i].port == res.dst_port){
					found=1;
					index=i;
					type=1;
					break;
				}
			}
			if(0==found){
				printf("package has no coressponding recver!\n");
				read_count+=1;
				*(ctx->recv_read_count) = read_count;
			}
		END_SINGLE_THREAD_DO
			if(1==found){
				int * data_addr = ctx->enroll_list[index].data_addr+int(ctx->enroll_list[index].cur_length/sizeof(int));
				move_data_recv(ctx,res.length,data_addr,res.addr_offset);

				BEGIN_SINGLE_THREAD_DO
					ctx->enroll_list[index].cur_length+=res.length;
					read_count+=1;
					*(ctx->recv_read_count) = read_count;
					if(ctx->enroll_list[index].cur_length == ctx->enroll_list[index].length){
						ctx->enroll_list[index].done = 1;
					}
				END_SINGLE_THREAD_DO
			}
	}
	
}