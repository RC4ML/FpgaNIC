#include "network_kernel.cuh"
#include "tool/log.hpp"
#include "main.h"

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
	int block_index = blockIdx.x;
	int block_num = gridDim.x;
	int index = blockIdx.x*blockDim.x+threadIdx.x;
	int index_in_blk = threadIdx.x;
	int total_threads = blockDim.x*gridDim.x;
	int block_threads = blockDim.x;
	int cnt=0;

	__shared__ unsigned int data[16];

	BEGIN_BLOCK_ZERO_DO
		cjdebug("buffer %d send\n",buffer_id);
	END_BLOCK_ZERO_DO
	for(int i=0;i<ctx->pre_pkg_cnt;i+=block_num){
		int p = i+block_index;
		int block_length	=	ctx->pre_length[p];
		int buffer_offset	=	ctx->pre_buffer_offset_bytes[p]/sizeof(int);
		size_t data_offset	=	ctx->pre_data_offset[p];

		BEGIN_BLOCK_ZERO_DO
			size_t cycle_count=clock64();
			while(ctx->send_write_count[buffer_id] > ctx->send_read_count[buffer_id]+(SINGLE_BUFFER_LENGTH-MAX_PACKAGE_LENGTH-OVERHEAD)){
				if( clock64()-cycle_count > size_t(15000000000L)){//1s = 3000000000
					cjdebug("check flow ctrl: %ld %ld\n",ctx->send_write_count[buffer_id],ctx->send_read_count[buffer_id]);
					// while(1);
					cu_sleep(5);
				};
			}
		END_BLOCK_ZERO_DO

		int op_num = int(block_length/sizeof(int));
		int iter_num = int(op_num/block_threads);
		for(int j=0;j<iter_num;j++){
			ctx->send_buffer[buffer_offset+block_threads*j+index_in_blk]	=	data_addr[data_offset+block_threads*j+index_in_blk];
		}

		BEGIN_BLOCK_ZERO_DO
			// printf("move a block data done! buffer:%d offset:%x \n",buffer_id,data_offset);//cjmark
			size_t cycle_count=clock64();
			while(ctx->mutex_sender!=p){
				if( clock64()-cycle_count > size_t(15000000000L)){//1s = 3000000000
					cjdebug("wait mutex_sender,p:%d mutex:%d\n",p,ctx->mutex_sender);
					while(1);
				};
			}
			// cjdebug("mutex,p:%d mutex:%d\n",p,ctx->mutex_sender);
			data[0] = block_length;//todo seq done
			data[1] = ctx->pre_buffer_offset_bytes[p];
			data[2] = ctx->buffer_info[buffer_id].session_id;
			data[3] = buffer_id;

			ctx->send_write_count[buffer_id]	+=	block_length;
		END_BLOCK_ZERO_DO
		write_bypass(ctx->send_data_cmd_bypass_reg,data);
		BEGIN_BLOCK_ZERO_DO
			ctx->mutex_sender=p+1;
		END_BLOCK_ZERO_DO
	}


	// 	// BEGIN_SINGLE_THREAD_DO//check cmd fifo overflow
	// 	// 	cnt++;
	// 	// 	if(cnt%32==0 && ctx->send_cmd_fifo_count[0]>500){
	// 	// 		printf("send_cmd_fifo_count %d\n",ctx->send_cmd_fifo_count[0]);
	// 	// 	}
	// 	// END_SINGLE_THREAD_DO
	// }
	BEGIN_SINGLE_THREAD_DO
		cjdebug("send data done,length=%fM\n",1.0*length/1024/1024);
	END_SINGLE_THREAD_DO

}

__device__ void _socket_send_pre(socket_context_t* ctx,int buffer_id,size_t length){
	int addr_base_offset_bytes = SINGLE_BUFFER_LENGTH*buffer_id+ctx->send_buffer_offset[buffer_id];
	int cnt=0;
	size_t data_offset_bytes=0;
	ctx->pre_pkg_cnt=0;
	ctx->mutex_sender=0;
	for(size_t i=0;i<length;i+=ctx->max_block_size){
		int block_length = min((unsigned long)(ctx->max_block_size),length-i);
		ctx->pre_data_offset[cnt]			=	data_offset_bytes/sizeof(int);
		ctx->pre_buffer_offset_bytes[cnt]	=	addr_base_offset_bytes;
		ctx->pre_length[cnt]				=	block_length;
		ctx->pre_pkg_cnt					++;
		data_offset_bytes					+=	block_length;
		addr_base_offset_bytes				+=	block_length;
		if(addr_base_offset_bytes>(SINGLE_BUFFER_LENGTH*buffer_id+SINGLE_BUFFER_LENGTH-MAX_PACKAGE_LENGTH)){
			addr_base_offset_bytes=0;
		}
		ctx->send_buffer_offset[buffer_id]	=	addr_base_offset_bytes;//useless
		cnt++;
	}
}

__global__ void socket_send_pre(socket_context_t* ctx,int* socket,size_t length, size_t max_block_size){
	int buffer_id;
	BEGIN_SINGLE_THREAD_DO
		ctx->max_block_size = max_block_size;
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
	END_SINGLE_THREAD_DO
	_socket_send_pre(ctx,buffer_id,length);
}
__global__ void socket_send(socket_context_t* ctx,int* socket,int * data_addr,size_t length){
	__shared__ int buffer_id;
	//verify socket
	BEGIN_BLOCK_ZERO_DO
		cjinfo("function socket_send called!\n");
		size_t t = clock64();
		//cjprint("send t= %ld\n",t);
		// int socket_id = *socket;
		int socket_id = 0;//todo, unknow bug
		if(false==check_socket_validation(ctx,socket_id)){
			cjerror("socket_send: socket %d does not exists!\n",socket_id);
			return;
		}
		
		if(ctx->socket_info[socket_id].valid == 0 ){
			cjerror("socket %d does not have connections!\n",socket_id);
			return;
		}

		buffer_id = ctx->socket_info[socket_id].buffer_id;
		cjdebug("send data,socket:%d  buffer_id:%d data_addr:%lx length:%ld\n",socket_id,buffer_id,data_addr,length);
	END_BLOCK_ZERO_DO
	_socket_send(ctx,buffer_id,data_addr,length);
}



__global__ void socket_send_offload_control(socket_context_t* ctx,int* socket,int * data_addr,size_t length,int cur_idx, unsigned int* ctrl_data){
	__shared__ int buffer_id;
	//verify socket
	BEGIN_BLOCK_ZERO_DO
		cjinfo("function socket_send called!\n");
		size_t t = clock64();
		//cjprint("send t= %ld\n",t);
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
	END_BLOCK_ZERO_DO
	
	int block_index = blockIdx.x;
	int block_num = gridDim.x;
	int index = blockIdx.x*blockDim.x+threadIdx.x;
	int index_in_blk = threadIdx.x;
	int total_threads = blockDim.x*gridDim.x;
	int block_threads = blockDim.x;

	// BEGIN_BLOCK_ZERO_DO
	// 	cjdebug("buffer %d send\n",buffer_id);
	// END_BLOCK_ZERO_DO

	// for(int i=0;i<ctx->pre_pkg_cnt;i+=block_num){
		int p = cur_idx+block_index;
		int block_length	=	ctx->pre_length[p];
		int buffer_offset	=	ctx->pre_buffer_offset_bytes[p]/sizeof(int);
		size_t data_offset	=	ctx->pre_data_offset[p];

		BEGIN_BLOCK_ZERO_DO
			size_t cycle_count=clock64();
			while(ctx->send_write_count[buffer_id] > ctx->send_read_count[buffer_id]+(SINGLE_BUFFER_LENGTH-MAX_PACKAGE_LENGTH-OVERHEAD)){
				if( clock64()-cycle_count > size_t(15000000000L)){//1s = 3000000000
					cjdebug("check flow ctrl: %ld %ld\n",ctx->send_write_count[buffer_id],ctx->send_read_count[buffer_id]);
					// while(1);
					cu_sleep(5);
				};
			}
		END_BLOCK_ZERO_DO

		int op_num = int(block_length/sizeof(int));
		int iter_num = int(op_num/block_threads);
		for(int j=0;j<iter_num;j++){
			ctx->send_buffer[buffer_offset+block_threads*j+index_in_blk]	=	data_addr[data_offset+block_threads*j+index_in_blk];
		}

		BEGIN_BLOCK_ZERO_DO
			size_t cycle_count=clock64();
			while(ctx->mutex_sender!=p){
				if( clock64()-cycle_count > size_t(15000000000L)){//1s = 3000000000
					cjdebug("wait mutex_sender,p:%d mutex:%d\n",p,ctx->mutex_sender);
					while(1);
				};
			}
			// cjdebug("mutex,p:%d mutex:%d\n",p,ctx->mutex_sender);
			// data[0] = block_length;//todo seq done
			// data[1] = ctx->pre_buffer_offset_bytes[p];
			// data[2] = ctx->buffer_info[buffer_id].session_id;
			// data[3] = buffer_id;

			ctrl_data[block_index*16+0] = block_length;
			ctrl_data[block_index*16+1] = ctx->pre_buffer_offset_bytes[p];
			ctrl_data[block_index*16+2] = ctx->buffer_info[buffer_id].session_id;
			ctrl_data[block_index*16+3] = buffer_id;

			ctx->send_write_count[buffer_id]	+=	block_length;
			ctx->mutex_sender=p+1;
		END_BLOCK_ZERO_DO
		// write_bypass(ctx->send_data_cmd_bypass_reg,data);

	// }
	// BEGIN_SINGLE_THREAD_DO
	// 	cjdebug("send data done,length=%fM\n",1.0*length/1024/1024);
	// END_SINGLE_THREAD_DO
}

__global__ void socket_send(socket_context_t* ctx,connection_t* connection,int * data_addr,size_t length){
	__shared__ int buffer_id;
	//verify connection
	BEGIN_SINGLE_THREAD_DO
		cjdebug("function socket_send connection type called!\n");
		size_t t = clock64();
		//cjprint("send t= %ld\n",t);
		if(connection->valid==0){
			cjerror("connection is not valid!\n");
			return;
		}
		buffer_id = connection->buffer_id;
	END_SINGLE_THREAD_DO
	_socket_send(ctx,buffer_id,data_addr,length);
}

__device__ void _socket_recv_data(socket_context_t* ctx,int buffer_id,int * data_addr,size_t length){
	int index = blockIdx.x*blockDim.x+threadIdx.x;
	int total_threads = blockDim.x*gridDim.x;
	int block_id = blockIdx.x;
	__shared__ size_t cur_length;
	__shared__ int cur_index,cur_index_mod;
	__shared__ int block_length,addr_base_offset;
	__shared__ int op_num,iter_num,rest_num;
	__shared__ clock_t s,e;
	int flag = 1;

	__shared__ clock_t s1[16],e1[16];
	__shared__ clock_t s2[16],e2[16];
	__shared__ clock_t s3[16],e3[16];
	int index1=0,index2=0,index3=0;
	
	BEGIN_BLOCK_ZERO_DO
		cur_length=0;
		cur_index=0;
		cur_index_mod=0;
		block_length=0;
		addr_base_offset=0;
		op_num=0;
		iter_num=0;
		rest_num=0;
	END_BLOCK_ZERO_DO

	BEGIN_BLOCK_ZERO_DO
		// cjdebug("enter _socket_recv_data function!\n");
		// size_t t=clock64();
		// cjprint("_socket_recv_data start t= %ld  blockid=%d\n",t,block_id);
	END_BLOCK_ZERO_DO
	while(1){
		BEGIN_BLOCK_ZERO_DO
			s1[index1]=clock64();
			cur_index = ctx->recv_fifo_rd[buffer_id][block_id];
			while(cur_index==ctx->recv_fifo_wr[buffer_id]){
				// cjdebug("waiting wr, block_id:%d buffer_id:%d rd:%d wr:%d cur_length:%ld\n",block_id,buffer_id,cur_index,ctx->recv_fifo_wr[buffer_id],cur_length);
				// cu_sleep(1);
			}
			e1[index1]=clock64();
			if(index1<15){
				index1++;
			}
			if(flag==1){
				s=clock64();
				// cjprint("_socket_recv_data see 1st pkg t= %ld\n",s);
				flag=0;
			}
			//printf("block_id:%d cur_index:%d  wr:%d\n",block_id,cur_index,ctx->recv_fifo_wr[buffer_id]);
			s2[index2]=clock64();
			cur_index_mod = cur_index%16;
			block_length=ctx->recv_fifo_length[buffer_id][cur_index_mod];
			addr_base_offset=ctx->recv_fifo_addr_offset[buffer_id][cur_index_mod];
			ctx->recv_fifo_rd[buffer_id][block_id] = cur_index+1;
			op_num = int(block_length/sizeof(int));
			iter_num = int(op_num/total_threads);
			rest_num = op_num%total_threads;
			cur_length+=block_length;
			e2[index2]=clock64();
			if(index2<15){
				index2++;
			}
		END_BLOCK_ZERO_DO
		s3[index3]=clock64();
		for(int i=0;i<iter_num;i++){
			//hll(ctx->recv_buffer+(addr_base_offset+total_threads*i+index),ctx->hll_mem);
			data_addr[total_threads*i+index]	=	ctx->recv_buffer[addr_base_offset+total_threads*i+index];
			//ctx->recv_buffer[addr_base_offset+total_threads*i+index]	=	0;
		}
		if(index<rest_num){//process tail data //todo
			data_addr[total_threads*iter_num+index]	=	ctx->recv_buffer[addr_base_offset+total_threads*iter_num+index];
			//ctx->recv_buffer[addr_base_offset+total_threads*iter_num+index]	=	0;
		}
		data_addr+=int(block_length/sizeof(int));
		e3[index3]=clock64();
		if(index3<15){
			index3++;
		}
		if(cur_length>=length){
			e=clock64();
			break;
		}
	}
	BEGIN_BLOCK_ZERO_DO
		if(cur_length==length){
			size_t t= clock64();
			//cjprint("recv done t= %ld blockid=%d\n",t,block_id);
			cjdebug("data recv done! %ld speed: %f\n",e-s,1.0*length/1024/1024/1024/((e-s)/1.41/1e9));
		}else if(cur_length>length){
			cjerror("data recv done,but data is a little bit more!\n");
		};
	END_BLOCK_ZERO_DO

	//	hll
	// BEGIN_SINGLE_THREAD_DO
	// 	for(int i=0;i<32;i++){
	// 		printf("%d\n",ctx->hll_mem[i]);
	// 	}
	// 	double s=0;
	// 	for(int i=0;i<65536;i++){
	// 		s += 1.0/(exp2((double)(ctx->hll_mem[i])));
	// 	}
	// 	double dv = 0.7213*65536*65536/(s);
	// 	printf("dv:%f\n",dv);
	// END_SINGLE_THREAD_DO
}

__global__ void socket_recv_ctrl(socket_context_t* ctx,int* socket,int * data_addr,size_t length){
	__shared__ int socket_id;
	__shared__ int buffer_id;
	BEGIN_SINGLE_THREAD_DO
		// cjdebug("enter socket_recv_ctrl function!\n");
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
		cjdebug("recv ctrl with socket:%d  buffer_id:%d	data_addr:%lx	length:%ld\n",socket_id,buffer_id,data_addr,length);
	END_SINGLE_THREAD_DO
	_socket_recv_ctrl(ctx,buffer_id,data_addr,length);
}

__global__ void socket_recv_ctrl(socket_context_t* ctx,connection_t* connection,int * data_addr,size_t length){
	__shared__ int buffer_id;
	BEGIN_SINGLE_THREAD_DO
		if(connection->valid==0){
			cjerror("connection is not valid!\n");
			return;
		}
	END_SINGLE_THREAD_DO
	_socket_recv_ctrl(ctx,connection->buffer_id,data_addr,length);
}

__device__ void _socket_recv_ctrl(socket_context_t* ctx,int buffer_id,int * data_addr,size_t length){
	int index = blockIdx.x*blockDim.x+threadIdx.x;
	__shared__ int session_id;
	__shared__ int block_length;
	__shared__ size_t cur_length;
	__shared__ unsigned long threshold,portion;
	__shared__ int flow_control_flag;
	__shared__ unsigned int data[16];

	__shared__ clock_t s1[16],e1[16];
	__shared__ clock_t s2[16],e2[16];
	__shared__ clock_t s3[16],e3[16];
	int index1=0,index2=0,index3=0;
	__shared__ clock_t s,e;
	int flag = 1;
	//verify connection
	BEGIN_SINGLE_THREAD_DO
		session_id=0;
		block_length=0;
		cur_length=0;
		flow_control_flag=0;
		// cjdebug("enter _socket_recv_ctrl function!\n");
		// size_t t=clock64();
		// cjprint("_socket_recv_ctrl start t= %ld\n",t);

		session_id = ctx->buffer_info[buffer_id].session_id;
		threshold = (unsigned long)(SINGLE_BUFFER_LENGTH*FLOW_CONTROL_RATIO*(ctx->buffer_read_count_record[buffer_id]));
		portion =  (unsigned long)(SINGLE_BUFFER_LENGTH*FLOW_CONTROL_RATIO);
	END_SINGLE_THREAD_DO
	while(1){
		BEGIN_SINGLE_THREAD_DO
			s1[index1]=clock64();
			int offset = int((SINGLE_BUFFER_LENGTH*buffer_id+ctx->recv_buffer_offset[buffer_id])/sizeof(int));
			size_t clock_start=clock64();
			while(ctx->recv_buffer[offset+1]==0){
				if(clock64()-clock_start > size_t(15000000000L)){
					cjprint("timeout at head, recv_buffer_offset=%d recv_rd_cnt:%ld\n",ctx->recv_buffer_offset[buffer_id],ctx->recv_read_count[buffer_id]);
					for(int i=0;i<16;i++){
						printf("%d ",ctx->recv_buffer[offset-16-i]);
					}
					printf("\n");
					for(int i=0;i<32;i++){
						printf("%d ",ctx->recv_buffer[offset+i]);
					}
					printf("\n");
					while(1);
				}
			}
			if(flag==1){
				s=clock64();
				// cjprint("_socket_recv_ctrl see 1st head t= %ld\n",s);
				flag=0;
			}
			block_length = ctx->recv_buffer[offset+1];
			// cjprint("block_length: %d\n",block_length);
			// for(int i=0;i<32;i++){
			// 	printf("%d ",ctx->recv_buffer[offset+i]);
			// }
			// printf("\n");

			int tail_offset = int(block_length/sizeof(int)) - int(64/sizeof(int));

			e1[index1]=clock64();
			if(index1<15){
				index1++;
			}
			s2[index2]=clock64();
			

			clock_start=clock64();
			while(ctx->recv_buffer[offset+1+tail_offset] != block_length){
				// cjdebug("waiting tail!\n");
				// cu_sleep(1);
				if(clock64()-clock_start > size_t(15000000000L)){
					cjprint("timeout at tail\n");
					while(1);
				}
			}
			e2[index2]=clock64();
			if(index2<15){
				index2++;
			}
			s3[index3]=clock64();
			ctx->recv_buffer[offset+1] = 0;
			ctx->recv_buffer[offset+1+tail_offset] = 0;
			ctx->recv_buffer_offset[buffer_id]+=64;
			block_length = block_length-128;
			int wr_index = ctx->recv_fifo_wr[buffer_id];
			int wr_index_mod = wr_index%16;
			for(int i=0;i<4;i++){//num blocks
				clock_start=clock64();
				while(wr_index - ctx->recv_fifo_rd[buffer_id][i]==16){
					if(clock64()-clock_start > size_t(15000000000L)){
						cjprint("timeout at data\n");
						//cjdebug("waiting rd, bufferId:%d rd0:%d rd1:%d wr:%d\n",buffer_id,ctx->recv_fifo_rd[buffer_id][0],ctx->recv_fifo_rd[buffer_id][1],wr_index);
						while(1);
					}
				}
			}
			ctx->recv_fifo_addr_offset[buffer_id][wr_index_mod] = int((SINGLE_BUFFER_LENGTH*buffer_id+ctx->recv_buffer_offset[buffer_id])/4);
			ctx->recv_fifo_length[buffer_id][wr_index_mod] = block_length;
			ctx->recv_fifo_wr[buffer_id]+=1;
			ctx->recv_read_count[buffer_id]+=block_length;
			ctx->recv_buffer_offset[buffer_id]+=(block_length+64);
			if(ctx->recv_buffer_offset[buffer_id]>(SINGLE_BUFFER_LENGTH-MAX_PACKAGE_LENGTH)){
				ctx->recv_buffer_offset[buffer_id] = 0;
			}
			cur_length+=block_length;
			//printf("%d %ld %ld\n",block_length,cur_length,length);
			e3[index3]=clock64();
			if(index3<15){
				index3++;
			}

			if(ctx->recv_read_count[buffer_id]>=threshold){
				ctx->buffer_read_count_record[buffer_id] += 1;
				threshold += portion;
				flow_control_flag=1;
				data[0] = (ctx->recv_read_count[buffer_id]);//todo seq
				data[1] = (ctx->recv_read_count[buffer_id])>>32;
				data[2] = session_id;
			}
		END_SINGLE_THREAD_DO
		if(flow_control_flag==1){
			write_bypass(ctx->recv_read_count_bypass_reg,data);
			flow_control_flag=0;
			// BEGIN_SINGLE_THREAD_DO
			// 	printf("write bypass: %ld\n",ctx->recv_read_count[buffer_id]);
			// END_SINGLE_THREAD_DO
		}
		if(cur_length==length){
			e=clock64();
			break;
		}
	}
	BEGIN_SINGLE_THREAD_DO
	if(cur_length==length){
		cjdebug("ctrl recv done! %ld cycles length=%fM\n",e-s,1.0*length/1024/1024);
	}else if(cur_length>length){
		cjerror("ctrl recv done,but data is a little bit more!\n");
	};
	// for(int i=0;i<1;i++){
	// 	printf("ctrl0 %ld %ld %ld\n",e1[i]-s1[i],e2[i]-s2[i],e3[i]-s3[i]);
	// }
	END_SINGLE_THREAD_DO

}
__device__ void _socket_recv(socket_context_t* ctx,int buffer_id,int * data_addr,size_t length){
	int index = blockIdx.x*blockDim.x+threadIdx.x;
	int total_threads = blockDim.x*gridDim.x;

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
		}
		//move_data_from_recv_buffer(ctx,buffer_id,block_length,data_addr);
		data_addr+=int(block_length/sizeof(int));
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
		}
	END_SINGLE_THREAD_DO
}
__global__ void socket_recv(socket_context_t* ctx,int* socket,int * data_addr,size_t length){
	__shared__ int socket_id;
	__shared__ int buffer_id;
	BEGIN_BLOCK_ZERO_DO
		// cjdebug("enter socket_recv function!\n");
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
	END_BLOCK_ZERO_DO
	_socket_recv_data(ctx,buffer_id,data_addr,length);
}

__global__ void socket_recv(socket_context_t* ctx,connection_t* connection,int * data_addr,size_t length){
	__shared__ int buffer_id;
	//verify connection
	BEGIN_BLOCK_ZERO_DO
		// cjdebug("enter socket_recv function!\n");
		if(connection->valid==0){
			cjerror("connection is not valid!\n");
			return;
		}
	END_BLOCK_ZERO_DO
	_socket_recv_data(ctx,connection->buffer_id,data_addr,length);
}

__global__ void send_kernel(socket_context_t* ctx,unsigned int *dev_buffer,fpga_registers_t registers,int node_type){

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

		ctx->send_cmd_fifo_count		=	registers.send_cmd_fifo_count;


		//bypass
		ctx->send_data_cmd_bypass_reg	=	registers.send_data_cmd_bypass_reg;
		ctx->recv_read_count_bypass_reg	=	registers.recv_read_count_bypass_reg;
		ctx->recv_buffer				=	dev_buffer+int(102*1024*1024/sizeof(int));
		
	END_SINGLE_THREAD_DO
		while(1){
			// BEGIN_SINGLE_THREAD_DO
				int res = read_info(ctx);
				if(res && node_type){
					cjdebug("#############kernel stop\n");
					break;
				}
				// #if SLOW_DEBUG
				// 	cu_sleep(1);
				// 	cjdebug("kernel pooling\n");
				// #endif
				// cu_sleep(1);
				// cjdebug("kernel pooling\n");
			// END_SINGLE_THREAD_DO
		}
	
}
