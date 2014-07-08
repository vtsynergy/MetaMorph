#include "afosr_cfd_mpi.h"

//Helper functions for asynchronous transfers

//minimal callback to free a host buffer
#ifdef WITH_OPENCL
void CL_CALLBACK opencl_free_host_cb(cl_event event, cl_int status, void * data) {
	free(data);
}
#endif //WITH_OPENCL
#ifdef WITH_CUDA
void CUDART_CB cuda_free_host_cb(cudaStream_t stream, cudaError_t status, void *data) {
	free(data);
}
#endif //WITH_CUDA

void rp_helper(request_record * rp_request) {
	//async H2D copy w/ callback to
	//free(host_packed_buf);
}


//The enqueue function
void register_mpi_request(request_record_type type, request_record * request) {
	recordQueueNode *record = malloc(sizeof(recordQueueNode));
	record->type = type;
	record->next = NULL;
	//When copying, we can treat the record as a rap_rec, since the union
	// will force it to use that much space, and it has all the elements
	// of the other three types (it will just copy unused garbage for the smaller ones)
	//TODO: Replace with memcpy?
	record->record.rap_rec.req = request->record.rap_rec.req;
	record->record.rap_rec.host_packed_buf = request->record.rap_rec.host_packed_buf;
	record->record.rap_rec.dev_packed_buf = request->record.rap_rec.dev_packed_buf;
	record->record.rap_rec.buf_size = request->record.rap_rec.buf_size;
	record->record.rap_rec.grid_size = request->record.rap_rec.grid_size;
	record->record.rap_rec.block_size = request->record.rap_rec.block_size;
	record->record.rap_rec.type = request->record.rap_rec.type;
	record->record.rap_rec.face = request->record.rap_rec.face;

	//Enqueue the node
	//FIXME: Make hazard aware! (Or at least put in the stubs to finish later once the HP
	// implementation from timers is finished.)
	record_queue_tail->next = record;
	record_queue_tail = record;
}

//The Dequeue function
void help_mpi_request() {
	//try to peek at a node from the record queue
	//FIXME: Make hazard-aware
	if (record_queue.next == NULL) return;
	recordQueueNode *curr = record_queue.next;
	//test if the request has completed, if not, return
	// no need to check record type yet, since they'll all have the request at the same offset

	//if it has completed, serviec the rest of the functionality in a type-dependent way
}

//A "forced wait until all requests finish" helper
// Meant to be used with accel_flush and accel_finish
void finish_mpi_requests() {

}

//Compound transfers, reliant on other library functions
//Essentially a wrapper for a contiguous buffer send
a_err accel_mpi_packed_face_send(int dst_rank, void *packed_buf, size_t buf_leng, int tag, MPI_Request req, accel_type_id type, int async) {
	a_err error;
	//If we have GPUDirect, simply copy the device buffer diectly
	#ifdef WITH_MPI_GPU_DIRECT
		#ifdef WITH_CUDA
			if (run_mode == accelModePreferCUDA) {
			//send directly
				//FIXME: add other data types
				if(async) {
					MPI_Isend(packed_buf, buf_leng, MPI_DOUBLE, dst_rank, tag, MPI_COMM_WORLD, &req);
					//FIXME: Build a record around the request, to be finished by a helper later
				} else {
					MPI_Send(packed_buf, buf_leng, MPI_DOUBLE, dst_rank, tag, MPI_COMM_WORLD);
				}
			
			} else
		#endif //WITH_CUDA
			{
	#endif //WITH_MPI_GPU_DIRECT
	//otherwise..
	//allocate a host buffer
	size_t type_size;
	void *packed_buf_host;
	switch (type) {
		case a_db:
			type_size = sizeof(double);
		break;

		case a_fl:
			type_size = sizeof(float);
		break;

		case a_ul:
			type_size = sizeof(unsigned long);
		break;

		case a_in:
			type_size = sizeof(int);
		break;

		case a_ui:
			type_size = sizeof(unsigned int);
		break;

		default:
			//Should be a No-op
			// for safety, just give it the size of a pointer
			type_size = sizeof(double *);
		break;
	}
	packed_buf_host = malloc(buf_leng*type_size);
	
	//copy into the host buffer
	//FIXME: Add proper callback-related functionality to allow the pack and copy to be asynchronous
	error |= accel_copy_d2h(packed_buf_host, packed_buf, size*type_size, 0);
	//Do the send
	//FIXME: bind the send routine to a cl_event/cu_event callback
	//FIXME: Record the request with the host buffer needing freeing
	if (async) {
		MPI_Isend(packed_buf_host, size, MPI_DOUBLE, dst_rank, tag, MPI_COMM_WORLD, &req);
	} else {
		MPI_Send(packed_buf_host, size, MPI_DOUBLE, dst_rank, tag, MPI_COMM_WORLD);
		free(packed_buf_host);
	}
	
	//FIXME:(later) remove th ehost buffer
	//Close the if/else checking if mode == CUDA
	#ifdef WITH_MPI_GPU_DIRECT
			}
	#endif
}

//Essentially a wrapper for a contiguous buffer receive
a_err accel_mpi_packed_face_recv(int src_rank, void *packed_buf, size_t buf_leng, int tag, MPI_Request req, accel_type_id type, int async) {
	a_err error;
	//If we have GPUDriect, simply receie directly to the device buffer
	#ifdef WITH_MPI_GPU_DIRECT
		#ifdef WITH_CUDA
			if (run_mode == accelModePreferCUDA) {
				//Do something
			} else
		#endif //WITH_CUDA
			{
	#endif //WITH_MPI_GPU_DIRECT
	//otherwise..
	//allocate a host buffer
	size_t type_size;
	void *packed_buf_host;
	switch (type) {
		case a_db:
			type_size = sizeof(double);
		break;

		case a_fl:
			type_size = sizeof(float);
		break;

		case a_ul:
			type_size = sizeof(unsigned long);
		break;

		case a_in:
			type_size = sizeof(int);
		break;

		case a_ui:
			type_size = sizeof(unsigned int);
		break;

		default:
			//Should be a No-op
			// for safety, just give it the size of a pointer
			type_size = sizeof(double *);
		break;
	}
	packed_buf_host = malloc(buf_leng*type_size);
	
	//Receive into the host buffer
	//FIXME: Add proper callback-related functionality to allow the pack and copy to be asynchronous
	if (async) {
		MPI_Irecv(packed_buf_host, size, MPI_DOUBLE, src_rank, tag, MPI_COMM_WORLD, &reg);
	} else {
		MPI_Recv(packed_buf_host, size, MPI_DOUBLE, src_rank, tag, MPI_COMM_WORLD, &reg);
	}
	//Copy into the device buffer
	error |= accel_copy_h2d(packed_buf, packed_buf_host, size*type_size, 0);
	//free the host buffer
	free(packed_buf_host);
	//Close the if/else checking if mode == CUDA
	#ifdef WITH_MPI_GPU_DIRECT
			}
	#endif // WITH_MPI_GPU_DIRECT
}

//A wrapper that, provided a face, will pack it, then send it
a_err accel_mpi_pack_and_send_face(a_dim3 *grid_size, a_dim3 *block_size, int dst_rank, accel_2d_face_indexed * face, void * buf, int tag, MPI_Request req, accel_type_id type, int async) {
	//allocate space for a packed buffer
	size_t size = 1;
	int i;
	for(i = 0; i < face->count; i++) size *= face->size[i];
	void *packed_buf, *packed_buf_host;
	size_t type_size;
	switch (type) {
		case a_db:
			type_size = sizeof(double);
		break;

		case a_fl:
			type_size = sizeof(float);
		break;

		case a_ul:
			type_size = sizeof(unsigned long);
		break;

		case a_in:
			type_size = sizeof(int);
		break;

		case a_ui:
			type_size = sizeof(unsigned int);
		break;

		default:
			//Should be a No-op
			// for safety, just give it the size of a pointer
			type_size = sizeof(double *);
		break;
	}
	a_err error = accel_alloc(&packed_buf, size*type_size);
	// call the device pack function
	//FIXME: add grid, block params
	//FIXME: Add proper callback-related functionality to allwo the pack and copy to be asynchronous
	error |= accel_pack_2d_face(grid_size, block_size, packed_buf, buf, face, type, 0);
	//do a send, if asynchronous, this should be a callback for when the async D2H copy finishes
	#ifdef WITH_MPI_GPU_DIRECT
		#ifdef WITH_CUDA
		//GPUDirect version
			if (run_mode == accelModePreferCUDA) { 
				//FIXME: add other data types
				if(async) {
					MPI_Isend(packed_buf, size, MPI_DOUBLE, dst_rank, tag, MPI_COMM_WORLD, &req);
				} else {
					MPI_Send(packed_buf, size, MPI_DOUBLE, dst_rank, tag, MPI_COMM_WORLD);
				}
			} else 
		#endif //WITH_CUDA
			{
	#endif //WITH_MPI_GPU_DIRECT
		
	packed_buf_host = malloc(size*type_size);
	//DO all variants via copy to host
	//FIXME: Add proper callback-related functionality to allow the pack and copy to be asynchronous
	error |= accel_copy_d2h(packed_buf_host, packed_buf, size*type_size, 0);
	if (async) {
		MPI_Isend(packed_buf_host, size, MPI_DOUBLE, dst_rank, tag, MPI_COMM_WORLD, &req);
	} else {
		MPI_Send(packeD_buf_host, size, MPI_DOUBLE, dst_rank, tag, MPI_COMM_WORLD);
		free(packed_buf_host);
	}
	//TODO: once async callback is implemented free malloced buffer	
	//TODO: add callback for asyncs
	//close the if/else for checking if mode == CUDA
	#ifdef WITH_MPI_GPU_DIRECT
			}
	#endif //WITH_MPI_GPU_DIRECT
	//FIXME: Free device packed_buf correctly (with callback if async)
}
//A wrapper that, provided a face, will receive a buffer, and unpack it
a_err accel_mpi_recv_and_unpack_face(int src_rank, accel_2d_face_indexed * face, void * buf, int tag, MPI_Request req, accel_type_id type, int async) {
	//allocate space to receive the packed buffer
	size_t size = 1;
	int i;
	for(i = 0; i < face->count; i++) size *= face->size[i];
	void *packed_buf, *packed_buf_host;
	size_t type_size;
	switch (type) {
		case a_db:
			type_size = sizeof(double);
		break;

		case a_fl:
			type_size = sizeof(float);
		break;

		case a_ul:
			type_size = sizeof(unsigned long);
		break;

		case a_in:
			type_size = sizeof(int);
		break;

		case a_ui:
			type_size = sizeof(unsigned int);
		break;

		default:
			//Should be a No-op
			// for safety, just give it the size of a pointer
			type_size = sizeof(double *);
		break;
	}
	a_err error = accel_alloc(&packed_buf, size*type_size);
	MPI_Status status; //FIXME: Should this be a function param?
	#ifdef WITH_MPI_GPU_DIRECT
		#ifdef WITH_CUDA
			//GPUDirect version
			if (run_mode == accelModePreferCUDA) {	
				//FIXME: add other data types
				if(async) {
					MPI_Irecv(packed_buf, size, MPI_DOUBLE, src_rank, tag, MPI_COMM_WORLD, &req);
				} else {
					MPI_Recv(packed_buf, size, MPI_DOUBLE, src_rank, tag, MPI_COMM_WORLD, &status);
				}
			} else 
		#endif //WITH_CUDA
			{
	#endif //WITH_MPI_GPU_DIRECT
	
	//Non-GPUDirect version
	packed_buf_host = malloc(size*type_size);
	//do a receive, if asynchronous, we need to implement a callback to launch the H2D copy
	//FIXME: Add proper callback-related functionality to allow the pack and copy to be asynchronous
	if (async) {
		MPI_Irecv(packed_buf_host, size, MPI_DOUBLE, src_rank, tag, MPI_COMM_WORLD, &reg);
	} else {
		MPI_Recv(packed_buf_host, size, MPI_DOUBLE, src_rank, tag, MPI_COMM_WORLD, &reg);
	}
	error |= accel_copy_h2d(packed_buf, packed_buf_host, size*type_size, 0);
	free(packed_buf_host);
	//TODO: once async callback is implemented free malloced buffer
	//TODO: add callback for asyncs
	//close the if/else for checking if mode == CUDA
	#ifdef WITH_MPI_GPU_DIRECT
			}
	#endif //WITH_MPI_GPU_DIRECT
	//FIXME: Free device packed_buf correctly (with callback if async)
	// call the device unpack function
	error |= accel_unpack_2d_face(grid_size, block_size, packed_buf, buf, face, type, 0);
	//

}


