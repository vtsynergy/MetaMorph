#include "afosr_cfd_mpi.h"

//Static sentinel head node, and dynamic tail pointer for the
// MPI request queue
recordQueueNode record_queue;
recordQueueNode *record_queue_tail;


//Helper functions for asynchronous transfers
void init_record_queue() {
	record_queue.type = sentinel;
	//No need to set the record, it won't be looked at
	record_queue.next = NULL;
	record_queue_tail = &record_queue;
}

//The enqueue function
void register_mpi_request(request_record_type type, request_record * request) {
	recordQueueNode *record =(recordQueueNode*) malloc(sizeof(recordQueueNode));
	record->type = type;
	record->next = NULL;
	//When copying, we can treat the record as a rap_rec, since the union
	// will force it to use that much space, and it has all the elements
	// of the other three types (it will just copy unused garbage for the smaller ones)
	//TODO: Replace with memcpy?
	memcpy(&(record->record.rap_rec), &(request->rap_rec), sizeof(recv_and_unpack_record));
	//record->record.rap_rec.req = request->rap_rec.req;
	//record->record.rap_rec.host_packed_buf = request->rap_rec.host_packed_buf;
	//record->record.rap_rec.dev_packed_buf = request->rap_rec.dev_packed_buf;
	//record->record.rap_rec.buf_size = request->rap_rec.buf_size;
	//record->record.rap_rec.grid_size = request->rap_rec.grid_size;
	//record->record.rap_rec.block_size = request->rap_rec.block_size;
	//record->record.rap_rec.type = request->rap_rec.type;
	//record->record.rap_rec.face = request->rap_rec.face;

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
	int flag;
	MPI_Status status;
	MPI_Test(curr->record.rap_rec.req, &flag, &status);
	if (flag) {
		//If the request completed, remove the node
		//FIXME: Hazardous!
		record_queue.next = curr->next;
	} else {
		//Don't do anything else, we need to remove nodes in-order
		// so no skipping ahead
		return;
	}
	switch (curr->type) {
		case sp_rec:
			//do seomthing
			sp_helper(&(curr->record));
		break;

		case rp_rec:
			//do something
			rp_helper(&(curr->record));
		break;

		case sap_rec:
			//Do seomthing
			sap_helper(&(curr->record));
		break;

		case rap_rec:
			//Do something
			rap_helper(&(curr->record));
		break;

		default:
			//shouldn't be reachable
		break;
	}
}

//A "forced wait until all requests finish" helper
// Meant to be used with accel_flush and accel_finish
void finish_mpi_requests() {
	//TODO: Implement "finish the world"
	//Use MPI_Wait rather than MPI Test
}

#ifdef WITH_CUDA
void CUDART_CB cuda_sp_isend_cb(cudaStream_t stream, cudaError_t status, void *data) {
	sp_callback_payload * call_pl = (sp_callback_payload *)data;
			MPI_Isend(call_pl->host_packed_buf, call_pl->buf_leng, call_pl->mpi_type, call_pl->dst_rank, call_pl->tag, MPI_COMM_WORLD, call_pl->req);
	//Assemble a request
	request_record * rec = (request_record *) malloc(sizeof(request_record));
	rec->sp_rec.req = call_pl->req;
	rec->sp_rec.host_packed_buf = call_pl->host_packed_buf;
	//and submit it
	register_mpi_request(sp_rec, rec);
	//once registered, all the params are copied, so the record can be freed
	free(rec);
	//once the Isend is invoked and the request is submitted, we can remove the payload
	free(data);
}
#endif //WITH_CUDA
#ifdef WITH_OPENCL
void CL_CALLBACK opencl_sp_isend_cb(cl_event event, cl_int status, void *data) {
	sp_callback_payload * call_pl = (sp_callback_payload *)data;
			MPI_Isend(call_pl->host_packed_buf, call_pl->buf_leng, call_pl->mpi_type, call_pl->dst_rank, call_pl->tag, MPI_COMM_WORLD, call_pl->req);
	//Assemble a request
	request_record * rec = (request_record *) malloc(sizeof(request_record));
	rec->sp_rec.req = call_pl->req;
	rec->sp_rec.host_packed_buf = call_pl->host_packed_buf;
	//and submit it
	register_mpi_request(sp_rec, rec);
	//once registered, all the params are copied, so the record can be freed
	free(rec);
	//once the Isend is invoked and the request is submitted, we can remove the payload
	free(data);
}
#endif //EITH_OPENCL

void sp_helper(request_record * sp_request) {
	free(sp_request->sp_rec.host_packed_buf);
}

//Compound transfers, reliant on other library functions
//Essentially a wrapper for a contiguous buffer send
a_err accel_mpi_packed_face_send(int dst_rank, void *packed_buf, void *packed_buf_host, size_t buf_leng, int tag, MPI_Request *req, accel_type_id type, int async) {
	a_err error;
	size_t type_size;
	MPI_Datatype mpi_type;
	switch (type) {
		case a_db:
			type_size = sizeof(double);
			mpi_type = MPI_DOUBLE;
		break;

		case a_fl:
			type_size = sizeof(float);
			mpi_type = MPI_FLOAT;
		break;

		case a_ul:
			type_size = sizeof(unsigned long);
			mpi_type = MPI_UNSIGNED_LONG;
		break;

		case a_in:
			type_size = sizeof(int);
			mpi_type = MPI_INT;
		break;

		case a_ui:
			type_size = sizeof(unsigned int);
			mpi_type = MPI_UNSIGNED;
		break;

		default:
			//Should be a No-op
			// for safety, just give it the size of a pointer
			type_size = sizeof(double *);
			//TODO: Put something appropriate in mpi_type;
		break;
	}
	//If we have GPUDirect, simply copy the device buffer diectly
	#ifdef WITH_MPI_GPU_DIRECT
		#ifdef WITH_CUDA
			if (run_mode == accelModePreferCUDA) {
			//send directly
				//FIXME: add other data types
				if(async) {
					MPI_Isend(packed_buf, buf_leng, mpi_type, dst_rank, tag, MPI_COMM_WORLD, req);
					//FIXME: Build a record around the request, to be finished by a helper later
				} else {
					MPI_Send(packed_buf, buf_leng, mpi_type, dst_rank, tag, MPI_COMM_WORLD);
				}
			
			} else
		#endif //WITH_CUDA
			{
	#endif //WITH_MPI_GPU_DIRECT
	//otherwise..
	//allocate a host buffer
	//FIXME: user app is now responsible for this
	//void *packed_buf_host = malloc(buf_leng*type_size);
	
	//Do the send
	//FIXME: bind the send routine to a cl_event/cu_event callback
	//FIXME: Record the request with the host buffer needing freeing
	if (async) {
		//Figure out whether to use CUDA or OpenCL callback
		accel_callback * call = (accel_callback*) malloc(sizeof(accel_callback));
		switch(run_mode) {
			#ifdef WITH_CUDA
				case accelModePreferCUDA:
					call->cudaCallback = cuda_sp_isend_cb;
				break;
			#endif //WITH_CUDA
			#ifdef WITH_OPENCL
				case accelModePreferOpenCL:
					call->openclCallback = opencl_sp_isend_cb;
				break;
			#endif
				default:
					//TODO: Do something
				break;
		}
		//Assemble the callback's necessary data elements
		sp_callback_payload * call_pl = (sp_callback_payload *)malloc(sizeof(sp_callback_payload));
		call_pl->host_packed_buf = packed_buf_host;
		call_pl->buf_leng = buf_leng;
		call_pl->mpi_type = mpi_type;
		call_pl->dst_rank = dst_rank;
		call_pl->tag = tag;
		call_pl->req = req;
		//Copy the buffer to the host, with the callback chain needed to complete the transfer
		error |= accel_copy_d2h_cb(packed_buf_host, packed_buf, buf_leng*type_size, 0, call, (void*)call_pl);
	} else {
		//copy into the host buffer
		error |= accel_copy_d2h(packed_buf_host, packed_buf, buf_leng*type_size, 0);
		MPI_Send(packed_buf_host, buf_leng, mpi_type, dst_rank, tag, MPI_COMM_WORLD);
		//FIXME: remove associated async frees
		//free(packed_buf_host);
	}
	//Close the if/else checking if mode == CUDA
	#ifdef WITH_MPI_GPU_DIRECT
			}
	#endif
}

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
	//async H2D copy w/ callback to free the temp host buffer
	//set up the mode_specific pointer to our free wrapper
	accel_callback * call = (accel_callback*) malloc(sizeof(accel_callback));
	switch(run_mode) {
		#ifdef WITH_CUDA
			case accelModePreferCUDA:
				call->cudaCallback = cuda_free_host_cb;
			break;
		#endif //WITH_CUDA
		#ifdef WITH_OPENCL
			case accelModePreferOpenCL:
				call->openclCallback = opencl_free_host_cb;
			break;
		#endif
			default:
				//TODO: Do something
			break;
	}
	//and invoke the async copy with the callback specified and the host pointer as the payload
	accel_copy_h2d_cb(rp_request->rp_rec.dev_packed_buf, rp_request->rp_rec.host_packed_buf, rp_request->rp_rec.buf_size, 1, call, rp_request->rp_rec.host_packed_buf);
}
	

//Essentially a wrapper for a contiguous buffer receive
a_err accel_mpi_packed_face_recv(int src_rank, void *packed_buf, void * packed_buf_host, size_t buf_leng, int tag, MPI_Request *req, accel_type_id type, int async) {
	a_err error;
	MPI_Status status;
	size_t type_size;
	MPI_Datatype mpi_type;
	switch (type) {
		case a_db:
			type_size = sizeof(double);
			mpi_type = MPI_DOUBLE;
		break;

		case a_fl:
			type_size = sizeof(float);
			mpi_type = MPI_FLOAT;
		break;

		case a_ul:
			type_size = sizeof(unsigned long);
			mpi_type = MPI_UNSIGNED_LONG;
		break;

		case a_in:
			type_size = sizeof(int);
			mpi_type = MPI_INT;
		break;

		case a_ui:
			type_size = sizeof(unsigned int);
			mpi_type = MPI_INT;
		break;

		default:
			//Should be a No-op
			// for safety, just give it the size of a pointer
			type_size = sizeof(double *);
		break;
	}
	
	//If we have GPUDriect, simply receive directly to the device buffer
	#ifdef WITH_MPI_GPU_DIRECT
		#ifdef WITH_CUDA
			if (run_mode == accelModePreferCUDA) {
				//Do something
				//FIXME: Support types other than double
				//rintf("GPU DIRECT MODE ACTIVE!\n");
				if(async) {
					MPI_Irecv(packed_buf, buf_leng, mpi_type, src_rank, tag, MPI_COMM_WORLD, req);
				} else {
					MPI_Recv(packed_buf, buf_leng, mpi_type, src_rank, tag, MPI_COMM_WORLD, &status);
				}
			} else
		#endif //WITH_CUDA
			{
	#endif //WITH_MPI_GPU_DIRECT
	//otherwise..	
	//allocate a host buffer
	//FIXME: user apps now responsible for this
	//void *packed_buf_host = malloc(buf_leng*type_size);
	
	//Receive into the host buffer
	//FIXME: Add proper callback-related functionality to allow the pack and copy to be asynchronous
	if (async) {
		MPI_Irecv(packed_buf_host, buf_leng, mpi_type, src_rank, tag, MPI_COMM_WORLD, req);
		//Assemble a request_record
		request_record * rec = (request_record *) malloc(sizeof(request_record));
		rec->rp_rec.req = req;
		rec->rp_rec.host_packed_buf = packed_buf_host;
		rec->rp_rec.dev_packed_buf = packed_buf;
		rec->rp_rec.buf_size = buf_leng*type_size; 
		//and submit it
		register_mpi_request(rp_rec, rec);
		//once registered, all the params are copied, so the record can be freed
		free(rec);
	} else {
		MPI_Recv(packed_buf_host, buf_leng, mpi_type, src_rank, tag, MPI_COMM_WORLD, &status);
		//Copy into the device buffer
		error |= accel_copy_h2d(packed_buf, packed_buf_host, buf_leng*type_size, 0);
		//free the host buffer
		//FIXME: remove associated async frees
		//free(packed_buf_host);
	}
	//Close the if/else checking if mode == CUDA
	#ifdef WITH_MPI_GPU_DIRECT
			}
	#endif // WITH_MPI_GPU_DIRECT
}

#ifdef WITH_CUDA
void CUDART_CB cuda_sap_isend_cb(cudaStream_t stream, cudaError_t status, void *data) {
	sap_callback_payload * call_pl = (sap_callback_payload *)data;
	if (call_pl->host_packed_buf != NULL) { //No GPUDirect
		MPI_Isend(call_pl->host_packed_buf, call_pl->buf_leng, call_pl->mpi_type, call_pl->dst_rank, call_pl->tag, MPI_COMM_WORLD, call_pl->req);
	} else {
		MPI_Isend(call_pl->dev_packed_buf, call_pl->buf_leng, call_pl->mpi_type, call_pl->dst_rank, call_pl->tag, MPI_COMM_WORLD, call_pl->req);
	}
	//Assemble a request
	request_record * rec = (request_record *) malloc(sizeof(request_record));
	rec->sap_rec.req = call_pl->req;
	rec->sap_rec.host_packed_buf = call_pl->host_packed_buf;
	rec->sap_rec.dev_packed_buf = call_pl->dev_packed_buf;
	//and submit it
	register_mpi_request(sap_rec, rec);
	//once registered, all the params are copied, so the record can be freed
	free(rec);
	//once the Isend is invoked and the request is submitted, we can remove the payload
	free(data);
}
#endif //WITH_CUDA
#ifdef WITH_OPENCL
void CL_CALLBACK opencl_sap_isend_cb(cl_event event, cl_int status, void *data) {
	sap_callback_payload * call_pl = (sap_callback_payload *)data;
	MPI_Isend(call_pl->host_packed_buf, call_pl->buf_leng, call_pl->mpi_type, call_pl->dst_rank, call_pl->tag, MPI_COMM_WORLD, call_pl->req);
	//Assemble a request
	request_record * rec = (request_record *) malloc(sizeof(request_record));
	rec->sap_rec.req = call_pl->req;
	rec->sap_rec.host_packed_buf = call_pl->host_packed_buf;
	rec->sap_rec.dev_packed_buf = call_pl->dev_packed_buf;
	//and submit it
	register_mpi_request(sap_rec, rec);
	//once registered, all the params are copied, so the record can be freed
	free(rec);
	//once the Isend is invoked and the request is submitted, we can remove the payload
	free(data);
}
#endif //EITH_OPENCL

void sap_helper(request_record * sap_request) {
	if (sap_request->sap_rec.host_packed_buf != NULL) free(sap_request->sap_rec.host_packed_buf);
	accel_free(sap_request->sap_rec.dev_packed_buf);
}

//A wrapper that, provided a face, will pack it, then send it
a_err accel_mpi_pack_and_send_face(a_dim3 *grid_size, a_dim3 *block_size, int dst_rank, accel_2d_face_indexed * face, void * buf, void * packed_buf, void * packed_buf_host, int tag, MPI_Request *req, accel_type_id type, int async) {
	//allocate space for a packed buffer
	size_t size = 1;
	int i;
	for(i = 0; i < face->count; i++) size *= face->size[i];
	//FIXME: User app is now responsible for allocing and managing these
	//void *packed_buf, *packed_buf_host;
	size_t type_size;
	MPI_Datatype mpi_type;
	switch (type) {
		case a_db:
			type_size = sizeof(double);
			mpi_type = MPI_DOUBLE;
		break;

		case a_fl:
			type_size = sizeof(float);
			mpi_type = MPI_FLOAT;
		break;

		case a_ul:
			type_size = sizeof(unsigned long);
			mpi_type = MPI_UNSIGNED_LONG;
		break;

		case a_in:
			type_size = sizeof(int);
			mpi_type = MPI_INT;
		break;

		case a_ui:
			type_size = sizeof(unsigned int);
			mpi_type = MPI_UNSIGNED;
		break;

		default:
			//Should be a No-op
			// for safety, just give it the size of a pointer
			type_size = sizeof(double *);
		break;
	}
	//FIXME: remove this, and any associated frees, app is now responsible
	a_err error;
	//a_err error = accel_alloc(&packed_buf, size*type_size);
	// call the device pack function
	//FIXME: add grid, block params
	//FIXME: Add proper callback-related functionality to allwo the pack and copy to be asynchronous

	//do a send, if asynchronous, this should be a callback for when the async D2H copy finishes
	#ifdef WITH_MPI_GPU_DIRECT
		#ifdef WITH_CUDA
		//GPUDirect version
			if (run_mode == accelModePreferCUDA) { 
				//FIXME: add other data types
				if(async) {
					//TODO: Move this into a callback for accel_pack_2d_face_cb
					accel_callback * call = (accel_callback*) malloc(sizeof(accel_callback));
					call->cudaCallback = cuda_sap_isend_cb;
					//Assemble the callback's necessary data elements
					sap_callback_payload * call_pl = (sap_callback_payload *)malloc(sizeof(sap_callback_payload));
					call_pl->host_packed_buf = NULL;
					call_pl->dev_packed_buf = packed_buf;
					call_pl->buf_leng = size;
					call_pl->mpi_type = mpi_type;
					call_pl->dst_rank = dst_rank;
					call_pl->tag = tag;
					call_pl->req = req;
					accel_pack_2d_face_cb(grid_size, block_size, packed_buf, buf, face, type, 1, call, call_pl);
					//MPI_Isend(packed_buf, size, mpi_type, dst_rank, tag, MPI_COMM_WORLD, req);
				} else {
					error |= accel_pack_2d_face(grid_size, block_size, packed_buf, buf, face, type, 0);
					MPI_Send(packed_buf, size, mpi_type, dst_rank, tag, MPI_COMM_WORLD);
					//FIXME: remove associated async free
					//accel_free(packed_buf);
				}
			} else 
		#endif //WITH_CUDA
			{
	#endif //WITH_MPI_GPU_DIRECT
	
	//FIXME Remove this and associate frees, application is now responsible for host buffer	
	//packed_buf_host = malloc(size*type_size);
	//DO all variants via copy to host
	//FIXME: Add proper callback-related functionality to allow the pack and copy to be asynchronous
	if (async) {
		accel_pack_2d_face(grid_size, block_size, packed_buf, buf, face, type, 1);
		//Figure out whether to use CUDA or OpenCL callback
		accel_callback * call = (accel_callback*) malloc(sizeof(accel_callback));
		switch(run_mode) {
			#ifdef WITH_CUDA
				case accelModePreferCUDA:
					call->cudaCallback = cuda_sap_isend_cb;
				break;
			#endif //WITH_CUDA
			#ifdef WITH_OPENCL
				case accelModePreferOpenCL:
					call->openclCallback = opencl_sap_isend_cb;
				break;
			#endif
				default:
					//TODO: Do something
				break;
		}
		//Assemble the callback's necessary data elements
		sap_callback_payload * call_pl = (sap_callback_payload *)malloc(sizeof(sap_callback_payload));
		call_pl->host_packed_buf = packed_buf_host;
		call_pl->dev_packed_buf = packed_buf;
		call_pl->buf_leng = size;
		call_pl->mpi_type = mpi_type;
		call_pl->dst_rank = dst_rank;
		call_pl->tag = tag;
		call_pl->req = req;
		accel_copy_d2h_cb(packed_buf_host, packed_buf, size*type_size, 1, call, call_pl);
	} else {
		error |= accel_pack_2d_face(grid_size, block_size, packed_buf, buf, face, type, 0);
		error |= accel_copy_d2h(packed_buf_host, packed_buf, size*type_size, 0);
		MPI_Send(packed_buf_host, size, mpi_type, dst_rank, tag, MPI_COMM_WORLD);
	//FIXME: remove asynchronous frees too
	//	accel_free(packed_buf);
	//	free(packed_buf_host);
	}
	//TODO: once async callback is implemented free malloced buffer	
	//TODO: add callback for asyncs
	//close the if/else for checking if mode == CUDA
	#ifdef WITH_MPI_GPU_DIRECT
			}
	#endif //WITH_MPI_GPU_DIRECT
	//FIXME: Free device packed_buf correctly (with callback if async)
}


#ifdef WITH_OPENCL
void CL_CALLBACK opencl_rap_freebufs_cb(cl_event event, cl_int status, void * data) {
	rap_callback_payload * call_pl = (rap_callback_payload *)data;
	if (call_pl->host_packed_buf != NULL) free(call_pl->host_packed_buf);
	free(call_pl->packed_buf);
	free(data);
}
#endif //WITH_OPENCL
#ifdef WITH_CUDA
void CUDART_CB cuda_rap_freebufs_cb(cudaStream_t stream, cudaError_t status, void *data) {
	rap_callback_payload * call_pl = (rap_callback_payload *)data;
	if (call_pl->host_packed_buf != NULL) free(call_pl->host_packed_buf);
	free(call_pl->packed_buf);
	free(data);
}
#endif //WITH_CUDA

void rap_helper(request_record *rap_request) {
	//async H2D copy w/ callback to free the temp host buffer
	//set up the mode_specific pointer to our free wrapper
	accel_callback * call = (accel_callback*) malloc(sizeof(accel_callback));
	switch(run_mode) {
		#ifdef WITH_CUDA
			case accelModePreferCUDA:
				call->cudaCallback = cuda_rap_freebufs_cb;
			break;
		#endif //WITH_CUDA
		#ifdef WITH_OPENCL
			case accelModePreferOpenCL:
				call->openclCallback = opencl_rap_freebufs_cb;
			break;
		#endif
			default:
				//TODO: Do something
			break;
	}
	//and invoke the async copy with the callback specified and the host pointer as the payload
	rap_callback_payload * call_pl = (rap_callback_payload *)malloc(sizeof(rap_callback_payload));
	call_pl->host_packed_buf = NULL;
	if(rap_request->rap_rec.host_packed_buf != NULL) {
		accel_copy_h2d(rap_request->rap_rec.dev_packed_buf, rap_request->rap_rec.host_packed_buf, rap_request->rap_rec.buf_size, 1);
		call_pl->host_packed_buf = rap_request->rap_rec.host_packed_buf;
	}
	accel_unpack_2d_face_cb(&(rap_request->rap_rec.grid_size), &(rap_request->rap_rec.block_size), rap_request->rap_rec.dev_packed_buf, rap_request->rap_rec.dev_buf, rap_request->rap_rec.face, rap_request->rap_rec.type, 1, call, call_pl);
}

//A wrapper that, provided a face, will receive a buffer, and unpack it
a_err accel_mpi_recv_and_unpack_face(a_dim3 * grid_size, a_dim3 * block_size, int src_rank, accel_2d_face_indexed * face, void * buf, void * packed_buf, void * packed_buf_host, int tag, MPI_Request * req, accel_type_id type, int async) {
	//allocate space to receive the packed buffer
	size_t size = 1;
	int i;
	for(i = 0; i < face->count; i++) size *= face->size[i];
	//FIXME: User app is now responsible for managing these
	//void *packed_buf, *packed_buf_host;
	size_t type_size;
	MPI_Datatype mpi_type;
	switch (type) {
		case a_db:
			type_size = sizeof(double);
			mpi_type = MPI_DOUBLE;
		break;

		case a_fl:
			type_size = sizeof(float);
			mpi_type = MPI_FLOAT;
		break;

		case a_ul:
			type_size = sizeof(unsigned long);
			mpi_type = MPI_UNSIGNED_LONG;
		break;

		case a_in:
			type_size = sizeof(int);
			mpi_type = MPI_INT;
		break;

		case a_ui:
			type_size = sizeof(unsigned int);
			mpi_type = MPI_UNSIGNED;
		break;

		default:
			//Should be a No-op
			// for safety, just give it the size of a pointer
			type_size = sizeof(double *);
		break;
	}
	//FIXME: remove this and associated frees
	a_err error;
	//a_err error = accel_alloc(&packed_buf, size*type_size);
	MPI_Status status; //FIXME: Should this be a function param?
	#ifdef WITH_MPI_GPU_DIRECT
		#ifdef WITH_CUDA
			//GPUDirect version
			if (run_mode == accelModePreferCUDA) {	
				if(async) {
					MPI_Irecv(packed_buf, size, mpi_type, src_rank, tag, MPI_COMM_WORLD, req);
					//Helper will recognize the null, and not do h2d transfer, but
					// will invoke unpack kernel
					//Assemble a request_record
					request_record * rec = (request_record *) malloc(sizeof(request_record));
					rec->rap_rec.req = req;
					rec->rap_rec.host_packed_buf = NULL;
					rec->rap_rec.dev_packed_buf = packed_buf;
					rec->rap_rec.buf_size = size*type_size;
					memcpy(&(rec->rap_rec.grid_size), grid_size, sizeof(a_dim3));
					memcpy(&(rec->rap_rec.block_size), block_size, sizeof(a_dim3));
					rec->rap_rec.type = type;
					rec->rap_rec.face = face;
					//and submit it
					register_mpi_request(rap_rec, rec);
					//once registered, all the params are copied, so the record can be freed
					free(rec);
				} else {
					MPI_Recv(packed_buf, size, mpi_type, src_rank, tag, MPI_COMM_WORLD, &status);
				}
			} else 
		#endif //WITH_CUDA
			{
	#endif //WITH_MPI_GPU_DIRECT
	
	//Non-GPUDirect version
	//FIXME: Remove this and associated frees
	//packed_buf_host = malloc(size*type_size);
	//do a receive, if asynchronous, we need to implement a callback to launch the H2D copy
	//FIXME: Add proper callback-related functionality to allow the pack and copy to be asynchronous
	if (async) {
		MPI_Irecv(packed_buf_host, size, mpi_type, src_rank, tag, MPI_COMM_WORLD, req);
		//TODO: Register the transfer, with host pointer set to packed_buf_host
		//Helper will recognize the non-null pointer and do an h2d transfer and unpack
		// with necessary callbacks to free both temp buffers
		request_record * rec = (request_record *) malloc(sizeof(request_record));
		rec->rap_rec.req = req;
		rec->rap_rec.host_packed_buf = packed_buf_host;
		rec->rap_rec.dev_packed_buf = packed_buf;
		rec->rap_rec.buf_size = size*type_size;
		memcpy(&(rec->rap_rec.grid_size), grid_size, sizeof(a_dim3));
		memcpy(&(rec->rap_rec.block_size), block_size, sizeof(a_dim3));
		rec->rap_rec.type = type;
		rec->rap_rec.face = face;
		//and submit it
		register_mpi_request(rap_rec, rec);
		//once registered, all the params are copied, so the record can be freed
		free(rec);
	} else {
		//receive the buffer
		MPI_Recv(packed_buf_host, size, mpi_type, src_rank, tag, MPI_COMM_WORLD, &status);
		//copy the buffer to the device
		error |= accel_copy_h2d(packed_buf, packed_buf_host, size*type_size, 0);
		//free the host temp buffer
		//FIXME: remove the async frees
		//free(packed_buf_host);
		//Unpack on the device
		error |= accel_unpack_2d_face(grid_size, block_size, packed_buf, buf, face, type, 0);
		//free the device temp buffer
		//FIXME: remove async frees
		//accel_free(packed_buf);
	}
	//TODO: once async callback is implemented free malloced buffer
	//TODO: add callback for asyncs
	//close the if/else for checking if mode == CUDA
	#ifdef WITH_MPI_GPU_DIRECT
			}
	#endif //WITH_MPI_GPU_DIRECT
	//FIXME: Free device packed_buf correctly (with callback if async)
}


