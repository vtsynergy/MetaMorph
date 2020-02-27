#include <string.h>
#include "metamorph_mpi.h"

//This pool manages host staging buffers
//These arrays are kept in sync to store the pointer and size of each buffer
void * host_buf_pool[META_MPI_POOL_SIZE];
//The size array moonlights as an occupied flag if size !=0
unsigned long host_buf_pool_size[META_MPI_POOL_SIZE];
unsigned long host_pool_token = 0;
#ifdef WITH_MPI_POOL_TIMING
double host_time = 0.0;
#endif

//This separate pool manages internal buffers for requests/callback payloads
void* internal_pool[META_MPI_POOL_SIZE];
unsigned long internal_pool_size[META_MPI_POOL_SIZE];
unsigned long internal_pool_token = 0;
#ifdef WITH_MPI_POOL_TIMING
double internal_time = 0.0;
#endif

//Alloc never moves the token, this lets the ring queue "ratchet" forward,
// which has the beneficial property of releasing anything that's been sitting
// in the pool for META_MPI_POOL_SIZE frees back to the OS.
//Additionally, this means the alloc token will always be on the oldest buffer
// (to avoid reallocating) but also be able to easily look back at the
// LOOKBACK-1 most recent buffers (to catch tight communication loops with
// heavy reuse of the same buffer).
//NOT THREADSAFE
void * pool_alloc(size_t size, int isHost) {
	void * retval;
#ifdef WITH_MPI_POOL_TIMING
	struct timeval start, end;
	gettimeofday(&start, NULL);
#endif
#ifndef WITH_MPI_POOL
	retval = malloc(size);
#else

	unsigned long * pool_size;
	void ** pool;
	unsigned long * token;
	if (isHost) {
		pool_size = host_buf_pool_size;
		pool = host_buf_pool;
		token = &host_pool_token;
	} else {
		pool_size = internal_pool_size;
		pool = internal_pool;
		token = &internal_pool_token;
	}

	//If there's something in the slot (size != 0)
	if (pool_size[*token & POOL_MASK]) {
		int i = 0;
		//roll backwards up to META_MPI_POOL_LOOKBACK cells to look for a big enough buffer
		for (i = 0; i < META_MPI_POOL_LOOKBACK && pool_size[(*token-i) & POOL_MASK] < size; i++);
		//if we found a slot of sufficient size (stopped before i == LOOKBACK)
		if (i < META_MPI_POOL_LOOKBACK) {
			//claim its pointer
			retval = pool[(*token-i) & POOL_MASK];
			//then notify (possibly other threads) that it's not available by zeroing out the size
			pool_size[(*token-i) & POOL_MASK] = 0;

		} else { //Otherwise just pull what's at the token, realloc it and move on
			//When things are pushed back to the pool, they are added back with their minimum size, so they might actually be bigger than they seem. In these cases realloc should just resize the buffer in-place.
			pool_size[*token & POOL_MASK] = 0;
			retval = realloc(pool[*token & POOL_MASK], size);
			//oh no, something went wrong
			if (retval == NULL) fprintf(stderr, "ERROR: in pool_alloc, realloc of pool buffer was unsuccessful!\n");

		}

	} else { //If there's not, just malloc directly
		retval = malloc(size);
	}
	//Sanitize the amount of space the caller requested.
	memset(retval, 0, size);
#endif
#ifdef WITH_MPI_POOL_TIMING
	gettimeofday(&end, NULL);
	if (isHost) {
		host_time += (end.tv_sec-start.tv_sec)*1000000.0+(end.tv_usec-start.tv_usec);
	} else {
		internal_time += (end.tv_sec-start.tv_sec)*1000000.0+(end.tv_usec-start.tv_usec);
	}
#endif
	return retval;
}

//Only free moves the token, this is what causes the "ratchet ring queue" behavior
// If the same buffer is used META_MPI_POOL_SIZE times in a row, all other pool buffers will get freed by it constantly being dropped one cell forward
//NOT THREADSAFE
//Uses different pools for bulk user data and internal management structs
void pool_free(void * buf, size_t size, int isUserData) {
#ifdef WITH_MPI_POOL_TIMING
	struct timeval start, end;
	gettimeofday(&start, NULL);
#endif
#ifndef WITH_MPI_POOL
	free(buf);
#else

	unsigned long * pool_size;
	void ** pool;
	unsigned long * token;
	if (isUserData) {
		pool_size = host_buf_pool_size;
		pool = host_buf_pool;
		token = &host_pool_token;
	} else {
		pool_size = internal_pool_size;
		pool = internal_pool;
		token = &internal_pool_token;
	}
	//If there's something in this slot, it must be the oldest buffer
	if (pool_size[*token & POOL_MASK]) {
		//so free it
		free(pool[*token & POOL_MASK]);
	}
	//place it
	pool[*token & POOL_MASK] = buf;
	//and update the size
	pool_size[*token & POOL_MASK] = size;
	//and move the token forward to the "next oldest" slot
	(*token)++;
#endif
#ifdef WITH_MPI_POOL_TIMING
	gettimeofday(&end, NULL);
	if (isHost) {
		host_time += (end.tv_sec-start.tv_sec)*1000000.0+(end.tv_usec-start.tv_usec);
	} else {
		internal_time += (end.tv_sec-start.tv_sec)*1000000.0+(end.tv_usec-start.tv_usec);
	}
#endif
	return;
}

//Returns the MPI_Datatype matching the given meta_type_id
// If a non-NULL size pointer is provided, it returns the sizeof the meta_type_id there as well
MPI_Datatype get_mpi_type(meta_type_id type, size_t * size) {
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
		mpi_type = MPI_DOUBLE;
		break;
	}
	if (size != NULL)
		*size = type_size;
	return mpi_type;
}
//Static sentinel head node, and dynamic tail pointer for the
// MPI request queue
//Spawn the queue in the uninit state to force calling the init function
recordQueueNode record_queue = { uninit, { NULL, NULL }, NULL };
recordQueueNode *record_queue_tail;

void meta_mpi_init(int * argc, char *** argv) {
	MPI_Init(argc, argv);
	host_pool_token = 0;
	internal_pool_token = 0;
	init_record_queue();
	memset(host_buf_pool_size, 0, sizeof(unsigned long) * META_MPI_POOL_SIZE);
	memset(internal_pool_size, 0, sizeof(unsigned long) * META_MPI_POOL_SIZE);
}

void meta_mpi_finalize() {
	//finish up any outstanding requests
	finish_mpi_requests();
	//clean up anything left in the ring buffer
	//pool_free doesn't ever dereference the provided pointer, so just calling it on zero-length null pointers POOL_SIZE times, is guaranteed to clear it
	//(since they are specified as zero-length buffers, the pool logic prevents them from ever being dereferenced or freed)
	int i;
	for (i = 0; i < META_MPI_POOL_SIZE; i++)
		pool_free(NULL, 0, 1);
	for (i = 0; i < META_MPI_POOL_SIZE; i++)
		pool_free(NULL, 0, 0);
#ifdef WITH_MPI_POOL_TIMING
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	fprintf(stderr, "MPI Pool Internal time rank[%d]: [%f] Host time: [%f]\n", rank, internal_time, host_time);
#endif
	MPI_Finalize();
}

//Helper functions for asynchronous transfers
__attribute__((constructor(103))) void init_record_queue() {
	record_queue.type = sentinel;
	//No need to set the record, it won't be looked at
	record_queue.next = NULL;
	record_queue_tail = &record_queue;
}

//The enqueue function
void register_mpi_request(request_record_type type, request_record * request) {
	//If the queue hasn't been setup, do so now
	//FIXME: Not threadsafe until only one thread can init and the rest block
	if (record_queue.type == uninit)
		init_record_queue();
	recordQueueNode *record = (recordQueueNode*) pool_alloc(
			sizeof(recordQueueNode), 0);
	record->type = type;
	record->next = NULL;
	//When copying, we can treat the record as a rap_rec, since the union
	// will force it to use that much space, and it has all the elements
	// of the other three types (it will just copy unused garbage for the smaller ones)
	//TODO: Replace with memcpy?
	memcpy(&(record->record.rap_rec), &(request->rap_rec),
			sizeof(recv_and_unpack_record));
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
	//If the queue hasn't been setup, do so now
	//FIXME: Not threadsafe until only one thread can init and the rest block
	if (record_queue.type == uninit)
		init_record_queue();
	//try to peek at a node from the record queue
	//FIXME: Make hazard-aware
	if (record_queue.next == NULL)
		return;
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
// Meant to be used with meta_flush and meta_finish
a_err finish_mpi_requests() {
	//printf("FINISH TRIGGERED\n");
	//TODO: Implement "finish the world"
	//Use MPI_Wait rather than MPI Test
	if (record_queue.type == uninit)
		init_record_queue();
	//try to peek at a node from the record queue
	//FIXME: Make hazard-aware
	int flag;
	MPI_Status status;
	while (record_queue.next != NULL) {
		recordQueueNode *curr = record_queue.next;
		//MPI_Wait(curr->record.rap_rec.req, &status);
		//MPI_Test(curr->record.rap_rec.req, &flag, &status);
		//if (flag) {
		//If the request completed, remove the node
		//FIXME: Hazardous!
		//	record_queue.next = curr->next;
		//} else {
		//Don't do anything else, we need to remove nodes in-order
		// so no skipping ahead
		//	continue;
		//}
		do {
			flag = 0;
			MPI_Test(curr->record.rap_rec.req, &flag, &status);
		} while (!flag);
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
		//printf("HELPER DONE\n");
		//FIXME(Make record queue hazard-aware!);
		//It must have finished the request, pull it off the queue
		record_queue.next = curr->next;
		//If this node is the last one, redirect the tail to the sentinel
		if (record_queue_tail == curr && record_queue.next == NULL)
			record_queue_tail = &record_queue;
		//FIXME(Is there anything inside a record that needs to be freed here);
		pool_free(curr, sizeof(recordQueueNode), 0);
	}
}

void sp_isend_cb(meta_callback * data) {
	sp_callback_payload * call_pl = (sp_callback_payload *)(data->data_payload);
	size_t type_size;
	MPI_Datatype mpi_type = get_mpi_type(call_pl->type, &type_size);
	MPI_Isend(call_pl->host_packed_buf, call_pl->buf_leng, mpi_type, call_pl->dst_rank, call_pl->tag, MPI_COMM_WORLD, call_pl->req);
	//Assemble a request
	request_record * rec = (request_record *) pool_alloc(sizeof(request_record), 0);
	rec->sp_rec.req = call_pl->req;
	rec->sp_rec.host_packed_buf = call_pl->host_packed_buf;
	rec->sp_rec.buf_size = call_pl->buf_leng*type_size;
	//and submit it
	register_mpi_request(sp_rec, rec);
	//once registered, all the params are copied, so the record can be freed
	pool_free(rec, sizeof(request_record), 0);
	//once the Isend is invoked and the request is submitted, we can remove the payload
	pool_free(call_pl, sizeof(sp_callback_payload), 0);
	pool_free(data, sizeof(meta_callback), 0);
}

void sp_helper(request_record * sp_request) {
	//printf("SP_HELPER FIRED!\n");
	pool_free(sp_request->sp_rec.host_packed_buf, sp_request->sp_rec.buf_size,
			1);
}

//Compound transfers, reliant on other library functions
//Essentially a wrapper for a contiguous buffer send
a_err meta_mpi_packed_face_send(int dst_rank, void *packed_buf, size_t buf_leng,
		int tag, MPI_Request *req, meta_type_id type, int async) {
	a_err error = 0;
	size_t type_size;
	MPI_Datatype mpi_type = get_mpi_type(type, &type_size);
	//If we have GPUDirect, simply copy the device buffer diectly
#ifdef WITH_MPI_GPU_DIRECT
	if (run_mode == metaModePreferCUDA) {
		//send directly
		//FIXME: add other data types
		//fprintf(stderr, "GPUDirect send: %x\n", packed_buf);
		if(async) {
			MPI_Isend(packed_buf, buf_leng, mpi_type, dst_rank, tag, MPI_COMM_WORLD, req);
			//FIXME: Build a record around the request, to be finished by a helper later
		} else {
			MPI_Send(packed_buf, buf_leng, mpi_type, dst_rank, tag, MPI_COMM_WORLD);
		}

	} else
	{
#endif //WITH_MPI_GPU_DIRECT
	//otherwise..
	//allocate a host buffer
	//FIXME(POOL try to alloc from the buffer pool);
	void *packed_buf_host = pool_alloc(buf_leng * type_size, 1);

	//Do the send
	if (async) {
		//Figure out whether to use CUDA or OpenCL callback
		meta_callback * call = (meta_callback*) pool_alloc(
				sizeof(meta_callback), 0);
		call->callback_mode = run_mode;
		call->callback_func = sp_isend_cb;
		//Assemble the callback's necessary data elements
		sp_callback_payload * call_pl = (sp_callback_payload *) pool_alloc(
				sizeof(sp_callback_payload), 0);
		call_pl->host_packed_buf = packed_buf_host;
		call_pl->buf_leng = buf_leng;
		call_pl->type = type;
		call_pl->dst_rank = dst_rank;
		call_pl->tag = tag;
		call_pl->req = req;
		call->data_payload = (void *) call_pl;
		//Copy the buffer to the host, with the callback chain needed to complete the transfer
		error |= meta_copy_d2h_cb(packed_buf_host, packed_buf,
				buf_leng * type_size, 0, call, NULL);
	} else {
		//copy into the host buffer
		error |= meta_copy_d2h(packed_buf_host, packed_buf,
				buf_leng * type_size, 0, NULL);
		MPI_Send(packed_buf_host, buf_leng, mpi_type, dst_rank, tag,
				MPI_COMM_WORLD);
		//FIXME: remove associated async frees
		pool_free(packed_buf_host, buf_leng * type_size, 1);
	}
	//Close the if/else checking if mode == CUDA
#ifdef WITH_MPI_GPU_DIRECT
}
#endif
}

//minimal callback to free a host buffer

void free_host_cb(meta_callback * data) {
	rp_callback_payload * call_pl = (rp_callback_payload *)(data->data_payload);
	pool_free(call_pl->host_packed_buf, call_pl->buf_size, 1);
	pool_free(call_pl, sizeof(rp_callback_payload), 0);
	pool_free(data, sizeof(meta_callback), 0);
}

void rp_helper(request_record * rp_request) {
	//Paul, added this short circuit to prevent GPU-direct RPs from trying to do a H2D copy and free a host buffer
	if (rp_request->rp_rec.host_packed_buf == NULL)
		return;
	//printf("RP_HELPER FIRED!\n");
	//async H2D copy w/ callback to free the temp host buffer
	//set up the mode_specific pointer to our free wrapper
	meta_callback * call = (meta_callback*) pool_alloc(sizeof(meta_callback),
			0);
	call->callback_mode = run_mode; 
	call->callback_func = free_host_cb;
	rp_callback_payload * call_pl = (rp_callback_payload *) pool_alloc(
			sizeof(rp_callback_payload), 0);
	call_pl->host_packed_buf = rp_request->rp_rec.host_packed_buf;
	call_pl->buf_size = rp_request->rp_rec.buf_size;
	call->data_payload = call_pl;
	//and invoke the async copy with the callback specified and the host pointer as the payload
	meta_copy_h2d_cb(rp_request->rp_rec.dev_packed_buf,
			rp_request->rp_rec.host_packed_buf, rp_request->rp_rec.buf_size, 1,
			call, NULL);
}

//Essentially a wrapper for a contiguous buffer receive
a_err meta_mpi_packed_face_recv(int src_rank, void *packed_buf, size_t buf_leng,
		int tag, MPI_Request *req, meta_type_id type, int async) {
	a_err error = 0;
	MPI_Status status;
	size_t type_size;
	MPI_Datatype mpi_type = get_mpi_type(type, &type_size);

	//If we have GPUDriect, simply receive directly to the device buffer
#ifdef WITH_MPI_GPU_DIRECT
	if (run_mode == metaModePreferCUDA) {
		//Do something
		//FIXME: Support types other than double
		//rintf("GPU DIRECT MODE ACTIVE!\n");
		if(async) {
			MPI_Irecv(packed_buf, buf_leng, mpi_type, src_rank, tag, MPI_COMM_WORLD, req);
			//TODO  ? Paul 2015.09.09
			// I see no reason to register the request, there is no need for a helper to free the non-existent host buffer
			// If we eventually allow the user to "fast-forward" the registry to ensure a specific request completes, then we will need to add both this and send_packed to the registry
			//Assemble a request_record
			//request_record * rec = (request_record *) pool_alloc(sizeof(request_record), 0);
			//rec->rp_rec.req = req;
			//rec->rp_rec.host_packed_buf = NULL;
			//rec->rp_rec.dev_packed_buf = packed_buf;
			//rec->rp_rec.buf_size = buf_leng*type_size;
			//and submit it
			//register_mpi_request(rp_rec, rec);
			//once registered, all the params are copied, so the record can be freed
			//pool_free(rec, sizeof(request_record), 0);
		} else {
			MPI_Recv(packed_buf, buf_leng, mpi_type, src_rank, tag, MPI_COMM_WORLD, &status);
		}
	} else
	{
#endif //WITH_MPI_GPU_DIRECT
	//otherwise..	
	//allocate a host buffer
	void *packed_buf_host = pool_alloc(buf_leng * type_size, 1);

	//Receive into the host buffer
	//FIXME: Add proper callback-related functionality to allow the pack and copy to be asynchronous
	if (async) {
		MPI_Irecv(packed_buf_host, buf_leng, mpi_type, src_rank, tag,
				MPI_COMM_WORLD, req);
		//Assemble a request_record
		request_record * rec = (request_record *) pool_alloc(
				sizeof(request_record), 0);
		rec->rp_rec.req = req;
		rec->rp_rec.host_packed_buf = packed_buf_host;
		rec->rp_rec.dev_packed_buf = packed_buf;
		rec->rp_rec.buf_size = buf_leng * type_size;
		//and submit it
		register_mpi_request(rp_rec, rec);
		//once registered, all the params are copied, so the record can be freed
		pool_free(rec, sizeof(request_record), 0);
	} else {
		MPI_Recv(packed_buf_host, buf_leng, mpi_type, src_rank, tag,
				MPI_COMM_WORLD, &status);
		//Copy into the device buffer
		error |= meta_copy_h2d(packed_buf, packed_buf_host,
				buf_leng * type_size, 0, NULL);
		//free the host buffer
		//FIXME: remove associated async frees
		pool_free(packed_buf_host, buf_leng * type_size, 1);
	}
	//Close the if/else checking if mode == CUDA
#ifdef WITH_MPI_GPU_DIRECT
}
#endif // WITH_MPI_GPU_DIRECT
}

void sap_isend_cb(meta_callback * data) {
	sap_callback_payload * call_pl = (sap_callback_payload *)(data->data_payload);
	size_t type_size;
	MPI_Datatype mpi_type = get_mpi_type(call_pl->type, &type_size);
	if (call_pl->host_packed_buf != NULL) { //No GPUDirect
		MPI_Isend(call_pl->host_packed_buf, call_pl->buf_leng, mpi_type, call_pl->dst_rank, call_pl->tag, MPI_COMM_WORLD, call_pl->req);
	} else {
		FIXME(GPUDirect PAS triggered illegally);
		//GPUDirect send requires CUDA API calls, forbidden in callbacks
		MPI_Isend(call_pl->dev_packed_buf, call_pl->buf_leng, mpi_type, call_pl->dst_rank, call_pl->tag, MPI_COMM_WORLD, call_pl->req);
	}
	//Assemble a request
	request_record * rec = (request_record *) pool_alloc(sizeof(request_record), 0);
	rec->sap_rec.req = call_pl->req;
	rec->sap_rec.host_packed_buf = call_pl->host_packed_buf;
	rec->sap_rec.dev_packed_buf = call_pl->dev_packed_buf;
	rec->sap_rec.buf_size = type_size*call_pl->buf_leng;
	//and submit it
	register_mpi_request(sap_rec, rec);
	//once registered, all the params are copied, so the record can be freed
	pool_free(rec, sizeof(request_record), 0);
	//once the Isend is invoked and the request is submitted, we can remove the payload
	pool_free(call_pl, sizeof(sap_callback_payload), 0);
	pool_free(data, sizeof(meta_callback), 0);
}

void sap_helper(request_record * sap_request) {
	//printf("SAP_HELPER FIRED!\n");
	if (sap_request->sap_rec.host_packed_buf != NULL)
		pool_free(sap_request->sap_rec.host_packed_buf,
				sap_request->sap_rec.buf_size, 1);
	//meta_free(sap_request->sap_rec.dev_packed_buf);
}

//A wrapper that, provided a face, will pack it, then send it
a_err meta_mpi_pack_and_send_face(a_dim3 *grid_size, a_dim3 *block_size,
		int dst_rank, meta_face * face, void * buf,
		void * packed_buf, int tag, MPI_Request *req, meta_type_id type,
		int async) {
	//allocate space for a packed buffer
	size_t size = 1;
	int i;
	for (i = 0; i < face->count; i++)
		size *= face->size[i];
	//FIXME: User app is now responsible for allocing and managing these
	//void *packed_buf
	void *packed_buf_host;
	size_t type_size;
	MPI_Datatype mpi_type = get_mpi_type(type, &type_size);
	//FIXME: remove this, and any associated frees, app is now responsible
	a_err error = 0;
	//a_err error = meta_alloc(&packed_buf, size*type_size);
	// call the device pack function
	//FIXME: add grid, block params
	//FIXME: Add proper callback-related functionality to allwo the pack and copy to be asynchronous

	packed_buf_host = pool_alloc(size * type_size, 1);

	//do a send, if asynchronous, this should be a callback for when the async D2H copy finishes
#ifdef WITH_MPI_GPU_DIRECT
	//GPUDirect version
	if (run_mode == metaModePreferCUDA) {
		//FIXME: add other data types
		if(async) {
			//TODO: Move this into a callback for meta_pack_face_cb
			meta_callback * call = (meta_callback*) pool_alloc(sizeof(meta_callback), 0);
			call->callback_func = sap_isend_cb;
			call->callback_mode = metaModePreferCUDA;
			//Assemble the callback's necessary data elements
			sap_callback_payload * call_pl = (sap_callback_payload *) pool_alloc(sizeof(sap_callback_payload), 0);
			FIXME(GPUDirect PAS needs host buffer now);
			call_pl->host_packed_buf = packed_buf_host;
			call_pl->dev_packed_buf = packed_buf;
			call_pl->buf_leng = size;
			call_pl->type = type;
			call_pl->dst_rank = dst_rank;
			call_pl->tag = tag;
			call_pl->req = req;
			//fprintf(stderr, "SAP Callback PL: %p %p %d %d %p\n", call_pl->host_packed_buf, call_pl->dev_packed_buf, call_pl->buf_leng, call_pl->tag, call_pl->req);
			meta_pack_face(grid_size, block_size, packed_buf, buf, face, type, 1);
			meta_copy_d2h_cb(packed_buf_host, packed_buf, size*type_size, 1, call, call_pl);
			//This was no longer valid after switching to host staging, as required by GPUDirect/CUDACallback interaction
			//meta_pack_face_cb(grid_size, block_size, packed_buf, buf, face, type, 1, call, call_pl);
		} else {
			error |= meta_pack_face(grid_size, block_size, packed_buf, buf, face, type, 0);
			MPI_Send(packed_buf, size, mpi_type, dst_rank, tag, MPI_COMM_WORLD);
			//FIXME: remove associated async free
			//meta_free(packed_buf);
		}
	} else
	{
#endif //WITH_MPI_GPU_DIRECT

	//DO all variants via copy to host
	if (async) {
		meta_pack_face(grid_size, block_size, packed_buf, buf, face, type,
				1, NULL, NULL, NULL, NULL);
		//Figure out whether to use CUDA or OpenCL callback
		meta_callback * call = (meta_callback*) pool_alloc(
				sizeof(meta_callback), 0);
		call->callback_mode = run_mode;
		call->callback_func = sap_isend_cb;
		//Assemble the callback's necessary data elements
		sap_callback_payload * call_pl = (sap_callback_payload *) pool_alloc(
				sizeof(sap_callback_payload), 0);
		call_pl->host_packed_buf = packed_buf_host;
		call_pl->dev_packed_buf = packed_buf;
		call_pl->buf_leng = size;
		call_pl->type = type;
		call_pl->dst_rank = dst_rank;
		call_pl->tag = tag;
		call_pl->req = req;
		call->data_payload = call_pl;
		meta_copy_d2h_cb(packed_buf_host, packed_buf, size * type_size, 1, call, NULL);
	} else {
		error |= meta_pack_face(grid_size, block_size, packed_buf, buf, face,
				type, 0, NULL, NULL, NULL, NULL);
		error |= meta_copy_d2h(packed_buf_host, packed_buf, size * type_size,
				0, NULL);
		MPI_Send(packed_buf_host, size, mpi_type, dst_rank, tag,
				MPI_COMM_WORLD);
		//FIXME: remove asynchronous frees too
		//	meta_free(packed_buf);
		pool_free(packed_buf_host, size * type_size, 1);
	}
	//TODO: once async callback is implemented free malloced buffer	
	//TODO: add callback for asyncs
	//close the if/else for checking if mode == CUDA
#ifdef WITH_MPI_GPU_DIRECT
}
#endif //WITH_MPI_GPU_DIRECT
	//FIXME: Free device packed_buf correctly (with callback if async)
}

void rap_freebufs_cb(meta_callback * data) {
	rap_callback_payload * call_pl = (rap_callback_payload *)(data->data_payload);
	if (call_pl->host_packed_buf != NULL) pool_free(call_pl->host_packed_buf, call_pl->buf_size, 1);
	//free(call_pl->packed_buf);
	pool_free(call_pl, sizeof(rap_callback_payload), 0);
	pool_free(data, sizeof(meta_callback), 0);
}

void rap_helper(request_record *rap_request) {
	//printf("RAP_HELPER FIRED!\n");
	//async H2D copy w/ callback to free the temp host buffer
	//set up the mode_specific pointer to our free wrapper
	meta_callback * call = (meta_callback*) pool_alloc(sizeof(meta_callback),
			0);
	call->callback_mode = run_mode;
	call->callback_func = rap_freebufs_cb;
	//and invoke the async copy with the callback specified and the host pointer as the payload
	rap_callback_payload * call_pl = (rap_callback_payload *) pool_alloc(
			sizeof(rap_callback_payload), 0);
	call_pl->host_packed_buf = NULL;
	if (rap_request->rap_rec.host_packed_buf != NULL) {
		meta_copy_h2d(rap_request->rap_rec.dev_packed_buf,
				rap_request->rap_rec.host_packed_buf,
				rap_request->rap_rec.buf_size, 1, NULL);
		call_pl->host_packed_buf = rap_request->rap_rec.host_packed_buf;
		call_pl->buf_size = rap_request->rap_rec.buf_size;
	}
	call_pl->packed_buf = rap_request->rap_rec.dev_packed_buf;
	call->data_payload = (void*) call_pl;
	//check that the "NULL grid/block" flags aren't set with ternaries
	meta_unpack_face_cb(
			(rap_request->rap_rec.grid_size[0] == 0 ?
					NULL : &(rap_request->rap_rec.grid_size)),
			(rap_request->rap_rec.block_size[0] == 0 ?
					NULL : &(rap_request->rap_rec.block_size)),
			rap_request->rap_rec.dev_packed_buf, rap_request->rap_rec.dev_buf,
			rap_request->rap_rec.face, rap_request->rap_rec.type, 1, call, NULL, NULL, NULL, NULL);
}

//A wrapper that, provided a face, will receive a buffer, and unpack it
a_err meta_mpi_recv_and_unpack_face(a_dim3 * grid_size, a_dim3 * block_size,
		int src_rank, meta_face * face, void * buf,
		void * packed_buf, int tag, MPI_Request * req, meta_type_id type,
		int async) {
	//allocate space to receive the packed buffer
	size_t size = 1;
	int i;
	for (i = 0; i < face->count; i++)
		size *= face->size[i];
	//FIXME: User app is now responsible for managing these
	//void *packed_buf, 
	void *packed_buf_host;
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
	a_err error = 0;
	//a_err error = meta_alloc(&packed_buf, size*type_size);
	MPI_Status status; //FIXME: Should this be a function param?
#ifdef WITH_MPI_GPU_DIRECT
	//GPUDirect version
	if (run_mode == metaModePreferCUDA) {
		if(async) {
			MPI_Irecv(packed_buf, size, mpi_type, src_rank, tag, MPI_COMM_WORLD, req);
			//Helper will recognize the null, and not do h2d transfer, but
			// will invoke unpack kernel
			//Assemble a request_record
			request_record * rec = (request_record *) pool_alloc(sizeof(request_record), 0);
			rec->rap_rec.req = req;
			rec->rap_rec.host_packed_buf = NULL;
			rec->rap_rec.dev_packed_buf = packed_buf;
			rec->rap_rec.dev_buf = buf;
			rec->rap_rec.buf_size = size*type_size;
			if (grid_size != NULL)
			memcpy(&(rec->rap_rec.grid_size), grid_size, sizeof(a_dim3));
			else rec->rap_rec.grid_size[0] = NULL;
			if (block_size != NULL)
			memcpy(&(rec->rap_rec.block_size), block_size, sizeof(a_dim3));
			else rec->rap_rec.block_size[0] = NULL;
			rec->rap_rec.type = type;
			rec->rap_rec.face = face;
			//and submit it
			register_mpi_request(rap_rec, rec);
			//once registered, all the params are copied, so the record can be freed
			pool_free(rec, sizeof(request_record), 0);
		} else {
			MPI_Recv(packed_buf, size, mpi_type, src_rank, tag, MPI_COMM_WORLD, &status);
		}
	} else
	{
#endif //WITH_MPI_GPU_DIRECT

	//Non-GPUDirect version
	packed_buf_host = pool_alloc(size * type_size, 1);
	//do a receive, if asynchronous, we need to implement a callback to launch the H2D copy
	//FIXME: Add proper callback-related functionality to allow the pack and copy to be asynchronous
	if (async) {
		MPI_Irecv(packed_buf_host, size, mpi_type, src_rank, tag,
				MPI_COMM_WORLD, req);
		//TODO: Register the transfer, with host pointer set to packed_buf_host
		//Helper will recognize the non-null pointer and do an h2d transfer and unpack
		// with necessary callbacks to free both temp buffers
		request_record * rec = (request_record *) pool_alloc(
				sizeof(request_record), 0);
		rec->rap_rec.req = req;
		rec->rap_rec.host_packed_buf = packed_buf_host;
		rec->rap_rec.dev_packed_buf = packed_buf;
		rec->rap_rec.dev_buf = buf;
		rec->rap_rec.buf_size = size * type_size;
		//If they've provided a grid_size, copy it
		if (grid_size != NULL)
			memcpy(&(rec->rap_rec.grid_size), grid_size, sizeof(a_dim3));
		//Otherwise, set the x element to zero as a flag
		else
			rec->rap_rec.grid_size[0] = 0;
		//If they've provided a block_size, copy it
		if (block_size != NULL)
			memcpy(&(rec->rap_rec.block_size), block_size, sizeof(a_dim3));
		//Otherwise, set the x element to zero as a flag
		else
			rec->rap_rec.block_size[0] = 0;
		rec->rap_rec.type = type;
		rec->rap_rec.face = face;
		//and submit it
		register_mpi_request(rap_rec, rec);
		//once registered, all the params are copied, so the record can be freed
		pool_free(rec, sizeof(request_record), 0);
	} else {
		//receive the buffer
		MPI_Recv(packed_buf_host, size, mpi_type, src_rank, tag, MPI_COMM_WORLD,
				&status);
		//copy the buffer to the device
		error |= meta_copy_h2d(packed_buf, packed_buf_host, size * type_size,
				0, NULL);
		//free the host temp buffer
		pool_free(packed_buf_host, size * type_size, 1);
		//Unpack on the device
		error |= meta_unpack_face(grid_size, block_size, packed_buf, buf,
				face, type, 0, NULL, NULL, NULL, NULL);
		//free the device temp buffer
		//FIXME: remove async frees
		//meta_free(packed_buf);
	}
	//TODO: once async callback is implemented free malloced buffer
	//TODO: add callback for asyncs
	//close the if/else for checking if mode == CUDA
#ifdef WITH_MPI_GPU_DIRECT
}
#endif //WITH_MPI_GPU_DIRECT
	//FIXME: Free device packed_buf correctly (with callback if async)
}

