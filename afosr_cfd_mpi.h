/*
 * Implementation of all MPI-related functionality for the library
 *
 * MPI is considered a library PLUGIN, therefore it must be kept
 * entirely optional. The library should function (build) without skipping
 * a beat on single node/process configurations regardless of whether
 * the system has an MPI install available. It is implemented as a
 * plugin as a convenience, as all packing and D2H transfer functions
 * needed to implement an MPI exchange by hand are already provided.
 * This piece simply provides an abstracted, unified wrapper around
 * GPUDirect, via-host, and (potentially) other transfer primitives.
 *
 * Further, GPUDirect and other approaches to reducing/eliminating
 * staging of GPU buffers in the host before transfer are OPTIONAL
 * components of the MPI plugin. When suitable, direct transfers
 * should be be provided and supplant the host-staged versions via
 * a compiler option, such as "-D WITH_MPI_GPU_DIRECT".
 *
 * Regardless of whether GPUDirect modes are enabled, the user
 * application should not need to provide any temp/staging buffers,
 * or other supporting info. This should all be managed by the
 * plugin internally, and transparently.
*/

#ifndef AFOSR_CFD_MPI_H
#define AFOSR_CFD_MPI_H

//Include afosr_cfd.h to grab necessary pieces from the library,
// cores, other plugins, and headers required by cores
#ifndef AFOSR_CFD_H
	#include "afosr_cfd.h"
#endif

//Any special concerns
#ifdef WITH_CUDA
#endif

#ifdef WITH_OPENCL
#endif

#ifdef WITH_OPENMP
#endif

//Helper records for async transfers
typedef struct {
	MPI_Request * req;
	void * host_packed_buf;
} send_packed_record;

typedef struct {
	MPI_Request *req;
	void *host_packed_buf;
	void *dev_packed_buf;
	size_t buf_size;
} recv_packed_record;

typedef struct {
	MPI_Request *req;
	void *host_packed_buf;
	void *dev_packed_buf;
} pack_and_send_record;

//FIXME: Do we need to take copies of grid, block, and face, so the user can
// free/reuse theirs after the call and we can safely get rid of our copies at dequeue time?
typedef struct {
	MPI_Request *req;
	void *host_packed_buf;
	void *dev_packed_buf;
	size_t buf_size;
	a_dim3 *grid_size;
	a_dim3 *block_size;
	accel_type_id type;
	accel_2d_face_indexed *face;
} recv_and_unpack_record;

//Union needed to simplify queue node implementation
typedef union {
	send_packed_record sp_rec;
	recv_packed_record rp_rec;
	pack_and_send_record sap_rec;
	recv_and_unpack_record rap_rec;
} request_record;

//Enum which tells the node consumer what type of record to treat the union as
typedef enum {
	sentinel,
	sp_rec,
	rp_rec,
	sap_rec,
	rap_rec
} request_record_type;

//The record queue used to manage all async requests
typedef struct recordQueueNode {
	request_record_type type;
	request_record record;
	recordQueueNode *next;
} recordQueueNode;

recordQueueNode record_queue;
recordQueueNode *record_queue_tail;


//Helper functions for async transfers
void init_record_queue() {
	record_queue.type = sentinel;
	//No need to set the record, it won't be looked at
	record_queue.next = NULL;
	record_queue_tail = &record_queue;
}

//The enqueue function
void register_mpi_request(request_record_type type, request_record * request); 

//The Dequeue function
void help_mpi_request();

//A "forced wait until all requests finish" helper
// Meant to be used with accel_flush and accel_finish
void finish_mpi_requests();

//FIXME: Paul 2014.06.30 - HOST ONLY TRANSFERS AREN'T OUR JOB
// all library-implemented transfers assume device buffers!!

//Compound transfers, reliant on other library functions
//Essentially a wrapper for a contiguous buffer send
a_err accel_mpi_packed_face_send();

//Essentially a wrapper for a contiguous buffer receive
a_err accel_mpi_packed_face_recv();

//A wrapper that, provided a face, will pack it, then send it
a_err accel_mpi_pack_and_send_face();

//A wrapper that, provided a face, will receive a buffer, and unpack it
a_err accel_mpi_recv_and_unpack_face();

#endif //AFOSR_CFD_MPI_H
