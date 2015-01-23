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

#ifndef METAMORPH_MPI_H
#define METAMORPH_MPI_H
#include <mpi.h>

//Include metamorph.h to grab necessary pieces from the library,
// cores, other plugins, and headers required by cores
#ifndef METAMORPH_H
	#include "metamorph.h"
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
	a_dim3 grid_size;
	a_dim3 block_size;
	meta_type_id type;
	meta_2d_face_indexed *face;
	void *dev_buf;
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
	uninit = -1,
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
	struct recordQueueNode *next;
} recordQueueNode;

//The data elements needed for the send_packed callback
// for the initial D2H copy
//Allows invoking the MPI_Isend and registering the request
// to clean up the temp host memory after Isend completes
typedef struct sp_callback_payload {
	void * host_packed_buf;
	size_t buf_leng;
	MPI_Datatype mpi_type;
	int dst_rank;
	int tag;
	MPI_Request *req;
}sp_callback_payload;

typedef struct sap_callback_payload {
	void * host_packed_buf;
	void * dev_packed_buf;
	size_t buf_leng;
	MPI_Datatype mpi_type;
	int dst_rank;
	int tag;
	MPI_Request *req;
}sap_callback_payload;

typedef struct rap_callback_payload {
	void * host_packed_buf;
	void * packed_buf;
}rap_callback_payload;

void init_record_queue();
//Helper functions for async transfers

//The enqueue function
void register_mpi_request(request_record_type type, request_record * request); 

//The Dequeue function
void help_mpi_request();

//A "forced wait until all requests finish" helper
// Meant to be used with meta_flush and meta_finish
void finish_mpi_requests();

void rp_helper(request_record *rp_request);
void sp_helper(request_record *sp_request);
void rap_helper(request_record *rap_request);
void sap_helper(request_record *sap_request);

//FIXME: Paul 2014.06.30 - HOST ONLY TRANSFERS AREN'T OUR JOB
// all library-implemented transfers assume device buffers!!

//Compound transfers, reliant on other library functions
//Essentially a wrapper for a contiguous buffer send
a_err meta_mpi_packed_face_send();

//Essentially a wrapper for a contiguous buffer receive
a_err meta_mpi_packed_face_recv();

//A wrapper that, provided a face, will pack it, then send it
a_err meta_mpi_pack_and_send_face();

//A wrapper that, provided a face, will receive a buffer, and unpack it
a_err meta_mpi_recv_and_unpack_face();

extern meta_preferred_mode run_mode;
#endif //METAMORPH_MPI_H
