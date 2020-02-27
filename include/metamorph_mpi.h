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

/** The top-level user APIs **/
#ifndef METAMORPH_MPI_H
#define METAMORPH_MPI_H
#include <mpi.h>

//Include metamorph.h to grab necessary pieces from the library,
// cores, other plugins, and headers required by cores
#ifndef METAMORPH_H
#include "metamorph.h"
#endif

//POOL_SIZE must be a power of two for quick masks instead of modulo.
// Behavior is left undefined if this is violated.
#ifndef META_MPI_POOL_SIZE
#define META_MPI_POOL_SIZE 16
#endif

//To keep the structure lightweight, it's a simple ring array.
// This param specifies how many previous ring slots we should look
// at for one of appropriate size before just reallocating the current one.
// The default is half the size.
//This lookback is part of the "ratchet" behavior of the ring.
#ifndef META_MPI_POOL_LOOKBACK
#define META_MPI_POOL_LOOKBACK (META_MPI_POOL_SIZE>>1)
#endif

//Should never be user-edited, it's just for masking the bits needed for
// fast indexing
#define POOL_MASK (META_MPI_POOL_SIZE - 1)

//Any special concerns

#ifdef WITH_OPENMP
#endif

//This pool manages host staging buffers
//These arrays are kept in sync to store the pointer and size of each buffer
extern void * host_buf_pool[META_MPI_POOL_SIZE];
//The size array moonlights as an occupied flag if size !=0
extern unsigned long host_buf_pool_size[META_MPI_POOL_SIZE];
extern unsigned long host_pool_token;

//This separate pool manages internal buffers for requests/callback payloads
extern void * internal_pool[META_MPI_POOL_SIZE];
extern unsigned long internal_pool_size[META_MPI_POOL_SIZE];
extern unsigned long internal_pool_token;

void * pool_alloc(size_t size, int isHost);
void pool_free(void * bufi, size_t size, int isHost);

//Helper records for async transfers
typedef struct {
	MPI_Request * req;
	void * host_packed_buf;
	size_t buf_size;
} send_packed_record;

typedef struct {
	MPI_Request *req;
	void *host_packed_buf;
	size_t buf_size;
	void *dev_packed_buf;
} recv_packed_record;

typedef struct {
	MPI_Request *req;
	void *host_packed_buf;
	size_t buf_size;
	void *dev_packed_buf;
} pack_and_send_record;

//FIXME: Do we need to take copies of grid, block, and face, so the user can
// free/reuse theirs after the call and we can safely get rid of our copies at dequeue time?
typedef struct {
	MPI_Request *req;
	void *host_packed_buf;
	size_t buf_size;
	void *dev_packed_buf;
	a_dim3 grid_size;
	a_dim3 block_size;
	meta_type_id type;
	meta_face *face;
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
	uninit = -1, sentinel, sp_rec, rp_rec, sap_rec, rap_rec
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
	meta_type_id type;
	int dst_rank;
	int tag;
	MPI_Request *req;
} sp_callback_payload;

typedef struct sap_callback_payload {
	void * host_packed_buf;
	void * dev_packed_buf;
	size_t buf_leng;
	meta_type_id type;
	int dst_rank;
	int tag;
	MPI_Request *req;
} sap_callback_payload;

typedef struct rap_callback_payload {
	void * host_packed_buf;
	void * packed_buf;
	size_t buf_size;
} rap_callback_payload;

typedef struct rp_callback_payload {
	void * host_packed_buf;
	size_t buf_size;
} rp_callback_payload;

//Call this first
// sets up MPI_Init, and all MetaMorph specific MPI features
void meta_mpi_init(int *argc, char *** argv);

//Call this last
// finalizes all MetaMorph specific MPI features
// and calls MPI_finalize
void meta_mpi_finalize();

//setup the record queue sentinel
void init_record_queue();
//Helper functions for async transfers

//The enqueue function
void register_mpi_request(request_record_type type, request_record * request);

//The Dequeue function
void help_mpi_request();

//A "forced wait until all requests finish" helper
// Meant to be used with meta_flush and meta_finish
a_err finish_mpi_requests();

void rp_helper(request_record *rp_request);
void sp_helper(request_record *sp_request);
void rap_helper(request_record *rap_request);
void sap_helper(request_record *sap_request);

//Returns the MPI_Datatype, (and the type's size in size, if a non-NULL pointer is provided
MPI_Datatype get_mpi_type(meta_type_id type, size_t * size);

//FIXME: Paul 2014.06.30 - HOST ONLY TRANSFERS AREN'T OUR JOB
// all library-implemented transfers assume device buffers!!

//Compound transfers, reliant on other library functions
//Essentially a wrapper for a contiguous buffer send
a_err meta_mpi_packed_face_send(int dst_rank, void * packed_buf,
		size_t buf_leng, int tag, MPI_Request * req, meta_type_id type,
		int async);

//Essentially a wrapper for a contiguous buffer receive
a_err meta_mpi_packed_face_recv(int src_rank, void * packed_buf,
		size_t buf_leng, int tag, MPI_Request * req, meta_type_id type,
		int async);

//A wrapper that, provided a face, will pack it, then send it
a_err meta_mpi_pack_and_send_face(a_dim3 * grid_size, a_dim3 * block_size,
		int dst_rank, meta_face * face, void * buf,
		void * packed_buf, int tag, MPI_Request * req, meta_type_id type,
		int async);

//A wrapper that, provided a face, will receive a buffer, and unpack it
a_err meta_mpi_recv_and_unpack_face(a_dim3 * grid_size, a_dim3 * block_size,
		int src_rank, meta_face * face, void * buf,
		void * packed_buf, int tag, MPI_Request * req, meta_type_id type,
		int async);

extern meta_preferred_mode run_mode;
#endif //METAMORPH_MPI_H
