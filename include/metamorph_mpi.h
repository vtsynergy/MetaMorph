/** \file
 * Exposed functions and types for MetaMorph's MPI interoperability plugin
 */

/** The top-level user APIs **/
#ifndef METAMORPH_MPI_H
#define METAMORPH_MPI_H
#include <mpi.h>

// Include metamorph.h to grab necessary pieces from the library,
// cores, other plugins, and headers required by cores
#ifndef METAMORPH_H
#include "metamorph.h"
#endif

/**
 * Allocate a buffer from the MPI plugin's current pool
 * Intended for internal use to reuse staging buffers and internal data
 * structures
 * \param size The size of the buffer needed
 * \param isHost Whether the buffer is a large host staging buffer (true) or a
 * small internal data structure (small)
 */
void *pool_alloc(size_t size, int isHost);
/**
 * Release a buffer back to the MPI plugin's current pool
 * Intended for internal use to reuse staging buffers and internal data
 * structures
 * \param buf The buffer to return
 * \param size How big the buffer being returned was
 * \param isHost Whether the buffer is a large host staging buffer (true) or a
 * small internal data structure (small)
 */
void pool_free(void *buf, size_t size, int isHost);

/**
 * A helper record to manage asynchronous sends for send_packed
 */
typedef struct {
  /** The request returned from MPI_ISend */
  MPI_Request *req;
  /** The host staging buffer */
  void *host_packed_buf;
  /** The size of the sent buffer */
  size_t buf_size;
} send_packed_record;

/**
 * A helper record to manage asynchronous receives for recv_packed
 */
typedef struct {
  /** The request returned from MPI_IRecv */
  MPI_Request *req;
  /** The host staging buffer */
  void *host_packed_buf;
  /** The size of the received buffer */
  size_t buf_size;
  /** The device staging buffer */
  void *dev_packed_buf;
} recv_packed_record;

/**
 * A helper record to manage asynchronous sends for pack_and_send
 */
typedef struct {
  /** The request returned from MPI_ISend */
  MPI_Request *req;
  /** The host staging buffer */
  void *host_packed_buf;
  /** The size of the sent buffer */
  size_t buf_size;
  /** The device staging buffer */
  void *dev_packed_buf;
} pack_and_send_record;

//\todo FIXME: Do we need to take copies of compound operations' grid, block,
//and face, so the user can
// free/reuse theirs after the call and we can safely get rid of our copies at
// dequeue time?
/**
 * A helper record to manage asynchronous receives for recv_and_unpack
 */
typedef struct {
  /** The request returned from MPI_IRecv */
  MPI_Request *req;
  /** The host staging buffer */
  void *host_packed_buf;
  /** The size of the received buffer */
  size_t buf_size;
  /** The device staging buffer */
  void *dev_packed_buf;
  /** The grid size to use for the eventual unpacking kernel */
  a_dim3 grid_size;
  /** The block size to use for the eventual unpacking kernel */
  a_dim3 block_size;
  /** The data type to use for the eventual unpacking kernel */
  meta_type_id type;
  /** The face to eventually unpack the buffer to */
  meta_face *face;
  /** The device 3D buffer to eventually unpack the buffer to */
  void *dev_buf;
} recv_and_unpack_record;

/**
 * Union of the asynchronous transfer records to simplify queue implementation
 */
typedef union {
  /** The send_packed record */
  send_packed_record sp_rec;
  /** The recv_packed record */
  recv_packed_record rp_rec;
  /** The pack_and_send record */
  pack_and_send_record sap_rec;
  /** The recv_and_unpack record */
  recv_and_unpack_record rap_rec;
} request_record;

/**
 * Enum which tells the queue node consumer what type of record to treat the
 * union as
 */
typedef enum {
  uninit = -1,
  sentinel,
  sp_rec,
  rp_rec,
  sap_rec,
  rap_rec
} request_record_type;

/**
 * The nodes used in the record queue to manage all async requests
 */
typedef struct recordQueueNode {
  /** The type of the record for internal casting */
  request_record_type type;
  /** The actual record data */
  request_record record;
  /** Pointer to the next request */
  struct recordQueueNode *next;
} recordQueueNode;

/**
 * The data elements needed for the send_packed callback for the initial D2H
 * copy
 */
typedef struct sp_callback_payload {
  /** The host-side staging buffer */
  void *host_packed_buf;
  /** The length of the buffer to transfer */
  size_t buf_leng;
  /** The type of data being transfered */
  meta_type_id type;
  /** The rank of the receiver */
  int dst_rank;
  /** The MPI tag for the transfer, provided by the application */
  int tag;
  /** The MPI Request corresponding to the copy */
  MPI_Request *req;
} sp_callback_payload;

/**
 * The data elements needed for the pack_and_send callback for the initial pack
 * kernel
 */
typedef struct sap_callback_payload {
  /** The host-side staging buffer */
  void *host_packed_buf;
  /** The device-side packed staging buffer */
  void *dev_packed_buf;
  /** The length of the buffer to transfer */
  size_t buf_leng;
  /** The type of data being transfered */
  meta_type_id type;
  /** The rank of the receiver */
  int dst_rank;
  /** The MPI tag for the transfer, provided by the application */
  int tag;
  /** The MPI Request corresponding to the copy */
  MPI_Request *req;
} sap_callback_payload;

/**
 * The data elements needed for the recv_and_unpack MPI callback to transfer to
 * the device then unpack
 */
typedef struct rap_callback_payload {
  /** The host-side staging buffer */
  void *host_packed_buf;
  /** The device-side packed staging buffer */
  void *packed_buf;
  /** The size of the transferred buffer */
  size_t buf_size;
} rap_callback_payload;

/**
 * The data elements needed for the recv_packed MPI callback to transfer to the
 * device
 */
typedef struct rp_callback_payload {
  /** The host-side staging buffer */
  void *host_packed_buf;
  /** The size of the transferred buffer */
  size_t buf_size;
} rp_callback_payload;

/**
 * Call this first
 * sets up MPI_Init, and all MetaMorph specific MPI features
 * \param argc The number of command line args provided to main (will be
 * modified by MPI_Init)
 * \param argv The vector of command line character strings provided to main
 * (will be modified by MPI_Init)
 */
void meta_mpi_init(int *argc, char ***argv);

/**
 * Call this last
 * finalizes all MetaMorph specific MPI features and calls MPI_finalize
 */
void meta_mpi_finalize();

/** Setup the record queue sentinel internally */
void init_record_queue();

/**
 * Enqueue the MPI request to be serviced later
 * \param type What type of MPI request this record is
 * \param request The request payload
 */
void register_mpi_request(request_record_type type, request_record *request);

/**
 * The Dequeue function, check if the oldest MPI request has finished, and if
 * so, move it along Does not permit out-of-order operations, will not do
 * anything until the oldest request is done
 */
void help_mpi_request();

/**
 * A "forced wait until all requests finish" helper
 * Meant to be used with meta_flush and meta_finish
 */
void finish_mpi_requests();

/**
 * Internal, called by help_mpi_request
 * Help a recv_packed operation along
 * \param rp_request The request to help, which has already been dequeued
 */
void rp_helper(request_record *rp_request);
/**
 * Internal, called by help_mpi_request
 * Help a send_packed operation along
 * \param sp_request The request to help, which has already been dequeued
 */
void sp_helper(request_record *sp_request);
/**
 * Internal, called by help_mpi_request
 * Help a recv_and_unpack operation along
 * \param rap_request The request to help, which has already been dequeued
 */
void rap_helper(request_record *rap_request);
/**
 * Internal, called by help_mpi_request
 * Help a pack_and_send operation along
 * \param sap_request The request to help, which has already been dequeued
 */
void sap_helper(request_record *sap_request);

/**
 * Provide a wrapper to query the rank so that we both don't have to care which
 * MPI library the backend is built against for dlopen We also don't have to
 * know/care what communicator the MPI plugin is using (if it changes in the
 * future)
 * \param rank An address to return the rank in
 * \return the status of MPI_Comm_rank if MPI is not shut down yet, -1 otherwise
 */
a_err metaMPIRank(int *rank);
/**
 * Get the MPI data type
 * \param type the MetaMorph type to re-interpret
 * \param size A pointer to be filled with the sizeof the MPI data type, or NULL
 * if not needed
 * \return the corresponding MPI_Dataype
 */
MPI_Datatype get_mpi_type(meta_type_id type, size_t *size);

/**
 * Send a MetaMorph device buffer to another process
 * Typically this is implemented via transparent host staging
 * \param dst_rank The receiver's rank in MPI_COMM_WORLD
 * \param packed_buf The MetaMorph-allocated device buffer to transfer (must be
 * on current MetaMorph device)
 * \param buf_leng The length of the buffer to transfer (number of elements, not
 * size)
 * \param tag A tag to apply to the transfer
 * \param req An address to return a request corresponding to the underlying
 * MPI_Send/MPI_Isend
 * \param type The MetaMorph type the buffer contains
 * \param async Whether the transfer should be performed asynchronously (both
 * the device-to-host, and process-to-process), or not
 */
void meta_mpi_packed_face_send(int dst_rank, void *packed_buf, size_t buf_leng,
                               int tag, MPI_Request *req, meta_type_id type,
                               int async);

/**
 * Receive a MetaMorph device buffer from another process
 * Typically this is implemented via transparent host staging
 * \param src_rank The sender's rank in MPI_COMM_WORLD
 * \param packed_buf The MetaMorph-allocated device buffer to receive to (must
 * be on current MetaMorph device)
 * \param buf_leng The length of the buffer to receive (number of elements, not
 * size)
 * \param tag A tag to match to the receive
 * \param req An address to return a request corresponding to the underlying
 * MPI_Recv/MPI_Irecv
 * \param type The MetaMorph type the buffer contains
 * \param async Whether the transfer should be performed asynchronously (and
 * process-to-process, and host-to-device), or not
 */
void meta_mpi_packed_face_recv(int src_rank, void *packed_buf, size_t buf_leng,
                               int tag, MPI_Request *req, meta_type_id type,
                               int async);

/**
 * Pack a face slab of a 3D MetaMorph device buffer, then send it to another
 * process Typically this is implemented via transparent host staging
 * \param grid_size The number of blocks to use for the pack (pack kernels are
 * 1D)
 * \param block_size The muber of threads to use within each block of the pack
 * (pack kernels are 1D)
 * \param dst_rank The receiver's rank in MPI_COMM_WORLD
 * \param face The face specification of the 3D region to send
 * \param buf The MetaMorph-allocated 3D device buffer to pack from (must be on
 * current MetaMorph device)
 * \param packed_buf A MetaMorph-allocated device buffer of sufficient size to
 * fit the whole packed face for  transfer (must be on current MetaMorph device)
 * \param tag A tag to apply to the transfer
 * \param req An address to return a request corresponding to the underlying
 * MPI_Send/MPI_Isend
 * \param type The MetaMorph type the buffer contains
 * \param async Whether the transfer should be performed asynchronously (The
 * pack kernel, device-to-host transfer, and process-to-process transfer), or
 * not
 */
void meta_mpi_pack_and_send_face(a_dim3 *grid_size, a_dim3 *block_size,
                                 int dst_rank, meta_face *face, void *buf,
                                 void *packed_buf, int tag, MPI_Request *req,
                                 meta_type_id type, int async);

/**
 * Receive a MetaMorph packed device buffer from another process, and unpack it
 * Typically this is implemented via transparent host staging
 * \param grid_size The number of blocks to use for the unpack (unpack kernels
 * are 1D)
 * \param block_size The muber of threads to use within each block of the unpack
 * (unpack kernels are 1D)
 * \param src_rank The sender's rank in MPI_COMM_WORLD
 * \param face The face specification of the 3D region to receive
 * \param buf The MetaMorph-allocated 3D device buffer to unpack to (must be on
 * current MetaMorph device)
 * \param packed_buf The MetaMorph-allocated device buffer to receive to (must
 * be on current MetaMorph device)
 * \param tag A tag to match to the receive
 * \param req An address to return a request corresponding to the underlying
 * MPI_Recv/MPI_Irecv
 * \param type The MetaMorph type the buffer contains
 * \param async Whether the transfer should be performed asynchronously (and
 * process-to-process, and host-to-device), or not
 */
void meta_mpi_recv_and_unpack_face(a_dim3 *grid_size, a_dim3 *block_size,
                                   int src_rank, meta_face *face, void *buf,
                                   void *packed_buf, int tag, MPI_Request *req,
                                   meta_type_id type, int async);

#endif // METAMORPH_MPI_H
