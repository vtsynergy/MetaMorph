/** \file
 * OpenMP Backend exposed functions and types
 */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef METAMORPH_OPENMP_BACKEND_H
#define METAMORPH_OPENMP_BACKEND_H

#ifndef METAMORPH_H
#include "metamorph.h"
#endif

/** The OpenMP event type, for now it's just a mapping to a time management
 * struct */
typedef struct timespec openmpEvent;

#ifdef __OPENMPCC__
extern "C" {
#endif
/**
 * A simple wrapper around malloc or _mm_malloc to create an OpenMP buffer and
 * pass it back up to the backend-agnostic layer
 * \param ptr The address in which to return allocated buffer
 * \param size The number of bytes to allocate
 * \return -1 if the allocation failed, 0 if it succeeded
 */
a_err metaOpenMPAlloc(void **ptr, size_t size);
/**
 * Wrapper function around free/_mm_free to release a MetaMorph-allocated OpenMP
 * buffer
 * \param ptr The buffer to release
 * \return always returns 0 (success)
 */
a_err metaOpenMPFree(void *ptr);
/**
 * A wrapper for a OpenMP host-to-device copy
 * \param dst The destination buffer, a buffer allocated in MetaMorph's
 * currently-running OpenMP context
 * \param src The source buffer, a host memory region
 * \param size The number of bytes to copy from the host to the device
 * \param async whether the write should be asynchronous or blocking (currently
 * ignored, all transfers are synchronous)
 * \param call A callback to run when the transfer finishes, or NULL if none
 * \param ret_event The address of a meta_event with initialized openmpEvent[2]
 * payload in which to copy the events corresponding to the write back to
 * \return 0 on success
 * \todo FIXME implement OpenMP error codes
 */
a_err metaOpenMPWrite(void *dst, void *src, size_t size, a_bool async,
                      meta_callback *call, meta_event *ret_event);
/**
 * A wrapper for a OpenMP device-to-host copy
 * \param dst The destination buffer, a host memory region
 * \param src The source buffer, a buffer allocated in MetaMorph's
 * currently-running OpenMP context
 * \param size The number of bytes to copy from the device to the host
 * \param async whether the read should be asynchronous or blocking (currently
 * ignored, all transfers are synchronous)
 * \param call A callback to run when the transfer finishes, or NULL if none
 * \param ret_event The address of a meta_event with initialized openmpEvent[2]
 * payload in which to copy the events corresponding to the write back to
 * \return 0 on success
 * \todo FIXME implement OpenMP error codes
 */
a_err metaOpenMPRead(void *dst, void *src, size_t size, a_bool async,
                     meta_callback *call, meta_event *ret_event);
/**
 * A wrapper for a OpenMP device-to-device copy
 * \param dst The destination buffer, a buffer allocated in MetaMorph's
 * currently-running OpenMP context
 * \param src The source buffer, a buffer allocated in MetaMorph's
 * currently-running OpenMP context
 * \param size The number of bytes to copy
 * \param async whether the copy should be asynchronous or blocking (currently
 * ignored, all transfers are synchronous)
 * \param call A callback to run when the transfer finishes, or NULL if none
 * \param ret_event The address of a meta_event with initialized openmpEvent[2]
 * payload in which to copy the events corresponding to the write back to
 * \return 0 on success
 * \todo FIXME implement OpenMP error codes
 */
a_err metaOpenMPDevCopy(void *dst, void *src, size_t size, a_bool async,
                        meta_callback *call, meta_event *ret_event);
/**
 * Finish all outstanding OpenMP operations
 * \bug currently just an OpenMP barrier, all work is currently performed
 * synchronously
 * \return 0 on success
 */
a_err metaOpenMPFlush();
/**
 * Just a small wrapper around an openmpEvent allocator to keep the real
 * datatype exclusively inside the OpenMP backend
 * \param ret_event The address in which to save the pointer to the
 * newly-allocated openmpEvent[2]
 * \return 0 on success, -1 if the pointer is NULL
 * \todo FIXME Implement OpenMP error codes
 */
a_err metaOpenMPCreateEvent(void **ret_event);
/**
 * Just a small wrapper around a openmpEvent destructor to keep the real
 * datatype exclusively inside the OpenMP backend
 * \param event The address of the openmpEvent[2] to destroy
 * \return 0 on success, -1 if the pointer is already NULL
 */
a_err metaOpenMPDestroyEvent(void *event);
/**
 * A simple wrapper to get the elapsed time of a meta_event containing two
 * openmpEvents
 * \param ret_ms The address to save the elapsed time in milliseconds
 * \param event The meta_event (bearing a dynamically-allocated openmpEvent[2]
 * as its payload) to query
 * \return 0 on success, -1 of either the return pointer or the event payload is
 * NULL
 */
a_err metaOpenMPEventElapsedTime(float *ret_ms, meta_event event);
/**
 * Internal function to register a meta_callback with the OpenMP backend
 * \param call the meta_callback payload that should be invoked and filled when
 * triggered
 * \return 0 after returning from call
 * \bug \todo FIXME right now this doesn't register, it directly runs the function since callbacks are registered after running the kernel and all OpenMP kernels currently run synchronously
 */
a_err metaOpenMPRegisterCallback(meta_callback *call);

/**
 * Dot-product of identically-shaped subregions of two identically-shaped 3D
 * arrays this kernel works for 3D data only.
 * \param grid_size The number of blocks in each global dimension (outer loop
 * bounds, currently ignored)
 * \param block_size The dimensions of each block (inner loop bounds, currently
 * ignored)
 * \param data1 first input array (an OpenMP buffer residing on the
 * currently-active device)
 * \param data2 second input array (an OpenMP buffer residing on the
 * currently-active device)
 * \param array_size X, Y, and Z dimension sizes of the input arrays (must
 * match)
 * \param arr_start X, Y, and Z dimension start indicies for computing on a
 * subregion, to allow for a halo region
 * \param arr_end X, Y, and Z dimension end indicies for computing on a
 * subregion, to allow for a halo region
 * \param reduction_var the final dotproduct output (scalar) across all
 * workgroups, assumed to be initialized before the kernel (an OpenMP buffer
 * residing on the currently-active device)
 * \param type The supported MetaMorph data type that data1, data2, and
 * reduced_val contain (Currently: a_db, a_fl, a_ul, a_in, a_ui)
 * \param async Whether the kernel should be run asynchronously or blocking
 * (Currently NOOP, all OpenMP operations run synchronously)
 * \param call Register a callback to be automatically invoked when the kernel
 * finishes, or NULL if none
 * \param ret_event Address of an initialized event (with openmpEvent[2]
 * payload) in which to return the kernel's wrapping events
 * \return 0 on success
 * \todo implement OpenMP error codes
 */
a_err openmp_dotProd(size_t (*grid_size)[3], size_t (*block_size)[3],
                     void *data1, void *data2, size_t (*array_size)[3],
                     size_t (*arr_start)[3], size_t (*arr_end)[3],
                     void *reduction_var, meta_type_id type, int async,
                     meta_callback *call, meta_event *ret_event);
/**
 * Reduction sum of subregion of a 3D array
 * this kernel works for 3D data only.
 * \param grid_size The number of blocks in each global dimension (outer loop
 * bounds, currently ignored)
 * \param block_size The dimensions of each block (inner loop bounds, currently
 * ignored)
 * \param data input array (an OpenMP buffer residing on the currently-active
 * device)
 * \param array_size X, Y, and Z dimension sizes of the input arrays (must
 * match)
 * \param arr_start X, Y, and Z dimension start indicies for computing on a
 * subregion, to allow for a halo region
 * \param arr_end X, Y, and Z dimension end indicies for computing on a
 * subregion, to allow for a halo region
 * \param reduction_var the final dotproduct output (scalar) across all
 * workgroups, assumed to be initialized before the kernel (an OpenMP buffer
 * residing on the currently-active device)
 * \param type The supported MetaMorph data type that data1, data2, and
 * reduced_val contain (Currently: a_db, a_fl, a_ul, a_in, a_ui)
 * \param async Whether the kernel should be run asynchronously or blocking
 * (Currently NOOP, all OpenMP operations run synchronously)
 * \param call Register a callback to be automatically invoked when the kernel
 * finishes, or NULL if none
 * \param ret_event Address of an initialized event (with openmpEvent[2]
 * payload) in which to return the kernel's wrapping events
 * \return 0 on success
 * \todo implement OpenMP error codes
 */
a_err openmp_reduce(size_t (*grid_size)[3], size_t (*block_size)[3], void *data,
                    size_t (*array_size)[3], size_t (*arr_start)[3],
                    size_t (*arr_end)[3], void *reduction_var,
                    meta_type_id type, int async, meta_callback *call,
                    meta_event *ret_event);
/**
 * OpenMP wrapper to transpose a 2D array
 * \param grid_size The number of blocks in each global dimension (outer loop
 * bounds, currently ignored)
 * \param block_size The dimensions of each block (inner loop bounds, currently
 * ignored)
 * \param indata The input untransposed 2D array
 * \param outdata The output transposed 2D array
 * \param arr_dim_xy the X and Y dimensions of indata, Z is ignored
 * \param tran_dim_xy the X and Y dimensions of outdata, Z is ignored (currently
 * ignored)
 * \param type The data type to invoke the appropriate kernel for
 * \param async Whether the kernel should be run asynchronously or blocking
 * (Currently NOOP, all OpenMP operations run synchronously)
 * \param call Register a callback to be automatically invoked when the kernel
 * finishes, or NULL if none
 * \param ret_event Address of an initialized event (with openmpEvent[2]
 * payload) in which to return the kernel's wrapping events
 * \return 0 on success
 * \todo implement OpenMP error codes
 */
a_err openmp_transpose_face(size_t (*grid_size)[3], size_t (*block_size)[3],
                            void *indata, void *outdata,
                            size_t (*arr_dim_xy)[3], size_t (*tran_dim_xy)[3],
                            meta_type_id type, int async, meta_callback *call,
                            meta_event *ret_event);
/**
 * OpenMP wrapper for the face packing kernel
 * \param grid_size The number of blocks in each global dimension (outer loop
 * bounds, currently ignored)
 * \param block_size The dimensions of each block (inner loop bounds, currently
 * ignored)
 * \param packed_buf The output packed array
 * \param buf The input full array
 * \param face The face/slab to extract, returned from make_slab_from_3d
 * \param remain_dim At each subsequent layer of the face struct, the
 * *cummulative* size of all remaining descendant dimensions. Used for computing
 * the per-thread write offset, typically automatically pre-computed in the
 * backend-agnostic MetaMorph wrapper, meta_pack_face
 * \param type The type of data to run the kernel on
 * \param async Whether the kernel should be run asynchronously or blocking
 * (Currently NOOP, all OpenMP operations run synchronously)
 * \param call Register a callback to be automatically invoked when the kernel
 * finishes, or NULL if none
 * \param ret_event_k1 Address of an initialized event (with openmpEvent[2]
 * payload) in which to return the kernel event used for asynchronous calls and
 * timing
 * \param ret_event_c1 Address of an initialized event (with openmpEvent[2]
 * payload) in which to return the event for copying face->size used for
 * asynchronous calls and timing (currently unused in OpenMP)
 * \param ret_event_c2 Address of an initialized event (with openmpEvent[2]
 * payload) in which to return the event for copying face->stride used for
 * asynchronous calls and timing (currently unused in OpenMP)
 * \param ret_event_c3 Address of an initialized event (with openmpEvent[2]
 * payload) in which to return the event for copying remain_dim to
 * c_face_child_size event used for asynchronous calls and timing (currently
 * unused in OpenMP)
 * \return 0 on success
 * \todo implement OpenMP error codes
 * \warning Implemented as a 1D kernel, Y and Z grid/block parameters will be
 * ignored
 */
a_err openmp_pack_face(size_t (*grid_size)[3], size_t (*block_size)[3],
                       void *packed_buf, void *buf, meta_face *face,
                       int *remain_dim, meta_type_id type, int async,
                       meta_callback *call, meta_event *ret_event_k1,
                       meta_event *ret_event_c1, meta_event *ret_event_c2,
                       meta_event *ret_event_c3);
/**
 * OpenMP wrapper for the face unpacking kernel
 * \param grid_size The number of blocks in each global dimension (outer loop
 * bounds, currently ignored)
 * \param block_size The dimensions of each block (inner loop bounds, currently
 * ignored)
 * \param packed_buf The input packed array
 * \param buf The output full array
 * \param face The face/slab to extract, returned from make_slab_from_3d
 * \param remain_dim At each subsequent layer of the face struct, the
 * *cummulative* size of all remaining descendant dimensions. Used for computing
 * the per-thread write offset, typically automatically pre-computed in the
 * backend-agnostic MetaMorph wrapper, meta_pack_face
 * \param type The type of data to run the kernel on
 * \param async Whether the kernel should be run asynchronously or blocking
 * (Currently NOOP, all OpenMP operations run synchronously)
 * \param call Register a callback to be automatically invoked when the kernel
 * finishes, or NULL if none
 * \param ret_event_k1 Address of an initialized event (with openmpEvent[2]
 * payload) in which to return the kernel event used for asynchronous calls and
 * timing
 * \param ret_event_c1 Address of an initialized event (with openmpEvent[2]
 * payload) in which to return the event for copying face->size used for
 * asynchronous calls and timing (currently unused in OpenMP)
 * \param ret_event_c2 Address of an initialized event (with openmpEvent[2]
 * payload) in which to return the event for copying face->stride used for
 * asynchronous calls and timing (currently unused in OpenMP)
 * \param ret_event_c3 Address of an initialized event (with openmpEvent[2]
 * payload) in which to return the event for copying remain_dim to
 * c_face_child_size event used for asynchronous calls and timing (currently
 * unused in OpenMP)
 * \return 0 on success
 * \todo implement OpenMP error codes
 * \warning Implemented as a 1D kernel, Y and Z grid/block parameters will be
 * ignored
 */
a_err openmp_unpack_face(size_t (*grid_size)[3], size_t (*block_size)[3],
                         void *packed_buf, void *buf, meta_face *face,
                         int *remain_dim, meta_type_id type, int async,
                         meta_callback *call, meta_event *ret_event_k1,
                         meta_event *ret_event_c1, meta_event *ret_event_c2,
                         meta_event *ret_event_c3);

/**
 * CUDA wrapper for the 3D 7-point stencil averaging kernel
 * \param grid_size The number of blocks in each global dimension (outer loop
 * bounds, currently ignored)
 * \param block_size The dimensions of each block (inner loop bounds, currently
 * ignored)
 * \param indata The array of input read values
 * \param outdata The array of output write values
 * \param array_size The size of the input and output arrays
 * \param arr_start The start index for writing in each dimension (for avoiding
 * writes to halo cells)
 * \warning Assumes at least a one-thick halo region i.e. (arr_start[dim]-1 >=
 * 0)
 * \param arr_end The end index for writing in each dimension (for avoiding
 * writes to halo cells)
 * \warning Assumes at least a one-thick halo region i.e. (arr_end[dim]+1 <=
 * array_size[dim]-1)
 * \param type The type of data to run the kernel on
 * \param async Whether the kernel should be run asynchronously or blocking
 * (Currently NOOP, all OpenMP operations run synchronously)
 * \param call Register a callback to be automatically invoked when the kernel
 * finishes, or NULL if none
 * \param ret_event Address of an initialized event (with openmpEvent[2]
 * payload) in which to return the kernel's wrapping events
 * \return 0 on success
 * \todo implement OpenMP error codes
 */
a_err openmp_stencil_3d7p(size_t (*grid_size)[3], size_t (*block_size)[3],
                          void *indata, void *outdata, size_t (*array_size)[3],
                          size_t (*arr_start)[3], size_t (*arr_end)[3],
                          meta_type_id type, int async, meta_callback *call,
                          meta_event *ret_event);

#ifdef __OPENMPCC__
}
#endif

#endif // METAMORPH_OPENMP_BACKEND_H
