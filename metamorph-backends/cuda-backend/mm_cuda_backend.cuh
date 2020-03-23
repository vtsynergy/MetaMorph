/** \file
 * CUDA Backend exposed functions and types
 */

#include <cuda_runtime.h>

/** CUDA Back-End **/
#ifndef METAMORPH_CUDA_BACKEND_H
#define METAMORPH_CUDA_BACKEND_H

#ifndef METAMORPH_H
#include "metamorph.h"
#endif

//If the user doesn't override default threadblock size..
#ifndef METAMORPH_CUDA_DEFAULT_BLOCK
/**If no alternative default block size is provided, set a default */
#define METAMORPH_CUDA_DEFAULT_BLOCK dim3(16, 8, 1)
#endif

#ifdef __CUDACC__
extern "C" {
#endif
/**
 * A simple wrapper around cudaMalloc to create a buffer on the current CUDA device and pass it back up to the backend-agnostic layer
 * \param ptr The address in which to return allocated buffer
 * \param size The number of bytes to allocate
 * \return the CUDA error status of the cudaMalloc call
 */
  a_err metaCUDAAlloc(void** ptr, size_t size);
/**
 * Wrapper function around cudaFree to release a MetaMorph-allocated CUDA buffer
 * \param ptr The buffer to release
 * \return the result of cudaFree
 */
  a_err metaCUDAFree(void* ptr);
/**
 * A wrapper for a CUDA host-to-device copy
 * \param dst The destination buffer, a buffer allocated in MetaMorph's currently-running CUDA context
 * \param src The source buffer, a host memory region
 * \param size The number of bytes to copy from the host to the device
 * \param async whether the write should be asynchronous or blocking
 * \param call A callback to run when the transfer finishes, or NULL if none
 * \param ret_event The address of a meta_event with initialized cudaEvent_t[2] payload in which to copy the cudaEvent_ts corresponding to the write back to
 * \return the CUDA error status of the wrapped cudaMemcpy/Async
 */
  a_err metaCUDAWrite(void* dst, void* src, size_t size, a_bool async, meta_callback* call, meta_event * ret_event);
/**
 * A wrapper for a CUDA device-to-host copy
 * \param dst The destination buffer, a host memory region
 * \param src The source buffer, a CUDA buffer allocated in MetaMorph's currently-running CUDA context
 * \param size The number of bytes to copy from the device to the host
 * \param async whether the read should be asynchronous or blocking
 * \param call A callback to run when the transfer finishes, or NULL if none
 * \param ret_event The address of a meta_event with initialized cudaEvent_t[2] payload in which to copy the cudaEvent_ts corresponding to the read back to
 * \return the CUDA error status of the wrapped clMemcpy/Async
 */
  a_err metaCUDARead(void* dst, void* src, size_t size, a_bool async, meta_callback* call, meta_event * ret_event);
/**
 * A wrapper for a CUDA device-to-device copy
 * \param dst The destination buffer, a CUDA buffer allocated in MetaMorph's currently-running CUDA context
 * \param src The source buffer, a CUDA buffer allocated in MetaMorph's currently-running CUDA context
 * \param size The number of bytes to copy
 * \param async whether the read should be asynchronous or blocking
 * \param call A callback to run when the transfer finishes, or NULL if none
 * \param ret_event The address of a meta_event with initialized cudaEvent_t[2] payload in which to copy the cudaEvent_ts corresponding to the read back to
 * \return the CUDA error status of the wrapped clMemcpy/Async
 */
  a_err metaCUDADevCopy(void* dst, void* src, size_t size, a_bool async, meta_callback* call, meta_event * ret_event);
/**
 * Initialize The n-th CUDA device and switch to using it
 * \param accel Which device to use
 * \return the CUDA error status of the underlying cudaSetDevice call
 */
  a_err metaCUDAInitByID(a_int accel);
/**
 * Get the index of the currently-utilized CUDA device
 * \param accel The address in which to return the numerical ID
 * \return the CUDA error status of the underlying cudaGetDevice call
 */
  a_err metaCUDACurrDev(a_int* accel);
/**
 * Check whether the requested work sizes will fit on the current CUDA device
 * \param grid The requested number of work groups in each dimension
 * \param block The requested number of work items within each workgroup in each dimension
 * \return the CUDA error status of any underlying CUDA API calls.
 * \bug FIXME Implement
 */
  a_err metaCUDAMaxWorkSizes(a_dim3* grid, a_dim3* block);
/**
 * Finish all outstanding CUDA operations
 * \return The result of cudaThreadSynchronize
 */
  a_err metaCUDAFlush();
/**
 * Just a small wrapper around a cudaEvent_t allocator to keep the real datatype exclusively inside the CUDA backend
 * \param ret_event The address in which to save the pointer to the newly-allocated cudaEvent_t[2]
 * \return cudaErrorInvalid value if ret_event is NULL, otherwise the binary OR of the two cudaEventCreate calls' error codes
 */
a_err metaCUDACreateEvent(void** ret_event);
/**
 * Just a small wrapper around a cudaEvent_t destructor to keep the real datatype exclusively inside the CUDA backend
 * \param event The address of the cudaEvent_t[2] to destroy
 * \return cudaErrorInvalid value if event is NULL, otherwise the binary OR of the two cudaEventDestroy calls' error codes
 */
a_err metaCUDADestroyEvent(void* event);
/**
 * A simple wrapper to get the elapsed time of a meta_event containing two cudaEvent_ts
 * \param ret_ms The address to save the elapsed time in milliseconds
 * \param event The meta_event (bearing a dynamically-allocated cudaEvent_t[2] as its payload) to query
 * \return cudaErrorInvalidValue if either the return pointer or event payload is NULL, otherwise the status of the cudaEventElapsedTime call
 */
a_err metaCUDAEventElapsedTime(float* ret_ms, meta_event event);
/**
 * An internal function that can generically be generically registered as a CUDA callback which in turn triggers the data payload's meta_callback
 * This is just a pass-though to send the callback back up to the backend-agnostic layer
 * \param stream The CUDA stream on which the callback was registered
 * \param status The status of the triggering event
 * \param data The meta_callback payload which should be triggered
 */
void CUDART_CB  metaCUDACallbackHelper(cudaStream_t stream, cudaError_t status, void * data);
/**
 * Since metaCUDACallbackHelper doesn't directly face the user, provide a way for them to unpack the information it would receive from the CUDA for use in their application
 * \param call The (already-triggered) callback to get the status of
 * \param ret_stream The address in which to copy the callback's triggering stream
 * \param ret_status The address in which to copy the callback's triggering event's status
 * \param ret_data The address into which to copy the pointer to the triggering callback's data payload (necessarily a meta_callback)
 * \return cudaSuccess if the payload was successfully unpacked, or cudaErrorInvalidValue if the stream, status or data pointers are NULL
 */
a_err metaCUDAExpandCallback(meta_callback call, cudaStream_t * ret_stream, cudaError_t * ret_status, void** ret_data);
/**
 * Internal function to register a meta_callback with the CUDA backend via the metaCUDACallbackHelper
 * \param call the meta_callback payload that should be invoked and filled when triggered
 * \return cudaErrorInvalidValue if the callback is improperly-created, otherwise the result of cudaStreamAddCallback
 */
  a_err metaCUDARegisterCallback(meta_callback * call);
/**
 * Dot-product of identically-shaped subregions of two identically-shaped 3D arrays
 * this kernel works for 3D data only.
 * \param grid_size The number of blocks in each global dimension
 * \param block_size The dimensions of each block
 * \param data1 first input array (a CUDA buffer residing on the currently-active device)
 * \param data2 second input array (a CUDA buffer residing on the currently-active device)
 * \param array_size X, Y, and Z dimension sizes of the input arrays (must match)
 * \param arr_start X, Y, and Z dimension start indicies for computing on a subregion, to allow for a halo region
 * \param arr_end X, Y, and Z dimension end indicies for computing on a subregion, to allow for a halo region
 * \param reduced_val the final dotproduct output (scalar) across all workgroups, assumed to be initialized before the kernel (a CUDA buffer residing on the currently-active device)
 * \param type The supported MetaMorph data type that data1, data2, and reduced_val contain (Currently: a_db, a_fl, a_ul, a_in, a_ui)
 * \param async Whether the kernel should be run asynchronously or blocking
 * \param call Register a callback to be automatically invoked when the kernel finishes, or NULL if none
 * \param ret_event Address of an initialized event (with cudaEvent_t[2] payload) in which to return the kernel's wrapping events
 * \return cudaSucces if kernel is launched synchronously, otherwise the result of cudaThreadSynchronize
 */
  a_err cuda_dotProd(size_t (* grid_size)[3], size_t (* block_size)[3], void * data1, void * data2, size_t (* array_size)[3], size_t (* arr_start)[3], size_t (* arr_end)[3], void * reduced_val, meta_type_id type, int async, meta_callback * call, meta_event * ret_event);
/**
 * Sum Reduction of subregion of a 3D array
 * this kernel works for 3D data only.
 * \param grid_size The number of blocks in each global dimension
 * \param block_size The dimensions of each block
 * \param data input array (a CUDA buffer residing o
 * \param array_size X, Y, and Z dimension sizes of the input arrays (must match)
 * \param arr_start X, Y, and Z dimension start indicies for computing on a subregion, to allow for a halo region
 * \param arr_end X, Y, and Z dimension end indicies for computing on a subregion, to allow for a halo region
 * \param reduced_val the final dotproduct output (scalar) across all blocks, assumed to be initialized before the kernel (a CUDA buffer residing on the currently-active device)
 * \param type The supported MetaMorph data type that data1, data2, and reduced_val contain (Currently: a_db, a_fl, a_ul, a_in, a_ui)
 * \param async Whether the kernel should be run asynchronously or blocking
 * \param call Register a callback to be automatically invoked when the kernel finishes, or NULL if none
 * \param ret_event Address of an initialized event (with cudaEvent_t[2] payload) in which to return the kernel's wrapping events
 * \return cudaSucces if kernel is launched synchronously, otherwise the result of cudaThreadSynchronize
 */
  a_err cuda_reduce(size_t (* grid_size)[3], size_t (* block_size)[3], void * data, size_t (* array_size)[3], size_t (* arr_start)[3], size_t (* arr_end)[3], void * reduced_val, meta_type_id type, int async, meta_callback * call, meta_event * ret_event);
/**
 * CUDA wrapper to transpose a 2D array
 * \param grid_size The number of blocks in each dimension
 * \param block_size The size of a block in each dimension
 * \param indata The input untransposed 2D array
 * \param outdata The output transposed 2D array
 * \param arr_dim_xy the X and Y dimensions of indata, Z is ignored 
 * \param tran_dim_xy the X and Y dimensions of outdata, Z is ignored
 * \param type The data type to invoke the appropriate kernel for
 * \param async Whether the kernel should run asynchronously or block
 * \param call Register a callback to be automatically invoked when the kernel finishes, or NULL if none
 * \param ret_event Address of an initialized event (with cudaEvent_t[2] payload) in which to return the kernel's wrapping events
 * \return cudaSucces if kernel is launched synchronously, otherwise the result of cudaThreadSynchronize
 */
  a_err cuda_transpose_face(size_t (* grid_size)[3], size_t (* block_size)[3], void * indata, void * outdata, size_t (* arr_dim_xy)[3], size_t (* tran_dim_xy)[3], meta_type_id type, int async, meta_callback * call, meta_event * ret_event);
/**
 * CUDA wrapper for the face packing kernel
 * \param grid_size The number of blocks to run in the X dimension, Y and Z are ignored
 * \param block_size The size of a block in the X dimension, Y and Z are ignored
 * \param packed_buf The output packed array
 * \param buf The input full array
 * \param face The face/slab to extract, returned from make_slab_from_3d
 * \param remain_dim At each subsequent layer of the face struct, the *cummulative* size of all remaining descendant dimensions. Used for computing the per-thread write offset, typically automatically pre-computed in the backend-agnostic MetaMorph wrapper, meta_pack_face
 * \param type The type of data to run the kernel on
 * \param async whether to run the kernel asynchronously (1) or synchronously (0)
 * \param call Register a callback to be automatically invoked when the kernel finishes, or NULL if none
 * \param ret_event_k1 Address of an initialized event (with cudaEvent_t[2] payload) in which to return the kernel event used for asynchronous calls and timing
 * \param ret_event_c1 Address of an initialized event (with cudaEvent_t[2] payload) in which to return the event for the cudaMemcpyToSymbol of face->size event used for asynchronous calls and timing
 * \param ret_event_c2 Address of an initialized event (with cudaEvent_t[2] payload) in which to return the event for the cudaMemcpyToSymbol of face->stride event used for asynchronous calls and timing
 * \param ret_event_c3 Address of an initialized event (with cudaEvent_t[2] payload) in which to return the event for the cudaMemcpyToSymbol of remain_dim to c_face_child_size event used for asynchronous calls and timing
 * \return either the result of enqueuing the kernel if async or the result of cudaThreadSynchronize if sync
 * \warning Implemented as a 1D kernel, Y and Z grid/block parameters will be ignored
 */
  a_err cuda_pack_face(size_t (* grid_size)[3], size_t (* block_size)[3], void * packed_buf, void * buf, meta_face * face, int * remain_dim, meta_type_id type, int async, meta_callback * call, meta_event * ret_event_k1, meta_event * ret_event_c1, meta_event * ret_event_c2, meta_event * ret_event_c3);
/**
 * CUDA wrapper for the face unpacking kernel
 * \param grid_size The number of blocks to run in the X dimension, Y and Z are ignored
 * \param block_size The size of a block in the X dimension, Y and Z are ignored
 * \param packed_buf The input packed array
 * \param buf The output full array
 * \param face The face/slab to extract, returned from make_slab_from_3d
 * \param remain_dim At each subsequent layer of the face struct, the *cummulative* size of all remaining descendant dimensions. Used for computing the per-thread write offset, typically automatically pre-computed in the backend-agnostic MetaMorph wrapper, meta_pack_face
 * \param type The type of data to run the kernel on
 * \param async whether to run the kernel asynchronously (1) or synchronously (0)
 * \param call Register a callback to be automatically invoked when the kernel finishes, or NULL if none
 * \param ret_event_k1 Address of an initialized event (with cudaEvent_t[2] payload) in which to return the kernel event used for asynchronous calls and timing
 * \param ret_event_c1 Address of an initialized event (with cudaEvent_t[2] payload) in which to return the event for the cudaMemcpyToSymbol of face->size event used for asynchronous calls and timing
 * \param ret_event_c2 Address of an initialized event (with cudaEvent_t[2] payload) in which to return the event for the cudaMemcpyToSymbol of face->stride event used for asynchronous calls and timing
 * \param ret_event_c3 Address of an initialized event (with cudaEvent_t[2] payload) in which to return the event for the cudaMemcpyToSymbol of remain_dim to c_face_child_size event used for asynchronous calls and timing
 * \return either the result of enqueuing the kernel if async or the result of cudaThreadSynchronize if sync
 * \warning Implemented as a 1D kernel, Y and Z grid/block parameters will be ignored
 */
  a_err cuda_unpack_face(size_t (* grid_size)[3], size_t (* block_size)[3], void * packed_buf, void * buf, meta_face * face, int * remain_dim, meta_type_id type, int async, meta_callback * call, meta_event * ret_event_k1, meta_event * ret_event_c1, meta_event * ret_event_c2, meta_event * ret_event_c3);
/**
 * CUDA wrapper for the 3D 7-point stencil averaging kernel
 * \param grid_size The number of blocks to run in each dimension
 * \param block_size The size of a block in each dimension
 * \param indata The array of input read values
 * \param outdata The array of output write values
 * \param array_size The size of the input and output arrays
 * \param arr_start The start index for writing in each dimension (for avoiding writes to halo cells)
 * \warning Assumes at least a one-thick halo region i.e. (arr_start[dim]-1 >= 0)
 * \param arr_end The end index for writing in each dimension (for avoiding writes to halo cells)
 * \warning Assumes at least a one-thick halo region i.e. (arr_end[dim]+1 <= array_size[dim]-1)
 * \param type The type of data to run the kernel on
 * \param async whether to run the kernel asynchronously (1) or synchronously (0)
 * \param call Register a callback to be automatically invoked when the kernel finishes, or NULL if none
 * \param ret_event Address of an initialized event (with cudaEvent_t[2] payload) in which to return the kernel event used for asynchronous calls and timing
 * \return either the result of enqueuing the kernel if async or the result of cudaThreadSynchronize if sync
 */
  a_err cuda_stencil_3d7p(size_t (* grid_size)[3], size_t (* block_size)[3], void * indata, void * outdata, size_t (* array_size)[3], size_t (* arr_start)[3], size_t (* arr_end)[3], meta_type_id type, int async, meta_callback * call, meta_event * ret_event);
#ifdef __CUDACC__
}
#endif

#endif //METAMORPH_CUDA_BACKEND_H
