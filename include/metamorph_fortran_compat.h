/** \file
 * Implementation of the Fortran-compatible variants of some
 * top-level C library functions. Currently implemented as
 * an opt-in plugin with the -D WITH_FORTRAN compiler define.
 *
 * \todo Implement the remaining top-level C library functions:
 * communication interface, data marshaling, stencil,...
 *
 * These functions should be as efficient as possible, but the
 * C interface always takes priority w.r.t. performance. It is
 * the shortest path from user code to library code, and the
 * standard; Fortran compatibility is and should continue to be
 * a convenience plugin only. Further we can only make any
 * reasonable claims about correctness of the Fortran->C glue
 * code on GNU systems, compiled with our Makefile; if you use
 * some other compilation setup for whatever reason, all bets
 * are off.
 */

#ifndef METAMORPH_FOTRAN_COMPAT_H
#define METAMORPH_FORTRAN_COMPAT_H

#ifndef METAMORPH_H
#include "metamorph.h"
#endif

/**
 * Get the current time in seconds since the epoch
 * Resolution is whatever is supported by clock_gettime
 * \param sec Addres in which to return the time
 * \return status of the clock_gettime call
 */
int curr_ns_(double *sec);
/**
 * Fortran-compatible wrapper around meta_alloc
 * \param ptr Address to return the newly-allocated void * handle in
 * \param size Number of bytes to allocate
 * \return The error code returned by meta_alloc
 */
int meta_alloc_c_(void ** ptr, size_t * size);
/**
 * Fortran-compatible wrapper around meta_free
 * \param ptr The void * MetaMorph memory handle to release
 * \return The error code returned by meta_free
 */
int meta_free_c_(void * ptr);
/**
 * Fortran-compatible wrapper around meta_set_acc
 * \param accel The desired accelerator's ID on the desired backend
 * \param mode The desired backend mode to switch to
 * \return The error code returned by meta_set_acc
 */
int meta_set_acc_c_(int * accel, int * mode);
/**
 * Fortran-compatible wrapper around meta_get_acc
 * \param accel Address in which to return the currently-active device's ID within the current backend
 * \param mode Address in which to return the currently-active backend mode
 * \return The error code returned by meta_get_acc
 */
int meta_get_acc_c_(int * accel, int * mode);
/**
 * Fortran-compatible wrapper around meta_validate_worksize
 * \param grid_x Number of desired thread blocks in the X dimension
 * \param grid_y Number of desired thread blocks in the Y dimension
 * \param grid_z Number of desired thread blocks in the Z dimension
 * \param block_x Number of desired threads within each block in the X dimension
 * \param block_y Number of desired threads within each block in the Y dimension
 * \param block_z Number of desired threads within each block in the Z dimension
 * \return The error code returned by meta_validate_worksize
 */
int meta_validate_worksize_c_(size_t * grid_x, size_t * grid_y, size_t * grid_z,
		size_t * block_x, size_t * block_y, size_t * block_z);
/**
 * Fortran-compatible wrapper around meta_dotProd
 * \param grid_x Number of desired thread blocks in the X dimension
 * \param grid_y Number of desired thread blocks in the Y dimension
 * \param grid_z Number of desired thread blocks in the Z dimension
 * \param block_x Number of desired threads within each block in the X dimension
 * \param block_y Number of desired threads within each block in the Y dimension
 * \param block_z Number of desired threads within each block in the Z dimension
 * \param data1 The left matrix in the dot product operator, a Metamorph-allocated buffer on the current device
 * \param data2 The right matrix in the dot product operator, a Metamorph-allocated buffer on the current device
 * \param size_x Number of elements in data1 and data2 in the X dimension
 * \param size_y Number of elements in data1 and data2 in the Y dimension
 * \param size_z Number of elements in data1 and data2 in the Z dimension
 * \param start_x Index in data1 and data2 of the first element in the X dimension
 * \param start_y Index in data1 and data2 of the first element in the Y dimension
 * \param start_z Index in data1 and data2 of the first element in the Z dimension
 * \param end_x Index in data1 and data2 of the last element in the X dimension
 * \param end_y Index in data1 and data2 of the last element in the Y dimension
 * \param end_z Index in data1 and data2 of the last element in the Z dimension
 * \param reduction_var The final scalar dot product value, a Metamorph-allocated buffer on the current device
 * \param type The MetaMorph data type to interpret the data arrays as
 * \param async Whether the kernel should be run asynchronously or blocking
 * \return The error code returned by meta_dotProd
 */
int meta_dotprod_c_(size_t * grid_x, size_t * grid_y, size_t * grid_z,
		size_t * block_x, size_t * block_y, size_t * block_z, void * data1,
		void * data2, size_t * size_x, size_t * size_y, size_t * size_z,
		size_t * start_x, size_t * start_y, size_t * start_z, size_t * end_x,
		size_t * end_y, size_t * end_z, void * reduction_var, meta_type_id type,
		int * async);
/**
 * Fortran-compatible wrapper around meta_reduce
 * \param grid_x Number of desired thread blocks in the X dimension
 * \param grid_y Number of desired thread blocks in the Y dimension
 * \param grid_z Number of desired thread blocks in the Z dimension
 * \param block_x Number of desired threads within each block in the X dimension
 * \param block_y Number of desired threads within each block in the Y dimension
 * \param block_z Number of desired threads within each block in the Z dimension
 * \param data The matrix to perform sum reduction on, a Metamorph-allocated buffer on the current device
 * \param size_x Number of elements in data in the X dimension
 * \param size_y Number of elements in data in the Y dimension
 * \param size_z Number of elements in data in the Z dimension
 * \param start_x Index in data of the first element in the X dimension
 * \param start_y Index in data of the first element in the Y dimension
 * \param start_z Index in data of the first element in the Z dimension
 * \param end_x Index in data of the last element in the X dimension
 * \param end_y Index in data of the last element in the Y dimension
 * \param end_z Index in data of the last element in the Z dimension
 * \param reduction_var The final scalar reduction value, a Metamorph-allocated buffer on the current device
 * \param type The MetaMorph data type to interpret the data arrays as
 * \param async Whether the kernel should be run asynchronously or blocking
 * \return The error code returned by meta_reduce
 */
int meta_reduce_c_(size_t * grid_x, size_t * grid_y, size_t * grid_z,
		size_t * block_x, size_t * block_y, size_t * block_z, void * data,
		size_t * size_x, size_t * size_y, size_t * size_z, size_t * start_x,
		size_t * start_y, size_t * start_z, size_t * end_x, size_t * end_y,
		size_t * end_z, void * reduction_var, meta_type_id type, int * async);
/**
 * Fortran-compatible wrapper around meta_copy_h2d
 * \param dst The destination, a MetaMorph-allocated handle on the currently-active device
 * \param src The source, a host buffer
 * \param size The number of bytes to write to the device
 * \param async Whether the transfer should be asynchronous or blocking
 * \return The error code returned from meta_copy_h2d
 */
int meta_copy_h2d_c_(void * dst, void * src, size_t * size, int * async);
/**
 * Fortran-compatible wrapper around meta_copy_d2h
 * \param dst The destination, a host buffer
 * \param src The source, a MetaMorph-allocated handle on the currently-active device
 * \param size The number of bytes to read from the device
 * \param async Whether the transfer should be asynchronous or blocking
 * \return The error code returned from meta_copy_d2h
 */
int meta_copy_d2h_c_(void * dst, void * src, size_t * size, int * async);
/**
 * Fortran-compatible wrapper around meta_copy_d2d
 * \param dst The destination, a MetaMorph-allocated handle on the currently-active device
 * \param src The source, a MetaMorph-allocated handle on the currently-active device
 * \param size The number of bytes to copy on the device
 * \param async Whether the transfer should be asynchronous or blocking
 * \return The error code returned from meta_copy_d2d
*/
int meta_copy_d2d_c_(void * dst, void * src, size_t * size, int * async);

#endif //METAMORPH_FORTRAN_COMPAT_H
