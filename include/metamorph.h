/*
 * The core library header for the METAMORPH Accelerated CFD Library
 *
 * Created as part of the Air Force Office of Scientific Research
 *  (AFOSR) Computational Mathematics Program via Grant number
 *  FA9550-12-1-0442.
 *  Virginia Polytechnic Institute and State University, 2013-2016.
 *
 * OpenCL code is largely generated by CU2CL. CU2CL has been supported in part
 *  by NSF I/UCRC IIP-0804155 via the NSF Center for High-Performance
 *  Reconfigurable Computing.
 *
 * Authors:
 *  Paul Sathre and Ahmed Helal (Design and implementation)
 *  Sriram Chivukula (CUDA dot-product/reduce prototypes)
 *  Kaixi Hou (CUDA data marshaling prototypes)
 *  Anshuman Verma (FPGA back-end)
 */

/** The top-level user APIs **/
#ifndef METAMORPH_H
#define METAMORPH_H
#ifdef __cplusplus
extern "C" {
#endif
#ifdef __DEBUG__
//Anything needed for a debug build

//Yell and fail if you hit something that's known to be broken
#define FIXME(str) { \
		fprintf(stderr, "FIXME:" __FILE__ ":%d: " #str "!\n", __LINE__); \
		exit(-1); \
		}
#endif

//If not in debug mode, yell, but don't exit
#if !defined(FIXME) && !defined(NO_FIXME)
#define FIXME(str) { \
		fprintf(stderr, "FIXME:" __FILE__ ":%d: " #str "!\n", __LINE__); \
		}
#endif

//If we're not in debug, but they did define NO_FIXME (at their own risk..
// then just shut up about unfinished things
#ifndef FIXME
#define FIXME(str)
#endif

//TODO should fail just as hard as fixme
#define TODO FIXME

#include <stdio.h>

//#define ALIGNED_MEMORY
//#define ALIGNED_MEMORY_PAGE	64

//This needs to be here so that the back-ends can internally
// switch between implementations for different primitive types
typedef enum {
	a_db = 0, 		// double
	a_fl = 1, 		// float
	a_lo = 2, 		// long
	a_ul = 3, 		// unsigned long
	a_in = 4, 		// int
	a_ui = 5, 		// unsigned int
	a_sh = 6, 		// short
	a_us = 7, 		// unsigned short
	a_ch = 8, 		// char
	a_uc = 9 		// unsigned char
} meta_type_id;

//Define Marshaling option bitmasks
#define METAMORPH_MARSH_OPT_NONE (0)
#define METAMORPH_MARSH_OPT_TRANSPOSE (1)

//Define limits to the meta_face structure
#define METAMORPH_FACE_MAX_DEPTH (10)

//FIXME Transpose shouldn't use hardcoded constants
//FIXME even if it does, OpenCL will need to pass them during compilation
//Define limits to transpose
#define TRANSPOSE_TILE_DIM (16)
#define TRANSPOSE_TILE_BLOCK_ROWS (16)

//This needs to be here so back-ends can all use the face struct
// for optimized pack/unpack operations
typedef struct {
	int start; //Starting offset in uni-dimensional buffer
	int count; //The size of size and stride buffers (number of tree levels)
	int *size; // The number of samples at each tree level
	int *stride; //The distance between samples at each tree level
} meta_face;
meta_face * meta_get_face(int s, int c, int *si, int *st);
int meta_free_face(meta_face * face);
meta_face * make_slab_from_3d(int face, int ni, int nj, int nk, int thickness);

//These are global controls over which accelerator "back-end" are
// compiled in. Companion features (plugins), such as timers, should support
// conditional compilation for one or more accelerator back-ends, but
// SHOULD NOT, include the back-end headers themselves without #ifdef macros
// i.e. plugins should never fail to compile regardless of what back-end
// are selected, rather if none of their supported back-end are selected,
// they should just define down to a NOOP.

//TODO change core to back-end in file names
#ifdef WITH_CUDA
#ifndef METAMORPH_CUDA_BACKEND_H
#include "mm_cuda_backend.cuh"
#endif
#endif

#ifdef WITH_OPENCL
#ifndef METAMORPH_OPENCL_BACKEND_H
#include "mm_opencl_backend.h"
#endif
#endif

#ifdef WITH_OPENMP
#ifndef METAMORPH_OPENMP_BACKEND_H
#include "mm_openmp_backend.h"
#endif
#endif

//TODO move all these typedefs, the generic types need to be runtime controlled
//TODO should generic "accelerator" types be typedefs, unions, ...?
//TODO implement generic accelerators types so an accelerator model
// can be chosen at runtime

typedef double a_double;
typedef float a_float;
typedef int a_int;
typedef long a_long;
typedef short a_short;
typedef char a_char;
typedef unsigned char a_uchar;
typedef unsigned short a_ushort;
typedef unsigned int a_uint;
typedef unsigned long a_ulong;
typedef int a_err;
#if defined (__CUDACC__) || defined(__cplusplus) || defined(bool)
typedef bool a_bool;
#else
typedef enum boolean {false , true}a_bool;
#endif
typedef size_t a_dim3[3];

typedef struct HPRecType {
	void * HP[2];
	struct HPRecType * next;
	void * rlist; //TODO implement the rlist type
	int rcount;
	char active;
} HPRecType;
//Shared HP variables

#ifdef WITH_OPENCL
//These are not needed if they're declared in metamorph.c
//	extern cl_context meta_context;
//	extern cl_command_queue meta_queue;
//	extern cl_device_id meta_device;
void metaOpenCLFallBack();

#endif

typedef enum {
	//A special-purpose mode which indicates none has been declared
	// used by sentinel nodes in the timer plugin queues
	metaModeUnset = -1,
#ifdef WITH_CUDA
metaModePreferCUDA = 1,
#endif
#ifdef WITH_OPENCL
metaModePreferOpenCL = 2,
#endif
#ifdef WITH_OPENMP
metaModePreferOpenMP = 3,
#endif
	metaModePreferGeneric = 0
} meta_preferred_mode;

//Module Management
//TODO need a bitfield to store the type of backend
typedef enum {
  module_implements_none = 0,
  module_implements_cuda = 1,
  module_implements_opencl = 2,
  module_implements_openmp = 4,
  module_implements_all = 7,
  module_implements_timers = 8,
  module_implements_mpi = 16,
  module_implements_fortran = 32,
  module_implements_general = 64 //general operations not related to a backend
} a_module_implements_backend;

struct a_module_record;
typedef struct a_module_record {
  //void function pointer to the initializer
  void (*module_init)(void);// = NULL;
  //void function pointer to the deinitializer
  void (*module_deinit)(void);// = NULL;
  //A copy of a pointer to the module's registration function, useful if we need to automatically modify the registration within MM instead of the module itself
  struct a_module_record * (*module_registry_func)(struct a_module_record * record);
  //enum "bitfield" defining which backend(s) (or general) the module provides implementations for
  a_module_implements_backend implements;// = module_implements_none;
  char initialized;// = 0;
} a_module_record;

//Not meant for users, lets MM components lookup sets of related modules
int lookup_implementing_modules(a_module_record ** retRecords, size_t szRetRecords, a_module_implements_backend signature, a_bool matchAny);
//Triggers for user modules to interop with metamorph
a_err meta_register_module(a_module_record * (*module_registry_func)(a_module_record * record));
a_err meta_deregister_module(a_module_record * (*module_registry_func)(a_module_record * record));
a_err meta_reinitialize_modules(a_module_implements_backend module_type);
// Memory/Context Management
a_err meta_alloc(void ** ptr, size_t size);
a_err meta_free(void * ptr);
a_err meta_set_acc(int accel, meta_preferred_mode mode);
a_err meta_get_acc(int * accel, meta_preferred_mode * mode);
a_err meta_validate_worksize(a_dim3 * grid_size, a_dim3 * block_size);
a_err meta_flush();
//share meta_context with with existing software
#ifdef WITH_OPENCL
a_int meta_get_state_OpenCL(cl_platform_id * platform, cl_device_id * device,
		cl_context * context, cl_command_queue * queue);
a_int meta_set_state_OpenCL(cl_platform_id platform, cl_device_id device,
		cl_context context, cl_command_queue queue);
#endif

//Some OpenCL implementations (may) not provide the CL_CALLBACK convention
#ifdef WITH_OPENCL
#ifndef CL_CALLBACK
#define CL_CALLBACK
#endif
#endif
typedef union meta_callback {
#ifdef WITH_CUDA
void (CUDART_CB * cudaCallback)(cudaStream_t stream, cudaError_t status, void *data);
#endif //WITH_CUDA
#ifdef WITH_OPENCL
void (CL_CALLBACK * openclCallback)(cl_event event, cl_int status, void * data);
#endif //WITH_OPENCL
} meta_callback;

//FIXME: As soon as the MPI implementation is finished, if
// payloads are still not needed, remove this code

#ifdef WITH_CUDA
typedef struct cuda_callback_payload {
//Unneeded, stream is always 0 (for now)
//		cudaStream_t stream;
//Unneeded, errors are managed in the library function responsible for
//	setting the callback
//		cudaError_t status;
void * data;
}cuda_callback_payload;
#endif //WITH_CUDA

#ifdef WITH_OPENCL
typedef struct opencl_callback_payload {
//Unneeded, the event is provided by the library function responsible for
//	setting the callback
//		cl_event event;
//Unneeded, the status *MUST* always be CL_COMPLETE
//		cl_int status;
void * data;
}opencl_callback_payload;
#endif //WITH_OPENCL

typedef union meta_callback_payload {
#ifdef WITH_CUDA
cuda_callback_payload cuda_pl;
#endif //WITH_CUDA
#ifdef WITH_OPENCL
opencl_callback_payload opencl_pl;
#endif //WITH_OPENCL
} meta_callback_payload;

//FIXME: If custom payloads are not needed, clean up the code above

//Kernels and transfers with callback params, necessary for MPI helpers
// These are **NOT** intended to be used externally, only by the library itself
// The callback/payload structure is not built for user-specified callbacks
a_err meta_copy_h2d_cb(void * dst, void * src, size_t size, a_bool async,
	//char * event_name, cl_event * wait,
	meta_callback *call, void *call_pl);
a_err meta_copy_d2h_cb(void * dst, void * src, size_t size, a_bool async,
	//char * event_name, cl_event * wait,
	meta_callback *call, void *call_pl);
a_err meta_copy_d2d_cb(void * dst, void * src, size_t size, a_bool async,
	//char * event_name, cl_event * wait,
	meta_callback *call, void *call_pl);
a_err meta_transpose_face_cb(a_dim3 * grid_size, a_dim3 * block_size,
	void *indata, void *outdata, a_dim3 * arr_dim_xy, a_dim3 * tran_dim_xy,
	meta_type_id type, a_bool async, meta_callback *call, void *call_pl);
a_err meta_pack_face_cb(a_dim3 * grid_size, a_dim3 * block_size,
	void *packed_buf, void *buf, meta_face *face, meta_type_id type,
	a_bool async, meta_callback *call, void *call_pl);
a_err meta_unpack_face_cb(a_dim3 * grid_size, a_dim3 * block_size,
	void *packed_buf, void *buf, meta_face *face, meta_type_id type,
	a_bool async, meta_callback *call, void *call_pl);
a_err meta_dotProd_cb(a_dim3 * grid_size, a_dim3 * block_size, void * data1,
	void * data2, a_dim3 * array_size, a_dim3 * array_start, a_dim3 * array_end,
	void * reduction_var, meta_type_id type, a_bool async, meta_callback *call,
	void *call_pl);
a_err meta_reduce_cb(a_dim3 * grid_size, a_dim3 * block_size, void * data,
	a_dim3 * array_size, a_dim3 * array_start, a_dim3 * array_end,
	void * reduction_var, meta_type_id type, a_bool async, meta_callback *call,
	void *call_pl);
a_err meta_stencil_3d7p_cb(a_dim3 * grid_size, a_dim3 * block_size,
	void *indata, void *outdata, a_dim3 * array_size, a_dim3 * array_start,
	a_dim3 * array_end, meta_type_id type, a_bool async, meta_callback *call,
	void *call_pl);
a_err meta_csr_cb(size_t global_size, size_t local_size, void * csr_ap, void * csr_aj, void * csr_ax, void * x_loc, void * y_loc,
		size_t wg_size, meta_type_id type, a_bool async,
		// cl_event * wait,
		meta_callback *call, void *call_pl);
a_err meta_crc_cb(size_t global_size, size_t local_size, void * dev_input, int page_size, int num_words, int numpages, void * dev_output,
		meta_type_id type, a_bool async,
		// cl_event * wait,
		meta_callback *call, void *call_pl);
		
		
#ifdef WITH_TIMERS
#ifdef WITH_OPENCL
//getting a pointer to specific event  
a_err meta_get_event(char * qname, char * ename, cl_event ** e);
#endif // WITH_OPENCL
#endif // WITH_TIMERS

//Reduced-complexity calls
// These are the ones applications built on top of the library should use
// Memory copy host to device
a_err meta_copy_h2d(void * dst, void * src, size_t size, a_bool async);//, char * event_name, cl_event * wait);
// Memory copy device to host
a_err meta_copy_d2h(void * dst, void * src, size_t size, a_bool async);//, char * event_name, cl_event * wait);
// Memory copy device to device
a_err meta_copy_d2d(void * dst, void * src, size_t size, a_bool async);//, char * event_name, cl_event * wait);
a_err meta_transpose_face(a_dim3 * grid_size, a_dim3 * block_size,
	void *indata, void *outdata, a_dim3 * arr_dim_xy, a_dim3 * tran_dim_xy,
	meta_type_id type, a_bool async);
a_err meta_pack_face(a_dim3 * grid_size, a_dim3 * block_size,
	void *packed_buf, void *buf, meta_face *face, meta_type_id type,
	a_bool async);
a_err meta_unpack_face(a_dim3 * grid_size, a_dim3 * block_size,
	void *packed_buf, void *buf, meta_face *face, meta_type_id type,
	a_bool async);
a_err meta_dotProd(a_dim3 * grid_size, a_dim3 * block_size, void * data1,
	void * data2, a_dim3 * array_size, a_dim3 * array_start, a_dim3 * array_end,
	void * reduction_var, meta_type_id type, a_bool async);
a_err meta_reduce(a_dim3 * grid_size, a_dim3 * block_size, void * data,
	a_dim3 * array_size, a_dim3 * array_start, a_dim3 * array_end,
	void * reduction_var, meta_type_id type, a_bool async);
a_err meta_stencil_3d7p(a_dim3 * grid_size, a_dim3 * block_size, void *indata,
	void *outdata, a_dim3 * array_size, a_dim3 * array_start,
	a_dim3 * array_end, meta_type_id type, a_bool async);
a_err meta_csr(size_t global_size, size_t local_size, void * csr_ap, void * csr_aj, void * csr_ax, void * x_loc, void * y_loc, 
	size_t wg_size, meta_type_id type, a_bool async);//, cl_event * wait);
a_err meta_crc(size_t global_size, size_t local_size, void * dev_input, int page_size, int num_words, int numpages, void * dev_output, 
	meta_type_id type, a_bool async);//, cl_event * wait);

//MPI functions need access to all top-level calls and types
//MPI needs to be before timers, so that timers can output rank info - if available
#ifdef WITH_MPI
#ifndef METAMORPH_MPI_H
#include "metamorph_mpi.h"
#endif
#endif

//Event-based timers
//WITH_TIMERS needs to be here, so the header passthrough can give it the
// meta_preferred_mode enum
#ifdef WITH_TIMERS
#ifndef METAMORPH_TIMERS_H
#include "metamorph_timers.h"
#endif
#endif

//Fortran compatibility plugin needs access to all top-level calls
// and data types.
#ifdef WITH_FORTRAN
#ifndef METAMORPH_FORTRAN_COMPAT_H
#include "metamorph_fortran_compat.h"
#endif
#endif

#ifdef __cplusplus
}
#endif

#endif //METAMORPH_H
