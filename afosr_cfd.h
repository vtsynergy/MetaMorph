/*
 * The core library header for the AFOSR-BRI Accelerated CFD Library
 *
 * Implements control over which accelerator core libraries the 
 *  application is compiled against via C Preprocessor Macros.
 * The corresponding afosr_cfd.c implements runtime control over
 *  which compiled-in core library functions are actually used
 *  internally.
 * Combined these provide the user simple control over which CFD
 *  library modules must be avaiable (Compile-time selections)
 *  and how it should choose among them for a given run (Runtime
 *  selections.)
 *
 * Created as part of the AFOSR-BRI <<INSERT PROJECT NAME AND ##>>
 * Virginia Polytechnic Institue and State University, 2013
 *
 * Authors: Paul Sathre, Sriram Chivukula, Kaixi Hou, Tom Scogland
 *  Harold Trease, Hao Wang. <<ADD OTHERS>>
 */

#include <stdio.h>

#ifdef WITH_CUDA
//TODO implement CUDA versions
	#include <afosr_cfd_cuda_core.cuh>
#endif

#ifdef WITH_OPENCL
//TODO implement OpenCL versions
	#include <afosr_cfd_opencl_core.h>
#endif

#ifdef WITH_OPENMP
//TODO implement OpenMP versions
	#include <afosr_cfd_openmp_core.h>
#endif

//TODO move all these typedefs, the generic types need to be runtime controlled
//TODO should generic "accelerator" types be typedefs, unions, ...?
//#ifndef ONLY_CUDA || ONLY_OPENCL || ONLY_OPENMP)
	//TODO implement generic accelerators types so an accelerator model
	// can be chosen at runtime

//#elif defined ONLY_CUDA
typedef double a_double;
typedef float a_float;
typedef int a_int;
//typedef long long a_longlong;
typedef long a_long;
typedef short a_short;
typedef char a_char;
typedef unsigned char a_uchar;
typedef unsigned short a_ushort;
typedef unsigned int a_uint;
typedef unsigned long a_ulong;
typedef int a_err;
typedef size_t a_dim3[3];
//typedef cudaError_t a_err;
//typedef unsigned long long a_ulonglong;

//#elif defined ONLY_OPENCL
// #pragma OPENCL EXTENSION cl_khr_fp64 : enable
//typedef cl_double a_double;
//typedef cl_float a_float;
//typedef cl_int a_int;
//typedef cl_long a_long;
//typedef cl_short a_short;
//typedef cl_char a_char;
//typedef cl_uchar a_uchar;
//typedef cl_ushort a_ushort;
//typedef cl_uint a_uint;
//typedef cl_ulong a_ulong;
//typedef cl_int a_err;
#ifdef WITH_OPENCL
cl_context accel_context = NULL;
cl_command_queue accel_queue = NULL;
#endif
//#elif defined ONLY_OPENMP
//TODO implement OpenMP primitives, should be same as CUDA
//#endif

typedef enum {
	accelModePreferGeneric = 0,
#ifdef WITH_CUDA
	accelModePreferCUDA = 1,
#endif
#ifdef WITH_OPENCL
	accelModePreferOpenCL = 2,
#endif
#ifdef WITH_OPENMP
	accelModePreferOpenMP = 3
#endif
} accel_preferred_mode;

a_err accel_alloc(void ** ptr, size_t size);
a_err accel_free(void * ptr);
a_err accel_copy_h2d(void * dst, void * src, size_t size);
a_err choose_accel(int accel, accel_preferred_mode mode);
a_err get_accel(int * accel, accel_preferred_mode * mode);
a_err accel_reduce(a_dim3 * grid_size, a_dim3 * block_size, a_double * data1, a_double * data2, a_dim3 * array_size, a_dim3 * array_start, a_dim3 * array_end, a_double * reduction_var);
a_err accel_copy_d2h(void * dst, void * src, size_t size);

