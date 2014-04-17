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
 * Virginia Polytechnic Institue and State University, 2013-2014
 *
 * Authors: Paul Sathre
 */

#ifndef AFOSR_CFD_H
#define AFOSR_CFD_H

#include <stdio.h>

//These are global controls over which accelerator "cores" are
// compiled in. Plugins, such as timers, should support
// conditional compilation for one or more accelerator cores, but
// SHOULD NOT, include the core headers themselves without #ifdef macros
// i.e. plugins should never fail to compile regardless of what cores
// are selected, rather if none of their supported cores are selected,
// they should just define down to a NOOP.
#ifdef WITH_CUDA
	#include "afosr_cfd_cuda_core.cuh"
#endif

#ifdef WITH_OPENCL
	#include "afosr_cfd_opencl_core.h"
#endif

#ifdef WITH_OPENMP
	#include "afosr_cfd_openmp_core.h"
#endif




//TODO move all these typedefs, the generic types need to be runtime controlled
//TODO should generic "accelerator" types be typedefs, unions, ...?
//TODO implement generic accelerators types so an accelerator model
// can be chosen at runtime

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
typedef enum { false, true } a_bool;

#ifdef WITH_OPENCL
	extern cl_context accel_context;
	extern cl_command_queue accel_queue;
	extern cl_device_id accel_device;
#endif

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
a_err choose_accel(int accel, accel_preferred_mode mode);
a_err get_accel(int * accel, accel_preferred_mode * mode);
a_err accel_validate_worksize(a_dim3 * grid_size, a_dim3 * block_size);
a_err accel_dotProd(a_dim3 * grid_size, a_dim3 * block_size, a_double * data1, a_double * data2, a_dim3 * array_size, a_dim3 * array_start, a_dim3 * array_end, a_double * reduction_var, a_bool async);
a_err accel_reduce(a_dim3 * grid_size, a_dim3 * block_size, a_double * data, a_dim3 * array_size, a_dim3 * array_start, a_dim3 * array_end, a_double * reduction_var, a_bool async);
a_err accel_copy_h2d(void * dst, void * src, size_t size, a_bool async);
a_err accel_copy_d2h(void * dst, void * src, size_t size, a_bool async);
a_err accel_copy_d2d(void * dst, void * src, size_t size, a_bool async);

//Separate from which core libraries are compiled in, the users
// should decide whether to compiler with different metrics
// packages. First and foremost, event-based timers for their
// respective platforms.
//WITH_TIMERS needs to be here, so the header passthrough can give it the accel_preferred_mode enum
#ifdef WITH_TIMERS
	#ifndef AFOSR_CFD_TIMERS_H
		#include "afosr_cfd_timers.h"
	#endif
#endif

//Fortran compatibility plugin needs access to all top-level calls
// and data types.
#ifdef WITH_FORTRAN
	#ifndef AFOSR_CFD_FORTRAN_COMPAT_H
		#include "afosr_cfd_fortran_compat.h"
	#endif
#endif


#endif //AFOSR_CFD_H
