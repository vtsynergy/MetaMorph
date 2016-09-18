/*
 * Implementation of the Fortran-compatible variants of all
 * top-level C library functions. Currently implemented as
 * an opt-in plugin with the -D WITH_FORTRAN compiler define.
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

#include <sys/time.h>
int elapsed_(double *sec);
int meta_alloc_c_(void ** ptr, size_t * size);
int meta_free_c_(void * ptr);
int choose_accel_c_(int * accel, int * mode);
int get_accel_c_(int * accel, int * mode);
int meta_validate_worksize_c_(size_t * grid_x, size_t * grid_y, size_t * grid_z,
		size_t * block_x, size_t * block_y, size_t * block_z);
int meta_dotprod_c_(size_t * grid_x, size_t * grid_y, size_t * grid_z,
		size_t * block_x, size_t * block_y, size_t * block_z, void * data1,
		void * data2, size_t * size_x, size_t * size_y, size_t * size_z,
		size_t * start_x, size_t * start_y, size_t * start_z, size_t * end_x,
		size_t * end_y, size_t * end_z, void * reduction_var, meta_type_id type,
		int * async);
int meta_reduce_c_(size_t * grid_x, size_t * grid_y, size_t * grid_z,
		size_t * block_x, size_t * block_y, size_t * block_z, void * data,
		size_t * size_x, size_t * size_y, size_t * size_z, size_t * start_x,
		size_t * start_y, size_t * start_z, size_t * end_x, size_t * end_y,
		size_t * end_z, void * reduction_var, meta_type_id type, int * async);
int meta_copy_h2d_c_(void * dst, void * src, size_t * size, int * async);
int meta_copy_d2h_c_(void * dst, void * src, size_t * size, int * async);
int meta_copy_d2d_c_(void * dst, void * src, size_t * size, int * async);

#ifdef WITH_TIMERS
int meta_timers_init_c_();
int meta_timers_flush_c_();
int meta_timers_finish_c_();
#endif
#endif //METAMORPH_FORTRAN_COMPAT_H
