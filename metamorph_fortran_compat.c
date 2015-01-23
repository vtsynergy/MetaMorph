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
 * are off. ISO_C_BINDINGS *should* help.
 */
#include "metamorph_fortran_compat.h"

int elapsed_(double *sec ){
	extern int gettimeofday();
	struct timeval t;

	int stat = gettimeofday(&t, NULL);
	*sec = (double)(t.tv_sec*1000000.0+t.tv_usec);
	return (stat);
}
int meta_alloc_c_(void ** ptr, size_t * size) { return (int) meta_alloc(ptr, *size); }
int meta_free_c_(void * ptr) { return (int) meta_free(ptr); }
int choose_accel_c_(int * accel, int * mode) { return (int) choose_accel(*accel, (meta_preferred_mode) (*mode)); }
int get_accel_c_(int * accel, int * mode) { return (int) get_accel(accel, (meta_preferred_mode *) mode); }
int meta_validate_worksize_c_(size_t * grid_x, size_t * grid_y, size_t * grid_z, size_t * block_x, size_t * block_y, size_t * block_z) {
	a_dim3 grid, block;
	grid[0] = *grid_x, grid[1] = *grid_y, grid[2] = *grid_z;
	block[0] = *block_x, block[1] = *block_y, block[2] = *block_z;
	return (int) meta_validate_worksize(&grid, &block);
}
int meta_dotprod_c_(size_t * grid_x, size_t * grid_y, size_t * grid_z, size_t * block_x, size_t * block_y, size_t * block_z, void * data1, void * data2, size_t * size_x, size_t * size_y, size_t * size_z, size_t * start_x, size_t * start_y, size_t * start_z, size_t * end_x, size_t * end_y, size_t * end_z, void * reduction_var, meta_type_id type, int * async) {
	a_dim3 grid, block, size, start, end;
	grid[0] = *grid_x, grid[1] = *grid_y, grid[2] = *grid_z;
	block[0] = *block_x, block[1] = *block_y, block[2] = *block_z;
	size[0] = *size_x, size[1] = *size_y, size[2] = *size_z;
	start[0] = *start_x-1, start[1] = *start_y-1, start[2] = *start_z-1;
	end[0] = *end_x-1, end[1] = *end_y-1, end[2] = *end_z-1;
	return (int) meta_dotProd(&grid, &block, data1, data2, &size, &start, &end, reduction_var, type, (a_bool) *async);
}
int meta_reduce_c_(size_t * grid_x, size_t * grid_y, size_t * grid_z, size_t * block_x, size_t * block_y, size_t * block_z, void * data, size_t * size_x, size_t * size_y, size_t * size_z, size_t * start_x, size_t * start_y, size_t * start_z, size_t * end_x, size_t * end_y, size_t * end_z, void * reduction_var, meta_type_id type,  int * async) {
	a_dim3 grid, block, size, start, end;
	grid[0] = *grid_x, grid[1] = *grid_y, grid[2] = *grid_z;
	block[0] = *block_x, block[1] = *block_y, block[2] = *block_z;
	size[0] = *size_x, size[1] = *size_y, size[2] = *size_z;
	start[0] = *start_x-1, start[1] = *start_y-1, start[2] = *start_z-1;
	end[0] = *end_x-1, end[1] = *end_y-1, end[2] = *end_z-1;
	return (int) meta_reduce(&grid, &block, data, &size, &start, &end, reduction_var, type, (a_bool) *async);
}
int meta_copy_h2d_c_(void * dst, void * src, size_t * size, int * async) { return (int) meta_copy_h2d(dst, src, *size, (a_bool) *async);}
int meta_copy_d2h_c_(void * dst, void * src, size_t * size, int * async) { return (int) meta_copy_d2h(dst, src, *size, (a_bool) *async);}
int meta_copy_d2d_c_(void * dst, void * src, size_t * size, int * async) { return (int) meta_copy_d2d(dst, src, *size, (a_bool) *async);}

#ifdef WITH_TIMERS
int meta_timers_init_c_() { return (int) metaTimersInit();}
int meta_timers_flush_c_() { return (int) metaTimersFlush();}
int meta_timers_finish_c_() { return (int) metaTimersFinish();}
#endif
