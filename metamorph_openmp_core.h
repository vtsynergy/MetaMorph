
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef METAMORPH_OPENMP_CORE_H
#define METAMORPH_OPENMP_CORE_H

#ifndef METAMORPH_H
	#include "metamorph.h"
#endif

#ifdef __OPENMPCC__
extern "C" {
#endif

	int omp_dotProd(size_t (* grid_size)[3], size_t (* block_size)[3], void * data1, void * data2, size_t (* array_size)[3], size_t (* arr_start)[3],  size_t (* arr_end)[3], void * reduction_var, meta_type_id type, int async);
	int omp_reduce(size_t (* grid_size)[3], size_t (* block_size)[3], void * data, size_t (* array_size)[3], size_t (* arr_start)[3],  size_t (* arr_end)[3], void * reduction_var, meta_type_id type, int async);
	int omp_transpose_2d_face(size_t (* grid_size)[3], size_t (* block_size)[3], void * indata, void *outdata, size_t (* arr_dim_xy)[3], size_t (* tran_dim_xy)[3], meta_type_id type, int async);
	int omp_pack_2d_face(size_t (* grid_size)[3], size_t (* block_size)[3], void *packed_buf, void *buf, meta_2d_face_indexed *face, int *remain_dim, meta_type_id type, int async);
	int omp_unpack_2d_face(size_t (* grid_size)[3], size_t (* block_size)[3], void *packed_buf, void *buf, meta_2d_face_indexed *face, int *remain_dim, meta_type_id type, int async);


#ifdef __OPENMPCC__
}
#endif


#endif //METAMORPH_OPENMP_CORE_H