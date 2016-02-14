#include <cuda_runtime.h>
#ifndef METAMORPH_CUDA_CORE_H
#define METAMORPH_CUDA_CORE_H

#ifndef METAMORPH_H
	#include "metamorph.h"
#endif

//If the user doesn't override default threadblock size..
#ifndef METAMORPH_CUDA_DEFAULT_BLOCK
#define METAMORPH_CUDA_DEFAULT_BLOCK dim3(16, 8, 1)
#endif

#ifdef __CUDACC__
extern "C" {
#endif
	cudaError_t cuda_dotProd(size_t (* grid_size)[3], size_t (* block_size)[3], void * data1, void * data2, size_t (* array_size)[3], size_t (* arr_start)[3], size_t (* arr_end)[3], void * reduced_val, meta_type_id type, int async, cudaEvent_t ((*event)[2])); 
	cudaError_t cuda_reduce(size_t (* grid_size)[3], size_t (* block_size)[3], void * data, size_t (* array_size)[3], size_t (* arr_start)[3], size_t (* arr_end)[3], void * reduced_val, meta_type_id type, int async, cudaEvent_t ((*event)[2])); 
	cudaError_t cuda_transpose_2d_face(size_t (* grid_size)[3], size_t (* block_size)[3], void *indata, void *outdata, size_t (* arr_dim_xy)[3], size_t (* tran_dim_xy)[3], meta_type_id type, int async, cudaEvent_t ((*event)[2]));
	cudaError_t cuda_pack_2d_face(size_t (* grid_size)[3], size_t (* block_size)[3], void *packed_buf, void *buf, meta_2d_face_indexed *face, int *remain_dim, meta_type_id type, int async, cudaEvent_t ((*event_k1)[2]), cudaEvent_t ((*event_c1)[2]), cudaEvent_t ((*event_c2)[2]), cudaEvent_t ((*event_c3)[2]));
	cudaError_t cuda_unpack_2d_face(size_t (* grid_size)[3], size_t (* block_size)[3], void *packed_buf, void *buf, meta_2d_face_indexed *face, int *remain_dim, meta_type_id type, int async, cudaEvent_t ((*event_k1)[2]), cudaEvent_t ((*event_c1)[2]), cudaEvent_t ((*event_c2)[2]), cudaEvent_t ((*event_c3)[2]));
	cudaError_t cuda_stencil_3d7p(size_t (* grid_size)[3], size_t (* block_size)[3], void * indata, void * outdata, size_t (* array_size)[3], size_t (* arr_start)[3],  size_t (* arr_end)[3], meta_type_id type, int async, cudaEvent_t ((*event)[2]));

#ifdef __CUDACC__
}
#endif

#endif //METAMORPH_CUDA_CORE_H
