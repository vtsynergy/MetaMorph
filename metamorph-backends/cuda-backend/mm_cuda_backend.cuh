#include <cuda_runtime.h>

/** CUDA Back-End **/
#ifndef METAMORPH_CUDA_BACKEND_H
#define METAMORPH_CUDA_BACKEND_H

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
  a_err metaCUDAAlloc(void**, size_t);
  a_err metaCUDAFree(void*);
  a_err metaCUDAWrite(void*, void*, size_t, a_bool, meta_callback*, meta_event *);
  a_err metaCUDARead(void*, void*, size_t, a_bool, meta_callback*, meta_event *);
  a_err metaCUDADevCopy(void*, void*, size_t, a_bool, meta_callback*, meta_event *);
  a_err metaCUDAInitByID(a_int);
  a_err metaCUDACurrDev(a_int*);
  a_err metaCUDAMaxWorkSizes(a_dim3*, a_dim3*);
  a_err metaCUDAFlush();
a_err metaCUDACreateEvent(void**);
a_err metaCUDADestroyEvent(void*);
a_err metaCUDAExpandCallback(meta_callback call, cudaStream_t * ret_stream, cudaError_t * ret_status, void** ret_data);
  a_err metaCUDARegisterCallback(meta_callback *);
  a_err cuda_dotProd(size_t (*)[3], size_t (*)[3], void *, void *, size_t (*)[3], size_t (*)[3], size_t (*)[3], void *, meta_type_id, int, meta_callback *, meta_event *);
  a_err cuda_reduce(size_t (*)[3], size_t (*)[3], void *, size_t (*)[3], size_t (*)[3], size_t (*)[3], void *, meta_type_id, int, meta_callback *, meta_event *);
  a_err cuda_transpose_face(size_t (*)[3], size_t (*)[3], void *, void *, size_t (*)[3], size_t (*)[3], meta_type_id, int, meta_callback *, meta_event *);
  a_err cuda_pack_face(size_t (*)[3], size_t (*)[3], void *, void *, meta_face *, int *, meta_type_id, int, meta_callback *, meta_event *, meta_event *, meta_event *, meta_event *);
  a_err cuda_unpack_face(size_t (*)[3], size_t (*)[3], void *, void *, meta_face *, int *, meta_type_id, int, meta_callback *, meta_event *, meta_event *, meta_event *, meta_event *);
  a_err cuda_stencil_3d7p(size_t (*)[3], size_t (*)[3], void *, void *, size_t (*)[3], size_t (*)[3], size_t (*)[3], meta_type_id, int, meta_callback *, meta_event *);
#ifdef __CUDACC__
}
#endif

#endif //METAMORPH_CUDA_BACKEND_H
