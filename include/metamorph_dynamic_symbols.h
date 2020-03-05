#ifndef METAMORPH_DYNAMIC_SYMBOLS_H
#define METAMORPH_DYNAMIC_SYMBOLS_H
#ifdef __cplusplus
extern "C" {
#endif

#include "metamorph.h"
//A storage struct for dynamically loaded library handles, not meant for users but needs to be exposed to the plugins
struct backend_handles {
  void * openmp_be_handle;
  void * openmp_lib_handle;
  void * opencl_be_handle;
  void * opencl_lib_handle;
  void * cuda_be_handle;
  void * cuda_lib_handle;
};
struct plugin_handles {
  void * mpi_handle;
  void * profiling_handle;
};

#define CHECKED_DLSYM(lib, handle, sym, sym_ptr) {\
  sym_ptr = dlsym(handle, sym);\
  char *sym_err; \
  if ((sym_err = dlerror()) != NULL) {\
    fprintf(stderr, "Could not dynamically load symbol \"%s\" in library \"%s\", error: \"%s\"\n", sym, lib, sym_err);\
    exit(1);\
  }\
}

struct cuda_dyn_ptrs {
  a_err (* metaCUDAAlloc)(void**, size_t);
  a_err (* metaCUDAFree)(void*);
  a_err (* metaCUDAWrite)(void*, void*, size_t, a_bool, meta_callback*, meta_event *);
  a_err (* metaCUDARead)(void*, void*, size_t, a_bool, meta_callback*, meta_event *);
  a_err (* metaCUDADevCopy)(void*, void*, size_t, a_bool, meta_callback*, meta_event *);
  a_err (* metaCUDAInitByID)(a_int);
  a_err (* metaCUDACurrDev)(a_int*);
  a_err (* metaCUDAMaxWorkSizes)(a_dim3*, a_dim3*);
  a_err (* metaCUDAFlush)();
  a_err (* metaCUDACreateEvent)(void**);
  a_err (* metaCUDADestroyEvent)(void*);
  a_err (* metaCUDARegisterCallback)(meta_callback);
  a_err (* cuda_dotProd)(size_t (*)[3], size_t (*)[3], void *, void *, size_t (*)[3], size_t (*)[3], size_t (*)[3], void *, meta_type_id, int, meta_callback *, meta_event *);
  a_err (* cuda_reduce)(size_t (*)[3], size_t (*)[3], void *, size_t (*)[3], size_t (*)[3], size_t (*)[3], void *, meta_type_id, int, meta_callback *, meta_event *);
  a_err (* cuda_transpose_face)(size_t (*)[3], size_t (*)[3], void *, void *, size_t (*)[3], size_t (*)[3], meta_type_id, int, meta_callback *, meta_event *);
  a_err (* cuda_pack_face)(size_t (*)[3], size_t (*)[3], void *, void *, meta_face *, int *, meta_type_id, int, meta_callback *, meta_event *, meta_event *, meta_event *, meta_event *);
  a_err (* cuda_unpack_face)(size_t (*)[3], size_t (*)[3], void *, void *, meta_face *, int *, meta_type_id, int, meta_callback *, meta_event *, meta_event *, meta_event *, meta_event *);
  a_err (* cuda_stencil_3d7p)(size_t (*)[3], size_t (*)[3], void *, void *, size_t (*)[3], size_t (*)[3], size_t (*)[3], meta_type_id, int, meta_callback *, meta_event *);
};

struct opencl_dyn_ptrs {
  void (* metaOpenCLFallback)(void);
  a_err (* metaOpenCLAlloc)(void**, size_t);
  a_err (* metaOpenCLFree)(void*);
  a_err (* metaOpenCLWrite)(void*, void*, size_t, a_bool, meta_callback*, meta_event *);
  a_err (* metaOpenCLRead)(void*, void*, size_t, a_bool, meta_callback*, meta_event *);
  a_err (* metaOpenCLDevCopy)(void*, void*, size_t, a_bool, meta_callback*, meta_event *);
  a_err (* metaOpenCLInitByID)(a_int);
  a_err (* metaOpenCLCurrDev)(a_int*);
  a_err (* metaOpenCLMaxWorkSizes)(a_dim3*, a_dim3*);
  a_err (* metaOpenCLFlush)();
  a_err (* metaOpenCLCreateEvent)(void**);
  a_err (* metaOpenCLRegisterCallback)(meta_callback);
  a_err (* opencl_dotProd)(size_t (*)[3], size_t (*)[3], void *, void *, size_t (*)[3], size_t (*)[3], size_t (*)[3], void *, meta_type_id, int, meta_callback *, meta_event *);
  a_err (* opencl_reduce)(size_t (*)[3], size_t (*)[3], void *, size_t (*)[3], size_t (*)[3], size_t (*)[3], void *, meta_type_id, int, meta_callback *, meta_event *);
a_err (* opencl_transpose_face)(size_t (*)[3], size_t (*)[3], void *, void *, size_t (*)[3], size_t (*)[3], meta_type_id, int, meta_callback *, meta_event *);
a_err (* opencl_pack_face)(size_t (*)[3], size_t (*)[3], void *, void *, meta_face *, int *, meta_type_id, int, meta_callback *, meta_event *, meta_event *, meta_event *, meta_event *);
a_err (* opencl_unpack_face)(size_t (*)[3], size_t (*)[3], void *, void *, meta_face *, int *, meta_type_id, int, meta_callback *, meta_event *, meta_event *, meta_event *, meta_event *);
a_err (* opencl_stencil_3d7p)(size_t (*)[3], size_t (*)[3], void *, void *, size_t (*)[3], size_t (*)[3], size_t (*)[3], meta_type_id, int, meta_callback *, meta_event *);
a_err (* opencl_csr)(size_t (*)[3], size_t (*)[3], size_t, void *, void *, void *, void *, void *, meta_type_id, int, meta_callback *, meta_event *);
a_err (* opencl_crc)(void *, int, int, int, void *, meta_type_id, int, meta_callback *, meta_event *);
};

struct openmp_dyn_ptrs {
  a_err (* metaOpenMPAlloc)(void**, size_t);
  a_err (* metaOpenMPFree)(void*);
  a_err (* metaOpenMPWrite)(void*, void*, size_t, a_bool, meta_callback*, meta_event *);
  a_err (* metaOpenMPRead)(void*, void*, size_t, a_bool, meta_callback*, meta_event *);
  a_err (* metaOpenMPDevCopy)(void*, void*, size_t, a_bool, meta_callback*, meta_event *);
  a_err (* metaOpenMPFlush)();
  a_err (* metaOpenMPCreateEvent)(void**);
  a_err (* metaOpenMPDestroyEvent)(void*);
  a_err (* metaOpenMPRegisterCallback)(meta_callback *);
  a_err (* openmp_dotProd)(size_t (*)[3], size_t (*)[3], void *, void *, size_t (*)[3], size_t (*)[3], size_t (*)[3], void *, meta_type_id, int, meta_callback *, meta_event *);
  a_err (* openmp_reduce)(size_t (*)[3], size_t (*)[3], void *, size_t (*)[3], size_t (*)[3], size_t (*)[3], void *, meta_type_id, int, meta_callback *, meta_event *);
a_err (* openmp_transpose_face)(size_t (*)[3], size_t (*)[3], void *, void *, size_t (*)[3], size_t (*)[3], meta_type_id, int, meta_callback *, meta_event *);
a_err (* openmp_pack_face)(size_t (*)[3], size_t (*)[3], void *, void *, meta_face *, int *, meta_type_id, int, meta_callback *, meta_event *, meta_event *, meta_event *, meta_event *);
a_err (* openmp_unpack_face)(size_t (*)[3], size_t (*)[3], void *, void *, meta_face *, int *, meta_type_id, int, meta_callback *, meta_event *, meta_event *, meta_event *, meta_event *);
a_err (* openmp_stencil_3d7p)(size_t (*)[3], size_t (*)[3], void *, void *, size_t (*)[3], size_t (*)[3], size_t (*)[3], meta_type_id, int, meta_callback *, meta_event *);
  
};

#ifndef METAMORPH_TIMERS_H
#include "metamorph_timers.h"
#endif
struct profiling_dyn_ptrs {
a_err (* metaProfilingCreateTimer)(meta_timer **, meta_preferred_mode, size_t);
a_err (* metaProfilingEnqueueTimer)(meta_timer, metaProfilingBuiltinQueueType);
a_err (* metaProfilingDestroyTimer)(meta_timer *);
};

struct mpi_dyn_ptrs {
  a_err (* finish_mpi_requests)();
};

void meta_load_libs();
void meta_close_libs();
#ifdef __cplusplus
}
#endif
#endif //METAMORPH_DYNAMIC_SYMBOLS_H
