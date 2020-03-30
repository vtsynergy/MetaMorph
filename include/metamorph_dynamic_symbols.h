/** \file
 * Internal data structures for managing dynamically-loaded backends and plugins
 */

#ifndef METAMORPH_DYNAMIC_SYMBOLS_H
#define METAMORPH_DYNAMIC_SYMBOLS_H
#ifdef __cplusplus
extern "C" {
#endif

#include "metamorph.h"
/** A storage struct for dynamically loaded library handles, not meant for users
 * but needs to be exposed to the plugins */
struct backend_handles {
  /** Handle from dlopen for libmm_openmp_backend.so */
  void *openmp_be_handle;
  /** Handle from dlopen for whatever the OpenMP implementation library is,
   * currently unused */
  void *openmp_lib_handle;
  /** Handle from dlopen for libmm_opencl_backend.so */
  void *opencl_be_handle;
  /** Handle from dlopen for libOpenCL.so */
  void *opencl_lib_handle;
  /** Handle from dlopen for libmm_cuda_backend.so */
  void *cuda_be_handle;
  /** Handle from dlopen for libcudart.so */
  void *cuda_lib_handle;
};
/** A storage strut for dynamically-loaded plugin library handles, not meant for
 * users but needs to be exposed to the backends */
struct plugin_handles {
  /** Handle from dlopen for libmm_mpi.so */
  void *mpi_handle;
  /** Handle from dlopen for libmm_profiling.so */
  void *profiling_handle;
};

/** Try to load a symbol by name from a given library that already has a handle
 * \param lib A char const string with the library name for diagnostic
 * \param handle The handle from dlopen to look for the symbol in
 * \param sym A char const string witht he symbol name to look for
 * \param sym_ptr The function pointer to save the retreived symbol to
 */
#define CHECKED_DLSYM(lib, handle, sym, sym_ptr)                               \
  {                                                                            \
    sym_ptr = dlsym(handle, sym);                                              \
    char *sym_err;                                                             \
    if ((sym_err = dlerror()) != NULL) {                                       \
      fprintf(stderr,                                                          \
              "Could not dynamically load symbol \"%s\" in library \"%s\", "   \
              "error: \"%s\"\n",                                               \
              sym, lib, sym_err);                                              \
      exit(1);                                                                 \
    }                                                                          \
  }

/**
 * Struct to hold CUDA wrapper function pointers that the core library needs to
 * reference
 */
struct cuda_dyn_ptrs {
  /** Dynamically-loaded pointer to the CUDA device allocate function */
  a_err (*metaCUDAAlloc)(void **, size_t);
  /** Dynamically-loaded pointer to the CUDA device free function */
  a_err (*metaCUDAFree)(void *);
  /** Dynamically-loaded pointer to the CUDA device write function */
  a_err (*metaCUDAWrite)(void *, void *, size_t, a_bool, meta_callback *,
                         meta_event *);
  /** Dynamically-loaded pointer to the CUDA device read function */
  a_err (*metaCUDARead)(void *, void *, size_t, a_bool, meta_callback *,
                        meta_event *);
  /** Dynamically-loaded pointer to the CUDA device copy function */
  a_err (*metaCUDADevCopy)(void *, void *, size_t, a_bool, meta_callback *,
                           meta_event *);
  /** Dynamically-loaded pointer to the function to initialize the n-th CUDA
   * device*/
  a_err (*metaCUDAInitByID)(a_int);
  /** Dynamically-loaded pointer to the function to query the current CUDA
   * device ID*/
  a_err (*metaCUDACurrDev)(a_int *);
  /** Dynamically-loaded pointer to the function to the function to validate the
   * CUDA work sizes */
  a_err (*metaCUDAMaxWorkSizes)(a_dim3 *, a_dim3 *);
  /** Dynamically-loaded pointer to the function to flush all outstanding CUDA
   * work */
  a_err (*metaCUDAFlush)();
  /** Dynamically-loaded pointer to the function to allocate and initialize two
   * cudaEvent_ts for a meta_event */
  a_err (*metaCUDACreateEvent)(void **);
  /** Dynamically-loaded pointer to the function to destroy two cudaEvent_ts
   * from a meta_event */
  a_err (*metaCUDADestroyEvent)(void *);
  /** Dynamically-loaded pointer to the function to register a callback through
   * the CUDA machinery*/
  a_err (*metaCUDARegisterCallback)(meta_callback);
  /** Dynamically-loaded pointer to the CUDA dot product function */
  a_err (*cuda_dotProd)(size_t (*)[3], size_t (*)[3], void *, void *,
                        size_t (*)[3], size_t (*)[3], size_t (*)[3], void *,
                        meta_type_id, int, meta_callback *, meta_event *);
  /** Dynamically-loaded pointer to the CUDA reduction sum function */
  a_err (*cuda_reduce)(size_t (*)[3], size_t (*)[3], void *, size_t (*)[3],
                       size_t (*)[3], size_t (*)[3], void *, meta_type_id, int,
                       meta_callback *, meta_event *);
  /** Dynamically-loaded pointer to the CUDA face transpose function */
  a_err (*cuda_transpose_face)(size_t (*)[3], size_t (*)[3], void *, void *,
                               size_t (*)[3], size_t (*)[3], meta_type_id, int,
                               meta_callback *, meta_event *);
  /** Dynamically-loaded pointer to the CUDA face packing function */
  a_err (*cuda_pack_face)(size_t (*)[3], size_t (*)[3], void *, void *,
                          meta_face *, int *, meta_type_id, int,
                          meta_callback *, meta_event *, meta_event *,
                          meta_event *, meta_event *);
  /** Dynamically-loaded pointer to the CUDA face unpacking function */
  a_err (*cuda_unpack_face)(size_t (*)[3], size_t (*)[3], void *, void *,
                            meta_face *, int *, meta_type_id, int,
                            meta_callback *, meta_event *, meta_event *,
                            meta_event *, meta_event *);
  /** Dynamically-loaded pointer to the CUDA jacobi stencil function */
  a_err (*cuda_stencil_3d7p)(size_t (*)[3], size_t (*)[3], void *, void *,
                             size_t (*)[3], size_t (*)[3], size_t (*)[3],
                             meta_type_id, int, meta_callback *, meta_event *);
};

/**
 * Struct to hold OpenCL wrapper function pointers that the core library needs
 * to reference
 */
struct opencl_dyn_ptrs {
  /** Dynamically-loaded pointer to the metaOpenCLFallback initializer function
   */
  void (*metaOpenCLFallback)(void);
  /** Dynamically-loaded pointer to the meta_destroy_OpenCL destructor function
   */
  a_int (*meta_destroy_OpenCL)();
  /** Dynamically-loaded pointer to the OpenCL device allocator function */
  a_err (*metaOpenCLAlloc)(void **, size_t);
  /** Dynamically-loaded pointer to the OpenCL device free function */
  a_err (*metaOpenCLFree)(void *);
  /** Dynamically-loaded pointer to the OpenCL device write function */
  a_err (*metaOpenCLWrite)(void *, void *, size_t, a_bool, meta_callback *,
                           meta_event *);
  /** Dynamically-loaded pointer to the OpenCL device read function */
  a_err (*metaOpenCLRead)(void *, void *, size_t, a_bool, meta_callback *,
                          meta_event *);
  /** Dynamically-loaded pointer to the OpenCL device copy function */
  a_err (*metaOpenCLDevCopy)(void *, void *, size_t, a_bool, meta_callback *,
                             meta_event *);
  /** Dynamically-loaded pointer to the OpenCL initializer by ID function */
  a_err (*metaOpenCLInitByID)(a_int);
  /** Dynamically-loaded pointer to the function to get the current OpenCL
   * device ID */
  a_err (*metaOpenCLCurrDev)(a_int *);
  /** Dynamically-loaded pointer to the function to check OpenCL work sizes */
  a_err (*metaOpenCLMaxWorkSizes)(a_dim3 *, a_dim3 *);
  /** Dynamically-loaded pointer to the function to flush the OpenCL queue */
  a_err (*metaOpenCLFlush)();
  /** Dynamically-loaded pointer to the function to allocate and initialize a
   * cl_event for inclusion in a meta_event */
  a_err (*metaOpenCLCreateEvent)(void **);
  /** Dynamically-loaded pointer to the function to register a meta_callback
   * through the OpenCL callback machinery */
  a_err (*metaOpenCLRegisterCallback)(meta_callback);
  /** Dynamically-loaded pointer to the OpenCL dot product function */
  a_err (*opencl_dotProd)(size_t (*)[3], size_t (*)[3], void *, void *,
                          size_t (*)[3], size_t (*)[3], size_t (*)[3], void *,
                          meta_type_id, int, meta_callback *, meta_event *);
  /** Dynamically-loaded pointer to the OpenCL sum reduction function */
  a_err (*opencl_reduce)(size_t (*)[3], size_t (*)[3], void *, size_t (*)[3],
                         size_t (*)[3], size_t (*)[3], void *, meta_type_id,
                         int, meta_callback *, meta_event *);
  /** Dynamically-loaded pointer to the OpenCL transpose function */
  a_err (*opencl_transpose_face)(size_t (*)[3], size_t (*)[3], void *, void *,
                                 size_t (*)[3], size_t (*)[3], meta_type_id,
                                 int, meta_callback *, meta_event *);
  /** Dynamically-loaded pointer to the OpenCL Face packing function */
  a_err (*opencl_pack_face)(size_t (*)[3], size_t (*)[3], void *, void *,
                            meta_face *, int *, meta_type_id, int,
                            meta_callback *, meta_event *, meta_event *,
                            meta_event *, meta_event *);
  /** Dynamically-loaded pointer to the OpenCL Face unpacking function */
  a_err (*opencl_unpack_face)(size_t (*)[3], size_t (*)[3], void *, void *,
                              meta_face *, int *, meta_type_id, int,
                              meta_callback *, meta_event *, meta_event *,
                              meta_event *, meta_event *);
  /** Dynamically-loaded pointer to the OpenCL Jacobi Stencil function */
  a_err (*opencl_stencil_3d7p)(size_t (*)[3], size_t (*)[3], void *, void *,
                               size_t (*)[3], size_t (*)[3], size_t (*)[3],
                               meta_type_id, int, meta_callback *,
                               meta_event *);
  /** Dynamically-loaded pointer to the OpenCL SPMV function */
  a_err (*opencl_csr)(size_t (*)[3], size_t (*)[3], size_t, void *, void *,
                      void *, void *, void *, meta_type_id, int,
                      meta_callback *, meta_event *);
  /** Dynamically-loaded pointer to the OpenCL CRC function */
  a_err (*opencl_crc)(void *, int, int, int, void *, meta_type_id, int,
                      meta_callback *, meta_event *);
};

/**
 * Struct to hold OpenMP wrapper function pointers that the core library needs
 * to reference
 */
struct openmp_dyn_ptrs {
  /** Dynamically-loaded pointer to the OpenMP allocator function */
  a_err (*metaOpenMPAlloc)(void **, size_t);
  /** Dynamically-loaded pointer to the OpenMP free function */
  a_err (*metaOpenMPFree)(void *);
  /** Dynamically-loaded pointer to the OpenMP host-to-device write function */
  a_err (*metaOpenMPWrite)(void *, void *, size_t, a_bool, meta_callback *,
                           meta_event *);
  /** Dynamically-loaded pointer to the OpenMP device-to-host read function */
  a_err (*metaOpenMPRead)(void *, void *, size_t, a_bool, meta_callback *,
                          meta_event *);
  /** Dynamically-loaded pointer to the OpenMP device-to-device copy function */
  a_err (*metaOpenMPDevCopy)(void *, void *, size_t, a_bool, meta_callback *,
                             meta_event *);
  /** Dynamically-loaded pointer to the function to finish any outstanding
   * OpenMP work */
  a_err (*metaOpenMPFlush)();
  /** Dynamically-loaded pointer to the function to create an openmpEvent */
  a_err (*metaOpenMPCreateEvent)(void **);
  /** Dynamically-loaded pointer to the function to destroy an openmpEvent */
  a_err (*metaOpenMPDestroyEvent)(void *);
  /** Dynamically-loaded pointer to the function to register a callback function
   * with the OpenMP backend */
  a_err (*metaOpenMPRegisterCallback)(meta_callback *);
  /** Dynamically-loaded pointer to the OpenMP dot product function */
  a_err (*openmp_dotProd)(size_t (*)[3], size_t (*)[3], void *, void *,
                          size_t (*)[3], size_t (*)[3], size_t (*)[3], void *,
                          meta_type_id, int, meta_callback *, meta_event *);
  /** Dynamically-loaded pointer to the OpenMP reduction sum function */
  a_err (*openmp_reduce)(size_t (*)[3], size_t (*)[3], void *, size_t (*)[3],
                         size_t (*)[3], size_t (*)[3], void *, meta_type_id,
                         int, meta_callback *, meta_event *);
  /** Dynamically-loaded pointer to the OpenMP transpose function */
  a_err (*openmp_transpose_face)(size_t (*)[3], size_t (*)[3], void *, void *,
                                 size_t (*)[3], size_t (*)[3], meta_type_id,
                                 int, meta_callback *, meta_event *);
  /** Dynamically-loaded pointer to the OpenMP face packing function */
  a_err (*openmp_pack_face)(size_t (*)[3], size_t (*)[3], void *, void *,
                            meta_face *, int *, meta_type_id, int,
                            meta_callback *, meta_event *, meta_event *,
                            meta_event *, meta_event *);
  /** Dynamically-loaded pointer to the OpenMP face unpacking function */
  a_err (*openmp_unpack_face)(size_t (*)[3], size_t (*)[3], void *, void *,
                              meta_face *, int *, meta_type_id, int,
                              meta_callback *, meta_event *, meta_event *,
                              meta_event *, meta_event *);
  /** Dynamically-loaded pointer to the OpenMP 3D &-point Jacobi stencil
   * function */
  a_err (*openmp_stencil_3d7p)(size_t (*)[3], size_t (*)[3], void *, void *,
                               size_t (*)[3], size_t (*)[3], size_t (*)[3],
                               meta_type_id, int, meta_callback *,
                               meta_event *);
};

#ifndef METAMORPH_PROFILING_H
#include "metamorph_profiling.h"
#endif
/**
 * Struct to hold Profiling functions the main library needs to reference, to
 * create and enqueue timers, and forcibly flush currently-held timers
 */
struct profiling_dyn_ptrs {
  /** Pointer to function to flush all curently-held timing results */
  a_err (*metaTimersFinish)();
  /** Pointer to function to create a new timer */
  a_err (*metaProfilingCreateTimer)(meta_timer **, meta_preferred_mode, size_t);
  /** Pointer to function to enqueue the new timer */
  a_err (*metaProfilingEnqueueTimer)(meta_timer, metaProfilingBuiltinQueueType);
};

/**
 * Struct to hold MPI wrapper functions the main library may need to reference,
 * for now just destructor and flush
 */
struct mpi_dyn_ptrs {
  /** Dynamically-loaded pointer to the MPIFinalize wrapper */
  a_err (*meta_mpi_finalize)();
  /** Dynamically-loaded pointer to the function to finish outstanding MPI work
   * without finalizing MPI */
  void (*finish_mpi_requests)();
};

/** The loader function that looks for each of the backends and plugins and
 * tries to lookup the required symbols */
void meta_load_libs();
/** The cleanup function that dlcloses each of the backends or plugins that was
 * successfully loaded */
void meta_close_libs();
#ifdef __cplusplus
}
#endif
#endif // METAMORPH_DYNAMIC_SYMBOLS_H
