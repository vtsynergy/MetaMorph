#include <dlfcn.h>
#include "metamorph_dynamic_symbols.h"
struct backend_handles backends = {NULL};
struct cuda_dyn_ptrs cuda_symbols = {NULL};
struct opencl_dyn_ptrs opencl_symbols = {NULL};
struct openmp_dyn_ptrs openmp_symbols = {NULL};
struct plugin_handles plugins = {NULL};
struct profiling_dyn_ptrs profiling_symbols = {NULL};
struct mpi_dyn_ptrs mpi_symbols = {NULL};
//Globally-set capability flag
a_module_implements_backend core_capability = module_uninitialized;

void meta_load_libs() {
  if (core_capability != module_uninitialized) return;
  core_capability = module_implements_general | module_implements_fortran; 
  //do auto-discovery of each backend and plugin
  //FIXME Not sure if we need RTLD_DEEPBIND
  backends.cuda_be_handle = dlopen("libmm_cuda_backend.so", RTLD_NOW | RTLD_GLOBAL);
  backends.cuda_lib_handle = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
  if (backends.cuda_be_handle == NULL) fprintf(stderr, "No CUDA backend detected\n");
  else if (backends.cuda_lib_handle == NULL) fprintf(stderr, "No CUDA runtime detected\n");
  else {
    core_capability |= module_implements_cuda;
    CHECKED_DLSYM("libmm_cuda_backend.so", backends.cuda_be_handle, "metaCUDAAlloc", cuda_symbols.metaCUDAAlloc);
    CHECKED_DLSYM("libmm_cuda_backend.so", backends.cuda_be_handle, "metaCUDAFree", cuda_symbols.metaCUDAFree);
    CHECKED_DLSYM("libmm_cuda_backend.so", backends.cuda_be_handle, "metaCUDAWrite", cuda_symbols.metaCUDAWrite);
    CHECKED_DLSYM("libmm_cuda_backend.so", backends.cuda_be_handle, "metaCUDARead", cuda_symbols.metaCUDARead);
    CHECKED_DLSYM("libmm_cuda_backend.so", backends.cuda_be_handle, "metaCUDADevCopy", cuda_symbols.metaCUDADevCopy);
    CHECKED_DLSYM("libmm_cuda_backend.so", backends.cuda_be_handle, "metaCUDAInitByID", cuda_symbols.metaCUDAInitByID);
    CHECKED_DLSYM("libmm_cuda_backend.so", backends.cuda_be_handle, "metaCUDACurrDev", cuda_symbols.metaCUDACurrDev);
    CHECKED_DLSYM("libmm_cuda_backend.so", backends.cuda_be_handle, "metaCUDAMaxWorkSizes", cuda_symbols.metaCUDAMaxWorkSizes);
    CHECKED_DLSYM("libmm_cuda_backend.so", backends.cuda_be_handle, "metaCUDAFlush", cuda_symbols.metaCUDAFlush);
    CHECKED_DLSYM("libmm_cuda_backend.so", backends.cuda_be_handle, "metaCUDACreateEvent", cuda_symbols.metaCUDACreateEvent);
    CHECKED_DLSYM("libmm_cuda_backend.so", backends.cuda_be_handle, "metaCUDADestroyEvent", cuda_symbols.metaCUDADestroyEvent);
    CHECKED_DLSYM("libmm_cuda_backend.so", backends.cuda_be_handle, "metaCUDARegisterCallback", cuda_symbols.metaCUDARegisterCallback);
    CHECKED_DLSYM("libmm_cuda_backend.so", backends.cuda_be_handle, "cuda_dotProd", cuda_symbols.cuda_dotProd);
    CHECKED_DLSYM("libmm_cuda_backend.so", backends.cuda_be_handle, "cuda_reduce", cuda_symbols.cuda_reduce);
    CHECKED_DLSYM("libmm_cuda_backend.so", backends.cuda_be_handle, "cuda_transpose_face", cuda_symbols.cuda_transpose_face);
    CHECKED_DLSYM("libmm_cuda_backend.so", backends.cuda_be_handle, "cuda_pack_face", cuda_symbols.cuda_pack_face);
    CHECKED_DLSYM("libmm_cuda_backend.so", backends.cuda_be_handle, "cuda_unpack_face", cuda_symbols.cuda_unpack_face);
    CHECKED_DLSYM("libmm_cuda_backend.so", backends.cuda_be_handle, "cuda_stencil_3d7p", cuda_symbols.cuda_stencil_3d7p);

    fprintf(stderr, "CUDA backend found\n");
  }
  backends.opencl_be_handle = dlopen("libmm_opencl_backend.so", RTLD_NOW | RTLD_GLOBAL);
  backends.opencl_lib_handle = dlopen("libOpenCL.so", RTLD_NOW | RTLD_GLOBAL);
  if (backends.opencl_be_handle == NULL) fprintf(stderr, "No OpenCL backend detected\n");
  else if (backends.opencl_lib_handle == NULL) fprintf(stderr, "No OpenCL runtime detected\n");
  else {
    core_capability |= module_implements_opencl;
    CHECKED_DLSYM("libmm_opencl_backend.so", backends.opencl_be_handle, "metaOpenCLFallback", opencl_symbols.metaOpenCLFallback);
    CHECKED_DLSYM("libmm_opencl_backend.so", backends.opencl_be_handle, "metaOpenCLAlloc", opencl_symbols.metaOpenCLAlloc);
    CHECKED_DLSYM("libmm_opencl_backend.so", backends.opencl_be_handle, "metaOpenCLFree", opencl_symbols.metaOpenCLFree);
    CHECKED_DLSYM("libmm_opencl_backend.so", backends.opencl_be_handle, "metaOpenCLWrite", opencl_symbols.metaOpenCLWrite);
    CHECKED_DLSYM("libmm_opencl_backend.so", backends.opencl_be_handle, "metaOpenCLRead", opencl_symbols.metaOpenCLRead);
    CHECKED_DLSYM("libmm_opencl_backend.so", backends.opencl_be_handle, "metaOpenCLDevCopy", opencl_symbols.metaOpenCLDevCopy);
    CHECKED_DLSYM("libmm_opencl_backend.so", backends.opencl_be_handle, "metaOpenCLInitByID", opencl_symbols.metaOpenCLInitByID);
    CHECKED_DLSYM("libmm_opencl_backend.so", backends.opencl_be_handle, "metaOpenCLCurrDev", opencl_symbols.metaOpenCLCurrDev);
    CHECKED_DLSYM("libmm_opencl_backend.so", backends.opencl_be_handle, "metaOpenCLMaxWorkSizes", opencl_symbols.metaOpenCLMaxWorkSizes);
    CHECKED_DLSYM("libmm_opencl_backend.so", backends.opencl_be_handle, "metaOpenCLFlush", opencl_symbols.metaOpenCLFlush);
    CHECKED_DLSYM("libmm_opencl_backend.so", backends.opencl_be_handle, "metaOpenCLCreateEvent", opencl_symbols.metaOpenCLCreateEvent);
    CHECKED_DLSYM("libmm_opencl_backend.so", backends.opencl_be_handle, "metaOpenCLRegisterCallback", opencl_symbols.metaOpenCLRegisterCallback);
    CHECKED_DLSYM("libmm_opencl_backend.so", backends.opencl_be_handle, "opencl_dotProd", opencl_symbols.opencl_dotProd);
    CHECKED_DLSYM("libmm_opencl_backend.so", backends.opencl_be_handle, "opencl_reduce", opencl_symbols.opencl_reduce);
    CHECKED_DLSYM("libmm_opencl_backend.so", backends.opencl_be_handle, "opencl_transpose_face", opencl_symbols.opencl_transpose_face);
    CHECKED_DLSYM("libmm_opencl_backend.so", backends.opencl_be_handle, "opencl_pack_face", opencl_symbols.opencl_pack_face);
    CHECKED_DLSYM("libmm_opencl_backend.so", backends.opencl_be_handle, "opencl_unpack_face", opencl_symbols.opencl_unpack_face);
    CHECKED_DLSYM("libmm_opencl_backend.so", backends.opencl_be_handle, "opencl_stencil_3d7p", opencl_symbols.opencl_stencil_3d7p);
    CHECKED_DLSYM("libmm_opencl_backend.so", backends.opencl_be_handle, "opencl_csr", opencl_symbols.opencl_csr);
    CHECKED_DLSYM("libmm_opencl_backend.so", backends.opencl_be_handle, "opencl_crc", opencl_symbols.opencl_crc);

    fprintf(stderr, "OpenCL backend found\n");
  }
  //If i understand the dynamic loader correctly, we should not need to explicitly load the runtime libs, they will be pulled in automatically be loading the backend
//  dlopen("libOpenCL.so", RTLD_NOW | RTLD_GLOBAL);
  backends.openmp_be_handle = dlopen("libmm_openmp_backend.so", RTLD_NOW | RTLD_GLOBAL);
  backends.openmp_lib_handle = dlopen("libomp.so", RTLD_NOW | RTLD_GLOBAL);
  if (backends.openmp_be_handle == NULL) fprintf(stderr, "No OpenMP backend detected\n");
//  else if (backends.openmp_lib_handle == NULL) fprintf(stderr, "No OpenMP runtime detected\n");
  else {
    core_capability |= module_implements_openmp;
    CHECKED_DLSYM("libmm_openmp_backend.so", backends.openmp_be_handle, "metaOpenMPAlloc", openmp_symbols.metaOpenMPAlloc);
    CHECKED_DLSYM("libmm_openmp_backend.so", backends.openmp_be_handle, "metaOpenMPFree", openmp_symbols.metaOpenMPFree);
    CHECKED_DLSYM("libmm_openmp_backend.so", backends.openmp_be_handle, "metaOpenMPWrite", openmp_symbols.metaOpenMPWrite);
    CHECKED_DLSYM("libmm_openmp_backend.so", backends.openmp_be_handle, "metaOpenMPRead", openmp_symbols.metaOpenMPRead);
    CHECKED_DLSYM("libmm_openmp_backend.so", backends.openmp_be_handle, "metaOpenMPDevCopy", openmp_symbols.metaOpenMPDevCopy);
    CHECKED_DLSYM("libmm_openmp_backend.so", backends.openmp_be_handle, "metaOpenMPFlush", openmp_symbols.metaOpenMPFlush);
    CHECKED_DLSYM("libmm_openmp_backend.so", backends.openmp_be_handle, "metaOpenMPCreateEvent", openmp_symbols.metaOpenMPCreateEvent);
    CHECKED_DLSYM("libmm_openmp_backend.so", backends.openmp_be_handle, "metaOpenMPDestroyEvent", openmp_symbols.metaOpenMPDestroyEvent);
    CHECKED_DLSYM("libmm_openmp_backend.so", backends.openmp_be_handle, "metaOpenMPRegisterCallback", openmp_symbols.metaOpenMPRegisterCallback);
    CHECKED_DLSYM("libmm_openmp_backend.so", backends.openmp_be_handle, "openmp_dotProd", openmp_symbols.openmp_dotProd);
    CHECKED_DLSYM("libmm_openmp_backend.so", backends.openmp_be_handle, "openmp_reduce", openmp_symbols.openmp_reduce);
    CHECKED_DLSYM("libmm_openmp_backend.so", backends.openmp_be_handle, "openmp_transpose_face", openmp_symbols.openmp_transpose_face);
    CHECKED_DLSYM("libmm_openmp_backend.so", backends.openmp_be_handle, "openmp_pack_face", openmp_symbols.openmp_pack_face);
    CHECKED_DLSYM("libmm_openmp_backend.so", backends.openmp_be_handle, "openmp_unpack_face", openmp_symbols.openmp_unpack_face);
    CHECKED_DLSYM("libmm_openmp_backend.so", backends.openmp_be_handle, "openmp_stencil_3d7p", openmp_symbols.openmp_stencil_3d7p);

    fprintf(stderr, "OpenMP backend found\n");
  }
  plugins.mpi_handle = dlopen("libmm_mpi.so", RTLD_NOW | RTLD_GLOBAL);
  if (plugins.mpi_handle == NULL) fprintf(stderr, "No MPI plugin detected\n");
  else {
    core_capability |= module_implements_mpi;
    CHECKED_DLSYM("libmm_mpi.so", plugins.mpi_handle, "finish_mpi_requests", mpi_symbols.finish_mpi_requests);
    fprintf(stderr, "MPI plugin found\n");
  }
  plugins.profiling_handle = dlopen("libmm_profiling.so", RTLD_NOW | RTLD_GLOBAL);
  if (plugins.profiling_handle == NULL) fprintf(stderr, "No profiling plugin detected\n");
  else {
    core_capability |= module_implements_profiling;
    CHECKED_DLSYM("libmm_profiling.so", plugins.profiling_handle, "metaProfilingCreateTimer", profiling_symbols.metaProfilingCreateTimer);
    CHECKED_DLSYM("libmm_profiling.so", plugins.profiling_handle, "metaProfilingEnqueueTimer", profiling_symbols.metaProfilingEnqueueTimer);
//    CHECKED_DLSYM("libmm_profiling.so", plugins.profiling_handle, "metaProfilingDestroyTimer", profiling_symbols.metaProfilingDestroyTimer);
    fprintf(stderr, "Profiling plugin found\n");
  }
}

void meta_close_libs() {
  if (core_capability == module_uninitialized) return;
  if (core_capability & module_implements_cuda) {
    dlclose(backends.cuda_be_handle);
    core_capability &= (~module_implements_cuda);
  }
  if (core_capability & module_implements_opencl) {
    dlclose(backends.opencl_be_handle);
    core_capability &= (~module_implements_opencl);
  }
  if (core_capability & module_implements_openmp) {
    dlclose(backends.openmp_be_handle);
    core_capability &= (~module_implements_openmp);
  }
  if (core_capability & module_implements_mpi) {
    dlclose(plugins.mpi_handle);
    core_capability &= (~module_implements_mpi);
  }
  if (core_capability & module_implements_profiling) {
    dlclose(plugins.profiling_handle);
    core_capability &= (~module_implements_profiling);
  }
  core_capability = module_uninitialized;
};
