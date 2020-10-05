/** Defines necessary to override the behavior of the functions in metamorph_shim.c with treating MetaMorph as an optional plugin.
 * Essentially these defines will attempt to dynamically load Metamorph inside the emulated functions, and if it is found, defer to it.
 * Only if it is not found will the emulation route be used.
 * This file is prepended to metamorph_shim.c automatically if MetaCL is run with --use-metamorph=OPTIONAL
 */
//The non-dynamic includes are preotected from duplication by header guards, we will need some types though
#include <CL/cl.h>
#include "metamorph.h"
#include "metamorph_opencl.h"
#include <dlfcn.h>
/** Try to load a symbol by name from a given library that already has a handle
 * \param lib A char const string with the library name for diagnostic
 * \param handle The handle from dlopen to look for the symbol in
 * \param sym A char const string witht he symbol name to look for
 * \param sym_ptr The function pointer to save the retreived symbol to
 */
#define CHECKED_DLSYM(lib, handle, sym, sym_ptr)                               \
  {                                                                            \
    *(void **)(&sym_ptr) = dlsym(handle, sym);                                              \
    char *sym_err;                                                             \
    if ((sym_err = dlerror()) != NULL) {                                       \
      fprintf(stderr,                                                          \
              "Could not dynamically load symbol \"%s\" in library \"%s\", "   \
              "warning: \"%s\"\n",                                               \
              sym, lib, sym_err);                                              \
    }                                                                          \
  }
struct __meta_emu_dyn_syms {
  void * metamorph_handle;
  void * metamorph_opencl_handle;
  void (*meta_init)();
  void (*meta_finalize)();
  meta_err (*meta_register_module)(meta_module_record *(*)(meta_module_record*));
  meta_err (*meta_deregister_module)(meta_module_record *(*)(meta_module_record*));
  meta_err (*meta_set_acc)(int, meta_preferred_mode);
  meta_err (*meta_get_acc)(int*, meta_preferred_mode*);
  void (*metaOpenCLFallback)();
  size_t (*metaOpenCLLoadProgramSource)(const char*, const char **, const char **);
  meta_cl_device_vendor (*metaOpenCLDetectDevice)(cl_device_id);
  meta_int (*meta_get_state_OpenCL)(cl_platform_id*, cl_device_id*, cl_context*, cl_command_queue*);
  meta_int (*meta_set_state_OpenCL)(cl_platform_id, cl_device_id, cl_context, cl_command_queue);
};

struct __meta_emu_dyn_syms __meta_syms = {NULL};
  //If dynamic, this is where you'd call the loader and bind the functions 
#define __META_INIT_DYN { \
  if (__meta_syms.metamorph_handle == NULL) { \
    __meta_syms.metamorph_handle = dlopen("libmetamorph.so", RTLD_NOW | RTLD_GLOBAL); \
  } \
  if (__meta_syms.metamorph_handle != NULL && __meta_syms.metamorph_opencl_handle == NULL) { \
    __meta_syms.metamorph_opencl_handle = dlopen("libmetamorph_opencl.so", RTLD_NOW | RTLD_GLOBAL); \
  } \
  if (__meta_syms.metamorph_handle != NULL) { \
    CHECKED_DLSYM("libmetamorph.so", __meta_syms.metamorph_handle, "meta_init", __meta_syms.meta_init); \
    CHECKED_DLSYM("libmetamorph.so", __meta_syms.metamorph_handle, "meta_finalize", __meta_syms.meta_finalize); \
    CHECKED_DLSYM("libmetamorph.so", __meta_syms.metamorph_handle, "meta_register_module", __meta_syms.meta_register_module); \
    CHECKED_DLSYM("libmetamorph.so", __meta_syms.metamorph_handle, "meta_deregister_module", __meta_syms.meta_deregister_module); \
    CHECKED_DLSYM("libmetamorph.so", __meta_syms.metamorph_handle, "meta_set_acc", __meta_syms.meta_set_acc); \
    CHECKED_DLSYM("libmetamorph.so", __meta_syms.metamorph_handle, "meta_get_acc", __meta_syms.meta_get_acc); \
  } \
  if (__meta_syms.metamorph_handle != NULL) { \
    CHECKED_DLSYM("libmetamorph_opencl.so", __meta_syms.metamorph_opencl_handle, "metaOpenCLFallback", __meta_syms.metaOpenCLFallback); \
    CHECKED_DLSYM("libmetamorph_opencl.so", __meta_syms.metamorph_opencl_handle, "metaOpenCLLoadProgramSource", __meta_syms.metaOpenCLLoadProgramSource); \
    CHECKED_DLSYM("libmetamorph_opencl.so", __meta_syms.metamorph_opencl_handle, "metaOpenCLDetectDevice", __meta_syms.metaOpenCLDetectDevice); \
    CHECKED_DLSYM("libmetamorph_opencl.so", __meta_syms.metamorph_opencl_handle, "meta_get_state_OpenCL", __meta_syms.meta_get_state_OpenCL); \
    CHECKED_DLSYM("libmetamorph_opencl.so", __meta_syms.metamorph_opencl_handle, "meta_set_state_OpenCL", __meta_syms.meta_set_state_OpenCL); \
  } \
  if (__meta_syms.meta_init != NULL) { \
    return; \
  } \
}
  //If dynamic, this is where you'd close the library and invalidate the symbols
#define __META_FINALIZE_DYN { \
  if (__meta_syms.metamorph_opencl_handle != NULL) { \
    __meta_syms.metaOpenCLFallback = NULL; \
    __meta_syms.metaOpenCLLoadProgramSource = NULL; \
    __meta_syms.metaOpenCLDetectDevice = NULL; \
    __meta_syms.meta_get_state_OpenCL = NULL; \
    __meta_syms.meta_set_state_OpenCL = NULL; \
    dlclose(__meta_syms.metamorph_opencl_handle); \
  } \
  if (__meta_syms.metamorph_handle != NULL) { \
    __meta_syms.meta_init = NULL; \
    __meta_syms.meta_finalize = NULL; \
    __meta_syms.meta_register_module = NULL; \
    __meta_syms.meta_deregister_module = NULL; \
    __meta_syms.meta_set_acc = NULL; \
    __meta_syms.meta_get_acc = NULL; \
    dlclose(__meta_syms.metamorph_handle); \
    return; \
  } \
}
  //If dynamic just pass through
#define __META_REGISTER_DYN { \
  if (__meta_syms.meta_register_module != NULL) { \
    return (*__meta_syms.meta_register_module)(module_registry_func); \
  } \
}
  //If dynamic, just pass through
#define __META_DEREGISTER_DYN { \
  if (__meta_syms.meta_deregister_module != NULL) { \
    return (*__meta_syms.meta_deregister_module)(module_registry_func); \
  } \
}
  //If dynamic, just pass through
#define __META_SET_ACC_DYN { \
  if (__meta_syms.meta_set_acc != NULL) { \
    return (*__meta_syms.meta_set_acc)(accel, mode); \
  } \
}
  //If dynamic, just pass through
#define __META_GET_ACC_DYN { \
  if (__meta_syms.meta_get_acc != NULL) { \
    return (*__meta_syms.meta_get_acc)(accel, mode); \
  } \
}
  //If dynamic, just pass through
#define __META_OCL_FALLBACK_DYN { \
  if (__meta_syms.metaOpenCLFallback != NULL) { \
    (*__meta_syms.metaOpenCLFallback)(); \
    return; \
  } \
}
  //If dynamic, just pass through
#define __META_OCL_LOAD_SRC_DYN { \
  if (__meta_syms.metaOpenCLLoadProgramSource != NULL) { \
    return (*__meta_syms.metaOpenCLLoadProgramSource)(filename, progSrc, foundFileDir); \
  } \
}
  //If dynamic, just pass through
#define __META_OCL_DETECT_DEV_DYN { \
  if (__meta_syms.metaOpenCLDetectDevice != NULL) { \
    return (*__meta_syms.metaOpenCLDetectDevice)(dev); \
  } \
}
  //If dynamic, just passthrough
#define __META_OCL_GET_STATE_DYN { \
  if (__meta_syms.meta_get_state_OpenCL != NULL) { \
    return (*__meta_syms.meta_get_state_OpenCL)(platform, device, context, queue); \
  } \
}
  //If dynamic, just pass through
#define __META_OCL_SET_STATE_DYN { \
  if (__meta_syms.meta_set_state_OpenCL != NULL) { \
    return (*__meta_syms.meta_set_state_OpenCL)(platform, device, context, queue); \
  } \
}
