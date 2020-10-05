/** \file Emulations of MetaMorph functions critical to MetaCL-generated programs
 * Contains macro references that will NOOP if MetaCL is run with --use-metamorph=DISABLED
 * However, if there is code before this comment block, MetaCL has been run with --use-metamorph=OPTIONAL, and the above macros will attempt to dynamically-load MetaMorph, and only defer to the emulation if the library is not found.
 */
#include <CL/cl.h>
#include "metamorph.h"
#include "metamorph_opencl.h"

cl_int curr_state;
cl_device_id * __meta_devices;
cl_platform_id * __meta_platforms;
cl_context * __meta_contexts;
cl_command_queue * __meta_queues;

#ifndef __META_INIT_DYN
#define __META_INIT_DYN
#endif
__attribute__((constructor(101))) void meta_init() {
  __META_INIT_DYN
  //In both cases, this is where you'd iterate over the OpenCL devices but creating the basic state tuple would be later (lazy init)
  fprintf(stderr, "EMULATE meta_int\n");
}
#ifndef __META_FINALIZE_DYN
#define __META_FINALIZE_DYN
#endif
__attribute__((destructor(101))) void meta_finalize() {
  //In both cases, this is where you'd release the OpenCL state and any internal objects
  __META_FINALIZE_DYN
  fprintf(stderr, "EMULATE meta_finalize\n");
}
#ifndef __META_REGISTER_DYN
#define __META_REGISTER_DYN
#endif
meta_err meta_register_module(
    meta_module_record *(*module_registry_func)(meta_module_record *record)) {
  __META_REGISTER_DYN
  //IF not dynamic, look at what the contract requires us to do, something about filling out hte record and then calling the init pointer if one is provided
  fprintf(stderr, "EMULATE meta_register_module\n");
  return -1;
}
#ifndef __META_DEREGISTER_DYN
#define __META_DEREGISTER_DYN
#endif
meta_err meta_deregister_module(
    meta_module_record *(*module_registry_func)(meta_module_record *record)) {
  __META_DEREGISTER_DYN
  //If not dynamic, look at what the contract requires us to do, something about invalidating and possibly freeing the record after calling the destructor pointer
  fprintf(stderr, "EMULATE meta_deregister_module\n");
  return -1;
}
#ifndef __META_SET_ACC_DYN
#define __META_SET_ACC_DYN
#endif
meta_err meta_set_acc(int accel, meta_preferred_mode mode) {
  __META_SET_ACC_DYN
  //If not dynamic just immediately create a new state tuple for the selected deviice, and call any registered initializer
  fprintf(stderr, "EMULATE meta_set_acc\n");
  //Confirm the mode is OpenCL, otherwise error
  return -1;
}
#ifndef __META_GET_ACC_DYN
#define __META_GET_ACC_DYN
#endif
meta_err meta_get_acc(int *accel, meta_preferred_mode *mode) {
  __META_GET_ACC_DYN
  //If not dynamic, just map the current state tuple's device into the list index
  //Always returrn OpenCL mode
  fprintf(stderr, "EMULATE meta_get_acc\n");
  return -1;
}
#ifndef __META_OCL_FALLBACK_DYN
#define __META_OCL_FALLBACK_DYN
#endif
void metaOpenCLFallback() {
  __META_OCL_FALLBACK_DYN
  //If not dynamic, just make sure we have a device array and initialize the zeroth device
  fprintf(stderr, "EMULATE metaOpenCLFallback\n");
}
#ifndef __META_OCL_LOAD_SRC_DYN
#define __META_OCL_LOAD_SRC_DYN
#endif
size_t metaOpenCLLoadProgramSource(const char *filename, const char **progSrc,
                                   const char **foundFileDir) {
  __META_OCL_LOAD_SRC_DYN
  //If not dynamic, do the dumb local file read without path
  //If it's not found, just return a NULL pointer and -1;
  //Always return . as the foundFileDir
  fprintf(stderr, "EMULATE metaOpenCLLoadProgramSource\n");
  return -1;
}
#ifndef __META_OCL_DETECT_DEV_DYN
#define __META_OCL_DETECT_DEV_DYN
#endif
meta_cl_device_vendor metaOpenCLDetectDevice(cl_device_id dev) {
  __META_OCL_DETECT_DEV_DYN
  //If not dynamic, the behavior will still be identical, cut and paste
  fprintf(stderr, "EMULATE metaOpenCLDetectDevice\n");
  return meta_cl_device_vendor_unknown;
}
#ifndef __META_OCL_GET_STATE_DYN
#define __META_OCL_GET_STATE_DYN
#endif
meta_int meta_get_state_OpenCL(cl_platform_id *platform, cl_device_id *device,
                            cl_context *context, cl_command_queue *queue) {
  __META_OCL_GET_STATE_DYN
  // f not dynamic, just NULL check each parameter and the current tuple, and set stuff. Very close to a cut-and paste
  fprintf(stderr, "EMULATE meta_get_state_OpenCL\n");
  return -1;
}
#ifndef __META_OCL_SET_STATE_DYN
#define __META_OCL_SET_STATE_DYN
#endif
meta_int meta_set_state_OpenCL(cl_platform_id platform, cl_device_id device,
                            cl_context context, cl_command_queue queue) {
  __META_OCL_SET_STATE_DYN
  //If not dynamic, accept whatever they provided and directly set the tuple
  fprintf(stderr, "EMULATE meta_set_state_OpenCL\n");
  //Call any registered initializers
  return -1;
}
