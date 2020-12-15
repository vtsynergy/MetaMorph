/** \file Emulations of MetaMorph functions critical to MetaCL-generated programs
 * Contains macro references that will NOOP if MetaCL is run with
 * --use-metamorph=DISABLED However, if there is code before this comment block,
 * MetaCL has been run with --use-metamorph=OPTIONAL, and the above macros will
 * attempt to dynamically-load MetaMorph, and only defer to the emulation if the
 * library is not found.
 *
 * This shim emulation library is (c) 2020-2021 Virginia Tech, and provided without
 * warranty subject to the terms of the LGPL 2.1 License. You may redistribute
 * this shim with your application or library in source or binary form according
 * to the terms of the license. This copyright covers ONLY this and other static
 * MetaCL-generated implementation files, and DOES NOT seek to extend copyright
 * over the application-specific files that MetaCL generates to interface with
 * your kernels.
 *
 *   This library is free software; you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation; either version 2.1 of the License, or (at your option) any later version.
 *
 *   This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

 *   You should have received a copy of the GNU Lesser General Public License along with this library; if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA 
 *
 * A copy of the current license terms may be obtained from https://github.com/vtsynergy/MetaMorph/blob/master/LICENSE
 */
#include <CL/opencl.h>
#include <string.h>
#include "metamorph.h"
#include "metamorph_opencl.h"

//-1 if uninitialized, [0-num_states) if found, num_states if set_state_OpenCL
// didn't match anything in [0-num_states)
cl_int __meta_curr_state = -1;
// Should equal the sum of devices across platforms
cl_uint __meta_num_states = -1;
// Should be size num_states + 1 (for the extra non-matching user-defined state
// at the end)
cl_device_id *__meta_devices;
cl_platform_id *__meta_platforms;
cl_context *__meta_contexts;
cl_command_queue *__meta_queues;

// Simple linked list to hold module registrations
struct __meta_reg_list;
struct __meta_reg_list {
  meta_module_record *record;
  struct __meta_reg_list *next;
} __meta_reg_list;

// List of module registrations
struct __meta_reg_list __meta_known_modules = {NULL};

// Append a new registration IFF the provided one isn't found
void __meta_reg_record(meta_module_record *record) {
  struct __meta_reg_list *curr = &__meta_known_modules;
  for (; curr != NULL; curr = curr->next) {
    if (curr->record == record)
      return;                 // we already have it
    if (curr->next == NULL) { // at the end and didn't find it, record it
      curr->next = (struct __meta_reg_list *)calloc(
          1, sizeof(struct __meta_reg_list)); // Sets next->next as NULL
      curr->next->record = record;            // Records the new record
    }
  }
}

// Remove a registration if we know about it, NOOP otherwise
void __meta_reg_remove(meta_module_record *record) {
  struct __meta_reg_list *curr = &__meta_known_modules;
  for (; curr != NULL; curr = curr->next) {
    if (curr->next != NULL && curr->next->record == record) { // Lookahead one
      struct __meta_reg_list *remove = curr->next;
      curr->next = remove->next;
      free(remove);
      return;
    }
  }
}

// Attempt to call the init function of all known modules
void __meta_module_reinit_all() {
  struct __meta_reg_list *curr = &__meta_known_modules;
  for (; curr != NULL; curr = curr->next) {
    if (curr->record != NULL && curr->record->module_init != NULL)
      (*(curr->record->module_init))();
  }
}

// Attempt to call the deinit funciton of all known modules then deregister all
void __meta_module_drop_all() {
  struct __meta_reg_list *curr = &__meta_known_modules;
  struct __meta_reg_list *next =
      NULL; // Save separately so we can copy the pointer before deregistration
            // frees the list node
  for (; curr != NULL; curr = next) {
    next = curr->next; // Record next before curr becomes invalid
    if (curr->record != NULL) {
      // This will call the destructor, remove curr from the list, and free it,
      // so it won't be a valid pointer after the call
      meta_deregister_module(curr->record->module_registry_func);
    }
  }
}

#ifndef __META_INIT_DYN
#define __META_INIT_DYN
#endif
__attribute__((constructor(101))) void meta_init() {
  __META_INIT_DYN
  // In both cases, this is where you'd iterate over the OpenCL devices but
  // creating the basic state tuple would be later (lazy init)
  __meta_num_states = 0;
  cl_uint num_plats;
  cl_int err = CL_SUCCESS;
  // Figure out number of platforms
  if ((err = clGetPlatformIDs(0, NULL, &num_plats)) != CL_SUCCESS) {
    fprintf(stderr, "Cannot query OpenCL platform count, error: %d\n", err);
    exit(err);
  }
  // Get pointers for all the platforms
  cl_platform_id *plats =
      (cl_platform_id *)calloc(num_plats, sizeof(cl_platform_id));
  if ((err = clGetPlatformIDs(num_plats, plats, NULL)) != CL_SUCCESS) {
    fprintf(stderr, "Cannot query OpenCL platform count, error: %d\n", err);
    exit(err);
  }
  // Figure out total number of devices
  int i = 0;
  cl_uint num_devs;
  for (i = 0; i < num_plats; i++) {
    num_devs = 0;
    if ((err = clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_ALL, 0, NULL,
                              &num_devs)) != CL_SUCCESS) {
      fprintf(stderr,
              "Cannot query OpenCL device  count on platform %d, error: %d\n",
              i, err);
      exit(err);
    }
    __meta_num_states += num_devs;
  }
  // Allocate internal arrays
  __meta_devices =
      (cl_device_id *)calloc(__meta_num_states + 1, sizeof(cl_device_id));
  __meta_platforms =
      (cl_platform_id *)calloc(__meta_num_states + 1, sizeof(cl_platform_id));
  __meta_contexts =
      (cl_context *)calloc(__meta_num_states + 1, sizeof(cl_context));
  __meta_queues = (cl_command_queue *)calloc(__meta_num_states + 1,
                                             sizeof(cl_command_queue));
  // Populate the state arrays
  int next_query = 0, curr_plat = -1;
  for (i = 0; i < __meta_num_states; i++) {
    num_devs = 0;
    if (i == next_query) {
      // Move the platform iterator forward
      curr_plat++;
      // Grab all the devices on the platform at once
      if ((err = clGetDeviceIDs(plats[curr_plat], CL_DEVICE_TYPE_ALL,
                                __meta_num_states - i, &__meta_devices[i],
                                &num_devs)) != CL_SUCCESS) {
        fprintf(stderr,
                "Cannot query OpenCL devices on platform %d, error: %d\n", i,
                err);
        exit(err);
      }
      // Update the next index we'll change platforms and query more devices at
      next_query += num_devs;
    }
    // Copy the current platform to the state slot
    __meta_platforms[i] = plats[curr_plat];
    // We don't actually initialize any contexts/queues here, we will do that
    // when a device is explicit selected or via fallback They are
    // NULL-initialized by virtue of the calloc
  }
}
#ifndef __META_FINALIZE_DYN
#define __META_FINALIZE_DYN
#endif
__attribute__((destructor(101))) void meta_finalize() {
  // In both cases, this is where you'd release the OpenCL state and any
  // internal objects
  __META_FINALIZE_DYN
  // Release any known modules
  __meta_module_drop_all();
  __meta_curr_state = -1;
  cl_int err = CL_SUCCESS;
  int i;
  for (i = 0; i < __meta_num_states; i++) {
    if (__meta_queues[i] != NULL) {
      if ((err = clReleaseCommandQueue(__meta_queues[i])) != CL_SUCCESS)
        fprintf(stderr, "Error releasing OpenCL Queue #%d: %d\n", i, err);
      __meta_queues[i] = NULL;
    }
    if (__meta_contexts[i] != NULL) {
      if ((err = clReleaseContext(__meta_contexts[i])) != CL_SUCCESS)
        fprintf(stderr, "Error releasing OpenCL Context #%d: %d\n", i, err);
      __meta_contexts[i] = NULL;
    }
    if (__meta_devices[i] != NULL) {
      if ((err = clReleaseDevice(__meta_devices[i])) != CL_SUCCESS)
        fprintf(stderr, "Error releasing OpenCL Device #%d: %d\n", i, err);
      __meta_devices[i] = NULL;
    }
  }
  __meta_num_states = -1;
  free(__meta_queues);
  free(__meta_contexts);
  free(__meta_devices);
  free(__meta_platforms);
}

#ifndef __META_REGISTER_DYN
#define __META_REGISTER_DYN
#endif
meta_err meta_register_module(
    meta_module_record *(*module_registry_func)(meta_module_record *record)) {
  __META_REGISTER_DYN
  // IF not dynamic, look at what the contract requires us to do, something
  // about filling out hte record and then calling the init pointer if one is
  // provided

  // Make sure the registration function is callable
  if (module_registry_func == NULL)
    return -1;
  // Call it once to make sure it's not registered
  meta_module_record *previous = (*(module_registry_func))(NULL);
  if (previous == NULL) { // Has no current registration
    meta_module_record *record =
        (meta_module_record *)calloc(1, sizeof(meta_module_record));
    // Provide it the record to fill out
    previous = (*(module_registry_func))(record);
    // Make sure the module accepted the record, if so, save it
    if (previous == NULL && (*(module_registry_func))(NULL) == record) {
      // Record the registration
      __meta_reg_record(record);
      // Invoke any initialization function
      if (record->module_init != NULL)
        (*(record->module_init))();
      return CL_SUCCESS;
    }
  }
  return -1;
}
#ifndef __META_DEREGISTER_DYN
#define __META_DEREGISTER_DYN
#endif
meta_err meta_deregister_module(
    meta_module_record *(*module_registry_func)(meta_module_record *record)) {
  __META_DEREGISTER_DYN
  // If not dynamic, look at what the contract requires us to do, something
  // about invalidating and possibly freeing the record after calling the
  // destructor pointer Make sure the registration function is callable
  if (module_registry_func == NULL)
    return -1;
  // Get the current registration
  meta_module_record *record = (*(module_registry_func))(NULL);
  // Invoke any destructor function
  if (record != NULL && record->module_deinit != NULL)
    (*(record->module_deinit))();
  // Unrecord the registration
  __meta_reg_remove(record);
  // Revoke it's registration
  meta_module_record *previous = (*(module_registry_func))(record);
  // Confirm it knows it's been revoked
  if (previous == record && (*(module_registry_func))(NULL) == NULL) {
    // Free the record
    free(record);
    return CL_SUCCESS;
  }
  return -1;
}
#ifndef __META_SET_ACC_DYN
#define __META_SET_ACC_DYN
#endif
meta_err meta_set_acc(int accel, meta_preferred_mode mode) {
  __META_SET_ACC_DYN
  // If not dynamic just immediately create a new state tuple for the selected
  // deviice, and call any registered initializer Confirm the mode is OpenCL,
  // otherwise error
  if (mode != metaModePreferOpenCL) {
    fprintf(stderr, "Error: MetaCL-generated MetaMorph emulation only supports "
                    "OpenCL mode\n");
    return -1;
  }
  // MetaMorph allows accel to be -1 which means "Pick for me", just default to
  // zeroth
  if (accel == -1)
    accel = 0;
  // Bind this to only the [0-num_devices) states, if they want the user-defined
  // num_devices-th state, they need to explicitly use set_state_OpenCL
  if (accel < 0 || accel >= __meta_num_states) {
    fprintf(stderr,
            "Error: cannot set OpenCL accelerator to unknown device %d\n",
            accel);
    return -1;
  }
  __meta_curr_state = accel;
  cl_int err = CL_SUCCESS;
  // Device and platform are already initialized by the constructor
  // Initialize a context if none exists
  if (__meta_contexts[__meta_curr_state] == NULL) {
    __meta_contexts[__meta_curr_state] = clCreateContext(
        NULL, 1, &__meta_devices[__meta_curr_state], NULL, NULL, &err);
    if (err != CL_SUCCESS) {
      fprintf(stderr, "Error creating OpenCL context for device #%d: %d\n",
              __meta_curr_state, err);
      return err;
    }
  }
  if (__meta_queues[__meta_curr_state] == NULL) {
    __meta_queues[__meta_curr_state] = clCreateCommandQueue(
        __meta_contexts[__meta_curr_state], __meta_devices[__meta_curr_state],
        CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS) {
      fprintf(stderr,
              "Error creating OpenCL Command Queue for device #%d: %d\n",
              __meta_curr_state, err);
      return err;
    }
  }
  // Fire any registered initializers
  __meta_module_reinit_all();
  return CL_SUCCESS;
}
#ifndef __META_GET_ACC_DYN
#define __META_GET_ACC_DYN
#endif
meta_err meta_get_acc(int *accel, meta_preferred_mode *mode) {
  __META_GET_ACC_DYN
  // If not dynamic, just map the current state tuple's device into the list
  // index Always returrn OpenCL mode
  if (accel == NULL || mode == NULL)
    return -1;
  *accel = __meta_curr_state;
  *mode = metaModePreferOpenCL;
  return CL_SUCCESS;
}
#ifndef __META_OCL_FALLBACK_DYN
#define __META_OCL_FALLBACK_DYN
#endif
void metaOpenCLFallback() {
  __META_OCL_FALLBACK_DYN
  // If not dynamic, just make sure we have a device array and initialize the
  // zeroth device Just use the zeroth device
  meta_set_acc(0, metaModePreferOpenCL);
}
#ifndef __META_OCL_LOAD_SRC_DYN
#define __META_OCL_LOAD_SRC_DYN
#endif
size_t metaOpenCLLoadProgramSource(const char *filename, const char **progSrc,
                                   const char **foundFileDir) {
  __META_OCL_LOAD_SRC_DYN
  // If not dynamic, do the dumb local file read without path
  // If it's not found, just return a NULL pointer and -1;
  // Always return . as the foundFileDir
  FILE *f = fopen(filename, "r");
  if (f == NULL) {
    fprintf(stderr, "Error: could not find OpenCL kernel file \"%s\n",
            filename);
    return CL_INVALID_PROGRAM;
  }
  fseek(f, 0, SEEK_END);
  size_t len = (size_t)ftell(f);
  *progSrc = (const char *)malloc(sizeof(char) * len);
  rewind(f);
  fread((void *)*progSrc, len, 1, f);
  fclose(f);
  return len;
}
#ifndef __META_OCL_DETECT_DEV_DYN
#define __META_OCL_DETECT_DEV_DYN
#endif
meta_cl_device_vendor metaOpenCLDetectDevice(cl_device_id dev) {
  __META_OCL_DETECT_DEV_DYN
  // If not dynamic, the behavior will still be identical, cut and paste
  meta_cl_device_vendor ret = meta_cl_device_vendor_unknown;
  // First, query the type of device
  cl_device_type type;
  clGetDeviceInfo(dev, CL_DEVICE_TYPE, sizeof(cl_device_type), &type, NULL);
  if (type & CL_DEVICE_TYPE_CPU)
    ret = (meta_cl_device_vendor)(ret | meta_cl_device_is_cpu);
  if (type & CL_DEVICE_TYPE_GPU)
    ret = (meta_cl_device_vendor)(ret | meta_cl_device_is_gpu);
  if (type & CL_DEVICE_TYPE_ACCELERATOR)
    ret = (meta_cl_device_vendor)(ret | meta_cl_device_is_accel);
  if (type & CL_DEVICE_TYPE_DEFAULT)
    ret = (meta_cl_device_vendor)(ret | meta_cl_device_is_default);

  // Then query the platform, and it's name
  size_t name_size;
  char *name;
  cl_platform_id plat;
  clGetDeviceInfo(dev, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &plat, NULL);
  clGetPlatformInfo(plat, CL_PLATFORM_NAME, 0, NULL, &name_size);
  name = (char *)calloc(sizeof(char), name_size + 1);
  clGetPlatformInfo(plat, CL_PLATFORM_NAME, name_size + 1, name, NULL);

  // and match it to known names
  if (strcmp(name, "NVIDIA CUDA") == 0)
    ret = (meta_cl_device_vendor)(ret | meta_cl_device_vendor_nvidia);
  if (strcmp(name, "AMD Accelerated Parallel Processing") == 0)
    ret = (meta_cl_device_vendor)(ret | meta_cl_device_vendor_amd_appsdk);
  // TODO AMD ROCM
  // TODO Intel CPU/GPU
  if (strcmp(name, "Intel(R) FPGA SDK for OpenCL(TM)") == 0)
    ret = (meta_cl_device_vendor)(ret | meta_cl_device_vendor_intelfpga);
  if (strcmp(name, "Intel(R) FPGA Emulation Platform for OpenCL(TM)") == 0)
    ret = (meta_cl_device_vendor)(ret | meta_cl_device_vendor_intelfpga);
  if (strcmp(name, "Altera SDK for OpenCL") == 0)
    ret = (meta_cl_device_vendor)(ret | meta_cl_device_vendor_intelfpga);
  // TODO Xilinx
  if (strcmp(name, "Portable Computing Language") == 0)
    ret = (meta_cl_device_vendor)(ret | meta_cl_device_vendor_pocl);

  if ((ret & meta_cl_device_vendor_mask) == meta_cl_device_vendor_unknown)
    fprintf(stderr,
            "Warning: OpenCL platform vendor \"%s\" is unrecognized, assuming "
            ".cl inputs and JIT support\n",
            name);
  free(name);
  return ret;
}
#ifndef __META_OCL_GET_STATE_DYN
#define __META_OCL_GET_STATE_DYN
#endif
meta_int meta_get_state_OpenCL(cl_platform_id *platform, cl_device_id *device,
                               cl_context *context, cl_command_queue *queue) {
  __META_OCL_GET_STATE_DYN
  // f not dynamic, just NULL check each parameter and the current tuple, and
  // set stuff. Very close to a cut-and paste
  //-1 is uninitialized, [0-num_states) are builtin, and num_states is a single
  // user-defined slot, all other values are NOOP
  if (__meta_curr_state > -1 && __meta_curr_state <= __meta_num_states) {
    if (platform != NULL)
      *platform = __meta_platforms[__meta_curr_state];
    if (device != NULL)
      *device = __meta_devices[__meta_curr_state];
    if (context != NULL)
      *context = __meta_contexts[__meta_curr_state];
    if (queue != NULL)
      *queue = __meta_queues[__meta_curr_state];
  }
  return __meta_curr_state;
}
#ifndef __META_OCL_SET_STATE_DYN
#define __META_OCL_SET_STATE_DYN
#endif
meta_int meta_set_state_OpenCL(cl_platform_id platform, cl_device_id device,
                               cl_context context, cl_command_queue queue) {
  __META_OCL_SET_STATE_DYN
  // If not dynamic, accept whatever they provided and directly set the tuple
  // Make sure the copy-from is bounded (set it to the custom slot if not set),
  // but don't change to the user-defined state yet
  if (__meta_curr_state < 0 || __meta_curr_state > __meta_num_states)
    __meta_curr_state = __meta_num_states;
  // Scan to see if a prepared state matches everything
  int eval_state;
  for (eval_state = 0; eval_state < __meta_num_states; eval_state++) {
    // If there's a complete match, switch to it and break;
    if (platform == __meta_platforms[eval_state] &&
        device == __meta_devices[eval_state] &&
        context == __meta_contexts[eval_state] &&
        queue == __meta_queues[eval_state]) {
      __meta_curr_state = eval_state;
      break;
    }
  }
  // Was not a prepared state, fill in the custom slot
  if (eval_state == __meta_num_states) {
    // Reuse any non-provided values from whatever the current state is;
    if (platform == NULL)
      __meta_platforms[eval_state] = __meta_platforms[__meta_curr_state];
    if (device == NULL)
      __meta_devices[eval_state] = __meta_devices[__meta_curr_state];
    if (context == NULL)
      __meta_contexts[eval_state] = __meta_contexts[__meta_curr_state];
    if (queue == NULL)
      __meta_queues[eval_state] = __meta_queues[__meta_curr_state];
    __meta_curr_state = eval_state;
  }
  // Fire any registered initializers
  __meta_module_reinit_all();
  return __meta_curr_state;
}
