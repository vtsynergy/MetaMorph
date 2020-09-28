/** \file
 * The workhorse pass-through of the library. This file implements
 *  control over which modes are compiled in, and generic wrappers
 *  for all functions which must be passed through to a specific
 *  mode's backend implementation.
 *
 * Should support targeting a specific class of device, a
 *  specific accelerator model (CUDA/OpenCL/OpenMP/...), and
 *  "choose best", which would be similar to Tom's CoreTSAR.
 *
 * For now, Generic, OpenMP, CUDA, and OpenCL are supported.
 * Generic simply uses environment variable "METAMORPH_MODE" to select
 *  a mode at runtime.
 * OpenCL mode also supports selecting the "-1st" device, which
 *  forces it to refer to environment variable "TARGET_DEVICE" to attempt
 *  to string match the name of an OpenCL device, defaulting to the zeroth
 *  if no match is identified.
 *
 * / \todo For OpenCL context switching, should we implement checks to ensure cl_mem
 *  regions are created w.r.t. the current context? Should it throw an error,
 *  or attempt to go through the stack and execute the call w.r.t. the correct
 *  context for the cl_mem pointer? What if a kernel launch mixes cl_mems from
 *  multiple contexts? TODO
 */

#include "metamorph.h"
#include "metamorph_dynamic_symbols.h"
#include <stdlib.h>
#include <string.h>
#ifdef ALIGNED_MEMORY
#include "xmmintrin.h"
#endif

/** Maintain a global state variable for the currently-in-use backend/execution
 * mode */
meta_preferred_mode run_mode = metaModePreferGeneric;

/** Forward declaration of the internal module-management list type */
struct meta_module_record_listelem;
/** Implementation of the internal module-management list type for recording
 * known modules */
typedef struct meta_module_record_listelem {
  /** A pointer to the record for each module */
  meta_module_record *rec;
  /** Th next module record */
  struct meta_module_record_listelem *next;
} meta_module_list;
/** Maintain a global list of all modules currently known by the MetaMorph core
 */
meta_module_list global_modules = {NULL, NULL};
/**
 * Internal: Check if a given module record is already known
 * \param record the record to check
 * \return true if the provided record is non-NULL and already in the list,
 * false otherwise
 * \todo Make Hazard-aware
 */
meta_bool __module_registered(meta_module_record *record) {
  if (record == NULL)
    return false;
  if (global_modules.next == NULL)
    return false;
  meta_module_list *elem;
  for (elem = global_modules.next; elem != NULL; elem = elem->next) {
    if (elem->rec == record)
      return true;
  }
  return false;
}
/**
 * Adds a module to our records, without initializing
 * \param record the module record to add
 * \return true iff added, false otherwise
 * \todo Make Hazard-aware
 */
meta_bool __record_module(meta_module_record *record) {
  if (record == NULL)
    return false;
  if (__module_registered(record) != false)
    return false;
  meta_module_list *new_elem = (meta_module_list *)malloc(sizeof(meta_module_list));
  new_elem->rec = record;
  new_elem->next = NULL;
  // begin hazards
  while (new_elem->next != global_modules.next) {
    new_elem->next = global_modules.next;
  }
  /// \todo atomic CAS the pointer in
  global_modules.next = new_elem;
  return true; /// \todo return the eventual true from CAS
}
/**
 * Removes a module from our records, does not deinitialize it
 * \param record the record for the module to remove
 * \return true iff removed, false otherwise
 * \todo Make Hazard-aware
 */
meta_bool __remove_module(meta_module_record *record) {
  if (record == NULL)
    return false;
  meta_module_list *curr = global_modules.next, *prev = &global_modules;
  // begin hazards
  while (prev->next == curr && curr != NULL) {
    /// \todo Ensure curr w/ atomic CAS
    if (curr->rec == record) {
      /// \todo Remove w/ atomic CAS
      prev->next = curr->next;
      free(curr); // Just the list wrapper
      return true;
    }
    curr = curr->next;
    prev = prev->next;
  }
  return false;
}

int lookup_implementing_modules(meta_module_record **retRecords,
                                size_t szRetRecords,
                                meta_module_implements_backend signature,
                                meta_bool matchAny) {
  int matches = 0, retIdx = 0;
  meta_module_list *elem = global_modules.next;
  for (; elem != NULL; elem = elem->next) {
    if (elem->rec != NULL &&
        (elem->rec->implements == signature ||
         (matchAny && (elem->rec->implements & signature)))) {
      if (retRecords != NULL && retIdx < szRetRecords) {
        retRecords[retIdx] = elem->rec;
        retIdx++;
      }
      matches++;
    }
  }
  return matches;
}
meta_err meta_register_module(
    meta_module_record *(*module_registry_func)(meta_module_record *record)) {
  // Do nothing if they don't specify a pointer
  if (module_registry_func == NULL)
    return -1;
  // Each module contains a referency to it's own registry object
  // We can check if it's registered by calling the registry function with a
  // NULL record
  meta_module_record *existing = (*module_registry_func)(NULL);
  // if there is an existing record, make sure we know about it
  if (existing != NULL) {
    __record_module(existing); // Don't need to check, __record_module will
                               // check before insertion
    // But don't call the registration function with the existing record,
    // because that will de-register it
    return 0;
  }
  // It has not already been registered

  meta_module_record *new_record =
      (meta_module_record *)calloc(sizeof(meta_module_record), 1);
  meta_module_record *returned = (*module_registry_func)(new_record);
  /// \todo make this threadsafe
  // If we trust that the registration function will only accept one registry,
  // then we just have to check that the one returned is the same as the one we
  // created. otherwise release the new one
  if (returned == NULL) {
    __record_module(
        new_record); // shouldn't be able to return false, because we should
                     // block recording records that aren't our own;
    // Only initialize if it's a new registration, do nothing if it's a
    // re-register It's been explicitly registered, immediately initialize it
    if (new_record->module_init != NULL)
      (*new_record->module_init)();
  } else if (new_record == returned) {
    // Somewhere between checking whether the module was registered and trying
    // to register it, someone else has already done so, just release our new
    // record and move on
    /// \todo if we want to support re-registration we will change this mechanism
    free(new_record);
  } // There should not be an else unless we support multiple registrations, in
    // which case returned will belong to the module's previous registration

  return 0;
}

meta_err meta_deregister_module(
    meta_module_record *(*module_registry_func)(meta_module_record *record)) {
  // If they gave us NULL, nothing to do
  if (module_registry_func == NULL)
    return -1;
  // Get the registration from the module
  meta_module_record *existing = (*module_registry_func)(NULL);
  // If the module doesn't think it's registered, we can't do anything with it
  if (existing == NULL)
    return 0;

  // If it does exist, deinitialize it, thenremove our record of it
  // Deinit has to occure before the module forgets its record so that the
  // module can inform MetaMorph when it is finish via the init flag
  if (existing->module_deinit != NULL) {
    while (existing->initialized) { // The module is responsible for updating
                                    // its registration when it has completed
                                    // deinitialization
      // Loop in case there are multiple stacked initializations that need to be
      // handled
      (*existing->module_deinit)(); // It has been deinitialized
    }
  }
  /// \todo make threadsafe
  // Tell the module to forget itself
  meta_module_record *returned = (*module_registry_func)(existing);
  // Check that the module indeed thinks it's deregistered (to distinguish from
  // the multiple-registration case that also immediately returns the same
  // record)
  meta_module_record *double_check = (*module_registry_func)(NULL);
  if (returned == existing &&
      double_check == NULL) { // The module has forgotten its registration
    // Then deinitialize the module
    __remove_module(
        existing);  // We have no record of it outside this function call
    free(existing); // Record memory released and this function's reference
                    // invalidated, we are done
  }
  return 0;
}

meta_err meta_reinitialize_modules(meta_module_implements_backend module_type) {
  if (global_modules.next == NULL)
    return -1;
  meta_module_list *elem;
  for (elem = global_modules.next; elem != NULL; elem = elem->next) {
    if (elem->rec != NULL && (elem->rec->implements & module_type) &&
        elem->rec->module_init != NULL)
      (*elem->rec->module_init)();
  }
  return 0;
}

// Make sure we can reference any dynamically-loaded capabilities
extern meta_module_implements_backend core_capability;
/** A global storage struct for handles for all the available backends and their
 * corresponding runtime libraries */
extern struct backend_handles backends;
/** A global storage struct for all the CUDA backend's function pointers, if
 * they are loaded */
extern struct cuda_dyn_ptrs cuda_symbols;
/** A global storage struct for all the OpenCL backend's function pointers, if
 * they are loaded */
extern struct opencl_dyn_ptrs opencl_symbols;
/** A global storage struct for all the OpenMP backend's function pointers, if
 * they are loaded */
extern struct openmp_dyn_ptrs openmp_symbols;
/** A global storage struct for handles for all the available plugins and their
 * corresponding runtime libraries */
extern struct plugin_handles plugins;
/** A global storage struct for the profiling plugin's function pointers, if it
 * is loaded */
extern struct profiling_dyn_ptrs profiling_symbols;
/** A global storage struct for the MPI plugin's function pointers, if it is
 * loaded */
extern struct mpi_dyn_ptrs mpi_symbols;

/**
 * The MetaMorph destructor, its primary purpose right now is just to call the
 * function that tears down dynamically-loaded backends and plugins in the
 * correct order
 */
void meta_finalize() { meta_close_libs(); }
/**
 * The MetaMorph constructor, its primary purpose is to search for any installed
 backends and plugins and load them
 * All plugins should check if the core library is loaded by checking
 core_capability for a state other than module_uninitialized, and if not call
 this . This should ensure that no matter what order things are linked, they are
 initialized in the correct order
 * This function is kept separate in expectation that eventually there will be
 initialization functionality that is not related to library loading
 */
__attribute__((constructor(101))) void meta_init() {
  //  atexit(meta_finalize);
  meta_load_libs();
}

/**
 * Convenience function to get the byte width of a selected type
 * \param type The MetaMorph type
 * \return the sizeof the corresponding actual type in bytes
 */
size_t get_atype_size(meta_type_id type) {
  switch (type) {
  case meta_db:
    return sizeof(double);
    break;

  case meta_fl:
    return sizeof(float);
    break;

  case meta_ul:
    return sizeof(unsigned long);
    break;

  case meta_in:
    return sizeof(int);
    break;

  case meta_ui:
    return sizeof(unsigned int);
    break;

  default:
    fprintf(stderr, "Error: no size retreivable for selected type [%d]!\n",
            type);
    return (size_t)-1;
    break;
  }
}

meta_err meta_alloc(void **ptr, size_t size) {
  /// \todo: should always set ret to a value
  meta_err ret = 0;
  switch (run_mode) {
  default:
  case metaModePreferGeneric:
    /// \todo implement generic (runtime choice) allocation
    break;

  case metaModePreferCUDA: {
    if (cuda_symbols.metaCUDAAlloc != NULL)
      ret = (*(cuda_symbols.metaCUDAAlloc))(ptr, size);
    else {
      fprintf(stderr, "CUDA backend improperly loaded or missing symbol "
                      "\"metaCUDAAlloc\"\n");
      /// \todo FIXME output a real error code
      ret = -1;
    }
  } break;

  case metaModePreferOpenCL: {
    if (opencl_symbols.metaOpenCLAlloc != NULL)
      ret = (*(opencl_symbols.metaOpenCLAlloc))(ptr, size);
    else {
      fprintf(stderr, "OpenCL backend improperly loaded or missing symbol "
                      "\"metaOpenCLAlloc\"\n");
      // FIXME output a real error code
      ret = -1;
    }
  } break;

  case metaModePreferOpenMP: {
    if (openmp_symbols.metaOpenMPAlloc != NULL)
      ret = (*(openmp_symbols.metaOpenMPAlloc))(ptr, size);
    else {
      fprintf(stderr, "OpenMP backend improperly loaded or missing symbol "
                      "\"metaOpenMPAlloc\"\n");
      /// \todo FIXME output a real error code
      ret = -1;
    }
  } break;
  }
  return (ret);
}

meta_err meta_free(void *ptr) {
  /// \todo: should always set ret to a value
  meta_err ret;
  switch (run_mode) {
  default:
  case metaModePreferGeneric:
    /// \todo implement generic (runtime choice) free
    break;

  case metaModePreferCUDA: {
    if (cuda_symbols.metaCUDAFree != NULL)
      ret = (*(cuda_symbols.metaCUDAFree))(ptr);
    else {
      fprintf(stderr, "CUDA backend improperly loaded or missing symbol "
                      "\"metaCUDAFree\"\n");
      /// \todo FIXME output a real error code
      ret = -1;
    }
  } break;

  case metaModePreferOpenCL: {
    if (opencl_symbols.metaOpenCLFree != NULL)
      ret = (*(opencl_symbols.metaOpenCLFree))(ptr);
    else {
      fprintf(stderr, "OpenCL backend improperly loaded or missing symbol "
                      "\"metaOpenCLFree\"\n");
      // FIXME output a real error code
      ret = -1;
    }
  } break;

  case metaModePreferOpenMP: {
    if (openmp_symbols.metaOpenMPFree != NULL)
      ret = (*(openmp_symbols.metaOpenMPFree))(ptr);
    else {
      fprintf(stderr, "OpenMP backend improperly loaded or missing symbol "
                      "\"metaOpenMPFree\"\n");
      /// \todo FIXME output a real error code
      ret = -1;
    }
  } break;
  }
  return (ret);
}

meta_err meta_copy_h2d(void *dst, void *src, size_t size, meta_bool async,
                       meta_callback *call, meta_event *ret_event) {
  meta_err ret;
  switch (run_mode) {
  default:
  case metaModePreferGeneric:
    /// \todo / \todo implement generic (runtime choice) H2D copy
    break;

  case metaModePreferCUDA: {
    if (cuda_symbols.metaCUDAWrite != NULL)
      ret = (*(cuda_symbols.metaCUDAWrite))(dst, src, size, async, call,
                                            ret_event);
    else {
      fprintf(stderr, "CUDA backend improperly loaded or missing symbol "
                      "\"metaCUDAWrite\"\n");
      /// \todo FIXME output a real error code
      ret = -1;
    }
  } break;

  case metaModePreferOpenCL: {
    if (opencl_symbols.metaOpenCLWrite != NULL)
      ret = (*(opencl_symbols.metaOpenCLWrite))(dst, src, size, async, call,
                                                ret_event);
    else {
      fprintf(stderr, "OpenCL backend improperly loaded or missing symbol "
                      "\"metaOpenCLWrite\"\n");
      /// \todo FIXME output a real error code
      ret = -1;
    }
  } break;

  case metaModePreferOpenMP: {
    if (openmp_symbols.metaOpenMPWrite != NULL)
      ret = (*(openmp_symbols.metaOpenMPWrite))(dst, src, size, async, call,
                                                ret_event);
    else {
      fprintf(stderr, "OpenMP backend improperly loaded or missing symbol "
                      "\"metaOpenMPWrite\"\n");
      /// \todo FIXME output a real error code
      ret = -1;
    }
  } break;
  }

  return (ret);
}

meta_err meta_copy_d2h(void *dst, void *src, size_t size, meta_bool async,
                       // char * event_name, cl_event * wait,
                       meta_callback *call, meta_event *ret_event) {
  meta_err ret;
  switch (run_mode) {
  default:
  case metaModePreferGeneric:
    /// \todo implement generic (runtime choice) H2D copy
    break;

  case metaModePreferCUDA: {
    if (cuda_symbols.metaCUDARead != NULL)
      ret = (*(cuda_symbols.metaCUDARead))(dst, src, size, async, call,
                                           ret_event);
    else {
      fprintf(stderr, "CUDA backend improperly loaded or missing symbol "
                      "\"metaCUDARead\"\n");
      /// \todo FIXME output a real error code
      ret = -1;
    }
  } break;

  case metaModePreferOpenCL: {
    if (opencl_symbols.metaOpenCLRead != NULL)
      ret = (*(opencl_symbols.metaOpenCLRead))(dst, src, size, async, call,
                                               ret_event);
    else {
      fprintf(stderr, "OpenCL backend improperly loaded or missing symbol "
                      "\"metaOpenCLRead\"\n");
      /// \todo FIXME output a real error code
      ret = -1;
    }
  } break;

  case metaModePreferOpenMP: {
    if (openmp_symbols.metaOpenMPRead != NULL)
      ret = (*(openmp_symbols.metaOpenMPRead))(dst, src, size, async, call,
                                               ret_event);
    else {
      fprintf(stderr, "OpenMP backend improperly loaded or missing symbol "
                      "\"metaOpenMPRead\"\n");
      /// \todo FIXME output a real error code
      ret = -1;
    }
  } break;
  }
  return (ret);
}

meta_err meta_copy_d2d(void *dst, void *src, size_t size, meta_bool async,
                       meta_callback *call, meta_event *ret_event) {
  meta_err ret;
  switch (run_mode) {
  default:
  case metaModePreferGeneric:
    /// \todo implement generic (runtime choice) H2D copy
    break;

  case metaModePreferCUDA: {
    if (cuda_symbols.metaCUDADevCopy != NULL)
      ret = (*(cuda_symbols.metaCUDADevCopy))(dst, src, size, async, call,
                                              ret_event);
    else {
      fprintf(stderr, "CUDA backend improperly loaded or missing symbol "
                      "\"metaCUDADevCopy\"\n");
      /// \todo FIXME output a real error code
      ret = -1;
    }
  } break;

  case metaModePreferOpenCL: {
    if (opencl_symbols.metaOpenCLDevCopy != NULL)
      ret = (*(opencl_symbols.metaOpenCLDevCopy))(dst, src, size, async, call,
                                                  ret_event);
    else {
      fprintf(stderr, "OpenCL backend improperly loaded or missing symbol "
                      "\"metaOpenCLDevCopy\"\n");
      /// \todo FIXME output a real error code
      ret = -1;
    }
  } break;

  case metaModePreferOpenMP: {
    if (openmp_symbols.metaOpenMPDevCopy != NULL)
      ret = (*(openmp_symbols.metaOpenMPDevCopy))(dst, src, size, async, call,
                                                  ret_event);
    else {
      fprintf(stderr, "OpenMP backend improperly loaded or missing symbol "
                      "\"metaOpenMPDevCopy\"\n");
      /// \todo FIXME output a real error code
      ret = -1;
    }
  } break;
  }
  return (ret);
}

meta_err meta_set_acc(int accel, meta_preferred_mode mode) {
  meta_err ret;
  run_mode = mode;
  switch (run_mode) {
  default:
  case metaModePreferGeneric:
    /// \todo support generic (auto-config) runtime selection
    /// \todo support "METAMORPH_MODE" environment variable.
    if (getenv("METAMORPH_MODE") != NULL) {
      if (strcmp(getenv("METAMORPH_MODE"), "CUDA") == 0)
        return meta_set_acc(accel, metaModePreferCUDA);
      if (core_capability | module_implements_opencl) {
        if (strcmp(getenv("METAMORPH_MODE"), "OpenCL") == 0 ||
            strcmp(getenv("METAMORPH_MODE"), "OpenCL_DEBUG") == 0)
          return meta_set_acc(accel, metaModePreferOpenCL);
      }

      if (strcmp(getenv("METAMORPH_MODE"), "OpenMP") == 0)
        return meta_set_acc(accel, metaModePreferOpenMP);

#ifdef WITH_CORETSAR
      if (strcmp(getenv("METAMORPH_MODE"), "CoreTsar") == 0) {
        fprintf(stderr, "CoreTsar mode not yet supported!\n");
        /// \todo implement whatever's required to get CoreTsar going...
      }
#endif

      fprintf(stderr,
              "Error: METAMORPH_MODE=\"%s\" not supported with specified "
              "compiler definitions.\n",
              getenv("METAMORPH_MODE"));
    }
    fprintf(stderr, "Generic Mode only supported with \"METAMORPH_MODE\" "
                    "environment variable set to one of:\n");
    if (core_capability | module_implements_cuda)
      fprintf(stderr, "\"CUDA\"\n");
    if (core_capability | module_implements_opencl)
      fprintf(stderr, "\"OpenCL\" (or \"OpenCL_DEBUG\")\n");
    if (core_capability | module_implements_openmp)
      fprintf(stderr, "\"OpenMP\"\n");
#ifdef WITH_CORETSAR
    /// \todo - Left as a stub to remind us that Generic Mode was intended to be used for
    // CoreTsar auto configuration across devices/modes
    fprintf(stderr, "\"CoreTsar\"\n");
#endif
    break;

  case metaModePreferCUDA: {
    if (cuda_symbols.metaCUDAInitByID != NULL)
      ret = (*(cuda_symbols.metaCUDAInitByID))(accel);
    else {
      fprintf(stderr, "CUDA backend improperly loaded or missing symbol "
                      "\"metaCUDAInitByID\"\n");
      /// \todo FIXME output a real error code
      ret = -1;
    }
  } break;

  case metaModePreferOpenCL: {
    if (opencl_symbols.metaOpenCLInitByID != NULL)
      ret = (*(opencl_symbols.metaOpenCLInitByID))(accel);
    else {
      fprintf(stderr, "OpenCL backend improperly loaded or missing symbol "
                      "\"metaOpenCLInitByID\"\n");
      /// \todo FIXME output a real error code
      ret = -1;
    }
  } break;

  case metaModePreferOpenMP:
    printf("OpenMP Mode selected\n");
    break;
  }
  return (ret);
}

meta_err meta_get_acc(int *accel, meta_preferred_mode *mode) {
  meta_err ret = 0;
  switch (run_mode) {
  default:
  case metaModePreferGeneric:
    /// \todo implement a generic response for which device was runtime selected
    // fprintf(stderr, "Generic Device Query not yet implemented!\n");
    *mode = metaModePreferGeneric;
    break;

  case metaModePreferCUDA: {
    *mode = metaModePreferCUDA;
    if (cuda_symbols.metaCUDACurrDev != NULL)
      ret = (*(cuda_symbols.metaCUDACurrDev))(accel);
    else {
      fprintf(stderr, "CUDA backend improperly loaded or missing symbol "
                      "\"metaCUDACurrDev\"\n");
      /// \todo FIXME output a real error code
      ret = -1;
    }
  } break;

  case metaModePreferOpenCL: {
    *mode = metaModePreferOpenCL;
    if (opencl_symbols.metaOpenCLCurrDev != NULL)
      ret = (*(opencl_symbols.metaOpenCLCurrDev))(accel);
    else {
      fprintf(stderr, "OpenCL backend improperly loaded or missing symbol "
                      "\"metaOpenCLCurrDev\"\n");
      /// \todo FIXME output a real error code
      ret = -1;
    }
  } break;

  case metaModePreferOpenMP:
    *mode = metaModePreferOpenMP;
    break;
  }
  return (ret);
}

meta_err meta_validate_worksize(meta_dim3 *grid_size, meta_dim3 *block_size) {
  meta_err ret;
  switch (run_mode) {
  default:
  case metaModePreferGeneric:
    /// \todo the only reason we should still be here is CoreTsar
    // Don't worry about doing anything until when/if we add that
    break;

  case metaModePreferCUDA:
    /// \todo implement whatever bounds checking is needed by CUDA
    return 0;
    break;

  case metaModePreferOpenCL: {
    if (opencl_symbols.metaOpenCLMaxWorkSizes != NULL)
      ret = (*(opencl_symbols.metaOpenCLMaxWorkSizes))(grid_size, block_size);
    else {
      fprintf(stderr, "OpenCL backend improperly loaded or missing symbol "
                      "\"metaOpenCLMaxWorkSizes\"\n");
      /// \todo FIXME output a real error code
      ret = -1;
    }
  } break;

  case metaModePreferOpenMP:
    /// \todo implement any bounds checking OpenMP may need
    return 0;
    break;
  }
  return (ret);
}

meta_err meta_flush() {
  meta_err ret = 0;
  switch (run_mode) {
  default:
  case metaModePreferGeneric:
    /// \todo implement a generic flush?
    break;

  case metaModePreferCUDA: {
    if (cuda_symbols.metaCUDAFlush != NULL)
      ret = (*(cuda_symbols.metaCUDAFlush))();
    else {
      fprintf(stderr, "CUDA backend improperly loaded or missing symbol "
                      "\"metaCUDAFlush\"\n");
      /// \todo FIXME output a real error code
      ret = -1;
    }
  } break;

  case metaModePreferOpenCL: {
    if (opencl_symbols.metaOpenCLFlush != NULL)
      ret = (*(opencl_symbols.metaOpenCLFlush))();
    else {
      fprintf(stderr, "OpenCL backend improperly loaded or missing symbol "
                      "\"metaOpenCLFlush\"\n");
      /// \todo FIXME output a real error code
      ret = -1;
    }
  } break;

  case metaModePreferOpenMP: {
    if (openmp_symbols.metaOpenMPFlush != NULL)
      ret = (*(openmp_symbols.metaOpenMPFlush))();
    else {
      fprintf(stderr, "OpenMP backend improperly loaded or missing symbol "
                      "\"metaOpenMPFlush\"\n");
      /// \todo FIXME output a real error code
      ret = -1;
    }
  } break;
  }
  // Flush all outstanding MPI work
  // We do this after flushing the GPUs as any packing will be finished
  if (mpi_symbols.finish_mpi_requests != NULL)
    (*(mpi_symbols.finish_mpi_requests))();
  return ret;
}

meta_err meta_init_event(meta_event *event) {
  meta_err ret = 0;
  if (event == NULL)
    return -1;
  event->mode = run_mode;
  switch (run_mode) {
  default:
  case metaModePreferGeneric:
    /// \todo implement a generic mode
    break;

  case metaModePreferCUDA: {
    if (cuda_symbols.metaCUDACreateEvent != NULL)
      ret = (*(cuda_symbols.metaCUDACreateEvent))(&(event->event_pl));
    else {
      fprintf(stderr, "CUDA backend improperly loaded or missing symbol "
                      "\"metaCUDACreateEvent\"\n");
      /// \todo FIXME output a real error code
      ret = -1;
    }
  } break;

  case metaModePreferOpenCL: {
    if (opencl_symbols.metaOpenCLCreateEvent != NULL)
      ret = (*(opencl_symbols.metaOpenCLCreateEvent))(&(event->event_pl));
    else {
      fprintf(stderr, "OpenCL backend improperly loaded or missing symbol "
                      "\"metaOpenCLCreateEvent\"\n");
      /// \todo FIXME output a real error code
      ret = -1;
    }
  } break;

  case metaModePreferOpenMP: {
    if (openmp_symbols.metaOpenMPCreateEvent != NULL)
      ret = (*(openmp_symbols.metaOpenMPCreateEvent))(&(event->event_pl));
    else {
      fprintf(stderr, "OpenMP backend improperly loaded or missing symbol "
                      "\"metaOpenMPCreateEvent\"\n");
      /// \todo FIXME output a real error code
      ret = -1;
    }
  } break;
  }
  return ret;
}

meta_err meta_destroy_event(meta_event event) {
  meta_err ret = 0;
  switch (run_mode) {
  default:
  case metaModePreferGeneric:
    /// \todo implement a generic mode
    break;

  case metaModePreferCUDA: {
    if (cuda_symbols.metaCUDADestroyEvent != NULL)
      ret = (*(cuda_symbols.metaCUDADestroyEvent))(&(event.event_pl));
    else {
      fprintf(stderr, "CUDA backend improperly loaded or missing symbol "
                      "\"metaCUDADestroyEvent\"\n");
      /// \todo FIXME output a real error code
      ret = -1;
    }
  } break;

  case metaModePreferOpenCL:
  case metaModePreferOpenMP: {
    if (event.event_pl != NULL) {
      free(event.event_pl);
    } else {
      /// \todo FIXME output a real error code
      ret = -1;
    }
  } break;
  }
  return ret;
}

meta_err meta_dotProd(meta_dim3 *grid_size, meta_dim3 *block_size, void *data1,
                      void *data2, meta_dim3 *array_size,
                      meta_dim3 *array_start, meta_dim3 *array_end,
                      void *reduction_var, meta_type_id type, meta_bool async,
                      meta_callback *call, meta_event *ret_event) {
  meta_err ret;

  /// \todo FIXME? Consider adding a compiler flag "UNCHECKED_EXPLICIT" to streamline out sanity checks like this
  // Before we do anything, sanity check the start/end/size
  if (array_start == NULL || array_end == NULL || array_size == NULL) {
    fprintf(stderr,
            "ERROR in meta_dotProd: array_start=[%p], array_end=[%p], or "
            "array_size=[%p] is NULL!\n",
            array_start, array_end, array_size);
    return -1;
  }
  int i;
  for (i = 0; i < 3; i++) {
    if ((*array_start)[i] < 0 || (*array_end)[i] < 0) {
      fprintf(stderr,
              "ERROR in meta_dotProd: array_start[%d]=[%ld] or "
              "array_end[%d]=[%ld] is negative!\n",
              i, (*array_start)[i], i, (*array_end)[i]);
      return -1;
    }
    if ((*array_size)[i] < 1) {
      fprintf(stderr,
              "ERROR in meta_dotProd: array_size[%d]=[%ld] must be >=1!\n", i,
              (*array_size)[i]);
      return -1;
    }
    if ((*array_start)[i] > (*array_end)[i]) {
      fprintf(stderr,
              "ERROR in meta_dotProd: array_start[%d]=[%ld] is after "
              "array_end[%d]=[%ld]!\n",
              i, (*array_start)[i], i, (*array_end)[i]);
      return -1;
    }
    if ((*array_end)[i] >= (*array_size)[i]) {
      fprintf(stderr,
              "ERROR in meta_dotProd: array_end[%d]=[%ld] is bigger than "
              "array_size[%d]=[%ld]!\n",
              i, (*array_end)[i], i, (*array_size)[i]);
      return -1;
    }
  }
  // Ensure the block is all powers of two
  // do not fail if not, but rescale and emit a warning
  if (grid_size != NULL && block_size != NULL) {
    int flag = 0;
    size_t new_block[3];
    size_t new_grid[3];
    for (i = 0; i < 3; i++) {
      new_block[i] = (*block_size)[i];
      new_grid[i] = (*grid_size)[i];
      // Bit-twiddle our way to the next-highest power of 2, from: (checked
      // 2015.01.06)
      // http://graphics.standford.edu/~seander/bithacks.html#RoundUpPowerOf2
      new_block[i]--;
      new_block[i] |= new_block[i] >> 1;
      new_block[i] |= new_block[i] >> 2;
      new_block[i] |= new_block[i] >> 4;
      new_block[i] |= new_block[i] >> 8;
      new_block[i] |= new_block[i] >> 16;
      new_block[i]++;
      if (new_block[i] != (*block_size)[i]) {
        flag = 1; // Trip the flag to emit a warning
        new_grid[i] = ((*block_size)[i] * (*grid_size)[i] - 1 + new_block[i]) /
                      new_block[i];
      }
    }
    if (flag) {
      fprintf(stderr,
              "WARNING in meta_dotProd: block_size={%ld, %ld, %ld} must be all "
              "powers of two!\n\tRescaled grid_size={%ld, %ld, %ld}, "
              "block_size={%ld, %ld, %ld} to\n\tnew_grid={%ld, %ld, %ld}, "
              "new_block={%ld, %ld, %ld}\n",
              (*block_size)[0], (*block_size)[1], (*block_size)[2],
              (*grid_size)[0], (*grid_size)[1], (*grid_size)[2],
              (*block_size)[0], (*block_size)[1], (*block_size)[2], new_grid[0],
              new_grid[1], new_grid[2], new_block[0], new_block[1],
              new_block[2]);
      (*grid_size)[0] = new_grid[0];
      (*grid_size)[1] = new_grid[1];
      (*grid_size)[2] = new_grid[2];
      (*block_size)[0] = new_block[0];
      (*block_size)[1] = new_block[1];
      (*block_size)[2] = new_block[2];
    }
  }

  switch (run_mode) {
  default:
  case metaModePreferGeneric:
    /// \todo implement a generic reduce
    break;

  case metaModePreferCUDA: {
    if (cuda_symbols.cuda_dotProd != NULL)
      ret = (*(cuda_symbols.cuda_dotProd))(
          grid_size, block_size, data1, data2, array_size, array_start,
          array_end, reduction_var, type, async, call, ret_event);
    else {
      fprintf(stderr, "CUDA backend improperly loaded or missing symbol "
                      "\"cuda_dotProd\"\n");
      /// \todo FIXME output a real error code
      ret = -1;
    }
  } break;

  case metaModePreferOpenCL: {
    if (opencl_symbols.opencl_dotProd != NULL)
      ret = (*(opencl_symbols.opencl_dotProd))(
          grid_size, block_size, data1, data2, array_size, array_start,
          array_end, reduction_var, type, async, call, ret_event);
    else {
      fprintf(stderr, "OpenCL backend improperly loaded or missing symbol "
                      "\"opencl_dotProd\"\n");
      /// \todo FIXME output a real error code
      ret = -1;
    }
  } break;

  case metaModePreferOpenMP: {
    if (openmp_symbols.openmp_dotProd != NULL)
      ret = (*(openmp_symbols.openmp_dotProd))(
          grid_size, block_size, data1, data2, array_size, array_start,
          array_end, reduction_var, type, async, call, ret_event);
    else {
      fprintf(stderr, "OpenMP backend improperly loaded or missing symbol "
                      "\"openmp_dotProd\"\n");
      /// \todo FIXME output a real error code
      ret = -1;
    }
  } break;
  }
  return (ret);
}

meta_err meta_reduce(meta_dim3 *grid_size, meta_dim3 *block_size, void *data,
                     meta_dim3 *array_size, meta_dim3 *array_start,
                     meta_dim3 *array_end, void *reduction_var,
                     meta_type_id type, meta_bool async, meta_callback *call,
                     meta_event *ret_event) {
  meta_err ret;

  /// \todo FIXME? Consider adding a compiler flag "UNCHECKED_EXPLICIT" to streamline out sanity checks like this
  // Before we do anything, sanity check the start/end/size
  if (array_start == NULL || array_end == NULL || array_size == NULL) {
    fprintf(stderr,
            "ERROR in meta_reduce: array_start=[%p], array_end=[%p], or "
            "array_size=[%p] is NULL!\n",
            array_start, array_end, array_size);
    return -1;
  }
  int i;
  for (i = 0; i < 3; i++) {
    if ((*array_start)[i] < 0 || (*array_end)[i] < 0) {
      fprintf(stderr,
              "ERROR in meta_reduce: array_start[%d]=[%ld] or "
              "array_end[%d]=[%ld] is negative!\n",
              i, (*array_start)[i], i, (*array_end)[i]);
      return -1;
    }
    if ((*array_size)[i] < 1) {
      fprintf(stderr,
              "ERROR in meta_reduce: array_size[%d]=[%ld] must be >=1!\n", i,
              (*array_size)[i]);
      return -1;
    }
    if ((*array_start)[i] > (*array_end)[i]) {
      fprintf(stderr,
              "ERROR in meta_reduce: array_start[%d]=[%ld] is after "
              "array_end[%d]=[%ld]!\n",
              i, (*array_start)[i], i, (*array_end)[i]);
      return -1;
    }
    if ((*array_end)[i] >= (*array_size)[i]) {
      fprintf(stderr,
              "ERROR in meta_reduce: array_end[%d]=[%ld] is bigger than "
              "array_size[%d]=[%ld]!\n",
              i, (*array_end)[i], i, (*array_size)[i]);
      return -1;
    }
  }
  // Ensure the block is all powers of two
  // do not fail if not, but rescale and emit a warning
  if (grid_size != NULL && block_size != NULL) {
    int flag = 0;
    size_t new_block[3];
    size_t new_grid[3];
    for (i = 0; i < 3; i++) {
      new_block[i] = (*block_size)[i];
      new_grid[i] = (*grid_size)[i];
      // Bit-twiddle our way to the next-highest power of 2, from: (checked
      // 2015.01.06)
      // http://graphics.standford.edu/~seander/bithacks.html#RoundUpPowerOf2
      new_block[i]--;
      new_block[i] |= new_block[i] >> 1;
      new_block[i] |= new_block[i] >> 2;
      new_block[i] |= new_block[i] >> 4;
      new_block[i] |= new_block[i] >> 8;
      new_block[i] |= new_block[i] >> 16;
      new_block[i]++;
      if (new_block[i] != (*block_size)[i]) {
        flag = 1; // Trip the flag to emit a warning
        new_grid[i] = ((*block_size)[i] * (*grid_size)[i] - 1 + new_block[i]) /
                      new_block[i];
      }
    }
    if (flag) {
      fprintf(stderr,
              "WARNING in meta_reduce: block_size={%ld, %ld, %ld} must be all "
              "powers of two!\n\tRescaled grid_size={%ld, %ld, %ld}, "
              "block_size={%ld, %ld, %ld} to\n\tnew_grid={%ld, %ld, %ld}, "
              "new_block={%ld, %ld, %ld}\n",
              (*block_size)[0], (*block_size)[1], (*block_size)[2],
              (*grid_size)[0], (*grid_size)[1], (*grid_size)[2],
              (*block_size)[0], (*block_size)[1], (*block_size)[2], new_grid[0],
              new_grid[1], new_grid[2], new_block[0], new_block[1],
              new_block[2]);
      (*grid_size)[0] = new_grid[0];
      (*grid_size)[1] = new_grid[1];
      (*grid_size)[2] = new_grid[2];
      (*block_size)[0] = new_block[0];
      (*block_size)[1] = new_block[1];
      (*block_size)[2] = new_block[2];
    }
  }

  switch (run_mode) {
  default:
  case metaModePreferGeneric:
    /// \todo implement a generic reduce
    break;

  case metaModePreferCUDA: {
    if (cuda_symbols.cuda_reduce != NULL)
      ret = (*(cuda_symbols.cuda_reduce))(
          grid_size, block_size, data, array_size, array_start, array_end,
          reduction_var, type, async, call, ret_event);
    else {
      fprintf(
          stderr,
          "CUDA backend improperly loaded or missing symbol \"cuda_reduce\"\n");
      /// \todo FIXME output a real error code
      ret = -1;
    }
  } break;

  case metaModePreferOpenCL: {
    if (opencl_symbols.opencl_reduce != NULL)
      ret = (*(opencl_symbols.opencl_reduce))(
          grid_size, block_size, data, array_size, array_start, array_end,
          reduction_var, type, async, call, ret_event);
    else {
      fprintf(stderr, "OpenCL backend improperly loaded or missing symbol "
                      "\"opencl_reduce\"\n");
      /// \todo FIXME output a real error code
      ret = -1;
    }
  } break;

  case metaModePreferOpenMP: {
    if (openmp_symbols.openmp_reduce != NULL)
      ret = (*(openmp_symbols.openmp_reduce))(
          grid_size, block_size, data, array_size, array_start, array_end,
          reduction_var, type, async, call, ret_event);
    else {
      fprintf(stderr, "OpenMP backend improperly loaded or missing symbol "
                      "\"openmp_reduce\"\n");
      /// \todo FIXME output a real error code
      ret = -1;
    }
  } break;
  }
  return (ret);
}

meta_face *meta_get_face(int s, int c, int *si, int *st) {
  // Unlike Kaixi's, we return a pointer copy, to ease Fortran implementation
  meta_face *face = (meta_face *)malloc(sizeof(meta_face));
  // We create our own copy of size and stride arrays to prevent
  // issues if the user unexpectedly reuses or frees the original pointer
  size_t sz = sizeof(int) * c;
  face->size = (int *)malloc(sz);
  face->stride = (int *)malloc(sz);
  memcpy((void *)face->size, (const void *)si, sz);
  memcpy((void *)face->stride, (const void *)st, sz);

  face->start = s;
  face->count = c;
}

// Simple deallocator for an meta_face type
// Assumes face, face->size, and ->stride are unfreed
// This is the only way a user should release a face returned
// from meta_get_face_index, and should not be used
// if the face was assembled by hand.
void meta_free_face(meta_face *face) {
  free(face->size);
  free(face->stride);
  free(face);
}

/**
 * Create the face structure for a given face and slab thickness of a contiguous
 * row-major 3D region 0 = XY face at the origin 1 = XY face at the far side 2 =
 * YZ face at the origin 3 = YZ face at the far side 4 = XZ face at the origin
 * 5 = XZ face at the far side
 * \param face which face index is desired
 * \param ni the X dimension of the region
 * \param nj the Y dimension of the region
 * \param nk the Z dimension of the region
 * \param thickness How many elements inward from the face are desired
 * \return a dynamically-allocated face corresponding to the selected slab
 */
meta_face *make_slab_from_3d(int face, int ni, int nj, int nk, int thickness) {
  meta_face *ret = (meta_face *)malloc(sizeof(meta_face));
  ret->count = 3;
  ret->size = (int *)malloc(sizeof(int) * 3);
  ret->stride = (int *)malloc(sizeof(int) * 3);
  // all even faces start at the origin, all others start at some offset
  // defined by the dimensions of the prism
  if (face & 1) {
    if (face == 1)
      ret->start = ni * nj * (nk - thickness);
    if (face == 3)
      ret->start = ni - thickness;
    if (face == 5)
      ret->start = ni * (nj - thickness);
  } else
    ret->start = 0;
  ret->size[0] = nk, ret->size[1] = nj, ret->size[2] = ni;
  if (face < 2)
    ret->size[0] = thickness;
  if (face > 3)
    ret->size[1] = thickness;
  if (face > 1 && face < 4)
    ret->size[2] = thickness;
  ret->stride[0] = ni * nj, ret->stride[1] = ni, ret->stride[2] = 1;
  printf("Generated Face:\n\tcount: %d\n\tstart: %d\n\tsize: %d %d "
         "%d\n\tstride: %d %d %d\n",
         ret->count, ret->start, ret->size[0], ret->size[1], ret->size[2],
         ret->stride[0], ret->stride[1], ret->stride[2]);
  return ret;
}

meta_err meta_transpose_face(meta_dim3 *grid_size, meta_dim3 *block_size,
                             void *indata, void *outdata, meta_dim3 *arr_dim_xy,
                             meta_dim3 *tran_dim_xy, meta_type_id type,
                             meta_bool async, meta_callback *call,
                             meta_event *ret_event) {
  meta_err ret;
  /// \todo FIXME? Consider adding a compiler flag "UNCHECKED_EXPLICIT" to streamline out sanity checks like this
  // Before we do anything, sanity check that trans_dim_xy fits inside
  // arr_dim_xy
  if (arr_dim_xy == NULL || tran_dim_xy == NULL) {
    fprintf(stderr,
            "ERROR in meta_transpose_face: arr_dim_xy=[%p] or tran_dim_xy=[%p] "
            "is NULL!\n",
            arr_dim_xy, tran_dim_xy);
    return -1;
  }
  int i;
  for (i = 0; i < 2; i++) {
    if ((*arr_dim_xy)[i] < 1 || (*tran_dim_xy)[i] < 1) {
      fprintf(stderr,
              "ERROR in meta_transpose_face: arr_dim_xy[%d]=[%ld] and "
              "tran_dim_xy[%d]=[%ld] must be >=1!\n",
              i, (*arr_dim_xy)[i], i, (*tran_dim_xy)[i]);
      return -1;
    }
    if ((*arr_dim_xy)[i] < (*tran_dim_xy)[i]) {
      fprintf(stderr,
              "ERROR in meta_transpose_face: tran_dim_xy[%d]=[%ld] must be <= "
              "arr_dim_xy[%d]=[%ld]!\n",
              i, (*tran_dim_xy)[i], i, (*arr_dim_xy)[i]);
      return -1;
    }
  }
  switch (run_mode) {
  default:
  case metaModePreferGeneric:
    /// \todo implement a generic reduce
    break;

  case metaModePreferCUDA: {
    if (cuda_symbols.cuda_transpose_face != NULL)
      ret = (*(cuda_symbols.cuda_transpose_face))(
          grid_size, block_size, indata, outdata, arr_dim_xy, tran_dim_xy, type,
          async, call, ret_event);
    else {
      fprintf(stderr, "CUDA backend improperly loaded or missing symbol "
                      "\"cuda_transpose_face\"\n");
      /// \todo FIXME output a real error code
      ret = -1;
    }
  } break;

  case metaModePreferOpenCL: {
    if (opencl_symbols.opencl_transpose_face != NULL)
      ret = (*(opencl_symbols.opencl_transpose_face))(
          grid_size, block_size, indata, outdata, arr_dim_xy, tran_dim_xy, type,
          async, call, ret_event);
    else {
      fprintf(stderr, "OpenCL backend improperly loaded or missing symbol "
                      "\"opencl_transpose_face\"\n");
      /// \todo FIXME output a real error code
      ret = -1;
    }
  } break;

  case metaModePreferOpenMP: {
    if (openmp_symbols.openmp_transpose_face != NULL)
      ret = (*(openmp_symbols.openmp_transpose_face))(
          grid_size, block_size, indata, outdata, arr_dim_xy, tran_dim_xy, type,
          async, call, ret_event);
    else {
      fprintf(stderr, "OpenMP backend improperly loaded or missing symbol "
                      "\"openmp_transpose_face\"\n");
      /// \todo FIXME output a real error code
      ret = -1;
    }
  } break;
  }
  return (ret);
}

meta_err meta_pack_face(meta_dim3 *grid_size, meta_dim3 *block_size,
                        void *packed_buf, void *buf, meta_face *face,
                        meta_type_id type, meta_bool async, meta_callback *call,
                        meta_event *ret_event_k1, meta_event *ret_event_c1,
                        meta_event *ret_event_c2, meta_event *ret_event_c3) {
  meta_err ret;
  /// \todo FIXME? Consider adding a compiler flag "UNCHECKED_EXPLICIT" to streamline out sanity checks like this
  // Before we do anything, sanity check that the face is set up
  if (face == NULL) {
    fprintf(stderr, "ERROR in meta_pack_face: face=[%p] is NULL!\n", face);
    return -1;
  }
  if (face->size == NULL || face->stride == NULL) {
    fprintf(stderr,
            "ERROR in meta_pack_face: face->size=[%p] or face->stride=[%p] is "
            "NULL!\n",
            face->size, face->stride);
    return -1;
  }

  // figure out what the aggregate size of all descendant branches are
  int *remain_dim = (int *)malloc(sizeof(int) * face->count);
  int i, j;
  remain_dim[face->count - 1] = 1;
  // This loop is backwards from Kaixi's to compute the child size in O(n)
  // rather than O(n^2) by recognizing that the size of nodes higher in the tree
  // is just the size of a child multiplied by the number of children, applied
  // upwards from the leaves
  for (i = face->count - 2; i >= 0; i--) {
    remain_dim[i] = remain_dim[i + 1] * face->size[i + 1];
    //	printf("Remain_dim[%d]: %d\n", i, remain_dim[i]);
  }

  //	for(i = 0; i < face->count; i++){
  //		remain_dim[i] = 1;
  //		for(j=i+1; j < face->count; j++) {
  //			remain_dim[i] *=face->size[j];
  //		}
  //			printf("Remain_dim[%d]: %d\n", i, remain_dim[i]);
  //	}

  switch (run_mode) {
  default:
  case metaModePreferGeneric:
    /// \todo implement a generic reduce
    break;

  case metaModePreferCUDA: {
    if (cuda_symbols.cuda_pack_face != NULL)
      ret = (*(cuda_symbols.cuda_pack_face))(
          grid_size, block_size, packed_buf, buf, face, remain_dim, type, async,
          call, ret_event_k1, ret_event_c1, ret_event_c2, ret_event_c3);
    else {
      fprintf(stderr, "CUDA backend improperly loaded or missing symbol "
                      "\"cuda_pack_face\"\n");
      /// \todo FIXME output a real error code
      ret = -1;
    }
  } break;

  case metaModePreferOpenCL: {
    if (opencl_symbols.opencl_pack_face != NULL)
      ret = (*(opencl_symbols.opencl_pack_face))(
          grid_size, block_size, packed_buf, buf, face, remain_dim, type, async,
          call, ret_event_k1, ret_event_c1, ret_event_c2, ret_event_c3);
    else {
      fprintf(stderr, "OpenCL backend improperly loaded or missing symbol "
                      "\"opencl_pack_face\"\n");
      /// \todo FIXME output a real error code
      ret = -1;
    }
  } break;

  case metaModePreferOpenMP: {
    if (openmp_symbols.openmp_pack_face != NULL)
      ret = (*(openmp_symbols.openmp_pack_face))(
          grid_size, block_size, packed_buf, buf, face, remain_dim, type, async,
          call, ret_event_k1, ret_event_c1, ret_event_c2, ret_event_c3);
    else {
      fprintf(stderr, "OpenMP backend improperly loaded or missing symbol "
                      "\"openmp_pack_face\"\n");
      /// \todo FIXME output a real error code
      ret = -1;
    }
  } break;
  }
  return (ret);
}

meta_err meta_unpack_face(meta_dim3 *grid_size, meta_dim3 *block_size,
                          void *packed_buf, void *buf, meta_face *face,
                          meta_type_id type, meta_bool async,
                          meta_callback *call, meta_event *ret_event_k1,
                          meta_event *ret_event_c1, meta_event *ret_event_c2,
                          meta_event *ret_event_c3) {
  meta_err ret;
  /// \todo FIXME? Consider adding a compiler flag "UNCHECKED_EXPLICIT" to streamline out sanity checks like this
  // Before we do anything, sanity check that the face is set up
  if (face == NULL) {
    fprintf(stderr, "ERROR in meta_unpack_face: face=[%p] is NULL!\n", face);
    return -1;
  }
  if (face->size == NULL || face->stride == NULL) {
    fprintf(stderr,
            "ERROR in meta_unpack_face: face->size=[%p] or face->stride=[%p] "
            "is NULL!\n",
            face->size, face->stride);
    return -1;
  }
  // figure out what the aggregate size of all descendant branches are
  int *remain_dim = (int *)malloc(sizeof(int) * face->count);
  int i;
  remain_dim[face->count - 1] = 1;
  // This loop is backwards from Kaixi's to compute the child size in O(n)
  // rather than O(n^2) by recognizing that the size of nodes higher in the tree
  // is just the size of a child multiplied by the number of children, applied
  // upwards from the leaves
  for (i = face->count - 2; i >= 0; i--) {
    remain_dim[i] = remain_dim[i + 1] * face->size[i + 1];
  }

  switch (run_mode) {
  default:
  case metaModePreferGeneric:
    /// \todo implement a generic reduce
    break;

  case metaModePreferCUDA: {
    if (cuda_symbols.cuda_unpack_face != NULL)
      ret = (*(cuda_symbols.cuda_unpack_face))(
          grid_size, block_size, packed_buf, buf, face, remain_dim, type, async,
          call, ret_event_k1, ret_event_c1, ret_event_c2, ret_event_c3);
    else {
      fprintf(stderr, "CUDA backend improperly loaded or missing symbol "
                      "\"cuda_unpack_face\"\n");
      /// \todo FIXME output a real error code
      ret = -1;
    }
  } break;

  case metaModePreferOpenCL: {
    if (opencl_symbols.opencl_unpack_face != NULL)
      ret = (*(opencl_symbols.opencl_unpack_face))(
          grid_size, block_size, packed_buf, buf, face, remain_dim, type, async,
          call, ret_event_k1, ret_event_c1, ret_event_c2, ret_event_c3);
    else {
      fprintf(stderr, "OpenCL backend improperly loaded or missing symbol "
                      "\"opencl_unpack_face\"\n");
      /// \todo FIXME output a real error code
      ret = -1;
    }
  } break;

  case metaModePreferOpenMP: {
    if (openmp_symbols.openmp_unpack_face != NULL)
      ret = (*(openmp_symbols.openmp_unpack_face))(
          grid_size, block_size, packed_buf, buf, face, remain_dim, type, async,
          call, ret_event_k1, ret_event_c1, ret_event_c2, ret_event_c3);
    else {
      fprintf(stderr, "OpenMP backend improperly loaded or missing symbol "
                      "\"openmp_unpack_face\"\n");
      /// \todo FIXME output a real error code
      ret = -1;
    }
  } break;
  }
  return (ret);
}

meta_err meta_stencil_3d7p(meta_dim3 *grid_size, meta_dim3 *block_size,
                           void *indata, void *outdata, meta_dim3 *array_size,
                           meta_dim3 *array_start, meta_dim3 *array_end,
                           meta_type_id type, meta_bool async,
                           meta_callback *call, meta_event *ret_event) {
  meta_err ret;

  /// \todo FIXME? Consider adding a compiler flag "UNCHECKED_EXPLICIT" to streamline out sanity checks like this
  // Before we do anything, sanity check the start/end/size
  if (array_start == NULL || array_end == NULL || array_size == NULL) {
    fprintf(stderr,
            "ERROR in meta_stencil: array_start=[%p], array_end=[%p], or "
            "array_size=[%p] is NULL!\n",
            array_start, array_end, array_size);
    return -1;
  }
  int i;
  for (i = 0; i < 3; i++) {
    if ((*array_start)[i] < 0 || (*array_end)[i] < 0) {
      fprintf(stderr,
              "ERROR in meta_stencil: array_start[%d]=[%ld] or "
              "array_end[%d]=[%ld] is negative!\n",
              i, (*array_start)[i], i, (*array_end)[i]);
      return -1;
    }
    if ((*array_size)[i] < 1) {
      fprintf(stderr,
              "ERROR in meta_stencil: array_size[%d]=[%ld] must be >=1!\n", i,
              (*array_size)[i]);
      return -1;
    }
    if ((*array_start)[i] > (*array_end)[i]) {
      fprintf(stderr,
              "ERROR in meta_stencil: array_start[%d]=[%ld] is after "
              "array_end[%d]=[%ld]!\n",
              i, (*array_start)[i], i, (*array_end)[i]);
      return -1;
    }
    if ((*array_end)[i] >= (*array_size)[i]) {
      fprintf(stderr,
              "ERROR in meta_stencil: array_end[%d]=[%ld] is bigger than "
              "array_size[%d]=[%ld]!\n",
              i, (*array_end)[i], i, (*array_size)[i]);
      return -1;
    }
  }
  // Ensure the block is all powers of two
  // do not fail if not, but rescale and emit a warning
  if (grid_size != NULL && block_size != NULL) {
    int flag = 0;
    size_t new_block[3];
    size_t new_grid[3];
    for (i = 0; i < 3; i++) {
      new_block[i] = (*block_size)[i];
      new_grid[i] = (*grid_size)[i];
      // Bit-twiddle our way to the next-highest power of 2, from: (checked
      // 2015.01.06)
      // http://graphics.standford.edu/~seander/bithacks.html#RoundUpPowerOf2
      new_block[i]--;
      new_block[i] |= new_block[i] >> 1;
      new_block[i] |= new_block[i] >> 2;
      new_block[i] |= new_block[i] >> 4;
      new_block[i] |= new_block[i] >> 8;
      new_block[i] |= new_block[i] >> 16;
      new_block[i]++;
      if (new_block[i] != (*block_size)[i]) {
        flag = 1; // Trip the flag to emit a warning
        new_grid[i] = ((*block_size)[i] * (*grid_size)[i] - 1 + new_block[i]) /
                      new_block[i];
      }
    }
    if (flag) {
      fprintf(stderr,
              "WARNING in meta_stencil: block_size={%ld, %ld, %ld} must be all "
              "powers of two!\n\tRescaled grid_size={%ld, %ld, %ld}, "
              "block_size={%ld, %ld, %ld} to\n\tnew_grid={%ld, %ld, %ld}, "
              "new_block={%ld, %ld, %ld}\n",
              (*block_size)[0], (*block_size)[1], (*block_size)[2],
              (*grid_size)[0], (*grid_size)[1], (*grid_size)[2],
              (*block_size)[0], (*block_size)[1], (*block_size)[2], new_grid[0],
              new_grid[1], new_grid[2], new_block[0], new_block[1],
              new_block[2]);
      (*grid_size)[0] = new_grid[0];
      (*grid_size)[1] = new_grid[1];
      (*grid_size)[2] = new_grid[2];
      (*block_size)[0] = new_block[0];
      (*block_size)[1] = new_block[1];
      (*block_size)[2] = new_block[2];
    }
  }

  switch (run_mode) {
  default:
  case metaModePreferGeneric:
    /// \todo implement a generic reduce
    break;

  case metaModePreferCUDA: {
    if (cuda_symbols.cuda_stencil_3d7p != NULL)
      ret = (*(cuda_symbols.cuda_stencil_3d7p))(
          grid_size, block_size, indata, outdata, array_size, array_start,
          array_end, type, async, call, ret_event);
    else {
      fprintf(stderr, "CUDA backend improperly loaded or missing symbol "
                      "\"cuda_stencil_3d7p\"\n");
      /// \todo FIXME output a real error code
      ret = -1;
    }
  } break;

  case metaModePreferOpenCL: {
    if (opencl_symbols.opencl_stencil_3d7p != NULL)
      ret = (*(opencl_symbols.opencl_stencil_3d7p))(
          grid_size, block_size, indata, outdata, array_size, array_start,
          array_end, type, async, call, ret_event);
    else {
      fprintf(stderr, "OpenCL backend improperly loaded or missing symbol "
                      "\"opencl_stencil_3d7p\"\n");
      /// \todo FIXME output a real error code
      ret = -1;
    }
  } break;

  case metaModePreferOpenMP: {
    if (openmp_symbols.openmp_stencil_3d7p != NULL)
      ret = (*(openmp_symbols.openmp_stencil_3d7p))(
          grid_size, block_size, indata, outdata, array_size, array_start,
          array_end, type, async, call, ret_event);
    else {
      fprintf(stderr, "OpenMP backend improperly loaded or missing symbol "
                      "\"openmp_stencil_3d7p\"\n");
      /// \todo FIXME output a real error code
      ret = -1;
    }
  } break;
  }
  return (ret);
}

meta_err meta_csr(meta_dim3 *grid_size, meta_dim3 *block_size,
                  size_t global_size, void *csr_ap, void *csr_aj, void *csr_ax,
                  void *x_loc, void *y_loc, meta_type_id type, meta_bool async,
                  meta_callback *call, meta_event *ret_event) {
  meta_err ret;

  switch (run_mode) {
  default:
  case metaModePreferGeneric:
    /// \todo implement a generic reduce
    break;

  case metaModePreferCUDA:
    fprintf(stderr, "CUDA CSR Not yet supported\n");
    break;

  case metaModePreferOpenCL: {
    if (opencl_symbols.opencl_csr != NULL)
      ret = (*(opencl_symbols.opencl_csr))(grid_size, block_size, global_size,
                                           csr_ap, csr_aj, csr_ax, x_loc, y_loc,
                                           type, async, call, ret_event);
    else {
      fprintf(stderr, "OpenCL backend improperly loaded or missing symbol "
                      "\"opencl_csr\"\n");
      /// \todo FIXME output a real error code
      ret = -1;
    }
  } break;

  case metaModePreferOpenMP:
    fprintf(stderr, "OpenMP CSR Not yet supported");
    break;
  }
  return (ret);
}

meta_err meta_crc(meta_dim3 *grid_size, meta_dim3 *block_size, void *dev_input,
                  int page_size, int num_words, int numpages, void *dev_output,
                  meta_type_id type, meta_bool async, meta_callback *call,
                  meta_event *ret_event) {
  meta_err ret;

  switch (run_mode) {
  default:
  case metaModePreferGeneric:
    /// \todo implement a generic crc
    break;

  case metaModePreferCUDA:
    fprintf(stderr, "CUDA CRC Not yet supported\n");
    break;

  case metaModePreferOpenCL: {
    if (opencl_symbols.opencl_crc != NULL)
      ret = (*(opencl_symbols.opencl_crc))(dev_input, page_size, num_words,
                                           numpages, dev_output, type, async,
                                           call, ret_event);
    else {
      fprintf(stderr, "OpenCL backend improperly loaded or missing symbol "
                      "\"opencl_crc\"\n");
      /// \todo FIXME output a real error code
      ret = -1;
    }
  } break;

  case metaModePreferOpenMP:
    fprintf(stderr, "OpenMP CRC Not yet supported");
    break;
  }
  return (ret);
}
