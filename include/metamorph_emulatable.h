/** \file
 * Portions of the core MetaMorph API that may be emulated by client programs
 *  that wish to leverage MetaMorph when it is available, but provide standalone
 *  fallbacks for use otherwise.
 *
 * (C) Virginia Polytechnic Institute and State University, 2013-2020.
 *  See attached LICENSE terms before continued use of the code.
 *
 */
#ifndef METAMORPH_EMULATABLE_H
#define METAMORPH_EMULATABLE_H
#ifdef __cplusplus
extern "C" {
#endif

/** Typedef for internal error type */
typedef int meta_err;
/** Define the set of supported execution modes */
typedef enum {
  /** A special-purpose mode which indicates none has been declared */
  metaModeUnset = -1,
  /** CUDA execution mode */
  metaModePreferCUDA = 1,
  /** OpenCL execution mode */
  metaModePreferOpenCL = 2,
  /** OpenMP execution mode */
  metaModePreferOpenMP = 3,
  /** Generic execution mode
   * Intended to support a "don't care, pick the best" adaptive mode */
  metaModePreferGeneric = 0
} meta_preferred_mode;

// Module Management
/// \todo need a bitfield to store the type of backend
typedef enum {
  module_uninitialized = -1,
  module_implements_none = 0,
  module_implements_cuda = 1,
  module_implements_opencl = 2,
  module_implements_openmp = 4,
  module_implements_all = 7,
  module_implements_profiling = 8,
  // A CoreTSAR backend would likely want all 3 and profiling
  //  module_implements_coretsar = 15,
  module_implements_mpi = 16,
  module_implements_fortran = 32,
  module_implements_general = 64 // general operations not related to a backend
} meta_module_implements_backend;

/** Forward declaration of the module record structure */
struct meta_module_record;
/** External MetaMorph modules need a well-defined interface for sharing
 * information with the MetaMorph core putting it in a structure allows us to
 * abstract some of it from the user and more easily adapt it if needs adjust in
 * the future
 */
typedef struct meta_module_record {
  /** Function pointer to the initializer for the module, should not be NULL
   * This function must attempt to auto-register if not already known to
   * MetaMorph Once it is finished it must set the initialized state of the
   * record to a non-zero value (this may change)
   */
  void (*module_init)(void);
  /** Function pointer to the destructor for the module, should not be NULL
   * This function must attempt to auto-deregister if its registration is still
   * valid
   */
  void (*module_deinit)(void);
  /** Pointer to the module's registration function, should not be NULL
   * This function is called multiple times to do different things depending on
   * the module's current state and the value of record IFF record is NULL, it
   * returns the pointer to the module's current registration (or NULL if
   * unregistered) IFF record is non-NULL and the module's current registration
   * is NULL, accept record as the new current registration, and populate the
   * record with the functino pointsrs and implements/status variables, then
   * return the old registration (NULL) IFF record is non-NULL and the module's
   * current registration is non-NULL, but matches record, this signifies a
   * deregistration request coming from MetaMorph. Set the current registration
   * to NULL, and return the old value IFF record is non-NULL and the module's
   * current registration is non-NULL, and does not match record, this is an
   * attempted re-registration, return whichever record is *not* accepted.
   * (Currently re-registrations should be rejected and thus return the value of
   * record)
   */
  struct meta_module_record *(*module_registry_func)(
      struct meta_module_record *record);
  /** enum "bitfield" defining which backend(s) (or general) the module provides
   * implementations for, typically defaults to module_implements_none, but
   * could be module_uninitialized */
  meta_module_implements_backend implements;
  /** An initialized state boolean, set and unset during calls to the init and
   * deinit function pointers May be deprecated in light of the
   * module_uninitialized value that was introduced */
  char initialized; // = 0;
} meta_module_record;
/**
 * Allow add-on modules to pass a pointer to their registration function to
 * metamorph-core, so that the core can then initialize and exchange the
 * appropriate internal data structs Any functions from the module should
 * implicitly call this if the module is not already registered, but users can
 * explicitly call it if they wish to pay initialization costs outside the
 * critical application
 * \param module_registry_func The function implementing the registration
 * contract as specified in the meta_module_record type
 * \return 0 if the module is successfully registered or was already registered,
 * -1 if the function pointer is NULL
 */
meta_err meta_register_module(
    meta_module_record *(*module_registry_func)(meta_module_record *record));
/**
 * Explicitly remove an external module from metamorph-core, causing it to
 * trigger any deconstruction code necessary to clean up the module Users *can*
 * explicitly call this if they will be done with MetaMorph for a while, but it
 * is implicitly called by the global MetaMorph constructor at program end on
 * any remaining modules
 * \param module_registry_func The function implementing the registration
 * contract as specified in the meta_module_record type
 * \return 0 if the module was successfully or already deregistered, -1 if the
 * function pointer is NULL
 */
meta_err meta_deregister_module(
    meta_module_record *(*module_registry_func)(meta_module_record *record));
/**
 * Switch MetaMorph to the device with the provided ID on the provided backend
 * \param accel the ID of the desired device
 * \param mode the desired mode
 * \return 0 on success, -1 or a backend-specific error code otherwise
 * \todo unpack OpenCL platform from the uint's high short, and the device from
 * the low short
 */
meta_err meta_set_acc(int accel, meta_preferred_mode mode);
/**
 * Get the currently-active backend mode and the in-use device's ID within it
 * \param accel The address in which to return the device's ID
 * \param mode The address in which to return the currently-active mode
 * \return 0 on success, -1 or an error code from the returned backend otherwise
 * \todo pack OpenCL platform into the uint's high short, and the device into
 * the low short
 */
meta_err meta_get_acc(int *accel, meta_preferred_mode *mode);
#ifdef __cplusplus
}
#endif

#endif // METAMORPH_EMULATABLE_H
