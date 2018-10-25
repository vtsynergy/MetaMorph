/**
 * A template showing what is required by a new module if a user wants to hand-create it
 *
 *
 *
*/

//The module must include metamorph.h for data types and function calls
#include "metamorph.h"

//If the plugin provides OpenCL, these global variables provide access to the current OpenCL state and should be used for initialization
#ifdef WITH_OPENCL
extern cl_context meta_context;
extern cl_command_queue meta_queue;
extern cl_device_id meta_device;
#endif

//The module must keep a copy of its own registration(s), and it should always be initialized to NULL, and only updated by calls to the registration function itself
//It does not necessarily have to be a single global variable as long as the registration function itself handles things predictably
a_module_record * my_module_registration = NULL;

//Forward declaration of the required registry function, for use in initializer/deinitializer
a_module_record * my_metamorph_module_registry(a_module_record * record);

//Most modules will need an initialization function
//(If it doesn't, you may supply a NULL pointer, and MetaMorph-core will not invoke anything
//Initializers must take and return only void
void my_module_init() {
  //Ordinarily, initialize is called during MetaMorph-core's registration process, but if this module is not yet registered, we must attempt to do so
  //It must only attempt to register if not already registered, and by calling registration will implicitly call itself, so instantly return;
  if (my_module_registration == NULL) {
    meta_register_module(&my_metamorph_module_registry); //To register, pass in the registry function
    return;
  }

  //Assuming a valid registration, continue on with the initialization code
  //The initializer must handle any backend-specific functionality via their APIs

  //TODO if all backends are supported (module_implements_all) do any special initialization for that case
  
  #ifdef WITH_CUDA
  //TODO if CUDA backend is supported (module_implements_cuda) do any special initialization
  #endif
  #ifdef WITH_OPENCL
  //OpenCL modules should always confirm that an OpenCL state exists, and if it doesn't create it
  if (meta_context == NULL) metaOpenCLFallBack();
  //FIXME use metaOpenCLLoadProgramSource to load program(s)
  //FIXME use clBuildProgram(..., meta_opencl_device, ...) to build them
  //FIXME use clCreateKernel() to build kernels
  #endif
  #ifdef WITH_OPENMP
  //TODO if OpenMP backend is supported (module_implements_openmp) do any special initialization
  #endif
  //General functionality (module_implements_general) can sometimes include initialization (for example MPI_Init, if MPI were implemented as an external module)

  //The module can decide for itself whether it can be initialized multiple times
  my_module_registration->initialized = 1;
}

//Most modules will need a deinitialization function
//(If it doesn't, you may supply a NULL pointer, and MetaMorph-core will not invoke anything
//Deinitializers must take and return only void
void my_module_deinit() {
  //Ordinarily, deinitialize is called after MetaMorph-core's deregistration process, but if this module is not yet deregistered, we must attempt to do so
  //It must only attempt to deregister if not already deregistered, and by calling deregistration will implicitly call itself, so instantly return;
  if (my_module_registration != NULL) {
    meta_deregister_module(&my_metamorph_module_registry); //To deregister, pass in the deregistry function
    return;
  }

  //Assuming a valid deregistration, continue on with the deinitialization code
  //The deinitializer must handle any backend-specific functionality via their APIs
  //TODO if all backends are supported (module_implements_all) do any special deinitialization for that case
  
  #ifdef WITH_CUDA
  //TODO if CUDA backend is supported (module_implements_cuda) do any special deinitialization
  #endif
  #ifdef WITH_OPENCL
  //TODO if OpenCL backend is supported (module_implments_opencl) the OpenCL state should still exist, remove from it
  #endif
  #ifdef WITH_OPENMP
  //TODO if OpenMP backend is supported (module_implements_openmp) do any special deinitialization
  #endif
  //General functionality (module_implements_general) can sometimes include deinitialization (for example MPI_Finalize, if MPI were implemented as an external module)
}


//The module must have a registration function that takes in a a_module_record * and returns an a_module_record *
// (If multiple registrations (for different devices/backends/optimizations/etc) are eventually allowed, it may be changeable by other internal methods.
// Regardless, this is the only function that tells MetaMorph-core this module exists)
a_module_record * my_metamorph_module_registry(a_module_record * record) {
  //If the registry function receives a NULL pointer, do nothing and return the current registration
  if (record == NULL) return my_module_registration;
  //If the registry function receives a non-NULL pointer and is not already registered, we either register, re-register, or de-register depending on the current registration
  //TODO Eventually this will be required to be done thread-safely
  a_module_record * old_registration = my_module_registration;
  //If the current record is NULL, for all intents and purposes this is the first time the module is registered
  if (old_registration == NULL) {
    my_module_registration = record; //Update our reference
    record->module_init = &my_module_init; //Inform MetaMorph-core where to find our initialization function (or NULL)
    record->module_deinit = &my_module_deinit; //Inform MetaMorph-core where to find our deinitialization function (or NULL)
    //Inform MetaMorph-core which backends this module supports
    record->implements = module_implements_none;
    //record->implements |= module_implements_all; //If all backends are supported
    #ifdef WITH_CUDA
    record->implements |= module_implements_cuda; //If some of its features support the CUDA backend
    #endif
    #ifdef WITH_OPENCL
    record->implements |= module_implements_opencl; //If some of its features support the OpenCL backend
    #endif
    #ifdef WITH_OPENMP
    record->implements |= module_implements_openmp; //If some of its features support the OpenMP backend
    #endif
    record->implements |= module_implements_general; //If it contains general operations to which the backend is irrelevant
  }
  //If the current record is non-NULL and different from what's provided, it's an attempted re-registration; handle it.
  if (old_registration != NULL && old_registration != record) { //In most cases re-registering shouldn't be allowed, return the new record unused
    return record;
  }
  //If the current record is the same as what's provided, it's an attempted de-registration
  if (old_registration == record) {
    //Remove our reference to it
    my_module_registration = NULL;
  }
  
  return old_registration;
}

//All functions in the module must support lazy registration (and thus initialization)
void my_module_kernel_function_wrapper(a_dim3 grid, a_dim3 block) {
  //If the module isn't registered, ensure it is
  if (my_module_registration == NULL) meta_register_module(&my_metamorph_module_registry);
  //If somehow the module is registered but not initialized (which shouldn't be possible), initialize it
  if (my_module_registration->initialized == 0) my_module_init();

  //FUNCTION IMPLEMENTATION GOES HERE (or calls to backend implementations if a top-level wrapper)

}


//Main should use explicit registration if they want to pay the initialization cost up-front
int main(int arg, const char * argv[]) {
 //Mode should be set before registration to avoid redundant initializations (otherwise the first registration will force MM to create an OpenCL state which may not be the one the user wants)
  meta_set_accel(0, metaModePreferOpenCL);
  meta_register_module(&my_metamorph_module_registry);
  meta_register_module(&another_metamorph_module_registry); //Multiple modules can be included, each should have its own header and registry function
  //TIME-SENSITIVE APPLICATION CODE
  my_module_kernel_function_wrapper(grid, block); //Will not have to wait to be initialized
  //END TIME-SENSITIVE APPLICATION CODE
  meta_deregister_module(&another_metamorph_module_registry);
  meta_deregister_module(&my_metamorph_module_registry);
}


//If the module will be part of a library that goes with MetaMorph, lazy initialization probably makes more sense
int my_library_macro_operation(void * data1, void *data2) {
  // Set up block sizes, etc
  my_module_kernel_function_wrapper(grid, block);
  another_modules_kernel_function_wrapper(grid, block, data1, data2);
}
