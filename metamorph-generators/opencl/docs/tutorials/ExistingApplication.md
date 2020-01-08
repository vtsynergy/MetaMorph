# WIP

# Porting an Existing OpenCL application to a MetaCL-generated interface

MetaCL is designed to encapsulate the vast majority of OpenCL's host-side boilerplate within logical explicit wrapper functions, and semi-automatic triggering of initialization and deconstruction work. As such, it is often worthwhile to port an existing application from "pure" OpenCL to MetaCL+MetaMorph to reduce developer workload and streamline application logic.

This tutorial will walk through a typical set of changes necessary for converting an existing pure OpenCL application to use a MetaCL-generated interface and a minimal set of MetaMorph OpenCL support features. It presumes MetaCL and MetaMorph have been installed according to [the installation tutorial](./InstallingMetaCL.md) and a single MetaCL-generated interface `myModule.c/h` via the `unified-output-file="myModule"` option detailed in [the MetaCL usage tutorial](./GenerateMetaCLInterface.md).

Integrating a MetaCL-ized interface into an existing application typically requires changes to both the build system (few), as well as application code. This tutorial presumes the build system is based on a simple monolithic Makefile.


Build System Changes
--------------------

Modifying the build system should effectively be about as difficult as adding two new source files / object file targets, and a few additional compiler flags. The first new target is to produce the MetaCL-generated interface from all the `.cl` kernel files, and the second is to compile the interface to a new `.o` object file. Existing host files which will reference the generated code may need their header search paths augmented appropriately. After these changes, the final step is simply to add the new object file and its two library dependencies to the final linking step(s).

For convenience and abstraction, the following make variables are used throughout the discussion below:
* `EXISTING_CL_SOURCES` A space-delimited list of all `.cl` kernel implementation files to be used by the OpenCL application. There is no need to manually list header files they may `#include`, as long as the appropriate `-I` arguments are provided in `KERNEL_CFLAGS`
* `CFLAGS` Existing program-wide compiler flags, such as include directories or conditional compilation macros. Space-delimited.
* `KERNEL_CFLAGS` Any compiler flags that are not typically used for compiling host files (i.e not in `CFLAGS`) but must be present in the `clBuildProgram` call for building kernel files. Space-delimited. (Does not include optimization flags like `-cl-fast-relaxed-math`.)
* `EXISTING_OBJ_FILES` The list of existing host-side object `.o` files. Space-delimited


## Add a Make target to regenerate the interface any time the `.cl` sources change

One of the benefits of MetaCL is that it allows the OpenCL host-to-device interface to be cheaply regenerated automatically. To ensure any changes to the kernel files are immediately represented in this interface, we want to create a new make target to regenerate the interface.

```Makefile
myModule.c : $(EXISTING_CL_SOURCES)
        /path/to/metaCL $(EXISTING_CL_SOURCES) --unified-output-file="myModule" -- -include opencl-c.h -I /path/to/metaCL/Clang/includes $(CFLAGS) ($KERNEL_CFLAGS)
```

**(Warning: Do not manually edit the generated files if you intend to regenerate them as part of your build process. Further, do not give any manually-written host files the same name as your kernel file or `unified-output-file`. MetaCL will currently destroy and replace its output files without warning.)**

## Create a new Make target to compile the generated interface any time it changes
```Makefile
myModule.o : myModule.c
        gcc -o myModule.o myModule.c $(CFLAGS)
```

## Add the MetaMorph header path and define to any host lines that need it
```Makefile
my_existing_file.o : my_existing_file.c
        gcc -o my_existing_file.o my_existing_file.c $(CFLAGS) -I /path/to/MetaMorph/include
```

## Add new elements to the final binary linking target
```Makefile
myFinalLinkedBinary : $(EXISTING_OBJ_FILES) myModule.o
        gcc -o myFinalLinkedBinary $(EXISTING_OBJ_FILES) myModule.o $(LDFLAGS) -L /path/to/MetaMorph/lib -lmetamorph -lmm_opencl_backend
```


Application Code Changes
------------------------

The main one-time effort of converting an existing OpenCL application to use a MetaCL-generated interface is removing any now-unnecessary existing OpenCL boilerplate, and where necessary, replacing them with simplified calls to either the MetaCL-generated or MetaMorph OpenCL APIs. While this work is largely application-specific, the remainder of the tutorial will walk through typical usage. Steps marked as **REQUIRED** must be performed for any MetaCL-ized application to work. The remainder can be applied incrementally and as-needed.

## Include generated interface header(s) and MetaMorph header
**REQUIRED**
```C
#define WITH_OPENCL
#include <metamorph.h>
#include "myModule.h
```

The general MetaMorph header implicitly includes the MetaMorph OpenCL API functions when `WITH_OPENCL` is defined, and is required in any source file that references either the MetaMorph OpenCL API or the MetaCL-generated API.


## Set MetaMorph to OpenCL mode and (optionally) choose a device
**REQUIRED**

MetaCL is part of the larger MetaMorph effort to develop a portable runtime _across_ heterogenous languages, but only provides interface-generation capabilities for OpenCL. As such, MetaMorph needs to be instructed to run in OpenCL mode via a call to `meta_set_acc`.

This call initializes all internal MetaMorph OpenCL backend machinery, including automatic discovery of all OpenCL devices on all platforms in the system. This call also provides an opportunity to explicitly select an OpenCL device, and for many applications is the simplest mechanism for doing so.

```C
meta_set_acc(<deviceNumber>, metaModePreferOpenCL);
```

`<deviceNumber>` can be any number from `0:N-1` and corresponds to the `N` OpenCL devices in the system, in the order they are discovered by automatically iterating over all OpenCL platforms. Additionally, if it is set to `-1`, all discovered devices are enumerated to `stderr` with their indices, and then a default is selected and initialized. This default is currently the `0th` enumerated device _unless_ the `TARGET_DEVICE` evironment variable is set to the exact device name reported during enumeration. If set, the lowest-index device with an exactly-matching name is chosen instead.

If an alternative method for setting the device is intended to be used (such as via `meta_set_state_OpenCL` detailed [below](#Sharing-OpenCL-state-with-MetaMorph)), it is safe to set the device number to `0` or `-1` without incurring significant performance penalty. (MetaMorph kernels are initialized on-demand, and kernels from a MetaCL-generated interface are not yet registered and thus are not initialized for the unused device.)


## Explicitly register the MetaCL-generated module with MetaMorph
The MetaMorph OpenCL API provides common functionality that MetaCL-generated code modules make use of, including auto-initialization and destruction. However, to take advantage of many of these features, modules must be registered with MetaMorph so that it can trigger the associated functions via function pointer. All generated functions will automatically attempt to register their containing module with MetaMorph if they detect that it has not yet been registered.

While this provides simple ease of use, it does force the whatever function triggered the registration to pay the runtime cost of initializing the entire module. In order to pay this cost up front and isolate it from performance-critical regions of code, we recommend explicitly registering the module by providing a pointer to the generated `registry` function to `meta_register_module`.

For example a MetaCL-generated module named `myModule` would use this explicit registration:
```C
meta_register_module(&metacl_myModule_registry);
```


## Sharing OpenCL state with MetaMorph

The MetaMorph OpenCL API is intentionally designed for interoperating with existing codes or libraries. As such, it supports _bidirectional_ sharing of the OpenCL state tuple(s) (platform + device + context + queue) it knows about. If you share a state **to** MetaMorph by calling `meta_set_state_OpenCL` it will presume the state is already initialized, and that all duty to release it lies in the user application. If you query a state **from** MetaMorph with `meta_get_state_OpenCL` that it has initialized, the user should not explicitly release any of the tuple elements (but **should** release any additional programs, kernels, or buffers added to it that are not managed by MetaMorph or a MetaCL-generated module).

`meta_set_state_OpenCL(cl_platform_id, cl_device_id, cl_context, cl_command_queue)` If you prefer to select an OpenCL device and create a context and queue yourself, you can share them **to** MetaMorph (and thus any registered MetaCL-generated modules) by calling `meta_set_state_OpenCL`. This call immediately sets the internal OpenCL state to the provided tuple and will trigger reinitializations for the shared device in any registered modules that implement OpenCL. (This call will also `clRetain...()` the associated objects to ensure that MetaMorph can safely `clRelease...()` them during deconstruction without invalidating them in any subsequent user code.)

`meta_get_state_OpenCL(cl_platform_id *, cl_device_id *, cl_context *, cl_command_queue)` If you prefer to let MetaMorph set up an OpenCL device but then need to share it with existing code or external libraries, it may be queried **from** MetaMorph by calling `meta_get_state_OpenCL` and providing pointers to copy the state into. (This call does not `clRetain...()` the shared objects and thus user code should **not** `clRelease...()` them. MetaMorph will do so automatically during its global deconstruction.)


## Remove old source management variables and calls
MetaCL self-manages the loading of OpenCL kernel implementation files (either `.cl` or `.aocx`, depending on selected OpenCL platform) via calls to a MetaMorph OpenCL API convenience function. As such, there is no longer a need to manually read existing kernel implementation into string constants with `#include`s at compile time, nor with `char *`s at runtime. Any associated variables and logic can be removed.

Further, the loader reads the `METAMORPH_OCL_KERNEL_PATH` environment variable for a list of search directories to attempt loading the kernels from. This allows different implementations to be loaded on a per-process basis.


## Remove existing cl_program and cl_kernel variables and initialization
MetaCL manages `cl_program` and `cl_kernel` variables on a per-queue basis. In most uses there should no longer be a need for raw access to them (and if there is, such power-users can extract their values from the management struct).

Additionally, MetaCL generates a single initialization function per output file (`metacl_<filename>_init()`), which triggers program loading, program building, and kernel creation. This should not typically be manually called, but rather is called implicitly when the generated module is registered to MetaMorph, and each time MetaMorph is switched to a new device thereafter.


## Set the kernel build arguments string for each kernel file
Since `cl_programs` no longer need to be explicitly managed by the user, a mechanism is needed to ensure that any custom build arguments necessary to tweak the JIT compilation of the kernels can still be provided. Currently, this mechanism is provided on a per-kernel-file basis via `extern char *` variables. These are named according to the input file with the following convention:

`__metacl_<input filename>_custom_args`

Any custom compiler arguments provided to the MetaCL invocation (such as `-I` include search paths or `-D` defines), as well as optimization flags (such as `-cl-fast-relaxed-math`) should be incorporated into a single C string, and the associated variable pointed to it. Each input file that must be JIT compiled with custom arguments must be separately assigned, though if the arguments are the same they may be pointed to the same C string.

_(This mechanism is likely to be further refined as MetaCL is developed to provide a greater guarantee of consistency between the MetaCL invocation and runtime, a cleaner interface, and simplified ease of use.)_


## Remove individual assignments of kernel arguments
MetaCL exposes a single host-side wrapper function for each kernel. Rather than individually passing each argument by reference via `clSetKernelArg`, the user can now simply call the generated wrapper function with all the arguments passed by value at once. If enabled, each argument assignment is automatically error-checked for safety, so associated manual checks can be removed.


## Remove unwrapped kernel enqueue calls.
MetaCL generates an enqueue call (`Task` or `NDRange` depending on kernel requirements) within each kernel's generated wrapper. If enabled, this call is automatically error-checked for safety. Additionally, the wrapper can be invoked in synchronous mode, in which case immediately after enqueing the target `cl_command_queue` will be flushed with `clFinish`, before returning to the caller.

## Replace existing global and local worksizes with MetaCL's dim3 types
**REQUIRED**
MetaCL kernel wrappers around NDRange kernels require the work dimensionality be set during invocation using the MetaMorph a_dim3 type, (an array of 3 size_t elements). Unused dimensions should be set to 1. If you wish to take advantage of OpenCL's automatic local sizing (i.e. a NULL localSize), simply set all three local dimensions to zero. If either size is NULL, then a single work group will be enqueued, using the size specified in either a `reqd_work_group_size` or `work_group_size_hint` attribute, or falling back to MetaMorph's compiled default 3D workgroup size.

If you have enabled the `cuda-grid-block` MetaCL option, then you should instead specify your worksizes using the the grid-of-blocks notation (i.e. "CUDA-style"), **not** the global/local notation. The grid dimensionality can easily be derived from the global and local size as below:
```C
a_dim3 grid, block;
block[0] = localSize[0]; block[1] = localSize[1]; block[2] = localSize[2];
grid[0] = (globalSize[0]/localSize[0]) + (globalSize[0] % localSize[0] > 0) ? 1 : 0;
grid[1] = (globalSize[1]/localSize[1]) + (globalSize[1] % localSize[1] > 0) ? 1 : 0;
grid[2] = (globalSize[2]/localSize[2]) + (globalSize[2] % localSize[2] > 0) ? 1 : 0;
```


## Remove existing cl_program and cl_kernel destruction
MetaCL generates a single destruction function for each output file (`metacl_<filename>_init()`), which releases the associated `cl_program`(s) and `cl_kernel`(s). As such, there is no longer a need to manually `clRelease...()` these objects. Furthermore, when the MetaCL-generated module is deregistered from MetaMorph, each initialized set of `cl_program`(s) and `cl_kernel`(s) is automatically deinitialized. (As noted [below](#Explicitly-deregister-generated-module-from-MetaMorph), deregistration can also be performed automatically at program end.)


## Explicitly deregister generated module from MetaMorph
MetaMorph's OpenCL backend automatically destructs any OpenCL state it knows about at program termination. This includes any modules which solely implement OpenCL, such as those generated by MetaCL.

However, if the user intends to continue execution of the application after all need for OpenCL has finished, they may explicitly deregister the module from MetaMorph, by calling `meta_deregister_module` with a pointer to the generated `registry` function. This will trigger the automatic destruction of all `cl_programs` and `cl_kernels` within the module as noted [above](#Replace-existing-global-and-local-worksizes-with-MetaCL's-grid-and-block), **without** releasing any state tuples (platform, device, context, and queue) MetaMorph knows about, nor `cl_programs` and `cl_kernels` from other modules.