# WIP

# Generating an OpenCL Host-to-Device Interface with MetaCL

MetaCL is a command-line tool to consume kernel files written in OpenCL C (typically `.cl` extension) and produce the corresponding host-side code. This host-side code includes wrapper functions to invoke the kernels themselves, as well as all the necessary boilerplate to JIT-compile the kernels (for devices that support JIT, like CPU and GPU) or load offline-compiled kernel implementations (such as `.aocx` files for Intel/Altera FPGA SDK for OpenCL). The generated code can also leverage simplified device selection, context swapping, and automatic initialization/deconstruction provided by MetaMorph's OpenCL backend.

MetaCL is implemented as a _[Clang Tool](https://clang.llvm.org/docs/LibTooling.html)_ and thus relies on a modern installation of Clang, and behaves much like a compiler. (In effect it is performing a pseudo source-to-source compilation of the kernel files to produce the host-to-device interface.)

This tutorial details the invocation of the MetaCL tool itself and presumes MetaCL/MetaMorph have been installed according to the [these instructions](./InstallingMetaCL.md) - if you have used different paths, directory names, or version of Clang, please remember to adapt the below code examples to your changes. After generating a host-to-device interface with MetaCL, please consult [the following tutorial](./ExistingApplication.md) for specific recommendations for how to use the generated code.

Structure of a MetaCL Invocation
--------------------------------
```bash
path/to/MetaMorph/metamorph-generators/opencl/metaCL <List of .cl files to process> [MetaCL options] -- -cl-std=<Version of OpenCL C to parse> -include opencl-c.h -I <path to Clang's implementation of opencl-c.h> [Project-specific compilation arguments] 
```

`path/to/MetaMorph/metamorph-generators/opencl/metaCL` This is the metaCL binary built during [installation](./InstallingMetaCL.md)

`<List of .cl files to process>` This is a required list of 1 or more files containing kernels implemented in OpenCL C (typically these have ".cl" extension). These files must contain only kernel code, but are preprocessed so may `#include` kernel data structures, `device` function prototypes, etc. (Be sure to specify necessary `-I <folder>` arguments in the `[Project-specific compilation arguments]` section of the invocation.)

`[MetaCL options]` An Optional set of _options_ to control MetaCL's code generation. A list of all that are currently supported is provided [below](#Supported-MetaCL-Options).

`--` The double-dash is a required component of a _Clang Tool's_ invocation. It separates arguments to the tool, i.e. MetaCL (to the left of the `--`) from the underlying compiler instance, i.e. Clang (to the right of the `--`)

`-cl-std=<Version of OpenCL C to parse>` The underlying Clang compiler instance needs to know which standard it is parsing. For MetaCL all inputs should be OpenCL C, so supported versions are `CL1.0` `CL1.1` `CL1.2` and `CL2.0`.

`-include opencl-c.h` The underlying Clang compiler instance needs definitions for all the OpenCL C kernel builtin functions. These are defined in a special `opencl-c.h` header that must currently be manually included.

`-I <path to Clang's implementation of opencl-c.h>` A path to the directory containing the required `opencl-c.h` header. (If the [installation tutorial](./InstallingMetaCL.md) was followed, it should be located in `/path/to/MetaMorph/metamorph-generators/opencl/clang_tarball/clang+llvm-6.0.1-x86_64-linux-gnu-ubuntu-16.04/lib/clang/6.0.1/include`.)

`[Project-specific compilation arguments]` Application-specific conditional compilation options. These must be the same as the options provided to the OpenCL JIT compiler at runtime --- or analagous offline compilation step ---, and currently only one set can be incorporated into the generated interface at a time.


Supported MetaCL Options
------------------------

`--unified-output-file="<filename>"` (Default: NONE) This MetaCL option specifies the name of the output “.c” and “.h” pair to produce, which will include the interface components from all _N_  input files. Without this option, it will produce a pair for each input file, with the input file’s name.

For example `./metaCL A.cl B.cl C.cl -- ...` would produce `A.c` `A.h` `B.c` `B.h` `C.c` and `C.h` with separate boilerplate, whereas `./metaCL A.cl B.cl C.cl --unified-output="myKernelWrappers" -- ...` would produce `myKernelWrappers.c` and `myKernelWrappers.h`, containing unified init/destroy boilerplate functions and wrappers for each of the functions in `A.cl` `B.cl` and `C.cl`. Kernel implementations remain separate in their respective `.cl` files and are still individually loaded by the generated boilerplate functions.

`--inline-error-check=<true/false>` (Default: TRUE) This MetaCL option can be used to _disable_ the generation of simple OpenCL Runtime API error checks in generated code, which may trade safety for a slight performance gain. If left on, every generated API call is immediately checked for a returned error. (However `clEnqueue..()` operations are **not** forced to `clFinish()`, rather the return code from the enqueue call is checked.)

`--split-wrappers=<true/false>` (Default: FALSE) This option can be used to _enable_ the generation of two additional wrappers for each kernel. One which only assigns all the kernel arguments (metacl_filename_kernelName_set_args), and one which only invokes the kernel (metacl_filename*kernelName_enqueue_again). This can be useful to pre-assign arguments once for an iterative kernel, without incurring the assignment overhead on subsequent calls.

`--cuda-grid-block=<true/false>` (Default: FALSE) By default, MetaCL generates launch wrappers assuming the execution configuration is specified according to the semantics of OpenCL's global and local worksizes, i.e. total work elements in a dimension, and work elements within a workgroup. By enabling this flag, the wrappers can instead be generated using the CUDA grid/block model, i.e. the total number of work groups in a dimension, and the number of work elements within a workgroup. This can be useful if you are converting an existing CUDA application to OpenCL, or are implementing an application which provides both CUDA and OpenCL implementations. (The MetaMorph API historically uses grid/blcok semantics across all backends, so you may also wish to enable this for consistency if you use any of the built-in kernels.)


Common Project-specific compilation arguments
---------------------------------------------

`-I <dir>` A header search path for `#include`d files. Will be searched in the order specified on the command line, **before** any system-wide paths are searched.

`-D ENABLE_SOME_MACRO` or `-D MY_MACRO=SOME_VALUE` Conditional compilation macros to control the behavior of `#ifdef`s, or otherwise substitute hardcoded constants. These must be set to the exact value used in the call to `clBuildProgram` (or analagous offline compiler).