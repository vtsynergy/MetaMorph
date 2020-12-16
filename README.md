# MetaMorph Beta Release (0.3b)

This is the third public beta release of our library framework for interoperable kernels on multi- and many-core Clusters, namely MetaMorph. This release refactors the internal library interaction model to a plugin-based architecture. This refactoring is designed to further support modular installs and allow individual binary packages to be leveraged without build-time knowledge of the available backends. User code may still enforce a hard dependency on a specific backend, but applications using only the core API should now be able to execute on any machine that has at least one backend installed.

The prior release (v0.2b) provided the new OpenCL generator "MetaCL", which given a set of OpenCL kernel implementation files will automatically produce a MetaMorph-compatible simplified host wrapper API. This API abstracts the vast majority of OpenCL program and kernel management boilerplate, and in many cases offers a significant productivity boost to host programming at low runtime cost. For additional information see the [MetaCL Readme](./metamorph-generators/opencl/README.md).

Since this release is a work-in-progress academic prototype, you may to face issues. Should you encounter any, please open a GitHub issue with details of your environnment and a minimal working example. MetaMorph will incorporate additional HPC kernels (e.g. sorting, dynamic programming, n-body) and run-time services developed by the Synergy Laboratory @ Virginia Tech (http://synergy.cs.vt.edu/). Email the authors for more details. 

Currently, the C interface always takes priority w.r.t. performance. It is the shortest path from user code to library code, and the standard. Fortran compatibility is and should continue to be a convenience plugin only. For more details, check out the library header files.

MetaMorph is created as part of the Air Force Office of Scientific Research (AFOSR) Computational Mathematics Program via Grant number FA9550-12-1-0442. 

MetaCL and underlying components of the MetaMorph OpenCL backend have been supported in part by NSF I/UCRC CNS-1266245 via NSF CHREC, and NSF I/UCRC CNS-1822080 via NSF SHREC.

OpenCL code is largely generated by CU2CL. CU2CL has been supported in part by NSF I/UCRC IIP-0804155 via NSF CHREC.

Authors:
	Paul Sathre and Ahmed Helal (MetaMorph Design and Implementation)
	Paul Sathre (MetaCL Implementation)
	Atharva Gondhalekar (MetaCL features)
	Sriram Chivukula (CUDA dot-product/reduce prototypes)
	Kaixi Hou (CUDA data marshaling prototypes)
	Anshuman Verma (OpenCL FPGA back-end)
	
	
Contact Email: 
	{sath6220, ammhelal}_at_vt.edu	

(c) Virginia Polytechnic Institute and State University, 2013-2020.


## News & Updates

Jun 10, 2020: Version 0.3b: Plugin-based dynamically loaded refactoring
Jan 15, 2020: Version 0.2b: Introduction of the MetaCL: OpenCL Kernel Interface Autogenerator
Nov 12, 2016: Version 0.1b


## Publications

* "MetaCL: Automated 'Meta' OpenCL Code Generation for High-Level Synthesis on FPGA." Paul Sathre, Atharva Gondhalekar, Mohamed Hassan, Wu-chun Feng. In Proceedings of the 24th Annual IEEE High Performance Extreme Computing Conference (HPEC '20), Waltham, Massachusetts, USA, September 2020. 

* “MetaMorph: A Library Framework for Interoperable Kernels on Multi- and Many-core Clusters.“ Ahmed E. Helal, Paul Sathre, Wu-chun Feng. In Proceedings of the IEEE/ACM International Conference for High Performance Computing, Networking, Storage and Analysis (SC|16), Salt Lake City, Utah, USA, November 2016.

* “MetaMorph: A Modular Library for Democratizing the Acceleration of Parallel Computing across Heterogeneous Devices.” Paul Sathre, Wu-chun Feng. In ACM/IEEE International Conference on High-Performance Computing, Networking, Storage, and Analysis (SC|14), New Orleans, LA, USA, November 2014 (poster).
	

## MetaMorph Overview

MetaMorph is designed to effectively utilize HPC systems that consist of multiple heterogeneous nodes with different hardware accelerators. It acts as middleware between the application code and compute devices, such as CPUs, GPUs, Intel MIC and FPGAs. MetaMorph hides the complexity of developing code for and executing on heterogeneous platform by acting as a unified “meta-platform.” The application developer needs only to call MetaMorph’s computation and communication APIs, and the operations are transparently mapped to the proper compute devices. MetaMorph uses a modular layered design, where each layer supports one of its core design principles and each module can be used relatively independently.


## MetaMorph Design

MetaMorph provides programmability, functional portability, and performance portability by abstracting software backends (currently, OpenMP, CUDA and OpenCL) behind a single interface. It achieves high performance by providing low-level implementations of common operations, based on the best-known solutions for a given compute platform. Moreover, the software back-ends are instantiated and individually tuned for the different heterogeneous and parallel computing platforms (currently, multicore CPUs, Intel MICs, AMD GPUs, and NVIDIA GPUs). 

MetaMorph provides an infrastructure for adding software back-ends for future compute devices — without end-user intervention or modifying the application. This provides the small population of early-adopter, architecture experts with a framework that enables them to dramatically extend the impact of their expertise to the wider community by expanding the library with new design patterns. So, rather than writing a kernel once for a single application, these experts can write that same kernel within the MetaMorph framework, provide it to the community, and allow it to be used across many applications. In addition, MetaMorph accelerates the development of new operations and computation/communication patterns by providing a compilation infrastructure and helper APIs that handle the boilerplate initialization and compilation and simplify data exchange between the host and accelerators, such that MetaMorph developers can focus on developing the new kernels.

Existing kernels (e.g., CUDA kernels) can be included in MetaMorph without re-factoring by adding their implementation directly into the interoperability layer (e.g., CUDA backend) and their C/Fortran interface in the abstraction layer. However, advanced features, like seamless execution on different accelerators within a node and across nodes, will work only if these kernels are implemented in all the different back-ends. Contribution of a kernel for even a single back-end is valuable as it provides the architecture-expert community a baseline from which to implement and integrate kernels for the remaining back-ends.


## MetaMorph Implementation

MetaMorph implements the offload/accelerator computation model, in which data is explicitly allocated, copied, and manipulated via kernels within the MetaMorph context. We realize it as a layered library of libraries. The top-level user APIs and platform-specific back-ends exists as separate shared library objects, with interfaces designated in shared header files. The top-level API improves the programmability of user applications by abstracting the backends,which provide accelerated kernels for each platform. It intercepts calls to the MetaMorph communication and computation kernels and transparently maps them to a back-end accelerator supported by the underlying platform. Back-ends are responsible for providing a standard C interface to the accelerated kernels. They are segregated from one another in order to allow separate compilation and encapsulation of platform-specific nuances. Consequently, if a given back-end requires special-purpose libraries or tools to build that are not present on the target machine, it can be easily excluded from a given build of the library as a whole without loss of function in the remaining back-ends.


## Dependencies

	make
	GNU C Library (glibc)
	
	Communication Interface
		MPI (tested with MPICH 3.1.4, MPICH 3.2 and OpenMPI 1.6.4)
	
	OpenMP-backend
		GNU GCC compiler (tested with gcc 4.5, 4.7.2, 4.8.2 and 4.9.2)
		Intel C/C++ Compiler (tested with icc 13.1)
		
	CUDA-backend
		NVIDIA CUDA toolkit (tested with nvcc 5, 6 and 7.5)
		
	OpenCL-backend
		GNU GCC compiler (tested with gcc 4.5, 4.7.2, 4.8.2 and 4.9.2)
		OpenCL libs
		
	MetaCL
		Clang and libClang >= 6.0 (static libraries)
		
		
## Installation

We recommend the use of pre-installed packages, which can be found at: https://vtsynergy.github.io/packages/. We will endeavour to keep this updated with major releases, as time allows. However, to leverage the newest features and bugfixes, a traditional make from this repository is also supported.

Configuration is managed through the use of command-line Make variables. The Makefile will attempt to locate packages and tools necessary to build the relevant backends and automatically configure itself. `make all` and `make install` will use this auto-configuration. The following overrides are available to assist with configuration:

* `DESTDIR` The root directory that `install` targets should place finished libraries, headers, and binaries into. By default assumed to be `/`, but can be placed in user-space for non-sudo installations
* `USE_OPENMP=<TRUE/FALSE>` Explicitly enable or disable building the OpenMP backend
* `USE_OPENCL=<TRUE/FALSE>` Explicitly enable or disable building the OpenCL backend
* `OPENCL_LIB_DIR` The path to the libraries of an OpenCL installation, i.e. where `$(OPENCL_LIB_DIR)/libOpenCL.so` is located
* `OPENCL_INCL_DIR` The path to the headers of an OpenCL installation, i.e. where `$(OPENCL_INCL_DIR)/CL/opencl.h` is located
* `USE_CUDA=<TRUE/FALSE>` Explicitly enable or disable building the CUDA backend
* `CUDA_LIB_DIR` The path to the root of a CUDA installation i.e. where `$(CUDA_LIB_DIR)/libcudart.so` and `$(CUDA_LIB_DIR)))/../bin/nvcc` are located
* `USE_MPI=<TRUE/FALSE>` Explicitly enable or disable building the MPI Plugin
* `MPI_DIR` The path to the root of an MPI installation, which provides MPI C compiler, i.e. where `$(MPI_DIR)/bin/mpicc` is located
* `USE_TIMERS=<TRUE/FALSE>` Explicitly enable or disable building the Timing Plugin
* `USE_FPGA=<BLANK/"INTEL">` **Work in progress, not needed for MetaCL-generated FPGA codes**. Whether to compile built-in kernels for FPGA **(Warning can be time-intensive)**

Many of these overrides will not need to be specified if relevant library, header, and binary paths are correctly set in your `LIBRARY_PATH`, `CPATH`, and `PATH` variables, respectively.

To build MetaCL, invoke the `make generators` target, which supports the following overrides to assist in locating a Clang installation:
* `CLANG_LIB_PATH` The path to the static Clang libraries (typically included with the `-dev` or `-devel` versions of `libClang`), i.e where `$(CLANG_LIB_PATH)/libclangTooling.a` is located




## Usage

* Set the root directories (MPICH_DIR and MM_DIR) in the top-level Makefile.
* To build the library with all the supported backends: `make metamorph_all`.  
* You may build the library with one or more backends using the additional make targets.
* To build the examples: $make examples. 
* Include metamorph/lib into the LD_LIBRARY_PATH environment variable.
* Change the working directory to metamorph/examples and run the executables using the app-specific options. 
* The OpenCL backend supports a configurable search path for kernel implementations, which can be assigned to the `METAMORPH_OCL_KERNEL_PATH` environment variable. Multiple directories are delimited with `:`, like a typical Unix PATH variable.


## License 

Please refer to the included [LICENSE](./LICENSE) file.

