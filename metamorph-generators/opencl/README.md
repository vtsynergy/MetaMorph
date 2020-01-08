# MetaCL

MetaCL is a host *auto-generator* for OpenCL. Given a set of `.cl` kernel files, it will produce encapsulated initialization, deconstruction, and launch wrappers for each kernel function it encounters. In doing so, it can ensure the host-side kernel interface is automatically consistent with the device-side implementation, *at compile time*, while simultaneously reducing the total effort spent manually implementing boilerplate for kernel management and launches. These wrappers are designed to build off of the MetaMorph OpenCL backend's existing capabilites for automatic device and context management, providing an easy on-ramp to implementing new OpenCL projects with MetaMorph.

Since this release is a work-in-progress academic prototype, you may to face issues. Should you encounter any, please open a GitHub issue with details of your environnment and a minimal working example. 

MetaCL and underlying components of the MetaMorph OpenCL backend have been supported in part by NSF I/UCRC CNS-1266245 via NSF CHREC, and NSF I/UCRC CNS-1822080 via NSF SHREC.


Authors:
	Paul Sathre (MetaCL Implementation)
	Atharva Gondhalekar (MetaCL features)
	
	
Contact Email: 
	sath6220_at_vt.edu	

(c) Virginia Polytechnic Institute and State University, 2018-2020.


## News & Updates

Jan 15, 2020: Version 0.2b



## MetaCL Implementation

MetaCL is implemented as a *Clang Tool*, which uses the Clang/LLVM compiler framework as a library to implement its code traversal, analysis, and generation functionality. Each OpenCL kernel file is parsed, preprocessed, and semantically analyzed by Clang, and then MetaCL walks the resulting abstract syntax trees to construct the corresponding host-side interface. During this process, any user-defined data types or other special cases within the kernel are identified and imported or enforced in the generated host API. (For example, a user-defined struct that uses device data types such as `float4` will be recursively imported and rewritten with the corresponding host data types, i.e. `cl_float4` to ensure the alignment is consistent between device and host.)

Thanks to Clang's robust and efficient design, MetaCL is able to run in seconds, and thus should be incorporated as part of a project's automated build system so that it invoked any time a kernel file is changed. This will ensure that any changes to the device interface are detected, regenerated, and enforced during host compilation.



## Dependencies

	make
	GNU C Library (glibc)
	
	Clang >= 6.0
	
	MetaMorph
	
	MetaMorph OpenCL-backend
		GNU GCC compiler (tested with gcc 4.5, 4.7.2, 4.8.2 and 4.9.2)
		OpenCL libs


## Usage

For detailed installation and usage instructions, please refer to the [MetaCL Tutorials](./docs/tutorials/README.md)


## License 

Please refer to MetaMorph's included [LICENSE](../../LICENSE) file.

