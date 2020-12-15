# MetaCL

MetaCL is a host *auto-generator* for OpenCL. Given a set of `.cl` kernel files, it will produce encapsulated initialization, deconstruction, and launch wrappers for each kernel function it encounters. In doing so, it can ensure the host-side kernel interface is automatically consistent with the device-side implementation, *at compile time*, while simultaneously reducing the total effort spent manually implementing boilerplate for kernel management and launches. These wrappers are designed to build off of the MetaMorph OpenCL backend's existing capabilites for automatic device and context management, providing an easy on-ramp to implementing new OpenCL projects with MetaMorph.

Since this release is a work-in-progress academic prototype, you may to face issues. Should you encounter any, please open a GitHub issue with details of your environnment and a minimal working example. 

MetaCL and underlying components of the MetaMorph OpenCL backend have been supported in part by NSF I/UCRC CNS-1266245 via NSF CHREC, and NSF I/UCRC CNS-1822080 via NSF SHREC.


Authors:
	Paul Sathre (MetaCL Implementation)
	Atharva Gondhalekar (MetaCL features)
	
	
Contact Email: 
	sath6220_at_vt.edu	

(c) Virginia Polytechnic Institute and State University, 2018-2021.


## News & Updates

* Jan 13, 2021: Version 0.3.1b: MetaCL with Standalone Interface support
* Jun 10, 2020: Version 0.3b: Integration with packaged MetaMorph v0.3b
* Jan 15, 2020: Version 0.2b


## Publications

* "MetaCL: Automated 'Meta' OpenCL Code Generation for High-Level Synthesis on FPGA." Paul Sathre, Atharva Gondhalekar, Mohamed Hassan, Wu-chun Feng. In Proceedings of the 24th Annual IEEE High Performance Extreme Computing Conference (HPEC '20), Waltham, Massachusetts, USA, September 2020. 


## MetaCL Implementation

MetaCL is implemented as a *Clang Tool*, which uses the Clang/LLVM compiler framework as a library to implement its code traversal, analysis, and generation functionality. Each OpenCL kernel file is parsed, preprocessed, and semantically analyzed by Clang, and then MetaCL walks the resulting abstract syntax trees to construct the corresponding host-side interface. During this process, any user-defined data types or other special cases within the kernel are identified and imported or enforced in the generated host API. (For example, a user-defined struct that uses device data types such as `float4` will be recursively imported and rewritten with the corresponding host data types, i.e. `cl_float4` to ensure the alignment is consistent between device and host.)

Thanks to Clang's robust and efficient design, MetaCL is able to run in seconds, and thus should be incorporated as part of a project's automated build system so that it invoked any time a kernel file is changed. This will ensure that any changes to the device interface are detected, regenerated, and enforced during host compilation.



## Dependencies

For building MetaCL
* make
* GNU C Library (glibc)
* GNU GCC compiler with support for C++11)
* Clang >= 6.0 (Tested with 6.0, 6.0.1, 7.0, 9.0.1 and 10.0)

For interface generation:
* Clang >= 6.0 libraries (Tested with 6.0, 6.0.1, 7.0, 9.0.1 and 10.0) (IFF *not* compiled with `METACL_LINK_STATIC=true`)
    * Note: In our testing, Clang 10 requires C++14 support and some additional LLVM libraries when linking statically
* Clang OpenCL kernel builting header: `opencl-c.h` needed when invoking metaCL

For compiling, linking, and executing program based on the auto-generated interface:
* MetaMorph (IFF `--use-metamorph` is set to `REQUIRED`, otherwise treated as optional or unused with `OPTIONAL` or `DISABLED`, respectively)
* MetaMorph OpenCL-backend (IFF `--use-metamorph=REQUIRED`)
* OpenCL headers, library, device(s)

The following Linux distributions have been tested, and should be supported based on what's available in their standard package managers. No additional configuration should be necessary unless things have been placed in non-standard locations by your system administrator or by a module/collections manager.

### Debian
#### 9 / 10
Presuming the use of the apt package manger, the following are needed:
* `build-essential`
* `libclang-<version>-dev` (`7` was tested for the 0.3.1b release)
* `llvm-<version>-dev` (must match the version of libclang used)
* `zlib1g-dev`
* `libncurses-dev` (in 10+, this is now correctly marked as a dependency of `llvm-7-dev` and should be installed automatically)

### Ubuntu
#### 18.04, 20.04
Presuming the use of the apt package manger, the following are needed:
* `build-essential`
* `libclang-<version>-dev` (`7` on 18.04 and `10` on 20.04 were tested for the 0.3.1b release)
* `llvm-<version>-dev` (must match the version of libclang used)
* `zlib1g-dev`
* `libncurses-dev` (in 20.04+, this is now correctly marked as a dependency of `llvm-10-dev` and should be installed automatically)`

### Centos
#### 7
Presuming the use of the yum package manager, the following are needed:
* group "Development Tools"
* `llvm-toolset-7.0-llvm-devel`
* `llvm-toolset-7.0-clang-devel`

Make should auto-detect the `/opt` directories the LLVM 7.0 Toolset packages are installed to. A Shared library build is the only option, as there are no Clang static libraries provided by the `llvm-toolset-7.0-llvm-static` package, nor is there an equivalent `clang-static` package.

### Fedora
#### 31
Presuming the use of the yum package manager, the following are needed:
* group "Development Tools"
* `llvm-devel`
* `clang-devel`

Make should auto-detect the headers and libraries in their standard `/usr` subdirectories. A Shared library build is the only option, as there are no Clang static libraries provided by the `llvm-static` package, nor is there an equivalent `clang-static` package.

#### Other Linux / Clang+LLVM from source
TODO
		
		
##Installation

Please follow the installation instructions in the top-level Makefile.


## Usage

For detailed installation and usage instructions, please refer to the [MetaCL Tutorials](./docs/tutorials/README.md)

Examples of invocation syntax can be found in the `metacl_module.c` make targets of https://github.com/vtsynergy/MetaCL-SNAP/blob/master/src/Makefile and https://github.com/vtsynergy/MetaCL-BabelStream/blob/master/OpenCL.make

#### Esoterica:
* On some non-systemwide Clang Installations, it is necessary to provide an additional header search directory (`-I <dir>`) to the Clang internal compiler (i.e. after the `--` that separates MetaCL options from Clang options). This must point to the directory that contains the Clang installation's copy of *their* OpenCL kernel headers (i.e. `/usr/lib/clang/6.0.1/include/opencl-c.h`) In the codes above this is captured in the `METACL_CFLAGS`, which are provided on the command line and would typically look like `METACL_CFLAGS="-I /usr/lib/clang/6.0.1/include" make -f OpenCL.make`. We hope to eliminate this requirement in a future release.


## License 

MetaCL, the *static* portions of the MetaMorph API it uses and produces (i.e. `metamorph.h` and `metamorph_opencl.h`), and the *static* emulation shims it produces to support `--use-metamorph=OPTIONAL` (i.e. `shim_dynamic.h`) and `--use-metamorph=DISABLED` (i.e. `metamorph_shim.c`), are governed by the same license as MetaMorph proper.
Please refer to MetaMorph's included [LICENSE](../../LICENSE) file.

The use of MetaCL extends *no* copyright whatsover to *your* code. (i.e. any `.cl` kernel(s) provided as input, as well as the *dynamic* output files that bear the names and signatures of *your* inputs, structs and kernel functions.) But no warranty is provided for such dynamically-generated code.

Our goal is to allow you as much flexibility with your generated code, while retaing access to the shared utility functions for the widest audience possible. If you have any questions about these terms, or need alternate licensing options, please contact us.

