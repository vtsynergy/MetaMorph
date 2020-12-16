#Most build options are just an explicit TRUE/FALSE
#Options that default on: USE_OPENMP, USE_TIMERS
#Options that attempt to autodetect: USE_CUDA, USE_OPENCL
#Options that must be explicitly enabled: USE_MPI, USE_FPGA, USE_MIC 

#Manually update package version information
#Use origin for a stronger "not-defined" check, and a defined but empty variable still returns true to ifdef
ifeq ($(origin VERSION),undefined)
VERSION=0.3.1b-rc1
endif

#define version_less
#LEFT_MAJOR=$(shell sed -i 's/\..*//' $(1)
#endef

#Configure root directories (usually just wherever you pulled MetaMorph and run the makefile from)
ifndef MM_DIR
export MM_DIR=$(shell pwd | sed 's/ /\\ /g')
endif
export MM_CORE=$(MM_DIR)/metamorph-core
export MM_MP=$(MM_DIR)/metamorph-backends/openmp-backend
export MM_CU=$(MM_DIR)/metamorph-backends/cuda-backend
export MM_CL=$(MM_DIR)/metamorph-backends/opencl-backend
export MM_EX=$(MM_DIR)/examples

export MM_LIB=$(MM_DIR)/lib
#TODO does it need a custom dir?
MM_AOCX_DIR=$(MM_LIB)

export MM_GEN_CL=$(MM_DIR)/metamorph-generators/opencl
OS := 
ARCH :=
VER :=
ifeq ($(shell uname),Linux)
ARCH = $(shell uname -m)#| sed 's/x86_//;s/i[3-6]86/32/')
ifneq (, $(shell which lsb_release 2>/dev/null))
OS = $(shell lsb_release -si)
VER = $(shell lsb_release -sr)
else
#grep the /etc/*-release file instead
OS = $(shell cat /etc/*-release | grep "^ID=" | sed 's/^.*=//g')
VER = $(shell cat /etc/*-release | grep "^VERSION_ID=" | sed 's/^.*=//g')
endif
OS := $(shell echo $(OS) | tr '[:upper:]' '[:lower:]')
endif
#check if 64-bit libs should be used
ifeq ($(shell arch),x86_64)
ARCH_64 = x86_64
else
ARCH_64 =
endif
#Debug default to off
ifndef DEBUG
DEBUG=FALSE
else
DEBUG := $(shell echo $(DEBUG) | tr '[:lower:]' '[:upper:]')
endif

#Tagging with package version
ifeq ($(origin VERSION_STR),undefined)
 ifneq ($(strip $(VERSION)),)
  VERSION_STR=-$(VERSION)
 else
  #what is happening
  VERSION_STR=
 endif
endif

#Timers default to on
ifndef USE_TIMERS
USE_TIMERS=TRUE
else
USE_TIMERS := $(shell echo $(USE_TIMERS) | tr '[:lower:]' '[:upper:]')
endif

#CHECK for an MPI environment
ifndef MPI_DIR
#attempt to autodetect
ifneq ($(shell which mpicc 2>/dev/null),)
MPI_DIR=$(patsubst %/bin/mpicc,%,$(shell which mpicc))
else
MPI_DIR=
endif
else
#confirm existence
ifneq ($(shell test -e $(MPI_DIR)/bin/mpicc && echo -n yes),yes) #check if mpicc exists
$(error bin/mpicc not found at MPI_DIR=$(MPI_DIR))
endif
endif
#If MPI is explicitly set, captialize
ifdef USE_MPI
USE_MPI := $(shell echo $(USE_MPI) | tr '[:lower:]' '[:upper:]')
#If it's enabled, but not found, error out
ifeq ($(USE_MPI),TRUE)
ifeq ($(MPI_DIR),)
$(error USE_MPI is set but no MPI environment found)
endif
endif
endif
#If not explicitly enabled, only activate if the directory exists
ifndef USE_MPI
ifneq ($(MPI_DIR),)
USE_MPI=TRUE
else
USE_MPI=FALSE
endif
endif


#Ensure CUDA environment
ifndef CUDA_LIB_DIR #Autodetect a cuda installation
ifeq ($(shell which nvcc 2>/dev/null),)
#none found
CUDA_LIB_DIR=
else
#found
CUDA_LIB_DIR=$(patsubst %/bin/nvcc,%,$(shell which nvcc))/$(if ARCH_64,lib64,lib)
#Make sure the installed CUDA version uses the maximum supported GCC
NVCC_VER=$(shell nvcc --version | grep "release" | awk '{print $5 }' | sed 's/,//')
GCC_VER=$(shell gcc --version | grep "gcc" | awk '{print $3}')
#CUDA 4.1 --> gcc-4.5
#CUDA 5.0 --> gcc-4.6
#CUDA 6.0 --> gcc-4.7
#CUDA 7.0 --> gcc-4.8 (4.9 if debian)
#CUDA 7.5 --> gcc-4.8 (4.9 if debian)
#CUDA 8 --> gcc-5.3
#CUDA 9 --> gcc-6
#CUDA 9.2 --> gcc-7
#CUDA 10.1 --> gcc-8
#CUDA 10.2 --> ?? gcc-8 but the toolkit doc says 7.3.0 for Ubuntu 18.04
NVCC=nvcc
#Long term we want to automatically match the installed CUDA and CC version, but there is broad variabilty between SDK, different OS versions, and different C compilers. For now just pass through an options string
ifdef NVCC_OPTS
NVCC += $(NVCC_OPTS)
#-ccbin gcc-4.9
endif
endif
else #User has provided one, check it exists
ifeq ($(shell test -e $(CUDA_LIB_DIR)/libcudart.so && echo -n yes),yes) #Check if the CUDA libs exist where they told us
NVCC=$(patsubst %/lib,%,$(patsubst %/lib64,%,$(CUDA_LIB_DIR)))/bin/nvcc
else
$(error Cannot find CUDA installation at CUDA_LIB_DIR=$(CUDA_LIB_DIR))
endif
endif

ifndef USE_CUDA #if not explicitly set
ifeq ($(CUDA_LIB_DIR),) #see if we found nvcc
#nope, disable
USE_CUDA=FALSE
else
#otherwise enable
USE_CUDA=TRUE
endif
else
#make uppercase if set by user
USE_CUDA := $(shell echo $(USE_CUDA) | tr '[:lower:]' '[:upper:]')
endif
#if true, make sure we have a working CUDA environment
ifeq ($(USE_CUDA),TRUE)
ifeq ($(CUDA_LIB_DIR),)
$(error CUDA_LIB_DIR is unset and not autodetected)
endif
endif


#Ensure at least one OpenCL environment
#if they explicitly set an environment they need to point us to both libs and headers
ifdef OPENCL_LIB_DIR
 ifdef OPENCL_INCL_DIR
  #both are set, validate things exist
  ifeq ($(shell test -e $(OPENCL_INCL_DIR)/CL/opencl.h && echo -n yes),yes)
   ifeq ($(shell test -e $(OPENCL_LIB_DIR)/libOpenCL.so && echo -n yes),yes)
   else
    $(error libOpenCL.so not found in OPENCL_LIB_DIR=$(OPENCL_LIB_DIR))
   endif
  else
   $(error CL/opencl.h not found in OPENCL_INCL_DIR=$(OPENCL_INCL_DIR))
  endif
 else
  $(error OPENCL_LIB_DIR set without setting OPENCL_INCL_DIR)
 endif
else
 ifdef OPENCL_INCL_DIR
  $(error OPENCL_INCL_DIR set without setting OPENCL_LIB_DIR)
 endif
endif
#NVIDIA installations
ifeq ($(OPENCL_LIB_DIR),)
ifneq ($(CUDA_LIB_DIR),)
OPENCL_LIB_DIR=$(CUDA_LIB_DIR)
OPENCL_INCL_DIR=$(patsubst %/lib,%,$(patsubst %/lib64,%,$(CUDA_LIB_DIR)))/include
endif
endif
#TODO AMD APP SDK installations (though deprecated)

#AMD ROCM installations
ifeq ($(OPENCL_LIB_DIR),)
ifeq ($(shell test -e /opt/rocm/opencl/include/CL/opencl.h && echo -n yes),yes)
OPENCL_LIB_DIR=/opt/rocm/opencl/lib/x86_64
OPENCL_INCL_DIR=/opt/rocm/opencl/include
endif
endif
ifeq ($(OPENCL_LIB_DIR),)
ifeq ($(shell test -e /opt/intel/opencl/lib64/libOpenCL.so && echo -n yes),yes)
ifeq ($(shell test -e /opt/intel/opencl/include/CL/opencl.h && echo -n yes),yes)
OPENCL_LIB_DIR=/opt/intel/opencl/lib64
OPENCL_INCL_DIR=/opt/intel/opencl/include
endif
endif
endif

#fallback to a find
ifeq ($(OPENCL_LIB_DIR),)
OPENCL_LIB_DIR=$(patsubst %/libOpenCL.so,%,$(shell find / -path /home -prune -o -name libOpenCL.so -print -quit 2>/dev/null))
endif
ifeq ($(OPENCL_INCL_DIR),)
OPENCL_INCL_DIR=$(patsubst %/CL/opencl.h,%,$(shell find / -path /home -prune -o -name opencl.h -print -quit 2>/dev/null))
endif
#endif #end autodetect/validate

ifndef USE_OPENCL #if not explicitly set
ifeq ($(OPENCL_LIB_DIR),) #see if we found an environment
#nope, disable
USE_OPENCL=FALSE
else
#otherwise enable
USE_OPENCL=TRUE
endif
else
#make uppercase if set by user
USE_OPENCL := $(shell echo $(USE_OPENCL) | tr '[:lower:]' '[:upper:]')
endif

ifeq ($(USE_OPENCL),TRUE)
ifeq ($(OPENCL_LIB_DIR),)
$(error OPENCL_LIB_DIR and OPENCL_INCL_DIR unset and not autodetected)
endif
endif
#TODO add FPGA options
#Set to one of INTEL, XILINX, or blank
ifndef USE_FPGA
USE_FPGA=
else
USE_FPGA := $(shell echo $(USE_FPGA) | tr '[:lower:]' '[:upper:]')
endif
ifndef OPENCL_SINGLE_KERNEL_PROGS
OPENCL_SINGLE_KERNEL_PROGS=
else
OPENCL_SINGLE_KERNEL_PROGS := $(shell echo $(OPENCL_SINGLE_KERNEL_PROGS) | tr '[:lower:]' '[:upper:]')
endif

#Ensure OpenMP (simple other than MIC stuff)
ifndef USE_OPENMP
USE_OPENMP=TRUE
else
USE_OPENMP := $(shell echo $(USE_OPENMP) | tr '[:lower:]' '[:upper:]')
endif
ifndef USE_MIC
USE_MIC=FALSE
else
USE_MIC := $(shell echo $(USE_MIC) | tr '[:lower:]' '[:upper:]')
endif
ifeq ($(USE_MIC),TRUE)
ifneq ($(USE_OPENMP),TRUE)
$(warning USE_MIC requires USE_OPENMP, overriding USE_OPENMP=$(USE_OPENMP))
USE_OPENMP=TRUE
endif
ifndef ICC_BIN
ICC_BIN=$(shell which icc 2>/dev/null)
else
#validate their path
ifneq ($(shell test -e $(ICC_BIN) && echo -n yes),yes)
$(error icc not found at ICC_BIN=$(ICC_BIN))
endif
endif
ifeq ($(ICC_BIN),)
$(error cannot compile for Intel MIC without ICC \(ICC_BIN=$(ICC_BIN)\))
endif
endif

ifndef FPGA_USE_EMULATOR
FPGA_USE_EMULATOR=FALSE
else
FPGA_USE_EMULATOR := $(shell echo $(FPGA_USE_EMULATOR) | tr '[:lower:]' '[:upper:]')
endif

#Paul above here is the discovery stuff, below here is the configuration stuff


BUILD_LIBS = $(MM_LIB)/libmetamorph.so
export G_TYPE = DOUBLE
ifeq ($(DEBUG),TRUE)
export OPT_LVL = -g -DDEBUG
else
export OPT_LVL = -O3
endif
export L_FLAGS= -fPIC -shared

CC=gcc
#CC=icc
CC_FLAGS = $(OPT_LVL) $(L_FLAGS)
INCLUDES  = -I $(MM_DIR)/include


#Configure compilation to the machine's availability
MM_DEPS= $(MM_CORE)/metamorph.c $(MM_CORE)/metamorph_fortran_compat.c $(MM_CORE)/metamorph_dynamic_symbols.c

#MPI features
ifeq ($(USE_MPI),TRUE)
BUILD_LIBS += $(MM_LIB)/libmetamorph_mpi.so
MPICC := $(MPI_DIR)/bin/mpicc -cc=$(CC)
endif

#timer features
ifeq ($(USE_TIMERS),TRUE)
BUILD_LIBS += $(MM_LIB)/libmetamorph_profiling.so
endif

#CUDA backend
ifeq ($(USE_CUDA),TRUE)
BUILD_LIBS += $(MM_LIB)/libmetamorph_cuda.so
endif

#OpenCL backend
ifeq ($(USE_OPENCL),TRUE)
 ifeq ($(USE_FPGA),INTEL)
  OPENCL_FLAGS += -D WITH_INTELFPGA
  BUILD_LIBS += $(MM_LIB)/libmetamorph_opencl_intelfpga.so
  ####################### FPGA ##################################
  # Where is the Intel(R) FPGA SDK for OpenCL(TM) software?
  #TODO clean up this lookup
  ifeq ($(wildcard $(ALTERAOCLSDKROOT)),)
   $(error Set ALTERAOCLSDKROOT to the root directory of the Intel\(R\) FPGA SDK for OpenCL\(TM\) software installation)
  endif
  ifeq ($(wildcard $(ALTERAOCLSDKROOT)/host/include/CL/opencl.h),)
   $(error Set ALTERAOCLSDKROOT to the root directory of the Intel\(R\) FPGA SDK for OpenCL\(TM\) software installation.)
  endif
  # OpenCL compile and link flags.
  #TODO cleanup these variables to be explicitly passed instead of exported
  export AOCL_COMPILE_CONFIG := $(shell aocl compile-config )
  export AOCL_LINK_CONFIG := $(shell aocl link-config )

  #TODO board should not be hardcoded, instead expose through INTEL_FPGA_AOC_FLAGS
  #TODO if we expose the flags do we need an explicit FPGA_USE_EMULATOR variable?
  ifeq ($(FPGA_USE_EMULATOR),TRUE)
   export AOC_DEF= -march=emulator -v --board bdw_fpga_v1.0
  else
   export AOC_DEF= -v --board bdw_fpga_v1.0
  endif

  #There is no reason this mechanic has to be specific to FPGA
  ifeq ($(OPENCL_SINGLE_KERNEL_PROGS),TRUE)
   export FPGA_DEF=-D WITH_INTELFPGA -D OPENCL_SINGLE_KERNEL_PROGS
  else
   #TODO These FPGA options should not be hardcoded, and in fact should be subsumed by the OPENCL_SINGLE_KERNEL_PROGS option
   export FPGA_DEF=-D WITH_INTELFPGA -D KERNEL_CRC -D FPGA_UNSIGNED_INTEGER
  endif
  #TODO Remove hardcode and expose as INTELFPGA_LIB_DIR
  export FPGA_LIB=-L /media/hdd/jehandad/altera_pro/16.0/hld/host/linux64/lib -L $(AALSDK)/lib


 else ifeq ($(USE_FPGA),XILINX)
  OPENCL_FLAGS += -D WITH_XILINXFPGA
  BUILD_LIBS += $(MM_LIB)/libmetamorph_opencl_xilinx.so
  $(error XILINX not yet supported)
 else ifeq ($(USE_FPGA),) #The non-FPGA implementation has an empty string
  BUILD_LIBS += $(MM_LIB)/libmetamorph_opencl.so
 else #They asked for an FPGA backend that is not explicitly supported
  $(error USE_FPGA=$(USE_FPGA) is not supported)
 endif
endif




ifeq ($(USE_OPENMP),TRUE)
 ifeq ($(USE_MIC),TRUE)
  ifeq ($(USE_MPI),TRUE)
   MPICC = $(MPI_DIR)/bin/mpicc -cc=$(ICC_BIN)
  else
   CC = $(ICC_BIN)
  endif

  CC_FLAGS += -mmic
  BUILD_LIBS += $(MM_LIB)/libmetamorph_openmp_mic.so
 else
  BUILD_LIBS += $(MM_LIB)/libmetamorph_openmp.so
 endif

 ifeq ($(CC),gcc)
  OPENMP_FLAGS += -fopenmp
 else
  OPENMP_FLAGS += -openmp
 endif
endif

export INCLUDES

#Move things to where they should be based on Linux FHS, everything under /usr as it's non-essential
#MetaCL --> /usr/bin/metaCL
#libraries --> /usr/lib/libmm ln -s --> /usr/lib/metamorph/libmm... (These might also make sense for /usr/local since which you want depend on the hardware and drivers in the system)
#headers --> /usr/include
#documentation --> /usr/share/doc
#OpenCL kernels? .cl and .aocx?? (These might make sense for /usr/local since which you want depend on the hardware in the system)
#Example binaries?
#Source-only examples?
#Base install directory. On many systems, according to the FHS ths would be /usr/local since the libraries are likely machine-specific. However Debian saves that path for the sysadmin's tools and pushes you to /usr instead
ifeq ($(OS),debian)
BASE_INSTALL_DIR=$(DESTDIR)/usr
else ifeq ($(OS),ubuntu)
BASE_INSTALL_DIR=$(DESTDIR)/usr
else
BASE_INSTALL_DIR=$(DESTDIR)/usr/local
endif
#Useful to have symbolic links in the main library directory
LINK_LIB_RDIR=lib
#Actual copies of the library may be somewhere else in the lib tree, in case multiple versions/implementations are co-installed
INSTALL_LIB_RDIR=$(LINK_LIB_RDIR)/metamorph
VERSIONED_LIB_RDIR=metamorph$(VERSION_STR)

MFLAGS := USE_CUDA=$(USE_CUDA) CUDA_LIB_DIR=$(CUDA_LIB_DIR) USE_OPENCL=$(USE_OPENCL) OPENCL_LIB_DIR=$(OPENCL_LIB_DIR) OPENCL_INCL_DIR=$(OPENCL_INCL_DIR) OPENCL_SINGLE_KERNEL_PROGS=$(OPENCL_SINGLE_KERNEL_PROGS) USE_OPENMP=$(USE_OPENMP) USE_MIC=$(USE_MIC) ICC_BIN=$(ICC_BIN) USE_TIMERS=$(USE_TIMERS) USE_MPI=$(USE_MPI) MPI_DIR=$(MPI_DIR) USE_FPGA=$(USE_FPGA) NVCC="$(NVCC)" MPICC="$(MPICC)" OPENMP_FLAGS="$(OPENMP_FLAGS)" OPENCL_FLAGS="$(OPENCL_FLAGS)" DESTDIR=$(DESTDIR) BASE_INSTALL_DIR=$(BASE_INSTALL_DIR) LINK_LIB_RDIR=$(LINK_LIB_RDIR) INSTALL_LIB_RDIR=$(INSTALL_LIB_RDIR) VERSIONED_LIB_DIR=$(VERSIONED_LIB_RDIR) $(MFLAGS)

METACL_FLAGS=
ifdef CLANG_LIB_PATH
METACL_FLAGS := CLANG_LIB_PATH=$(CLANG_LIB_PATH)
endif

.PHONY: all
all: $(BUILD_LIBS)

$(MM_LIB):
	if [ ! -d $(MM_LIB) ]; then mkdir -p $(MM_LIB); fi

$(MM_LIB)/libmetamorph.so: $(MM_LIB) $(MM_DEPS)
	$(CC) $(MM_DEPS) $(CC_FLAGS) $(INCLUDES) -L$(MM_LIB) $(MM_COMPONENTS) -o $(MM_LIB)/libmetamorph.so -ldl -shared -Wl,-soname,libmetamorph.so

$(MM_LIB)/libmetamorph_profiling.so: $(MM_LIB) $(MM_LIB)/libmetamorph.so $(MM_CORE)/metamorph_profiling.c
	$(CC) $(MM_CORE)/metamorph_profiling.c $(CC_FLAGS) $(INCLUDES) -L $(MM_LIB) -lmetamorph -o $(MM_LIB)/libmetamorph_profiling.so -shared -Wl,-soname,libmetamorph_profiling.so

$(MM_LIB)/libmetamorph_mpi.so: $(MM_LIB) $(MM_LIB)/libmetamorph.so $(MM_CORE)/metamorph_mpi.c
	$(MPICC) $(MM_CORE)/metamorph_mpi.c $(CC_FLAGS) $(INCLUDES) -I$(MPI_DIR)/include -L $(MM_LIB) -lmetamorph -L$(MPI_DIR)/lib -o $(MM_LIB)/libmetamorph_mpi.so -shared -Wl,-soname,libmetamorph_mpi.so

$(MM_LIB)/libmetamorph_openmp.so: $(MM_LIB) $(MM_LIB)/libmetamorph.so
	cd $(MM_MP) && $(MAKE) $(MFLAGS) libmetamorph_openmp.so

#TODO Make this happen transparently to this file, create a symlink in the backend's makefile	
$(MM_LIB)/libmetamorph_openmp_mic.so: $(MM_LIB) $(MM_LIB)/libmetamorph.so
	cd $(MM_MP) && $(MAKE) $(MFLAGS) libmetamorph_openmp_mic.so

$(MM_LIB)/libmetamorph_cuda.so: $(MM_LIB) $(MM_LIB)/libmetamorph.so
	cd $(MM_CU) && $(MAKE) $(MFLAGS) libmetamorph_cuda.so

$(MM_LIB)/libmetamorph_opencl.so: $(MM_LIB) $(MM_LIB)/libmetamorph.so
	cd $(MM_CL) && $(MAKE) $(MFLAGS) libmetamorph_opencl.so

#TODO Make this happen transparently to this file, create a symlink in the backend's makefile	
$(MM_LIB)/libmetamorph_opencl_intelfpga.so: $(MM_LIB) $(MM_LIB)/libmetamorph.so
	cd $(MM_CL) && $(MAKE) $(MFLAGS) libmetamorph_opencl_intelfpga.so

.PHONY: generators
generators: $(MM_GEN_CL)/metaCL


$(MM_GEN_CL)/metaCL: $(MM_GEN_CL)/metacl.cpp $(MM_GEN_CL)/metamorph_shim.c $(MM_GEN_CL)/shim_dynamic.h include/metamorph_emulatable.h $(MM_CL)/metamorph_opencl_emulatable.h
	cd $(MM_GEN_CL) && $(MAKE) metaCL

.PHONY: examples
examples: torus_ex

torus_ex:
	cd $(MM_EX) && $(MAKE) $(MFLAGS) torus_reduce_test

#Install should only do what's supported according to the config and auto-detected packages
#TODO add headers
.PHONY: install
install: all install-libraries install-headers install-metaCL

#these variants support the "only what's configured" install targets
.PHONY: install-libraries
install-libraries: $(foreach target,$(BUILD_LIBS),$(subst $(MM_LIB),$(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR),$(target)))

.PHONY: install-headers
install-headers: install-core-headers $(subst metamorph_cuda.h,metamorph_cuda.cuh,$(foreach target,$(BUILD_LIBS),$(subst .so,.h,$(subst libmetamorph,metamorph,$(subst $(MM_LIB),$(BASE_INSTALL_DIR)/include,$(target))))))

#install all should try to do everything
.PHONY: install-all
install-all: install-all-libraries install-all-headers install-metaCL

.PHONY: install-all-libraries
install-all-libraries: install-core-library install-backend-libraries install-plugin-libraries

.PHONY: install-core-library
install-core-library: $(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/libmetamorph.so

#Symbolic link to the actual library folder, in case we support multiple versioning down the road
$(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/libmetamorph.so: $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR)/libmetamorph.so
	#Remove any existing symlink
	@if [ -L $(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/libmetamorph.so ]; then rm $(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/libmetamorph.so; fi
	#Create a symlink in the main library directory
	ln -s ./metamorph/libmetamorph.so $(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/libmetamorph.so

#Copy to the actual library folder
$(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR)/libmetamorph.so: $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR) $(MM_LIB)/libmetamorph.so
	#Remove any existing copy
	@if [ -f $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR)/libmetamorph.so ]; then rm $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR)/libmetamorph.so; fi
	#Copy the current version
	cp $(MM_LIB)/libmetamorph.so $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR)/libmetamorph.so


#Create the actual library folder (versioned with a symlink to it if the version is non-empty)
ifneq ($(INSTALL_LIB_RDIR),$(LINK_LIB_RDIR)/$(VERSIONED_LIB_RDIR))
$(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR): $(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/$(VERSIONED_LIB_RDIR)
	#Link to the versioned directory
	@if [ ! -L $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR) ]; then ln -s ./$(VERSIONED_LIB_RDIR) $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR); fi

$(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/$(VERSIONED_LIB_RDIR):
else
$(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR):
endif
	#Ensure the directory exists
	@if [ ! -d $(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/$(VERSIONED_LIB_RDIR) ]; then mkdir -p $(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/$(VERSIONED_LIB_RDIR); fi

.PHONY: install-backends-all
install-backends-all: install-backend-libraries install-backend-headers

.PHONY: install-backend-libraries
install-backend-libraries: install-opencl-library install-cuda-library install-openmp-library

#Kernels remain separated, the OpenCL core can function without them for MetaCL-ized apps
.PHONY: install-opencl-all
install-opencl-all: install-opencl-library install-opencl-headers install-opencl-kernels

.PHONY: install-opencl-library
install-opencl-library: install-core-library $(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/libmetamorph_opencl.so install-opencl-kernels

#TODO Make this work generically with the FPGA backend via symlink
$(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/libmetamorph_opencl.so: $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR)/libmetamorph_opencl.so
	@if [ -L $(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/libmetamorph_opencl.so ]; then rm $(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/libmetamorph_opencl.so; fi
	ln -s ./metamorph/libmetamorph_opencl.so $(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/libmetamorph_opencl.so

$(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR)/libmetamorph_opencl.so: $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR) $(MM_LIB)/libmetamorph_opencl.so
	@if [ -f $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR)/libmetamorph_opencl.so ]; then rm $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR)/libmetamorph_opencl.so; fi
	cp $(MM_LIB)/libmetamorph_opencl.so $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR)/libmetamorph_opencl.so

#TODO Make this work generically with the FPGA backend aocx(s)
install-opencl-kernels: $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR) $(MM_DIR)/metamorph-backends/opencl-backend/metamorph_opencl.cl
	@if [ -f $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR)/metamorph_opencl.cl ]; then rm $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR)/metamorph_opencl.cl; fi
	cp $(MM_DIR)/metamorph-backends/opencl-backend/metamorph_opencl.cl $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR)/

.PHONY: install-cuda-all
install-cuda-all: install-cuda-library install-cuda-headers

.PHONY: install-cuda-library
install-cuda-library: install-core-library $(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/libmetamorph_cuda.so

$(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/libmetamorph_cuda.so: $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR)/libmetamorph_cuda.so
	@if [ -L $(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/libmetamorph_cuda.so ]; then rm $(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/libmetamorph_cuda.so; fi
	ln -s ./metamorph/libmetamorph_cuda.so $(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/libmetamorph_cuda.so

$(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR)/libmetamorph_cuda.so: $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR) $(MM_LIB)/libmetamorph_cuda.so
	@if [ -f $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR)/libmetamorph_cuda.so ]; then rm $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR)/libmetamorph_cuda.so; fi
	cp $(MM_LIB)/libmetamorph_cuda.so $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR)/libmetamorph_cuda.so

.PHONY: install-openmp-all
install-openmp-all: install-openmp-library install-openmp-headers

.PHONY: install-openmp-library
install-openmp-library: install-core-library $(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/libmetamorph_openmp.so

$(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/libmetamorph_openmp.so: $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR)/libmetamorph_openmp.so
	@if [ -L $(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/libmetamorph_openmp.so ]; then rm $(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/libmetamorph_openmp.so; fi
	ln -s ./metamorph/libmetamorph_openmp.so $(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/libmetamorph_openmp.so

$(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR)/libmetamorph_openmp.so: $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR) $(MM_LIB)/libmetamorph_openmp.so
	@if [ -f $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR)/libmetamorph_openmp.so ]; then rm $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR)/libmetamorph_openmp.so; fi
	cp $(MM_LIB)/libmetamorph_openmp.so $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR)/libmetamorph_openmp.so

.PHONY: install-plugins-all
install-plugins-all: install-mpi-all install-profiling-all

.PHONY: install-plugin-libraries
install-plugin-libraries: install-mpi-library install-profiling-library

.PHONY: install-mpi-all
install-mpi-all: install-mpi-library install-mpi-headers

.PHONY: install-mpi-library
install-mpi-library: install-core-library $(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/libmetamorph_mpi.so

$(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/libmetamorph_mpi.so: $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR)/libmetamorph_mpi.so
	@if [ -L $(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/libmetamorph_mpi.so ]; then rm $(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/libmetamorph_mpi.so; fi
	ln -s ./metamorph/libmetamorph_mpi.so $(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/libmetamorph_mpi.so

$(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR)/libmetamorph_mpi.so: $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR) $(MM_LIB)/libmetamorph_mpi.so
	@if [ -f $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR)/libmetamorph_mpi.so ]; then rm $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR)/libmetamorph_mpi.so; fi
	cp $(MM_LIB)/libmetamorph_mpi.so $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR)/libmetamorph_mpi.so

.PHONY: install-profiling-all
install-profiling-all: install-profiling-library install-profiling-headers

.PHONY: install-profiling-library
install-profiling-library: install-core-library $(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/libmetamorph_profiling.so

$(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/libmetamorph_profiling.so: $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR)/libmetamorph_profiling.so
	@if [ -L $(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/libmetamorph_profiling.so ]; then rm $(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/libmetamorph_profiling.so; fi
	ln -s ./metamorph/libmetamorph_profiling.so $(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/libmetamorph_profiling.so

$(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR)/libmetamorph_profiling.so: $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR) $(MM_LIB)/libmetamorph_profiling.so
	@if [ -f $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR)/libmetamorph_profiling.so ]; then rm $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR)/libmetamorph_profiling.so; fi
	cp $(MM_LIB)/libmetamorph_profiling.so $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR)/libmetamorph_profiling.so

.PHONY: install-all-headers
install-all-headers: install-core-headers install-backend-headers install-plugin-headers

.PHONY: install-core-headers
install-core-headers: install-main-header install-dynSym-header install-fortran-headers

$(BASE_INSTALL_DIR)/include:
	if [ ! -d $(BASE_INSTALL_DIR)/include ]; then mkdir -p $(BASE_INSTALL_DIR)/include; fi

.PHONY: install-main-header
install-main-header: $(BASE_INSTALL_DIR)/include/metamorph.h

$(BASE_INSTALL_DIR)/include/metamorph.h: $(BASE_INSTALL_DIR)/include $(BASE_INSTALL_DIR)/include/metamorph_emulatable.h include/metamorph.h
	@if [ -f $(BASE_INSTALL_DIR)/include/metamorph.h ]; then rm $(BASE_INSTALL_DIR)/include/metamorph.h; fi
	cp include/metamorph.h $(BASE_INSTALL_DIR)/include/

$(BASE_INSTALL_DIR)/include/metamorph_emulatable.h: $(BASE_INSTALL_DIR)/include include/metamorph_emulatable.h
	@if [ -f $(BASE_INSTALL_DIR)/include/metamorph_emulatable.h ]; then rm $(BASE_INSTALL_DIR)/include/metamorph_emulatable.h; fi
	cp include/metamorph_emulatable.h $(BASE_INSTALL_DIR)/include/

.PHONY: install-dynSym-header
install-dynSym-header: $(BASE_INSTALL_DIR)/include/metamorph_dynamic_symbols.h

$(BASE_INSTALL_DIR)/include/metamorph_dynamic_symbols.h: $(BASE_INSTALL_DIR)/include include/metamorph_dynamic_symbols.h
	@if [ -f $(BASE_INSTALL_DIR)/include/metamorph_dynamic_symbols.h ]; then rm $(BASE_INSTALL_DIR)/include/metamorph_dynamic_symbols.h; fi
	cp include/metamorph_dynamic_symbols.h $(BASE_INSTALL_DIR)/include/

.PHONY: install-fortran-headers
install-fortran-headers: $(BASE_INSTALL_DIR)/include/metamorph_fortran_compat.h $(BASE_INSTALL_DIR)/include/metamorph_fortran_header.F03

$(BASE_INSTALL_DIR)/include/metamorph_fortran_compat.h: $(BASE_INSTALL_DIR)/include include/metamorph_fortran_compat.h
	@if [ -f $(BASE_INSTALL_DIR)/include/metamorph_fortran_compat.h ]; then rm $(BASE_INSTALL_DIR)/include/metamorph_fortran_compat.h; fi
	cp include/metamorph_fortran_compat.h $(BASE_INSTALL_DIR)/include/

$(BASE_INSTALL_DIR)/include/metamorph_fortran_header.F03: $(BASE_INSTALL_DIR)/include include/metamorph_fortran_header.F03
	@if [ -f $(BASE_INSTALL_DIR)/include/metamorph_fortran_header.F03 ]; then rm $(BASE_INSTALL_DIR)/include/metamorph_fortran_header.F03; fi
	cp include/metamorph_fortran_header.F03 $(BASE_INSTALL_DIR)/include/

.PHONY: install-backend-headers
install-backend-headers: install-opencl-headers install-cuda-headers install-openmp-headers

.PHONY: install-opencl-headers
install-opencl-headers: $(BASE_INSTALL_DIR)/include/metamorph_opencl.h

$(BASE_INSTALL_DIR)/include/metamorph_opencl.h: $(BASE_INSTALL_DIR)/include $(BASE_INSTALL_DIR)/include/metamorph_opencl_emulatable.h metamorph-backends/opencl-backend/metamorph_opencl.h
	@if [ -f $(BASE_INSTALL_DIR)/include/metamorph_opencl.h ]; then rm $(BASE_INSTALL_DIR)/include/metamorph_opencl.h; fi
	cp metamorph-backends/opencl-backend/metamorph_opencl.h $(BASE_INSTALL_DIR)/include/

$(BASE_INSTALL_DIR)/include/metamorph_opencl_emulatable.h: $(BASE_INSTALL_DIR)/include metamorph-backends/opencl-backend/metamorph_opencl_emulatable.h
	@if [ -f $(BASE_INSTALL_DIR)/include/metamorph_opencl_emulatable.h ]; then rm $(BASE_INSTALL_DIR)/include/metamorph_opencl_emulatable.h; fi
	cp metamorph-backends/opencl-backend/metamorph_opencl_emulatable.h $(BASE_INSTALL_DIR)/include/

.PHONY: install-cuda-headers
install-cuda-headers: $(BASE_INSTALL_DIR)/include/metamorph_cuda.cuh

$(BASE_INSTALL_DIR)/include/metamorph_cuda.cuh: $(BASE_INSTALL_DIR)/include metamorph-backends/cuda-backend/metamorph_cuda.cuh
	@if [ -f $(BASE_INSTALL_DIR)/include/metamorph_cuda.cuh ]; then rm $(BASE_INSTALL_DIR)/include/metamorph_cuda.cuh; fi
	cp metamorph-backends/cuda-backend/metamorph_cuda.cuh $(BASE_INSTALL_DIR)/include/

.PHONY: install-openmp-headers
install-openmp-headers: $(BASE_INSTALL_DIR)/include/metamorph_openmp.h

$(BASE_INSTALL_DIR)/include/metamorph_openmp.h: $(BASE_INSTALL_DIR)/include metamorph-backends/openmp-backend/metamorph_openmp.h
	@if [ -f $(BASE_INSTALL_DIR)/include/metamorph_openmp.h ]; then rm $(BASE_INSTALL_DIR)/include/metamorph_openmp.h; fi
	cp metamorph-backends/openmp-backend/metamorph_openmp.h $(BASE_INSTALL_DIR)/include/

.PHONY: install-plugin-headers
install-plugin-headers: install-mpi-headers install-profiling-headers

.PHONY: install-mpi-headers
install-mpi-headers: $(BASE_INSTALL_DIR)/include/metamorph_mpi.h

$(BASE_INSTALL_DIR)/include/metamorph_mpi.h: $(BASE_INSTALL_DIR)/include include/metamorph_mpi.h
	@if [ -f $(BASE_INSTALL_DIR)/include/metamorph_mpi.h ]; then rm $(BASE_INSTALL_DIR)/include/metamorph_mpi.h; fi
	cp include/metamorph_mpi.h $(BASE_INSTALL_DIR)/include/

.PHONY: install-profiling-headers
install-profiling-headers: $(BASE_INSTALL_DIR)/include/metamorph_profiling.h

$(BASE_INSTALL_DIR)/include/metamorph_profiling.h: $(BASE_INSTALL_DIR)/include include/metamorph_profiling.h
	@if [ -f $(BASE_INSTALL_DIR)/include/metamorph_profiling.h ]; then rm $(BASE_INSTALL_DIR)/include/metamorph_profiling.h; fi
	cp include/metamorph_profiling.h $(BASE_INSTALL_DIR)/include/

install-templates:

install-examples: examples

$(BASE_INSTALL_DIR)/bin:
	if [ ! -d $(BASE_INSTALL_DIR)/bin ]; then mkdir -p $(BASE_INSTALL_DIR)/bin; fi

.PHONY: install-metaCL
install-metaCL: $(MM_GEN_CL)/metaCL $(BASE_INSTALL_DIR)/bin
	if [ -f $(BASE_INSTALL_DIR)/bin/metaCL ]; then rm $(BASE_INSTALL_DIR)/bin/metaCL; fi
	cp $(MM_GEN_CL)/metaCL $(BASE_INSTALL_DIR)/bin/metaCL

.PHONY: clean
clean:
	if [ -f $(MM_LIB)/libmetamorph.so ]; then rm $(MM_LIB)/libmetamorph.so; fi
	if [ -f $(MM_LIB)/libmetamorph_opencl.so ]; then rm $(MM_LIB)/libmetamorph_opencl.so; fi
	if [ -f $(MM_LIB)/libmetamorph_openmp.so ]; then rm $(MM_LIB)/libmetamorph_openmp.so; fi
	if [ -f $(MM_LIB)/libmetamorph_cuda.so ]; then rm $(MM_LIB)/libmetamorph_cuda.so; fi
	if [ -f $(MM_LIB)/libmetamorph_profiling.so ]; then rm $(MM_LIB)/libmetamorph_profiling.so; fi
	if [ -f $(MM_LIB)/libmetamorph_mpi.so ]; then rm $(MM_LIB)/libmetamorph_mpi.so; fi
	if [ -f $(MM_CU)/metamorph_cuda.o ]; then rm $(MM_CU)/metamorph_cuda.o; fi
	if [ -f $(MM_GEN_CL)/metaCL ]; then rm $(MM_GEN_CL)/metaCL; fi

.PHONY: uninstall
uninstall:
	#Core library and link
	if [ -f $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR)/libmetamorph.so ]; then rm $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR)/libmetamorph.so; fi
	if [ -L $(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/libmetamorph.so ]; then rm $(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/libmetamorph.so; fi
	#Backend libraries and links
	if [ -f $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR)/libmetamorph_opencl.so ]; then rm $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR)/libmetamorph_opencl.so; fi
	if [ -L $(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/libmetamorph_opencl.so ]; then rm $(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/libmetamorph_opencl.so; fi
	if [ -f $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR)/libmetamorph_cuda.so ]; then rm $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR)/libmetamorph_cuda.so; fi
	if [ -L $(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/libmetamorph_cuda.so ]; then rm $(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/libmetamorph_cuda.so; fi
	if [ -f $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR)/libmetamorph_openmp.so ]; then rm $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR)/libmetamorph_openmp.so; fi
	if [ -L $(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/libmetamorph_openmp.so ]; then rm $(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/libmetamorph_openmp.so; fi
	#Plugin libraries and links
	if [ -f $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR)/libmetamorph_mpi.so ]; then rm $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR)/libmetamorph_mpi.so; fi
	if [ -L $(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/libmetamorph_mpi.so ]; then rm $(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/libmetamorph_mpi.so; fi
	if [ -f $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR)/libmetamorph_profiling.so ]; then rm $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR)/libmetamorph_profiling.so; fi
	if [ -L $(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/libmetamorph_profiling.so ]; then rm $(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/libmetamorph_profiling.so; fi
	#Library install directory
	if [ -L $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR) ]; then rm $(BASE_INSTALL_DIR)/$(INSTALL_LIB_RDIR); fi
	if [ -d $(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/$(VERSIONED_LIB_RDIR) ]; then rmdir $(BASE_INSTALL_DIR)/$(LINK_LIB_RDIR)/$(VERSIONED_LIB_RDIR); fi
	#Core headers
	if [ -f $(BASE_INSTALL_DIR)/include/metamorph.h ]; then rm $(BASE_INSTALL_DIR)/include/metamorph.h; fi
	if [ -f $(BASE_INSTALL_DIR)/include/metamorph_emulatable.h ]; then rm $(BASE_INSTALL_DIR)/include/metamorph_emulatable.h; fi
	if [ -f $(BASE_INSTALL_DIR)/include/metamorph_dynamic_symbols.h ]; then rm $(BASE_INSTALL_DIR)/include/metamorph_dynamic_symbols.h; fi
	if [ -f $(BASE_INSTALL_DIR)/include/metamorph_fortran_compat.h ]; then rm $(BASE_INSTALL_DIR)/include/metamorph_fortran_compat.h; fi
	if [ -f $(BASE_INSTALL_DIR)/include/metamorph_fortran_header.F03 ]; then rm $(BASE_INSTALL_DIR)/include/metamorph_fortran_header.F03; fi
	#Backend headers
	if [ -f $(BASE_INSTALL_DIR)/include/metamorph_opencl.h ]; then rm $(BASE_INSTALL_DIR)/include/metamorph_opencl.h; fi
	if [ -f $(BASE_INSTALL_DIR)/include/metamorph_opencl_emulatable.h ]; then rm $(BASE_INSTALL_DIR)/include/metamorph_opencl_emulatable.h; fi
	if [ -f $(BASE_INSTALL_DIR)/include/metamorph_cuda.cuh ]; then rm $(BASE_INSTALL_DIR)/include/metamorph_cuda.cuh; fi
	if [ -f $(BASE_INSTALL_DIR)/include/metamorph_openmp.h ]; then rm $(BASE_INSTALL_DIR)/include/metamorph_openmp.h; fi
	#Plugin headers
	if [ -f $(BASE_INSTALL_DIR)/include/metamorph_mpi.h ]; then rm $(BASE_INSTALL_DIR)/include/metamorph_mpi.h; fi
	if [ -f $(BASE_INSTALL_DIR)/include/metamorph_profiling.h ]; then rm $(BASE_INSTALL_DIR)/include/metamorph_profiling.h; fi
	#MetaCL
	if [ -f $(BASE_INSTALL_DIR)/bin/metaCL ]; then rm $(BASE_INSTALL_DIR)/bin/metaCL; fi
	#Doxygen
	if [ -d $(BASE_INSTALL_DIR)/share/docs/metamorph ]; then rm -Rf $(BASE_INSTALL_DIR)/share/docs/metamorph; fi

refresh:
	rm $(MM_EX)/crc_alt $(MM_EX)/metamorph_opencl_intelfpga.aocx

doc:
	DOXY_PROJECT_NUMBER=$(shell git log -1 --format \"%h \(%cd\)\") doxygen Doxyfile

latex: doc
	cd docs/latex && make

$(BASE_INSTALL_DIR)/share/docs/metamorph:
	if [ ! -d $(BASE_INSTALL_DIR)/share/docs/metamorph ]; then mkdir -p $(BASE_INSTALL_DIR)/share/docs/metamorph; fi

.PHONY: install-docs
install-docs: doc $(BASE_INSTALL_DIR)/share/docs/metamorph
	cp -r $(MM_DIR)/docs/* $(BASE_INSTALL_DIR)/share/docs/metamorph/
