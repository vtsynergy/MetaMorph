#Most build options are just an explicit TRUE/FALSE
#Options that default on: USE_OPENMP, USE_TIMERS
#Options that attempt to autodetect: USE_CUDA, USE_OPENCL
#Options that must be explicitly enabled: USE_MPI, USE_FPGA, USE_MIC 

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
ifneq (, $(shell which lsb_release))
OS = $(shell lsb_release -si)
OS := $(shell echo $(OS) | tr '[:upper:]' '[:lower:]')
ARCH = $(shell uname -m)#| sed 's/x86_//;s/i[3-6]86/32/')
VER = $(shell lsb_release -sr)
endif
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

#Timers default to on
ifndef USE_TIMERS
USE_TIMERS=TRUE
else
USE_TIMERS := $(shell echo $(USE_TIMERS) | tr '[:lower:]' '[:upper:]')
endif

#CHECK for an MPI environment
ifndef MPI_DIR
#attempt to autodetect
ifneq ($(shell which mpicc),)
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
ifeq ($(shell which nvcc),)
#none found
CUDA_LIB_DIR=
else
#found
CUDA_LIB_DIR=$(patsubst %/bin/nvcc,%,$(shell which nvcc))/$(if ARCH_64,lib64,lib)
NVCC=nvcc -ccbin gcc-4.9
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
BUILD_LIBS += $(MM_LIB)/libmm_mpi.so
MPICC := $(MPI_DIR)/bin/mpicc -cc=$(CC)
endif

#timer features
ifeq ($(USE_TIMERS),TRUE)
BUILD_LIBS += $(MM_LIB)/libmm_profiling.so
endif

#CUDA backend
ifeq ($(USE_CUDA),TRUE)
BUILD_LIBS += $(MM_LIB)/libmm_cuda_backend.so
endif

#OpenCL backend
ifeq ($(USE_OPENCL),TRUE)
 ifeq ($(USE_FPGA),INTEL)
  OPENCL_FLAGS += -D WITH_INTELFPGA
  BUILD_LIBS += $(MM_LIB)/libmm_opencl_intelfpga_backend.so
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
  BUILD_LIBS += $(MM_LIB)/libmm_opencl_xilinx_backend.so
  $(error XILINX not yet supported)
 else ifeq ($(USE_FPGA),) #The non-FPGA implementation has an empty string
  BUILD_LIBS += $(MM_LIB)/libmm_opencl_backend.so
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
  BUILD_LIBS += $(MM_LIB)/libmm_openmp_mic_backend.so
 else
  BUILD_LIBS += $(MM_LIB)/libmm_openmp_backend.so
 endif

 ifeq ($(CC),gcc)
  OPENMP_FLAGS += -fopenmp
 else
  OPENMP_FLAGS += -openmp
 endif
endif

export INCLUDES

MFLAGS := USE_CUDA=$(USE_CUDA) CUDA_LIB_DIR=$(CUDA_LIB_DIR) USE_OPENCL=$(USE_OPENCL) OPENCL_LIB_DIR=$(OPENCL_LIB_DIR) OPENCL_INCL_DIR=$(OPENCL_INCL_DIR) OPENCL_SINGLE_KERNEL_PROGS=$(OPENCL_SINGLE_KERNEL_PROGS) USE_OPENMP=$(USE_OPENMP) USE_MIC=$(USE_MIC) ICC_BIN=$(ICC_BIN) USE_TIMERS=$(USE_TIMERS) USE_MPI=$(USE_MPI) MPI_DIR=$(MPI_DIR) USE_FPGA=$(USE_FPGA) CC="$(CC)" NVCC="$(NVCC)" MPICC="$(MPICC)" OPENMP_FLAGS="$(OPENMP_FLAGS)" OPENCL_FLAGS="$(OPENCL_FLAGS)" $(MFLAGS)

.PHONY: all
all: $(BUILD_LIBS)

$(MM_LIB):
	if [ ! -d $(MM_LIB) ]; then mkdir -p $(MM_LIB); fi

$(MM_LIB)/libmetamorph.so: $(MM_LIB) $(MM_DEPS)
	$(CC) $(MM_DEPS) $(CC_FLAGS) $(INCLUDES) -L$(MM_LIB) $(MM_COMPONENTS) -o $(MM_LIB)/libmetamorph.so -ldl -shared -Wl,-soname,libmetamorph.so

$(MM_LIB)/libmm_profiling.so: $(MM_LIB) $(MM_CORE)/metamorph_profiling.c
	$(CC) $(MM_CORE)/metamorph_profiling.c $(CC_FLAGS) $(INCLUDES) -o $(MM_LIB)/libmm_profiling.so -shared -Wl,-soname,libmm_profiling.so

$(MM_LIB)/libmm_mpi.so: $(MM_LIB) $(MM_CORE)/metamorph_mpi.c
	$(MPICC) $(MM_CORE)/metamorph_mpi.c $(CC_FLAGS) $(INCLUDES) -I$(MPI_DIR)/include -L$(MPI_DIR)/lib -o $(MM_LIB)/libmm_mpi.so -shared -Wl,-soname,libmm_mpi.so

$(MM_LIB)/libmm_openmp_backend.so: $(MM_LIB)	
	cd $(MM_MP) && $(MAKE) $(MFLAGS) libmm_openmp_backend.so

#TODO Make this happen transparently to this file, create a symlink in the backend's makefile	
$(MM_LIB)/libmm_openmp_mic_backend.so: $(MM_LIB)
	cd $(MM_MP) && $(MAKE) $(MFLAGS) libmm_openmp_backend_mic.so

$(MM_LIB)/libmm_cuda_backend.so: $(MM_LIB)
	cd $(MM_CU) && $(MAKE) $(MFLAGS) libmm_cuda_backend.so

$(MM_LIB)/libmm_opencl_backend.so: $(MM_LIB)
	cd $(MM_CL) && $(MAKE) $(MFLAGS) libmm_opencl_backend.so

#TODO Make this happen transparently to this file, create a symlink in the backend's makefile	
$(MM_LIB)/libmm_opencl_intelfpga_backend.so: $(MM_LIB)
	cd $(MM_CL) && $(MAKE) $(MFLAGS) libmm_opencl_intelfpga_backend.so

.PHONY: generators
generators: $(MM_GEN_CL)/metaCL


$(MM_GEN_CL)/metaCL:
	cd $(MM_GEN_CL) && $(MAKE) metaCL

.PHONY: examples
examples: torus_ex

torus_ex:
	cd $(MM_EX) && $(MAKE) $(MFLAGS) torus_reduce_test

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
LINK_LIB_DIR=$(BASE_INSTALL_DIR)/lib
#Actual copies of the library may be somewhere else in the lib tree, in case multiple versions/implementations are co-installed
INSTALL_LIB_DIR=$(LINK_LIB_DIR)/metamorph-0.3-rc1

#Should install and link on the same libraries as all, that's what the foreach is for
.PHONY: install
install: all $(foreach target,$(BUILD_LIBS),$(subst $(MM_LIB),$(LINK_LIB_DIR),$(target)))

.PHONY: install-core-library
install-core-library: $(LINK_LIB_DIR)/libmetamorph.so

#Symbolic link to the actual library folder, in case we support multiple versioning down the road
$(LINK_LIB_DIR)/libmetamorph.so: $(INSTALL_LIB_DIR)/libmetamorph.so
	#Remove any existing symlink
	@if [ -L $(LINK_LIB_DIR)/libmetamorph.so ]; then rm $(LINK_LIB_DIR)/libmetamorph.so; fi
	#Create a symlink in the main library directory
	ln -s $(INSTALL_LIB_DIR)/libmetamorph.so $(LINK_LIB_DIR)/libmetamorph.so

#Copy to the actual library folder
$(INSTALL_LIB_DIR)/libmetamorph.so: $(INSTALL_LIB_DIR) $(MM_LIB)/libmetamorph.so
	#Remove any existing copy
	@if [ -f $(INSTALL_LIB_DIR)/libmetamorph.so ]; then rm $(INSTALL_LIB_DIR)/libmetamorph.so; fi
	#Copy the current version
	cp $(MM_LIB)/libmetamorph.so $(INSTALL_LIB_DIR)/libmetamorph.so

#Create the actual library folder
$(INSTALL_LIB_DIR):
	#Ensure the directory exists
	@if [ ! -d $(INSTALL_LIB_DIR) ]; then mkdir -p $(INSTALL_LIB_DIR); fi

.PHONY: install-opencl-library
install-opencl-library: install-core-library $(LINK_LIB_DIR)/libmm_opencl_backend.so

$(LINK_LIB_DIR)/libmm_opencl_backend.so: $(INSTALL_LIB_DIR)/libmm_opencl_backend.so
	@if [ -L $(LINK_LIB_DIR)/libmm_opencl_backend.so ]; then rm $(LINK_LIB_DIR)/libmm_opencl_backend.so; fi
	ln -s $(INSTALL_LIB_DIR)/libmm_opencl_backend.so $(LINK_LIB_DIR)/libmm_opencl_backend.so

$(INSTALL_LIB_DIR)/libmm_opencl_backend.so: $(INSTALL_LIB_DIR) $(MM_LIB)/libmm_opencl_backend.so
	@if [ -f $(INSTALL_LIB_DIR)/libmm_opencl_backend.so ]; then rm $(INSTALL_LIB_DIR)/libmm_opencl_backend.so; fi
	cp $(MM_LIB)/libmm_opencl_backend.so $(INSTALL_LIB_DIR)/libmm_opencl_backend.so

.PHONY: install-cuda-library
install-cuda-library: install-core-library $(LINK_LIB_DIR)/libmm_cuda_backend.so

$(LINK_LIB_DIR)/libmm_cuda_backend.so: $(INSTALL_LIB_DIR)/libmm_cuda_backend.so
	@if [ -L $(LINK_LIB_DIR)/libmm_cuda_backend.so ]; then rm $(LINK_LIB_DIR)/libmm_cuda_backend.so; fi
	ln -s $(INSTALL_LIB_DIR)/libmm_cuda_backend.so $(LINK_LIB_DIR)/libmm_cuda_backend.so

$(INSTALL_LIB_DIR)/libmm_cuda_backend.so: $(INSTALL_LIB_DIR) $(MM_LIB)/libmm_cuda_backend.so
	@if [ -f $(INSTALL_LIB_DIR)/libmm_cuda_backend.so ]; then rm $(INSTALL_LIB_DIR)/libmm_cuda_backend.so; fi
	cp $(MM_LIB)/libmm_cuda_backend.so $(INSTALL_LIB_DIR)/libmm_cuda_backend.so

.PHONY: install-openmp-library
install-openmp-library: install-core-library $(LINK_LIB_DIR)/libmm_openmp_backend.so

$(LINK_LIB_DIR)/libmm_openmp_backend.so: $(INSTALL_LIB_DIR)/libmm_openmp_backend.so
	@if [ -L $(LINK_LIB_DIR)/libmm_openmp_backend.so ]; then rm $(LINK_LIB_DIR)/libmm_openmp_backend.so; fi
	ln -s $(INSTALL_LIB_DIR)/libmm_openmp_backend.so $(LINK_LIB_DIR)/libmm_openmp_backend.so

$(INSTALL_LIB_DIR)/libmm_openmp_backend.so: $(INSTALL_LIB_DIR) $(MM_LIB)/libmm_openmp_backend.so
	@if [ -f $(INSTALL_LIB_DIR)/libmm_openmp_backend.so ]; then rm $(INSTALL_LIB_DIR)/libmm_openmp_backend.so; fi
	cp $(MM_LIB)/libmm_openmp_backend.so $(INSTALL_LIB_DIR)/libmm_openmp_backend.so

.PHONY: install-mpi-library
install-mpi-library: install-core-library $(LINK_LIB_DIR)/libmm_mpi.so

$(LINK_LIB_DIR)/libmm_mpi.so: $(INSTALL_LIB_DIR)/libmm_mpi.so
	@if [ -L $(LINK_LIB_DIR)/libmm_mpi.so ]; then rm $(LINK_LIB_DIR)/libmm_mpi.so; fi
	ln -s $(INSTALL_LIB_DIR)/libmm_mpi.so $(LINK_LIB_DIR)/libmm_mpi.so

$(INSTALL_LIB_DIR)/libmm_mpi.so: $(INSTALL_LIB_DIR) $(MM_LIB)/libmm_mpi.so
	@if [ -f $(INSTALL_LIB_DIR)/libmm_mpi.so ]; then rm $(INSTALL_LIB_DIR)/libmm_mpi.so; fi
	cp $(MM_LIB)/libmm_mpi.so $(INSTALL_LIB_DIR)/libmm_mpi.so

.PHONY: install-profiling-library
install-profiling-library: install-core-library $(LINK_LIB_DIR)/libmm_profiling.so

$(LINK_LIB_DIR)/libmm_profiling.so: $(INSTALL_LIB_DIR)/libmm_profiling.so
	@if [ -L $(LINK_LIB_DIR)/libmm_profiling.so ]; then rm $(LINK_LIB_DIR)/libmm_profiling.so; fi
	ln -s $(INSTALL_LIB_DIR)/libmm_profiling.so $(LINK_LIB_DIR)/libmm_profiling.so

$(INSTALL_LIB_DIR)/libmm_profiling.so: $(INSTALL_LIB_DIR) $(MM_LIB)/libmm_profiling.so
	@if [ -f $(INSTALL_LIB_DIR)/libmm_profiling.so ]; then rm $(INSTALL_LIB_DIR)/libmm_profiling.so; fi
	cp $(MM_LIB)/libmm_profiling.so $(INSTALL_LIB_DIR)/libmm_profiling.so

install-core-headers:
	@if [ -d $(CASE_THAT_CONTROLS_HEADERS) ]; then cp -r $(MM_DIR)/include $(DESTDIR)/usr/include/metamorph;

install-opencl-headers:

install-cude-headers:

install-openmp-headers:

install-opencl-kernels:

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
	if [ -f $(MM_LIB)/libmm_opencl_backend.so ]; then rm $(MM_LIB)/libmm_opencl_backend.so; fi
	if [ -f $(MM_LIB)/libmm_openmp_backend.so ]; then rm $(MM_LIB)/libmm_openmp_backend.so; fi
	if [ -f $(MM_LIB)/libmm_cuda_backend.so ]; then rm $(MM_LIB)/libmm_cuda_backend.so; fi
	if [ -f $(MM_LIB)/libmm_profiling.so ]; then rm $(MM_LIB)/libmm_profiling.so; fi
	if [ -f $(MM_LIB)/libmm_mpi.so ]; then rm $(MM_LIB)/libmm_mpi.so; fi
	if [ -f $(MM_CU)/mm_cuda_backend.o ]; then rm $(MM_CU)/mm_cuda_backend.o; fi
	if [ -f $(MM_GEN_CL)/metaCL ]; then rm $(MM_GEN_CL)/metaCL; fi

.PHONY: uninstall
uninstall:
	#Core library and link
	if [ -f $(INSTALL_LIB_DIR)/libmetamorph.so ]; then rm $(INSTALL_LIB_DIR)/libmetamorph.so; fi
	if [ -L $(LINK_LIB_DIR)/libmetamorph.so ]; then rm $(LINK_LIB_DIR)/libmetamorph.so; fi
	#Backend libraries and links
	if [ -f $(INSTALL_LIB_DIR)/libmm_opencl_backend.so ]; then rm $(INSTALL_LIB_DIR)/libmm_opencl_backend.so; fi
	if [ -L $(LINK_LIB_DIR)/libmm_opencl_backend.so ]; then rm $(LINK_LIB_DIR)/libmm_opencl_backend.so; fi
	if [ -f $(INSTALL_LIB_DIR)/libmm_cuda_backend.so ]; then rm $(INSTALL_LIB_DIR)/libmm_cuda_backend.so; fi
	if [ -L $(LINK_LIB_DIR)/libmm_cuda_backend.so ]; then rm $(LINK_LIB_DIR)/libmm_cuda_backend.so; fi
	if [ -f $(INSTALL_LIB_DIR)/libmm_openmp_backend.so ]; then rm $(INSTALL_LIB_DIR)/libmm_openmp_backend.so; fi
	if [ -L $(LINK_LIB_DIR)/libmm_openmp_backend.so ]; then rm $(LINK_LIB_DIR)/libmm_openmp_backend.so; fi
	#Plugin libraries and links
	if [ -f $(INSTALL_LIB_DIR)/libmm_mpi.so ]; then rm $(INSTALL_LIB_DIR)/libmm_mpi.so; fi
	if [ -L $(LINK_LIB_DIR)/libmm_mpi.so ]; then rm $(LINK_LIB_DIR)/libmm_mpi.so; fi
	if [ -f $(INSTALL_LIB_DIR)/libmm_profiling.so ]; then rm $(INSTALL_LIB_DIR)/libmm_profiling.so; fi
	if [ -L $(LINK_LIB_DIR)/libmm_profiling.so ]; then rm $(LINK_LIB_DIR)/libmm_profiling.so; fi
	#Library install directory
	if [ -d $(INSTALL_LIB_DIR) ]; then rmdir $(INSTALL_LIB_DIR); fi
	#Core headers
	#Backend headers
	#Plugin headers
	#MetaCL
	if [ -f $(BASE_INSTALL_DIR)/bin/metaCL ]; then rm $(BASE_INSTALL_DIR)/bin/metaCL; fi
	#Doxygen
	if [ -d $(BASE_INSTALL_DIR)/share/docs/metamorph ]; then rm -Rf $(BASE_INSTALL_DIR)/share/docs/metamorph; fi

refresh:
	rm $(MM_EX)/crc_alt $(MM_EX)/mm_opencl_intelfpga_backend.aocx

doc:
	DOXY_PROJECT_NUMBER=$(shell git log -1 --format \"%h \(%cd\)\") doxygen Doxyfile

latex: doc
	cd docs/latex && make

$(BASE_INSTALL_DIR)/share/docs/metamorph:
	if [ ! -d $(BASE_INSTALL_DIR)/share/docs/metamorph ]; then mkdir -p $(BASE_INSTALL_DIR)/share/docs/metamorph; fi

.PHONY: install-docs
install-docs: doc $(BASE_INSTALL_DIR)/share/docs/metamorph
	cp -r $(MM_DIR)/docs/* $(BASE_INSTALL_DIR)/share/docs/metamorph/
