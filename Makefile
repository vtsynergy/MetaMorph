#Most build options are just an explicit TRUE/FALSE
#Options that default on: USE_OPENMP
#Options that attempt to autodetect: USE_CUDA, USE_OPENCL
#Options that must be explicitly enabled: USE_TIMERS, USE_MPI, USE_FPGA, USE_MIC 

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
#check if 64-bit libs should be used
ifeq ($(shell arch),x86_64)
ARCH_64 = x86_64
else
ARCH_64 =
endif

#Timers default to off
ifndef USE_TIMERS
USE_TIMERS=FALSE
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
ifdef USE_MPI
#make uppercase
USE_MPI := $(shell echo $(USE_MPI) | tr '[:lower:]' '[:upper:]')
ifeq ($(USE_MPI),TRUE)
ifeq ($(MPI_DIR),)
$(error USE_MPI is set but no MPI environment found)
endif
endif
else
USE_MPI=FALSE
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




export G_TYPE = DOUBLE
#export OPT_LVL = -g -DDEBUG
export OPT_LVL = -O3
export L_FLAGS= -fPIC -shared

CC=gcc
#CC=icc
CC_FLAGS = $(OPT_LVL) $(L_FLAGS)
INCLUDES  = -I $(MM_DIR)/include

ifndef FPGA_USE_EMULATOR
FPGA_USE_EMULATOR=FALSE
else
FPGA_USE_EMULATOR := $(shell echo $(FPGA_USE_EMULATOR) | tr '[:lower:]' '[:upper:]')
endif

#Configure compilation to the machine's availability
MM_DEPS= $(MM_CORE)/metamorph.c
MM_COMPONENTS=
#MPI features
ifeq ($(USE_MPI),TRUE)
MM_COMPONENTS += -D WITH_MPI $(MM_CORE)/metamorph_mpi.c -I$(MPI_DIR)/include -L$(MPI_DIR)/lib
MM_DEPS += $(MM_CORE)/metamorph_mpi.c
CC := $(MPI_DIR)/bin/mpicc -cc=$(CC)
endif
#timer features
ifeq ($(USE_TIMERS),TRUE)
MM_COMPONENTS += -D WITH_TIMERS $(MM_CORE)/metamorph_timers.c
MM_DEPS += $(MM_CORE)/metamorph_timers.c
endif

#CUDA backend
ifeq ($(USE_CUDA),TRUE)
MM_COMPONENTS += -D WITH_CUDA -lmm_cuda_backend -L$(CUDA_LIB_DIR) -lcudart
MM_DEPS += libmm_cuda_backend.so
INCLUDES += -I$(MM_CU)
endif

#OpenCL backend
ifeq ($(USE_OPENCL),TRUE)
 MM_COMPONENTS += -D WITH_OPENCL
 INCLUDES += -I$(MM_CL)
 ifeq ($(USE_FPGA),INTEL)
  MM_COMPONENTS += -D WITH_INTELFPGA -lmm_opencl_intelfpga_backend
  MM_DEPS += libmm_opencl_intelfpga_backend.so
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

  #-D WITH_TIMERS
  #There is no reason this mechanic has to be specific to FPGA
  ifeq ($(OPENCL_SINGLE_KERNEL_PROGS),TRUE)
   export FPGA_DEF=-D WITH_OPENCL -D WITH_INTELFPGA -D OPENCL_SINGLE_KERNEL_PROGS
  else
   #TODO These FPGA options should not be hardcoded, and in fact should be subsumed by the OPENCL_SINGLE_KERNEL_PROGS option
   export FPGA_DEF=-D WITH_OPENCL -D WITH_INTELFPGA -D WITH_TIMERS -D KERNEL_CRC -D FPGA_UNSIGNED_INTEGER
  endif
  #export FPGA_DEF=-D WITH_OPENCL -D WITH_TIMERS -D WITH_INTELFPGA -D KERNEL_STENCIL -D FPGA_DOUBLE
  #export FPGA_LIB=/home/jehandad/rte/opt/altera/aocl-rte/host/linux64/lib/
  #TODO Remove hardcode and expose as INTELFPGA_LIB_DIR
  export FPGA_LIB=-L /media/hdd/jehandad/altera_pro/16.0/hld/host/linux64/lib -L $(AALSDK)/lib


 else ifeq ($(USE_FPGA),XILINX)
  MM_COMPONENTS += -D WITH_XILINXFPGA -lmm_opencl_xilinx_backend
  MM_DEPS += libmm_opencl_xilinx_backend.so
  $(error XILINX not yet supported)
 else ifeq ($(USE_FPGA),) #The non-FPGA implementation has an empty string
  MM_COMPONENTS += -lmm_opencl_backend
  MM_DEPS += libmm_opencl_backend.so
 else #They asked for an FPGA backend that is not explicitly supported
  $(error USE_FPGA=$(USE_FPGA) is not supported)
 endif
 MM_COMPONENTS += -I$(OPENCL_INCL_DIR) -L$(OPENCL_LIB_DIR) -lOpenCL
endif




ifeq ($(USE_OPENMP),TRUE)
MM_COMPONENTS += -D WITH_OPENMP
INCLUDES += -I$(MM_MP)

ifeq ($(USE_MIC),TRUE)

ifeq ($(USE_MPI),TRUE)
CC = $(MPI_DIR)/bin/mpicc -cc=$(ICC_BIN)
else
CC = $(ICC_BIN)
endif

CC_FLAGS += -mmic
MM_COMPONENTS += -lmm_openmp_backend_mic
MM_DEPS += libmm_openmp_mic_backend.so
else
MM_COMPONENTS += -lmm_openmp_backend
MM_DEPS += libmm_openmp_backend.so
endif

ifeq ($(CC),gcc)
CC_FLAGS += -fopenmp
else
CC_FLAGS += -openmp
endif
endif

export INCLUDES

MFLAGS := USE_CUDA=$(USE_CUDA) CUDA_LIB_DIR=$(CUDA_LIB_DIR) USE_OPENCL=$(USE_OPENCL) OPENCL_LIB_DIR=$(OPENCL_LIB_DIR) OPENCL_INCL_DIR=$(OPENCL_INCL_DIR) OPENCL_SINGLE_KERNEL_PROGS=$(OPENCL_SINGLE_KERNEL_PROGS) USE_OPENMP=$(USE_OPENMP) USE_MIC=$(USE_MIC) ICC_BIN=$(ICC_BIN) USE_TIMERS=$(USE_TIMERS) USE_MPI=$(USE_MPI) MPI_DIR=$(MPI_DIR) USE_FPGA=$(USE_FPGA) CC="$(CC)" NVCC="$(NVCC)" $(MFLAGS)

#.PHONY: metamorph_all examples
.PHONY: libmetamorph.so examples
all: libmetamorph.so
#metamorph_all: libmetamorph.so libmetamorph_mp.so libmetamorph_mic.so libmetamorph_cl.so libmetamorph_cu.so

#libmetamorph.so: libmm_openmp_backend.so libmm_cuda_backend.so libmm_opencl_backend.so
#ifeq ($(USE_MPI),TRUE)
#	mpicc $(MM_CORE)/metamorph.c $(MM_CORE)/metamorph_timers.c $(MM_CORE)/metamorph_mpi.c $(CC_FLAGS) $(INCLUDES) -L $(MM_LIB) -D WITH_OPENMP -D WITH_OPENCL -D WITH_CUDA -D WITH_TIMERS -D WITH_MPI -I $(MPICH_DIR)/include -L $(MPICH_DIR)/lib -L /usr/local/cuda/lib64 -lmm_openmp_backend  -lmm_cuda_backend -lmm_opencl_backend -lOpenCL -lcudart -o $(MM_LIB)/libmetamorph.so
#else
#	$(CC) $(MM_CORE)/metamorph.c $(MM_CORE)/metamorph_timers.c $(CC_FLAGS) $(INCLUDES) -L $(MM_LIB) -D WITH_OPENMP -D WITH_OPENCL -D WITH_CUDA -D WITH_TIMERS -L /usr/local/cuda/lib64 -lmm_openmp_backend  -lmm_cuda_backend -lmm_opencl_backend -lOpenCL -lcudart -o $(MM_LIB)/libmetamorph.so
#endif
libmetamorph.so: $(MM_DEPS)
	$(CC) $(MM_CORE)/metamorph.c $(CC_FLAGS) $(INCLUDES) -L$(MM_LIB) $(MM_COMPONENTS) -o $(MM_LIB)/libmetamorph.so -shared -Wl,-soname,libmetamorph.so

#these old single-backend targets are deprecated now that the above is modular
libmetamorph_mp.so: libmm_openmp_backend.so
ifeq ($(USE_MPI),TRUE)
	mpicc -cc=$(CC) $(MM_CORE)/metamorph.c $(MM_CORE)/metamorph_timers.c $(MM_CORE)/metamorph_mpi.c $(CC_FLAGS) $(INCLUDES) -I $(MPICH_DIR)/include -L $(MM_LIB) -L $(MPICH_DIR)/lib -D WITH_OPENMP -D WITH_TIMERS -D WITH_MPI  -lmm_openmp_backend -o $(MM_LIB)/libmetamorph_mp.so
else
	$(CC) $(MM_CORE)/metamorph.c $(MM_CORE)/metamorph_timers.c $(CC_FLAGS) $(INCLUDES) -L $(MM_LIB) -D WITH_OPENMP -D WITH_TIMERS -lmm_openmp_backend -o $(MM_LIB)/libmetamorph_mp.so
endif

libmetamorph_mic.so: libmm_openmp_mic_backend.so
ifeq ($(USE_MPI),TRUE)
	mpicc -cc=icc $(MM_CORE)/metamorph.c $(MM_CORE)/metamorph_timers.c $(MM_CORE)/metamorph_mpi.c $(CC_FLAGS) -openmp -mmic $(INCLUDES) -L $(MM_LIB) -D WITH_OPENMP -D WITH_TIMERS -D WITH_MPI -lmm_openmp_mic_backend -I $(MPICH_DIR)-mic/include -L $(MPICH_DIR)-mic/lib -o $(MM_LIB)/libmetamorph_mic.so
else
	icc $(MM_CORE)/metamorph.c $(MM_CORE)/metamorph_timers.c $(CC_FLAGS) -openmp -mmic $(INCLUDES) -L $(MM_LIB) -D WITH_OPENMP -D WITH_TIMERS -lmm_openmp_mic_backend -o $(MM_LIB)/libmetamorph_mic.so
endif

libmetamorph_cu.so: libmm_cuda_backend.so
ifeq ($(USE_MPI),TRUE)
	$(CC) $(MM_CORE)/metamorph.c $(MM_CORE)/metamorph_timers.c $(MM_CORE)/metamorph_mpi.c $(CC_FLAGS) $(INCLUDES) -L $(MM_LIB) -D WITH_CUDA -D WITH_TIMERS -D WITH_MPI -I $(MPICH_DIR)/include -L $(MPICH_DIR)/lib -L /usr/local/cuda/lib64 -lmm_cuda_backend -lcudart -o $(MM_LIB)/libmetamorph_cu.so
else
	$(CC) $(MM_CORE)/metamorph.c $(MM_CORE)/metamorph_timers.c $(CC_FLAGS) $(INCLUDES) -L $(MM_LIB) -D WITH_CUDA -D WITH_TIMERS -L /usr/local/cuda/lib64 -lmm_cuda_backend -lcudart -o $(MM_LIB)/libmetamorph_cu.so
endif
	
libmetamorph_cl.so: libmm_opencl_backend.so
ifeq ($(USE_MPI),TRUE)
	$(CC) $(MM_CORE)/metamorph.c $(MM_CORE)/metamorph_timers.c $(MM_CORE)/metamorph_mpi.c $(CC_FLAGS) $(INCLUDES) -L $(MM_LIB) -D WITH_OPENCL -D WITH_TIMERS -D WITH_MPI -I $(MPICH_DIR)/include -L $(MPICH_DIR)/lib -lmm_opencl_backend -lOpenCL -o $(MM_LIB)/libmetamorph_cl.so
else
	$(CC) $(MM_CORE)/metamorph.c $(MM_CORE)/metamorph_timers.c $(CC_FLAGS) $(AOCL_COMPILE_CONFIG) $(INCLUDES) -L $(MM_LIB) $(FPGA_LIB) -D WITH_OPENCL -D WITH_TIMERS -lmm_opencl_backend -lOpenCL -o $(MM_LIB)/libmetamorph_cl.so
endif

#/media/hdd/jehandad/altera_pro/16.0/hld/host/linux64/lib 
libmm_openmp_backend.so:	
	cd $(MM_MP) && $(MFLAGS) $(MAKE) libmm_openmp_backend.so
	
libmm_openmp_mic_backend.so:
	cd $(MM_MP) && $(MFLAGS) $(MAKE) libmm_openmp_backend_mic.so

libmm_cuda_backend.so:
	cd $(MM_CU) && $(MFLAGS) $(MAKE) libmm_cuda_backend.so

libmm_opencl_backend.so:
	cd $(MM_CL) && $(MFLAGS) $(MAKE) libmm_opencl_backend.so

libmm_opencl_intelfpga_backend.so:
	cd $(MM_CL) && $(MFLAGS) $(MAKE) libmm_opencl_intelfpga_backend.so

generators: metaCL

metaCL:
	cd $(MM_GEN_CL) && $(MAKE) metaCL
	
examples: torus_ex

torus_ex:
	cd $(MM_EX) && $(MFLAGS) $(MAKE) torus_reduce_test
#	cd $(MM_EX) && $(MAKE) torus_reduce_test_mp torus_reduce_test_mic torus_reduce_test_cu torus_reduce_test_cl $(MFLAGS)


#Dependency never added to repo
#csr_ex:
#	cd $(MM_EX) && $(MFLAGS) $(MAKE) csr_alt
#	cd $(MM_EX) && $(MAKE) torus_reduce_test_mp torus_reduce_test_mic torus_reduce_test_cu torus_reduce_test_cl $(MFLAGS)

#dependency never added to repo
#crc_ex:
#	cd $(MM_EX) && $(MFLAGS) $(MAKE) crc_alt
clean:
	rm $(MM_LIB)/libmetamorph*.so $(MM_LIB)/libmm*.so $(MM_CU)/mm_cuda_backend.o

refresh:
	rm $(MM_EX)/crc_alt $(MM_EX)/mm_opencl_intelfpga_backend.aocx

doc:
	DOXY_PROJECT_NUMBER=$(shell git log -1 --format \"%h \(%cd\)\") doxygen Doxyfile

latex: doc
	cd docs/latex && make
