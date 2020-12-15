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
#Function to test that a necessary file exists
define NEWLINE


endef
CHECK_FILE_EXISTS = $(if $(shell test -e $(path)/$(filename)  && echo -n yes),,Cannot find $(path)/$(filename)$(NEWLINE))

#TODO In the long run these should be exposed to the user
CXX := g++
CXXFLAGS := -O2 -fno-rtti
ifeq ($(OS), ubuntu)
 ifeq ($(shell expr $(VER) \>= 20.04), 1)
  CXXFLAGS := $(CXXFLAGS) -std=c++14
 else
  CXXFLAGS := $(CXXFLAGS) -std=c++11
 endif
else ifeq ($(OS), debian)
 ifeq ($(VER), testing)
  $(warning "Debian version $(VER)")
  CXXFLAGS := $(CXXFLAGS) -std=c++14
 else ifeq ($(shell expr $(VER) \> 10), 1)
  CXXFLAGS := $(CXXFLAGS) -std=c++14
 else
  CXXFLAGS := $(CXXFLAGS) -std=c++11
 endif
else
 CXXFLAGS := $(CXXFLAGS) -std=c++11
endif
#Required header files
LLVM_REQD_INCLUDES := llvm/Support/raw_ostream.h
CLANG_REQD_INCLUDES := clang/AST/Attr.h \
	clang/ASTMatchers/ASTMatchFinder.h \
	clang/ASTMatchers/ASTMatchers.h \
	clang/Tooling/CommonOptionsParser.h \
	clang/Tooling/Tooling.h
#These can all be manually set to override which llvm/clang is used, otherwise we'll attempt to set a default
#Centos 7 has the clang headers in /usr/include/clang/AST/Attr.h etc
#so does Fedora 31
#Debian 9 has them in /usr/lib/llvm-<VERSION>/include/clang/AST/Attr.h
#so does Ubuntu 18.04
ifndef LLVM_INCLUDE_PATH
LLVM_INCLUDE_PATH :=
#Since debian and ubuntu explicitly version their directories, just grab the last listed
ifeq ($(OS), ubuntu)
LLVM_INCLUDE_PATH := $(shell ls -d /usr/lib/llvm-*/include | tail -n1)
else ifeq ($(OS), debian)
LLVM_INCLUDE_PATH := $(shell ls -d /usr/lib/llvm-*/include | tail -n1)
else ifeq ($(OS), centos)
#The only ones new enough to actually use on Centos 7 would actually be in /opt/rh/llvm-toolset-<VERSION>/root/usr/include
LLVM_INCLUDE_PATH := /usr/include/
path = $(LLVM_INCLUDE_PATH)
 ifneq ($(foreach filename, $(LLVM_REQD_INCLUDES), $(CHECK_FILE_EXISTS)),)
  TOOLSET_PATH := $(shell ls -d /opt/rh/llvm-toolset-*/root/usr/include | tail -n1)
  ifneq ($(TOOLSET_PATH),)
   LLVM_INCLUDE_PATH := $(TOOLSET_PATH)
  endif
 endif
else ifeq ($(OS), rhel)
LLVM_INCLUDE_PATH := /usr/include/
path = $(LLVM_INCLUDE_PATH)
 ifneq ($(foreach filename, $(LLVM_REQD_INCLUDES), $(CHECK_FILE_EXISTS)),)
  TOOLSET_PATH := $(shell ls -d /opt/rh/llvm-toolset-*/root/usr/include | tail -n1)
  ifneq ($(TOOLSET_PATH),)
   LLVM_INCLUDE_PATH := $(TOOLSET_PATH)
  endif
 endif
else ifeq ($(OS), fedora)
LLVM_INCLUDE_PATH := /usr/include/
else
$(error Unrecognized Operating system, must manually set LLVM_INCLUDE_PATH)
endif
endif
ifndef CLANG_INCLUDE_PATH
CLANG_INCLUDE_PATH :=
#All of the ones we've tested use the same directory for package installs, but that may not be the case for manual installs, so they'll remain separate
ifeq ($(OS), ubuntu)
CLANG_INCLUDE_PATH := $(LLVM_INCLUDE_PATH)
else ifeq ($(OS), debian)
CLANG_INCLUDE_PATH := $(LLVM_INCLUDE_PATH)
else ifeq ($(OS), centos)
CLANG_INCLUDE_PATH := $(LLVM_INCLUDE_PATH)
else ifeq ($(OS), rhel)
CLANG_INCLUDE_PATH := $(LLVM_INCLUDE_PATH)
else ifeq ($(OS), fedora)
CLANG_INCLUDE_PATH := $(LLVM_INCLUDE_PATH)
else
$(error Unrecognized Operating system, must manually set CLANG_INCLUDE_PATH)
endif
endif
#Validate all the headers exist
path = $(LLVM_INCLUDE_PATH)
MISSING_HEADERS := $(foreach filename, $(LLVM_REQD_INCLUDES), $(CHECK_FILE_EXISTS))
path = $(CLANG_INCLUDE_PATH)
MISSING_HEADERS := $(MISSING_HEADERS) $(foreach filename, $(CLANG_REQD_INCLUDES), $(CHECK_FILE_EXISTS))
ifneq ($(strip $(MISSING_HEADERS)),)
 $(error $(MISSING_HEADERS))
endif
#Construct the include line
CLANG_INCLUDES := -I$(CLANG_INCLUDE_PATH)
ifneq ($(LLVM_INCLUDE_PATH), $(CLANG_INCLUDE_PATH))
CLANG_INCLUDES := $(CLANG_INCLUDES) -I$(LLVM_INCLUDE_PATH)
endif

#Required static libraries
LLVM_REQD_STATIC_LIBS := LLVMOption \
	LLVMSupport \
	LLVMBitReader \
	LLVMMC \
	LLVMMCParser \
	LLVMCore \
	LLVMBinaryFormat \
	LLVMProfileData
ifeq ($(OS), ubuntu)
 ifeq ($(shell expr $(VER) \>= 20.04), 1)
#Technically these are LLVM-10 requirements, might need to eventually add LLVM/Clang version detection that is separate from OS
  LLVM_REQD_STATIC_LIBS := $(LLVM_REQD_STATIC_LIBS) \
	LLVMRemarks \
	LLVMFrontendOpenMP \
	LLVMBitstreamReader \
	LLVMDemangle
 endif
else ifeq ($(OS), debian)
 ifeq ($(VER), testing)
  LLVM_REQD_STATIC_LIBS := $(LLVM_REQD_STATIC_LIBS) \
	LLVMRemarks \
	LLVMFrontendOpenMP \
	LLVMBitstreamReader \
	LLVMDemangle
 else ifeq ($(shell expr $(VER) \> 10), 1)
  LLVM_REQD_STATIC_LIBS := $(LLVM_REQD_STATIC_LIBS) \
	LLVMRemarks \
	LLVMFrontendOpenMP \
	LLVMBitstreamReader \
	LLVMDemangle
 endif
endif
CLANG_REQD_STATIC_LIBS := clangAST \
	clangASTMatchers \
	clangTooling \
	clangBasic \
	clangDriver \
	clangFrontend \
	clangFrontendTool \
	clangLex \
	clangSema \
	clangSerialization \
	clangParse \
	clangEdit \
	clangAnalysis
#Required shared libraries
LLVM_REQD_SHARED_LIBS := LLVM
CLANG_REQD_SHARED_LIBS := clang \
	clangAST \
	clangASTMatchers \
	clangTooling \
	clangBasic \
	clangFrontend
ifeq ($(OS), fedora) #On 31 at least
 CLANG_REQD_SHARED_LIBS := $(CLANG_REQD_SHARED_LIBS) \
	clangSerialization
endif
#Choose static or shared libs
#Debian apt packages only support static linking, their libclang.so is missing a ton of symbols
ifdef METACL_LINK_STATIC
METACL_LINK_STATIC := $(shell echo $(METACL_LINK_STATIC) | tr '[:upper:]' '[:lower:]')
else
 ifeq ($(OS), debian)
  METACL_LINK_STATIC := true
 else ifeq ($(OS), ubuntu)
  METACL_LINK_STATIC := true
 endif
endif

ifeq ($(METACL_LINK_STATIC), true)
 LLVM_REQD_LIBS := $(LLVM_REQD_STATIC_LIBS)
 CLANG_REQD_LIBS := $(CLANG_REQD_STATIC_LIBS)
 LIB_EXT := .a
else
 LLVM_REQD_LIBS := $(LLVM_REQD_SHARED_LIBS)
 CLANG_REQD_LIBS := $(CLANG_REQD_SHARED_LIBS)
 LIB_EXT := .so
endif
#Centos 7 has the clang libraries in /usr/lib64/clang-private/libclangASTMatchers.so, or with llvm-static /usr/lib64/llvm/libclangASTMatchers.a
# I think we need the static ones though
#Fedora 31 has them in /usr/lib64/libclangASTMatchers.a
#Debian 9 has them in /usr/lib/llvm-<VERISON>/lib/libclangASTMatchers.a
#so does Ubuntu 18.04
ifndef LLVM_LIBRARY_PATH
LLVM_LIBRARY_PATH :=
ifeq ($(OS), ubuntu)
LLVM_LIBRARY_PATH := $(shell ls -d /usr/lib/llvm-*/lib | tail -n1)
else ifeq ($(OS), debian)
LLVM_LIBRARY_PATH := $(shell ls -d /usr/lib/llvm-*/lib | tail -n1)
else ifeq ($(OS), centos)
LLVM_LIBRARY_PATH := /usr/lib$(subst _,,$(subst x86,,$(ARCH)))/llvm
path = $(LLVM_LIBRARY_PATH)
 ifneq ($(foreach filename, $(foreach shortname, $(LLVM_REQD_LIBS), lib$(shortname)$(LIB_EXT)), $(CHECK_FILE_EXISTS)),)
  TOOLSET_PATH := $(shell ls -d /opt/rh/llvm-toolset-*/root/usr/lib$(subst _,,$(subst x86,,$(ARCH))) | tail -n1)
  ifneq ($(TOOLSET_PATH),)
   LLVM_LIBRARY_PATH := $(TOOLSET_PATH)
  endif
 endif
else ifeq ($(OS), rhel)
LLVM_LIBRARY_PATH := /usr/lib$(subst _,,$(subst x86,,$(ARCH)))/llvm
path = $(LLVM_LIBRARY_PATH)
 ifneq ($(foreach filename, $(foreach shortname, $(LLVM_REQD_LIBS), lib$(shortname)$(LIB_EXT)), $(CHECK_FILE_EXISTS)),)
  TOOLSET_PATH := $(shell ls -d /opt/rh/llvm-toolset-*/root/usr/lib$(subst _,,$(subst x86,,$(ARCH))) | tail -n1)
  ifneq ($(TOOLSET_PATH),)
   LLVM_LIBRARY_PATH := $(TOOLSET_PATH)
  endif
 endif
else ifeq ($(OS), fedora)
LLVM_LIBRARY_PATH := /usr/lib$(subst _,,$(subst x86,,$(ARCH)))
else
$(error Unrecognized Operating system, must manually set LLVM_LIBRARY_PATH)
endif
endif
ifndef CLANG_LIBRARY_PATH
CLANG_LIBRARY_PATH :=
#All of the ones we've tested use the same directory for package installs, but that may not be the case for manual installs, so they'll remain separate
ifeq ($(OS), ubuntu)
CLANG_LIBRARY_PATH := $(LLVM_LIBRARY_PATH)
else ifeq ($(OS), debian)
CLANG_LIBRARY_PATH := $(LLVM_LIBRARY_PATH)
else ifeq ($(OS), centos)
CLANG_LIBRARY_PATH := $(LLVM_LIBRARY_PATH)
else ifeq ($(OS), rhel)
CLANG_LIBRARY_PATH := $(LLVM_LIBRARY_PATH)
else ifeq ($(OS), fedora)
CLANG_LIBRARY_PATH := $(LLVM_LIBRARY_PATH)
else
$(error Unrecognized Operating system, must manually set CLANG_LIBRARY_PATH)
endif
endif
#Validate all the libraries exist
path = $(LLVM_LIBRARY_PATH)
MISSING_LIBS := $(foreach filename, $(foreach shortname, $(LLVM_REQD_LIBS), lib$(shortname)$(LIB_EXT)), $(CHECK_FILE_EXISTS))
path = $(CLANG_LIBRARY_PATH)
MISSING_LIBS := $(MISSING_LIBS) $(foreach filename, $(foreach shortname, $(CLANG_REQD_LIBS), lib$(shortname)$(LIB_EXT)), $(CHECK_FILE_EXISTS))
ifneq ($(strip $(MISSING_LIBS)),)
 $(error $(MISSING_LIBS))
endif
#Construct the link line
#Add directories
CLANG_LIBS := -L$(CLANG_LIBRARY_PATH) -Wl,-rpath=$(CLANG_LIBRARY_PATH),-rpath-link=$(CLANG_LIBRARY_PATH)
ifneq ($(LLVM_LIBRARY_PATH), $(CLANG_LIBRARY_PATH))
CLANG_LIBS := $(CLANG_LIBS) -L$(LLVM_LIBRARY_PATH) -Wl,-rpath=$(LLVM_LIBRARY_PATH),-rpath-link=$(LLVM_LIBRARY_PATH)
endif
#Add libraries with a link group to address circular dependencies
CLANG_LIBS := $(CLANG_LIBS) \
	-Wl,--start-group \
        $(foreach lib, $(LLVM_REQD_LIBS), -l$(lib)) \
        $(foreach lib, $(CLANG_REQD_LIBS), -l$(lib)) \
	-Wl,--end-group \
	-lpthread
##Append any additional dependencies
ifeq ($(OS), ubuntu)
CLANG_LIBS := $(CLANG_LIBS) -lz -lncurses
else ifeq ($(OS), debian)
CLANG_LIBS := $(CLANG_LIBS) -lz -lncurses
endif

all: metaCL


TEXT_OBJS = metamorph_emulatable.o metamorph_opencl_emulatable.o metamorph_shim.o shim_dynamic.o
metamorph_emulatable.o: ../../include/metamorph_emulatable.h
	cp $< ./
	ld -r -b binary -o $@ $(subst ../../include/,,$<)
	rm ./$(subst ../../include/,,$<)

metamorph_opencl_emulatable.o: ../../metamorph-backends/opencl-backend/metamorph_opencl_emulatable.h
	cp $< ./
	ld -r -b binary -o $@ $(subst ../../metamorph-backends/opencl-backend/,,$<)
	rm ./$(subst ../../metamorph-backends/opencl-backend/,,$<)

metamorph_shim.o: metamorph_shim.c
	ld -r -b binary -o $@ $<

shim_dynamic.o: shim_dynamic.h
	ld -r -b binary -o $@ $<

metaCL: metacl.o $(TEXT_OBJS) | $(CLANG_LIB_PATH)
	$(CXX) -o metaCL metacl.o $(TEXT_OBJS) $(CLANG_LIBS) $(CLANG_LDFLAGS)

metacl.o: metacl.cpp
	$(CXX) $(CXXFLAGS) $(CLANG_INCLUDES) metacl.cpp -c -o metacl.o

