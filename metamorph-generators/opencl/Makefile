OS := 
ARCH :=
VER :=
ifeq ($(shell uname),Linux)
ifneq (, $(shell which lsb_release))
#FIXME For now just fallback on trying to do a pre-compiled install
#FIXME For now we just construct the version strings unchecked
#TODO expand distributions/version
#TODO provide a last-ditch fallback of building from source
OS = $(shell lsb_release -si)
OS := $(shell echo $(OS) | tr '[:upper:]' '[:lower:]')
ARCH = $(shell uname -m | sed 's/x86_//;s/i[3-6]86/32/')
VER = $(shell lsb_release -sr)
endif
endif

#For now default assume we're on the tested Ubuntu 18.04 configuration
ifndef LLVM_INSTALL_PATH
LLVM_INSTALL_PATH := /usr/lib/llvm-6.0
endif
#Autodetect Ubuntu 18.04/Clang 6.0 apt install
#TODO Detect more default distribution locations
ifeq ($(shell test -e $(LLVM_INSTALL_PATH)/lib/libclangTooling.a  && echo -n yes),yes)
LLVM_BUILD_PATH := $(LLVM_INSTALL_PATH)
else
LLVM_BASE_PATH := clang_tarball
LLVM_VERSION := 6.0.1
LLVM_OS_STRING := $(ARCH)-linux-gnu-$(OS)-$(VER)
LLVM_TARBALL := clang+llvm-$(LLVM_VERSION)-$(LLVM_OS_STRING)
LLVM_WGET := http://releases.llvm.org/$(LLVM_VERSION)/$(LLVM_TARBALL).tar.xz
LLVM_BUILD_PATH := $(LLVM_BASE_PATH)/$(LLVM_TARBALL)
$(LLVM_BUILD_PATH): | $(LLVM_BASE_PATH)/$(LLVM_TARBALL).tar.xz
	tar -xf $(LLVM_BASE_PATH)/$(LLVM_TARBALL).tar.xz -C $(LLVM_BASE_PATH)

$(LLVM_BASE_PATH)/$(LLVM_TARBALL).tar.xz: | $(LLVM_BASE_PATH)
	echo $(LLVM_BASE_PATH)/$(LLVM_TARBALL).tar.xz && cd $(LLVM_BASE_PATH) && wget $(LLVM_WGET)

$(LLVM_BASE_PATH):
	mkdir -p $(LLVM_BASE_PATH)
endif


LLVM_BIN_PATH := $(LLVM_BUILD_PATH)/bin
LLVM_HEADER_PATH := $(LLVM_BUILD_PATH)/include

CLANG_INCLUDES := -I$(LLVM_HEADER_PATH) 

CLANG_LIBS :=

BUILD_DIR := 

CXX := g++

CXXFLAGS := -g -O0 -std=c++11 -fno-rtti

ifeq ($(OS), ubuntu)
ifeq ($(VER), 18.04)
#TODO safety check these exist
CLANG_LDFLAGS := -ltinfo -lz
endif
endif
#CLANG_LDFLAGS := `$(LLVM_BIN_PATH)/llvm-config --cxxflags`

CLANG_LIBS := -L $(LLVM_BUILD_PATH)/lib \
	-Wl,--start-group \
	-lclangAST \
	-lclangASTMatchers \
	-lclangTooling \
	-lclangBasic \
	-lclangDriver \
	-lclangFrontend \
	-lclangFrontendTool \
	-lclangLex \
	-lclangSema \
	-lclangSerialization \
	-lclangParse \
	-lclangEdit \
	-lclangAnalysis \
	-lLLVMOption \
	-lLLVMSupport \
	-lLLVMBitReader \
	-lLLVMMC \
	-lLLVMMCParser \
	-lLLVMCore \
	-lLLVMBinaryFormat \
	-lLLVMProfileData \
	-Wl,--end-group \
	-lpthread

all: metaCL

metaCL: metacl.cpp | $(LLVM_BUILD_PATH)
	$(CXX) $(CXXFLAGS) $(CLANG_INCLUDES) metacl.cpp -o metaCL $(CLANG_LIBS) $(CLANG_LDFLAGS)


