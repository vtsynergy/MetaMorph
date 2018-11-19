LLVM_BASE_PATH := clang_tarball
LLVM_VERSION := 6.0.1
LLVM_OS_STRING := x86_64-linux-gnu-ubuntu-16.04
LLVM_TARBALL := clang+llvm-$(LLVM_VERSION)-$(LLVM_OS_STRING)
LLVM_WGET := http://releases.llvm.org/$(LLVM_VERSION)/$(LLVM_TARBALL).tar.xz
LLVM_BUILD_PATH := $(LLVM_BASE_PATH)/$(LLVM_TARBALL)
LLVM_BIN_PATH := $(LLVM_BUILD_PATH)/bin
LLVM_HEADER_PATH := $(LLVM_BUILD_PATH)/include

CLANG_INCLUDES := -I$(LLVM_HEADER_PATH) 

CLANG_LIBS :=

BUILD_DIR := 

CXX := g++

CXXFLAGS := -g -O0 -std=c++11 -fno-rtti

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
	$(CXX) $(CXXFLAGS) $(CLANG_INCLUDES) metacl.cpp -o metaCL $(CLANG_LDFLAGS) $(CLANG_LIBS)

$(LLVM_BUILD_PATH): | $(LLVM_BASE_PATH)/$(LLVM_TARBALL).tar.xz
	tar -xf $(LLVM_BASE_PATH)/$(LLVM_TARBALL).tar.xz -C $(LLVM_BASE_PATH)

$(LLVM_BASE_PATH)/$(LLVM_TARBALL).tar.xz: | $(LLVM_BASE_PATH)
	echo $(LLVM_BASE_PATH)/$(LLVM_TARBALL).tar.xz && cd $(LLVM_BASE_PATH) && wget $(LLVM_WGET)

$(LLVM_BASE_PATH):
	mkdir -p $(LLVM_BASE_PATH)

