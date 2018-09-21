LLVM_BUILD_PATH := ../clang-6.0.1-ubuntu-16.04/clang+llvm-6.0.1-x86_64-linux-gnu-ubuntu-16.04
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

all: metagen-cl

metagen-cl: metagen-cl.cpp
	$(CXX) $(CXXFLAGS) $(CLANG_INCLUDES) metagen-cl.cpp -o metagen-cl $(CLANG_LDFLAGS) $(CLANG_LIBS)
