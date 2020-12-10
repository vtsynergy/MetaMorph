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
#Function to test that a necessary file exists
CHECK_FILE_EXISTS = $(if $(shell test -e $(path)/$(filename)  && echo -n yes),,"Cannot find $(path)/$(filename)\n")

#TODO In the long run these should be exposed to the user
CXX := g++
CXXFLAGS := -O2 -std=c++11 -fno-rtti

#These can all be manually set to override which llvm/clang is used, otherwise we'll attempt to set a default
#Centos 7 has the clang headers in /usr/include/clang/AST/Attr.h etc
#so does Fedora 31
#Debian 9 has them in /usr/include/llvm-<VERSION>/include/clang/AST/Attr.h
#so does Ubuntu 18.04
ifndef LLVM_INCLUDE_PATH
LLVM_INCLUDE_PATH :=
endif
ifndef LLVM_LIBRARY_PATH
LLVM_LIBRARY_PATH :=
endif
#Centos 7 has the clang libraries in /usr/lib64/clang-private/libclangASTMatchers.so, or with llvm-static /usr/lib64/llvm/libclangASTMatchers.a
# I think we need the static ones though
#Fedora 31 has them in /usr/lib64/libclangASTMatchers.a
#Debian 9 has them in /usr/lib/llvm-<VERISON>/lib/libclangASTMatchers.a
#so does Ubuntu 18.04
ifndef CLANG_INCLUDE_PATH
CLANG_INCLUDE_PATH :=
endif
ifndef CLANG_LIBRARY_PATH
CLANG_LIBRARY_PATH :=
endif

#Required header files
LLVM_REQD_INCLUDES := llvm/Support/raw_ostream.h
CLANG_REQD_INCLUDES := clang/AST/Attr.h \
	clang/ASTMatchers/ASTMatchFinder.h \
	clang/ASTMatchers/ASTMatchers.h \
	clang/Tooling/CommonOptionsParser.h \
	clang/Tooling/Tooling.h
#Validate all the headers exist
path = $(LLVM_INCLUDE_PATH)
MISSING_HEADERS := $(foreach filename, $(LLVM_REQD_INCLUDES), $(CHECK_FILE_EXISTS))
path = $(CLANG_INCLUDE_PATH)
MISSING_HEADERS := $(MISSING_HEADERS) $(foreach filename, $(CLANG_REQD_INCLUDES), $(CHECK_FILE_EXISTS))
#Construct the include line
CLANG_INCLUDES := -I$(CLANG_INCLUDE_PATH)
ifneq ($(LLVM_INCLUDE_PATH), $(CLANG_INCLUDE_PATH))
CLANG_INCLUDES := $(CLANG_INCLUDES) -I$(LLVM_INCLUDE_PATH)
endif

#Required static libraries
LLVM_REQD_LIBS := LLVMOption \
	LLVMSupport \
	LLVMBitReader \
	LLVMMC \
	LLVMMCParser \
	LLVMCore \
	LLVMBinaryFormat \
	LLVMProfileData
CLANG_REQD_LIBS := clangAST \
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
#Validate all the libraries exist
path = $(LLVM_LIBRARY_PATH)
MISSING_LIBS := $(foreach filename, $(foreach shortname, $(LLVM_REQD_LIBS), lib$(shortname).a), $(CHECK_FILE_EXISTS))
path = $(CLANG_LIBRARY_PATH)
MISSING_LIBS := $(MISSING_LIBS) $(foreach filename, $(foreach shortname, $(CLANG_REQD_LIBS), lib$(shortname).a), $(CHECK_FILE_EXISTS))
#Construct the link line
CLANG_LIBS := -L$(CLANG_LIBRARY_PATH)
ifneq ($(LLVM_LIBRARY_PATH), $(CLANG_LIBRARY_PATH))
CLANG_LIBS := $(CLANG_LIBS) -L$(LLVM_LIBRARY_PATH)
endif
CLANG_LIBS := $(CLANG_LIBS) \
	-Wl,--start-group \
        $(foreach lib, $(LLVM_REQD_LIBS), -l$(lib)) \
        $(foreach lib, $(CLANG_REQD_LIBS), -l$(lib)) \
	-Wl,--end-group \
	-lpthread
ifeq ($(OS), ubuntu)
ifeq ($(VER), 18.04)
CLANG_LDFLAGS := -ltinfo -lz
endif
endif
ifeq ($(OS), debian)
CLANG_LDFLAGS := -ltinfo -lz
endif
ifeq ($(OS),)
CLANG_LDFLAGS := -lncurses
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
	@echo $(MISSING_HEADERS)
	@echo $(MISSING_LIBS)
	$(CXX) $(CXXFLAGS) $(CLANG_INCLUDES) metacl.cpp -c -o metacl.o

