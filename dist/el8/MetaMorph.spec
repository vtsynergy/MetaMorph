Name:           MetaMorph
Version:        0.3b
Release:        1%{?dist}
Summary:        MetaMorph heterogenous runtime and support utilities

License:        LGPLv2.1+
URL:            https://github.com/vtsynergy/%{name}
Source0:        https://github.com/vtsynergy/%{name}/archive/v%{version}.tar.gz
Patch0:		Centos8_MetaCL_MakeFix.patch

#TODO: figure ou how to have a generic cuda-compiler version
%define _cuda_major 11
%define _cuda_minor 0
%define _cuda_ver %{_cuda_major}-%{_cuda_minor}
BuildRequires:	gcc-c++, cuda-compiler-%{_cuda_ver}, cuda-libraries-devel-%{_cuda_ver}, (mpich-devel or openmpi-devel), clang-devel >= 6.0, llvm-devel >= 6.0, llvm-static >= 6.0, ncurses-devel
Requires:       (metamorph-openmp or metamorph-opencl or metamorph-cuda)


%if %( if [ -f %{_libdir}/mpich/bin/mpicc  ]; then echo "1" ; else echo "0" ; fi )
%define _mpidir %{_libdir}/mpich
%else
%define _mpidir %{_libdir}/openmpi
%endif
#TODO build debug symbols once upstream supports it
%global debug_package %{nil}

%description
MetaMorph Core Library, which implements the core API and dynamicallyloaded bridge across supported backends


%prep
%setup
%patch0 -p1
#TODO: support pull from github

%build
%make_build
DESTDIR=%{buildroot} PATH=%{_usr}/local/cuda-%{_cuda_major}.%{_cuda_minor}/bin:$PATH VERSION=%{version} MPI_DIR=%{_mpidir} LIBRARY_PATH=%{_usr}/local/cuda-%{_cuda_major}.%{_cuda_minor}/targets/%{_arch}-%{_os}/lib:$LIBRARY_PATH CLANG_LIB_PATH=%{_libdir} USE_CUDA=TRUE USE_OPENCL=TRUE USE_OPENMP=TRUE USE_MPI=TRUE USE_FORTRAN=TRUE USE_TIMERS=TRUE make all generators

%install
rm -rf $RPM_BUILD_ROOT
#%make_install
#TODO
VERSION=%{version} DESTDIR=%{buildroot} make install-all

%clean
make clean

%files
%license LICENSE
%doc README.md
#TODO
%{_usr}/local/lib/libmetamorph.so
%dir %{_usr}/local/lib
#symlink, not dir
%{_usr}/local/lib/metamorph
%dir %{_usr}/local/lib/metamorph-%{version}
%{_usr}/local/lib/metamorph-%{version}/libmetamorph.so


%changelog
* Wed Jul  8 2020 Paul Sathre
- #TODO


%package openmp
Summary:MetaMorph OpenMP Backend
%description openmp
Implements OpenMP versions of MetaMorph's top-level functions
Requires: metamorph
%files openmp
%{_usr}/local/lib/libmetamorph_openmp.so
%{_usr}/local/lib/metamorph-%{version}/libmetamorph_openmp.so

%package cuda
Summary: MetaMorph CUDA Backend
%description cuda
Implements CUDA versions of MetaMorph's top-level functions

Requires: metamorph, libcuda1
%files cuda
%{_usr}/local/lib/libmetamorph_cuda.so
%{_usr}/local/lib/metamorph-%{version}/libmetamorph_cuda.so

%package opencl
Summary: MetaMorph OpenCL Backend
%description opencl
Implements the OpenCL versions of MetaMorph's top-level functions
It also provides OpenCL-specific functionality to support general OpenCL applications, particularly those created with MetaCL

Requires: metamorph, opencl-icd | amd-opencl-icd, libopencl-1.1-1
%files opencl
%{_usr}/local/lib/libmetamorph_opencl.so
%{_usr}/local/lib/metamorph-%{version}/libmetamorph_opencl.so
%{_usr}/local/lib/metamorph-%{version}/metamorph_opencl.cl


%package mpi
Summary: MetaMorph MPI Interoperability plugin
%description mpi
Provides operations to send and receive MetaMorph device buffers across MPI

Requires: metamorph, mpi-default-bin
%files mpi
%{_usr}/local/lib/libmetamorph_mpi.so
%{_usr}/local/lib/metamorph-%{version}/libmetamorph_mpi.so

%package profiling
Summary: MetaMorph Profiling plugin
%description profiling
Provides transparent profiling of MetaMorph API calls via backend-native events.

Requires: metamorph
%files profiling
%{_usr}/local/lib/libmetamorph_profiling.so
%{_usr}/local/lib/metamorph-%{version}/libmetamorph_profiling.so

%package dev
Summary: MetaMorph core headers
%description dev
Files necessary to develop a new application based on only the core MetaMorph capabities

Requires: metamorph
%files dev
%dir %{_usr}/local/include
%{_usr}/local/include/metamorph.h
%{_usr}/local/include/metamorph_dynamic_symbols.h
%{_usr}/local/include/metamorph_fortran_compat.h
%{_usr}/local/include/metamorph_fortran_header.F03

%package openmp-dev
Summary: MetaMorph OpenMP Backend Headers
%description openmp-dev
Files necessary to develop a new application using features specific to the OpenMP backend

Requires: metamorph-dev, metamorph-openmp
%files openmp-dev
%{_usr}/local/include/metamorph_openmp.h

%package cuda-dev
Summary: MetaMorph CUDA Backend headers
%description cuda-dev
Files necessary to develop a new application using features specific to the CUDA backend

Requires: metamorph-dev, metamorph-cuda, cuda-headers
%files cuda-dev
%{_usr}/local/include/metamorph_cuda.cuh

%package opencl-dev
Summary: MetaMorph OpenCL Backend Header
%description opencl-dev
Files necessary to develop a new application using features specific to the OpenCL Backend, including those necessary to support MetaCL-generated OpenCL applications.

Requires: metamorph-dev, metamorph-opencl, opencl-headers
%files opencl-dev
%{_usr}/local/include/metamorph_opencl.h


%package mpi-dev
Summary: MetaMorph MPI Interoperability headers 
%description mpi-dev
Files necessary to develop a new application using functionality from the MPI plugin

Requires: metamorph-dev, metamorph-mpi, mpi-headers
%files mpi-dev
%{_usr}/local/include/metamorph_mpi.h

%package profiling-dev
Summary: MetaMorph Profiling header
%description profiling-dev
Files necessary to direectly utilize MetaMorph's backend-native timing infrastructure.
Only necessary for developers adding timers to features that lie outside the core MetaMorph API, or wishing to explitly flush or othewise manage the timers outside their own implicit constructor/destructor.
%files profiling-dev
%{_usr}/local/include/metamorph_profiling.h


%package -n metacl
Summary: MetaCL OpenCL Host Autogenerator.
%description -n metacl
MetaCL is an autogenerator for OpenCL host code based on the Clang/LLVM compiler framework.
Given a set of OpencL kernel file(s), it generates the appropriate boilerplate to initialize the kernels and invoke them, wrapped inside a convenient and simplified API.
Generated code is dependent on metamorph-opencl amd metamorh to run, and corresponding dev packages to compile, but the MetaCL tool itself does not.

Requires: libclang (>=6.0)

%files -n metacl
%{_usr}/local/bin/metaCL
