Name:           metaCL
Version:        0.3.1b
Release:        1%{?dist}
Summary:        MetaCL: OpenCL Device-to-Host interface autogenerator

%global debug_package %{nil}

License:        LGPLv2.1+
URL:            https://github.com/vtsynergy/MetaMorph
Source0:        https://github.com/vtsynergy/MetaMorph/archive/v%{version}.tar.gz

BuildRequires:	gcc-c++, make, ncurses-devel, llvm-toolset-7.0-llvm-devel, llvm-toolset-7.0-clang-devel
Requires:       llvm-toolset-7.0-llvm-devel
Requires:       llvm-toolset-7.0-clang-devel


%description
MetaCL is an autogenerator for OpenCL host code based on the Clang/LLVM compiler framework.
Given a set of OpencL kernel file(s), it generates the appropriate boilerplate to initialize the kernels and invoke them, wrapped inside a convenient and simplified API.
Generated codes processed with --use-metamorph=disabled have only an OpenCL runtime as a dependency, but additional functionality is available when paired with the MetaMorph OpenCL backend.

%prep
%setup -n MetaMorph-%{version}

%build
#%make_build
#sets a standard rpath in some cases, set QA_RPATHS=0x0001 during rpmbuild
DESTDIR=%{buildroot} VERSION=%{version} make generators

%install
rm -rf $RPM_BUILD_ROOT
#%make_install
#TODO
VERSION=%{version} DESTDIR=%{buildroot} make install-metaCL

%clean
make clean

%files
%{_usr}/local/bin/metaCL


%changelog
* Fri Jan 22 2021 Paul Sathre
- Initial working SPEC implementation
