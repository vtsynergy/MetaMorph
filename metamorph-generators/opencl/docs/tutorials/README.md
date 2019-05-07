# Index of MetaCL tutorials

MetaCL and these associated tutorials are governed by the same [License](../../../../LICENSE) as MetaMorph. Please ensure you accept the permissions, limitations, and conditions specified therein before proceeding.


Installation
------------

A brief introduction to installing the MetaCL code generator and the MetaMorph (OpenCL-backend) dependency. Subsequent tutorials will assume the same setup, paths, etc. as specified here.
[Installation Tutorial](./InstallingMetaCL.md)


MetaCL Invocation
-----------------

A brief guide to invoking the MetaCL tool on the command line to consume OpenCL kernel (.cl) files, and generate corresponding host-to-device interface wrappers (.c/.h) files. Covers typical use cases and explains all current code generation options.
[How to use MetaCL](./GenerateMetaCLInterface.md)


Porting an Existing Application to a MetaCL-generated Interface
---------------------------------------------------------------

MetaCL saves developer effort by automatically generating the host-to-device interface for OpenCL kernels directly from the kernel implementation. For an existing code this has the benefit of releasing developers from the burden of managing the OpenCL boilerplate themselves so they can instead focus on the application logic. This tutorial walks through typical best practices for permanently converting an existing OpenCL application to use a MetaCL-generated interface, including regenerating the interface as part of the applications build process.
[MetaCL-izing an Existing Application](./ExistingApplication.md)