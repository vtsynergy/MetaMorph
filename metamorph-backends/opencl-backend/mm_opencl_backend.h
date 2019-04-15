/** \file
 * Exposed OpenCL backend functions, defines, and data structures
 */

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/** OpenCL Back-End **/
#ifndef METAMORPH_OPENCL_BACKEND_H
#define METAMORPH_OPENCL_BACKEND_H

#ifndef METAMORPH_H
#include "../../include/metamorph.h"
#endif

//If the user doesn't override default threadblock size..
#ifndef METAMORPH_OCL_DEFAULT_BLOCK_3D
#define METAMORPH_OCL_DEFAULT_BLOCK_3D {16, 8, 1}
#endif
#ifndef METAMORPH_OCL_DEFAULT_BLOCK_2D
#define METAMORPH_OCL_DEFAULT_BLOCK_2D {16, 8}
#endif
#ifndef METAMORPH_OCL_DEFAULT_BLOCK_1D
#define METAMORPH_OCL_DEFAULT_BLOCK_1D 16
#endif

#ifndef METAMORPH_OCL_KERNEL_PATH
#define METAMORPH_OCL_KERNEL_PATH ""
#endif

#if defined(__OPENCLCC__) || defined(__cplusplus)
extern "C" {
#endif

//This is a simple enum to store important details about the type of device
// and which vendor is providing the implementation
//Currently ontly used to check for Altera/IntelFPGA at runtime to load .aocx files rather than .cl
typedef enum meta_cl_device_vendor {
  meta_cl_device_vendor_unknown = 0,
  meta_cl_device_vendor_nvidia = 1,
  meta_cl_device_vendor_amd_appsdk = 2,
  meta_cl_device_vendor_amd_rocm = 3,
  meta_cl_device_vendor_intel = 4,
  meta_cl_device_vendor_intelfpga = 5,
  meta_cl_device_vendor_xilinx = 6,
  meta_cl_device_vendor_pocl = 7,
  meta_cl_device_vendor_mask = (1 << 8) - 1,
  meta_cl_device_is_cpu = (1 << 8),
  meta_cl_device_is_gpu = (1 << 9),
  meta_cl_device_is_accel = (1 << 10),
  meta_cl_device_is_default = (1 << 11)
} meta_cl_device_vendor;

/**
 * \brief The struct for managing an entire device/queue's OpenCL state
 *
 *
 * This is the exposed object for managing multiple OpenCL devices with the
 *  library's built-in stack.
 * These frame objects are allocated and freed by the Init and Destroy calls, respectively.
 * As a precursor to making the library threadsafe, all Stack operations (push, top and pop)
 * make a copy of the frame's contents, to keep the frame itself thread-private.
 * \todo TODO remove compile-time INTELFPGA ifdefs and appropriately adjust the struct to handle swapping to/from FPGA devices at runtime.
 * \todo TODO support single-kernel-programs mode with runtime FPGA swap
 */
typedef struct metaOpenCLStackFrame {

        /** The OpenCL platform for this state */
	cl_platform_id platform;
        /** The OpenCL device for this state */
	cl_device_id device;
        /** The OpenCL context for this state */
	cl_context context;
        /** The OpenCL queue for this state */
	cl_command_queue queue;
	/** The initialization status of the platform/device/context/queue */
	unsigned char state_init
	/** The initialization status of the kernels for this state */
	unsigned char kernels_init;

//Trades host StackFrame size for smaller programs (sometimes necessary for FPGA)
#if  (defined(WITH_INTELFPGA) && defined(OPENCL_SINGLE_KERNELPROGS))
	//If we have separate binaries per kernel, then we need separate buffers
	const char *metaCLbin_reduce_db;
	size_t metaCLbinLen_reduce_db;
	const char *metaCLbin_reduce_fl;
	size_t metaCLbinLen_reduce_fl;
	const char *metaCLbin_reduce_ul;
	size_t metaCLbinLen_reduce_ul;
	const char *metaCLbin_reduce_in;
	size_t metaCLbinLen_reduce_in;
	const char *metaCLbin_reduce_ui;
	size_t metaCLbinLen_reduce_ui;
	const char *metaCLbin_dotProd_db;
	size_t metaCLbinLen_dotProd_db;
	const char *metaCLbin_dotProd_fl;
	size_t metaCLbinLen_dotProd_fl;
	const char *metaCLbin_dotProd_ul;
	size_t metaCLbinLen_dotProd_ul;
	const char *metaCLbin_dotProd_in;
	size_t metaCLbinLen_dotProd_in;
	const char *metaCLbin_dotProd_ui;
	size_t metaCLbinLen_dotProd_ui;
	const char *metaCLbin_transpose_2d_face_db;
	size_t metaCLbinLen_transpose_2d_face_db;
	const char *metaCLbin_transpose_2d_face_fl;
	size_t metaCLbinLen_transpose_2d_face_fl;
	const char *metaCLbin_transpose_2d_face_ul;
	size_t metaCLbinLen_transpose_2d_face_ul;
	const char *metaCLbin_transpose_2d_face_in;
	size_t metaCLbinLen_transpose_2d_face_in;
	const char *metaCLbin_transpose_2d_face_ui;
	size_t metaCLbinLen_transpose_2d_face_ui;
	const char *metaCLbin_pack_2d_face_db;
	size_t metaCLbinLen_pack_2d_face_db;
	const char *metaCLbin_pack_2d_face_fl;
	size_t metaCLbinLen_pack_2d_face_fl;
	const char *metaCLbin_pack_2d_face_ul;
	size_t metaCLbinLen_pack_2d_face_ul;
	const char *metaCLbin_pack_2d_face_in;
	size_t metaCLbinLen_pack_2d_face_in;
	const char *metaCLbin_pack_2d_face_ui;
	size_t metaCLbinLen_pack_2d_face_ui;
	const char *metaCLbin_unpack_2d_face_db;
	size_t metaCLbinLen_unpack_2d_face_db;
	const char *metaCLbin_unpack_2d_face_fl;
	size_t metaCLbinLen_unpack_2d_face_fl;
	const char *metaCLbin_unpack_2d_face_ul;
	size_t metaCLbinLen_unpack_2d_face_ul;
	const char *metaCLbin_unpack_2d_face_in;
	size_t metaCLbinLen_unpack_2d_face_in;
	const char *metaCLbin_unpack_2d_face_ui;
	size_t metaCLbinLen_unpack_2d_face_ui;
	const char *metaCLbin_stencil_3d7p_db;
	size_t metaCLbinLen_stencil_3d7p_db;
	const char *metaCLbin_stencil_3d7p_fl;
	size_t metaCLbinLen_stencil_3d7p_fl;
	const char *metaCLbin_stencil_3d7p_ul;
	size_t metaCLbinLen_stencil_3d7p_ul;
	const char *metaCLbin_stencil_3d7p_in;
	size_t metaCLbinLen_stencil_3d7p_in;
	const char *metaCLbin_stencil_3d7p_ui;
	size_t metaCLbinLen_stencil_3d7p_ui;
	const char *metaCLbin_csr_db;
	size_t metaCLbinLen_csr_db;
	const char *metaCLbin_csr_fl;
	size_t metaCLbinLen_csr_fl;
	const char *metaCLbin_csr_ul;
	size_t metaCLbinLen_csr_ul;
	const char *metaCLbin_csr_in;
	size_t metaCLbinLen_csr_in;
	const char *metaCLbin_csr_ui;
	size_t metaCLbinLen_csr_ui;
//	const char *metaCLbin_crc_db;
//	size_t metaCLbinLen_crc_db;
//	const char *metaCLbin_crc_fl;
//	size_t metaCLbinLen_crc_fl;
//	const char *metaCLbin_crc_ul;
//	size_t metaCLbinLen_crc_ul;
//	const char *metaCLbin_crc_in;
//	size_t metaCLbinLen_crc_in;
	const char *metaCLbin_crc_ui;
	size_t metaCLbinLen_crc_ui;
#else
	//If we still have access to the raw source, no need for separate copies
	/** The unified kernel source (if single-kernel-programs is off or they are handled by conditional JIT) */
	const char *metaCLProgSrc;
	/** Length in bytes of the unified kernel source */
	size_t metaCLProgLen;
#endif


#ifndef OPENCL_SINGLE_KERNEL_PROGS

	/** Unified OpenCL program object */
	cl_program program_opencl_core;
#else
	cl_program program_reduce_db;
	cl_program program_reduce_fl;
	cl_program program_reduce_ul;
	cl_program program_reduce_in;
	cl_program program_reduce_ui;
	cl_program program_dotProd_db;
	cl_program program_dotProd_fl;
	cl_program program_dotProd_ul;
	cl_program program_dotProd_in;
	cl_program program_dotProd_ui;
	cl_program program_transpose_2d_face_db;
	cl_program program_transpose_2d_face_fl;
	cl_program program_transpose_2d_face_ul;
	cl_program program_transpose_2d_face_in;
	cl_program program_transpose_2d_face_ui;
	cl_program program_pack_2d_face_db;
	cl_program program_pack_2d_face_fl;
	cl_program program_pack_2d_face_ul;
	cl_program program_pack_2d_face_in;
	cl_program program_pack_2d_face_ui;
	cl_program program_unpack_2d_face_db;
	cl_program program_unpack_2d_face_fl;
	cl_program program_unpack_2d_face_ul;
	cl_program program_unpack_2d_face_in;
	cl_program program_unpack_2d_face_ui;
	cl_program program_stencil_3d7p_db;
	cl_program program_stencil_3d7p_fl;
	cl_program program_stencil_3d7p_ul;
	cl_program program_stencil_3d7p_in;
	cl_program program_stencil_3d7p_ui;
	cl_program program_csr_db;
	cl_program program_csr_fl;
	cl_program program_csr_ul;
	cl_program program_csr_in;
	cl_program program_csr_ui;
//	cl_program program_crc_db;
//	cl_program program_crc_fl;
//	cl_program program_crc_ul;
//	cl_program program_crc_in;
	cl_program program_crc_ui;
#endif

	/** Double-precision reduction (sum) kernel */
	cl_kernel kernel_reduce_db;
	/** Single-precision reduction (sum) kernel */
	cl_kernel kernel_reduce_fl;
	/** Unsigned long reduction (sum) kernel */
	cl_kernel kernel_reduce_ul;
	/** Integer reduction (sum) kernel */
	cl_kernel kernel_reduce_in;
	/** Unsigned integer reduction (sum) kernel */
	cl_kernel kernel_reduce_ui;
	/** Double-precision dot product kernel */
	cl_kernel kernel_dotProd_db;
	/** Single-precision dot product kernel */
	cl_kernel kernel_dotProd_fl;
	/** Unsigned long dot product kernel */
	cl_kernel kernel_dotProd_ul;
	/** Integer dot product kernel */
	cl_kernel kernel_dotProd_in;
	/** Unsigned integer dot product kernel */
	cl_kernel kernel_dotProd_ui;
	/** Double-precision 2D face transpose kernel */
	cl_kernel kernel_transpose_2d_face_db;
	/** Single-precision 2D face transpose kernel */
	cl_kernel kernel_transpose_2d_face_fl;
	/** Unsigned long 2D face transpose kernel */
	cl_kernel kernel_transpose_2d_face_ul;
	/** Integer 2D face transpose kernel */
	cl_kernel kernel_transpose_2d_face_in;
	/** Unsigned Integer 2D face transpose kernel */
	cl_kernel kernel_transpose_2d_face_ui;
	/** Double-precision "slab" packing kernel */
	cl_kernel kernel_pack_2d_face_db;
	/** Single-precision "slab" packing kernel */
	cl_kernel kernel_pack_2d_face_fl;
	/** Unsigned Long "slab" packing kernel */
	cl_kernel kernel_pack_2d_face_ul;
	/** Integer "slab" packing kernel */
	cl_kernel kernel_pack_2d_face_in;
	/** Unsigned integer "slab" packing kernel */
	cl_kernel kernel_pack_2d_face_ui;
	/** Double-precision "slab" unpacking kernel */
	cl_kernel kernel_unpack_2d_face_db;
	/** Single-precision "slab" unpacking kernel */
	cl_kernel kernel_unpack_2d_face_fl;
	/** Unsigned long "slab" unpacking kernel */
	cl_kernel kernel_unpack_2d_face_ul;
	/** Integer "slab" unpacking kernel */
	cl_kernel kernel_unpack_2d_face_in;
	/** Unsigned integer "slab" unpacking kernel */
	cl_kernel kernel_unpack_2d_face_ui;
	/** Double-precision 3D 7-point "average" stencil kernel */
	cl_kernel kernel_stencil_3d7p_db;
	/** Single-precision 3D 7-point "average" stencil kernel */
	cl_kernel kernel_stencil_3d7p_fl;
	/** Unsigned long 3D 7-point "average" stencil kernel */
	cl_kernel kernel_stencil_3d7p_ul;
	/** Integer 3D 7-point "average" stencil kernel */
	cl_kernel kernel_stencil_3d7p_in;
	/** Unsigned integer 3D 7-point "average" stencil kernel */
	cl_kernel kernel_stencil_3d7p_ui;
	/** Double-precision SPMV kernel for CSR storage format */
	cl_kernel kernel_csr_db;
	/** Single-precision SPMV kernel for CSR storage format */
	cl_kernel kernel_csr_fl;
        /** Unsigned long SPMV kernel for CSR storage format */
        cl_kernel kernel_csr_ul;
        /** Integer SPMV kernel for CSR storage format */
	cl_kernel kernel_csr_in;
        /** Unsigned Integer SPMV kernel for CSR storage format */
	cl_kernel kernel_csr_ui;
        /** Cyclic redundacy check kernel (all data treated as unsigned int) */
	cl_kernel kernel_crc_ui;

        /** Constant address space storage for the size of the face-specification arrays */
	cl_mem constant_face_size;
        /** Constant address space storage for the stride array for face specification */
	cl_mem constant_face_stride;
	/** Constant address space storage for the "child" extent array for face specification */
	cl_mem constant_face_child_size;
	/** A singleton global variable of sufficient size to store the final result of on-device reductions */
	cl_mem red_loc;

} metaOpenCLStackFrame;
//TODO these shouldn't need to be exposed to the user, unless there's a CUDA call we need to emulate
void metaOpenCLPushStackFrame(metaOpenCLStackFrame * frame);

metaOpenCLStackFrame * metaOpenCLTopStackFrame();

metaOpenCLStackFrame * metaOpenCLPopStackFrame();

//start everything for a frame
cl_int metaOpenCLInitStackFrame(metaOpenCLStackFrame ** frame, cl_int device);

//stop everything for a frame
cl_int metaOpenCLDestroyStackFrame(metaOpenCLStackFrame * frame);

//support initialization of a default frame as well as environment variable
// -based control, via $TARGET_DEVICE="Some Device Name"
cl_int metaOpenCLInitStackFrameDefault(metaOpenCLStackFrame ** frame);

size_t metaOpenCLLoadProgramSource(const char *filename, const char **progSrc);

//explicitly initialize kernels, instead of automatically
cl_int metaOpenCLInitCoreKernels();

//Given a device, detect the type and supporting implementation
meta_cl_device_vendor metaOpenCLDetectDevice(cl_device_id dev);

cl_int opencl_dotProd(size_t (*grid_size)[3], size_t (*block_size)[3],
		void * data1, void * data2, size_t (*array_size)[3],
		size_t (*arr_start)[3], size_t (*arr_end)[3], void * reduced_val,
		meta_type_id type, int async, cl_event * event);
cl_int opencl_reduce(size_t (*grid_size)[3], size_t (*block_size)[3],
		void * data, size_t (*array_size)[3], size_t (*arr_start)[3],
		size_t (*arr_end)[3], void * reduced_val, meta_type_id type, int async,
		cl_event * event);
cl_int opencl_transpose_face(size_t (*grid_size)[3], size_t (*block_size)[3],
		void *indata, void *outdata, size_t (*arr_dim_xy)[3],
		size_t (*tran_dim_xy)[3], meta_type_id type, int async,
		cl_event * event);
cl_int opencl_pack_face(size_t (*grid_size)[3], size_t (*block_size)[3],
		void *packed_buf, void *buf, meta_face *face,
		int *remain_dim, meta_type_id type, int async, cl_event * event_k1,
		cl_event * event_c1, cl_event *event_c2, cl_event *event_c3);
cl_int opencl_unpack_face(size_t (*grid_size)[3], size_t (*block_size)[3],
		void *packed_buf, void *buf, meta_face *face,
		int *remain_dim, meta_type_id type, int async, cl_event * event_k1,
		cl_event * event_c1, cl_event *event_c2, cl_event *event_c3);
cl_int opencl_stencil_3d7p(size_t (*grid_size)[3], size_t (*block_size)[3],
		void * indata, void * outdata, size_t (*array_size)[3],
		size_t (*arr_start)[3], size_t (*arr_end)[3], meta_type_id type,
		int async, cl_event * event);
cl_int opencl_csr(size_t global_size, size_t local_size,
		void * csr_ap, void * csr_aj, void * csr_ax, void * x_loc, void * y_loc, 
		meta_type_id type, int async,
		// cl_event * wait, 
		cl_event * event);
cl_int opencl_crc(size_t global_size, size_t local_size,
		void * dev_input, int page_size, int num_words, int numpages, void * dev_output, 
		meta_type_id type, int async,
		// cl_event * wait, 
		cl_event * event);


#if defined(__OPENCLCC__) || defined(__cplusplus)
}
#endif

#endif //METAMORPH_OPENCL_BACKEND_H
