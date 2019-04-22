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

#ifndef METAMORPH_OCL_DEFAULT_BLOCK_3D
/** Default 3D workgroup dimension */
#define METAMORPH_OCL_DEFAULT_BLOCK_3D {16, 8, 1}
#endif
#ifndef METAMORPH_OCL_DEFAULT_BLOCK_2D
/** Default 2D workgroup dimension */
#define METAMORPH_OCL_DEFAULT_BLOCK_2D {16, 8}
#endif
#ifndef METAMORPH_OCL_DEFAULT_BLOCK_1D
/** Default 1D workgroup dimension */
#define METAMORPH_OCL_DEFAULT_BLOCK_1D 16
#endif

#ifndef METAMORPH_OCL_KERNEL_PATH
/** Default kernel search path
 * \warning this guaranteed define only exists to ensure the value is defined, this is intended to be set via the Makefile to grab absolute paths to the relevant source and build directories
 */
#define METAMORPH_OCL_KERNEL_PATH ""
#endif

#if defined(__OPENCLCC__) || defined(__cplusplus)
extern "C" {
#endif

/** This is a simple enum to store important details about the type of device and which vendor is providing the implementation
 * Currently ontly used to check for Altera/IntelFPGA at runtime to load .aocx files rather than .cl
 */
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

/// \todo TODO OpenCL stack operations probably shouldn't need to be exposed to the user, unless there's a CUDA call we need to emulate, or some reason a module would need direct access, rather than querying the global state
/**
 * Record a new OpenCL state frame on the stack, then swap the global OpenCL state variables to it and reinitializes any registered module that has the module_implements_opencl flag set
 * \param frame A new *complete* frame object to *copy* to the stack
 * \warning Once copied the original should be freed (not Destroyed)
 */
void metaOpenCLPushStackFrame(metaOpenCLStackFrame * frame);

/**
 * Look at a copy of whatever the current frame is without removing or replacing it.
 * Kernel wrappers will typically call this to get the the appropriate cl_kernel to enqueue
 * \return a pointer to a copy of the top frame
 * \warning the returned copy should be freed, not Destroyed
 */
metaOpenCLStackFrame * metaOpenCLTopStackFrame();

/**
 * Retrieve the current top frame, and remove it
 * Intended to be used just to deconstruct a state (or if you want to remove it from MetaMorph's knowledge
 * \return a pointer to the top frame
 * \warning after this call MetaMorph will have forgotten about the contents of this frame, it must be Destroyed or manually Released
 */
metaOpenCLStackFrame * metaOpenCLPopStackFrame();

/**
 * Allocate a new frame and initialize the OpenCL state objects for it (platform, device, context, queue, and internal buffers)
 * \param frame the address of a frame * in which to return the allocated/initialized frame
 * \param device which of the [0:N-1] devices aggregated across all platforms in the system to use, or -1 for "don't care, pick something",which triggers metaOpenCLInitStackFrameDefault
 * \return an OpenCL error code if anything went wrong, otherwise CL_SUCCESS
 * \bug Does not currently return anything
 * \todo TODO Not all OpenCL API calls are being checked
 * \warning does not read or build cl_programs or cl_kernels, see metaOpenCLInitCoreKernels
 */
cl_int metaOpenCLInitStackFrame(metaOpenCLStackFrame ** frame, cl_int device);

/**
 * \brief Release all the OpenCL state and programs/kernels this frame refers to
 *
 * In order it 1) Checks if each cl_kernel is initialized, if so, clReleases it
 * 2) clReleases internal buffers
 * 3) checks if each cl_program is initialized, if so, clReleases it
 * 4) clReleases the queue and context
 * 5) frees all the program source/binary buffers
 * \param frame The frame to deconstruct
 * \return an OpenCL error code if anything went wrong, or CL_SUCCESS otherwise
 * \todo TODO not all OpenCL API calls are being error checked
 * \bug doesn't currently return anything
 * \warning, does not pop any stack nodes or free the passed frame, those are the caller's responsibility
 */ 
cl_int metaOpenCLDestroyStackFrame(metaOpenCLStackFrame * frame);

//support initialization of a default frame as well as environment variable
// -based control, via $TARGET_DEVICE="Some Device Name"
/**
 * \brief Catchall "Make sure *some* device is initialized before trying to run any OpenCL commands
 * Warns that automatic OpenCL devices selection is used, and lists the name and MetaMorph device ID (index) of all devices on all platforms in the system
 * Chooses an OpenCL device and reports it as a stderr warning, as the user did not explicitly pick a device.
 * Will either choose the zeroth device among all [0:N-1] devices in the system, or if the TARGET_DEVICE=\<string\>
 *  environment variable is set, will try to find the first device with an exactly-matching name string
 * Once a device is chosen it calls metaOpenCLInitStackFrame to initialize a frame for it
 * \param frame The address of a frame pointer in which to return the newly-initialized default frame
 * \return an OpenCL error code if anything went wrong, otherwise CL_SUCCESS
 * \todo TODO not all OpenCL API calls are being checked
 * \bug if metaOpenCLInitStackFrame is not explicitly returning a sane error code, in most cases neither will this
 */
cl_int metaOpenCLInitStackFrameDefault(metaOpenCLStackFrame ** frame);

/**
 * \brief Load a specified OpenCL kernel implementation
 *
 * Attempts to load the OpenCL kernel implementation specified by filename with a configurable search path.
 * If the environment variable METAMORPH_OCL_KERNEL_PATH is set (syntax like a regular path variable \<dir1\>:\<dir2\>:...\<dirN\>), scan through those directories in order for the specified filename. If not set or not found, then scan through the compile-time configure directories. If still not found, emit a warning to stderr and return a -1 program length
 * \param filename a pointer to a NULL-terminated string with the desired filename
 * \warning the filename should be specified *without* any path information or else it will be concatenated onto the search paths
 * \param progSrc The address of a character pointer in which to return the address of the complete NULL-terminated string that is read in
 * \return The number of bytes read into progSrc, or -1 if the file is not found
 */ 
size_t metaOpenCLLoadProgramSource(const char *filename, const char **progSrc);

/**
 * Initialize all the builtin kernels for the current frame.
 * Separate from frame initialization to omit the cost for modules which do not make use of builtin kernels
 * \return an OpenCL error code if anything went wrong, otherwise CL_SUCCESS
 */
cl_int metaOpenCLInitCoreKernels();

/**
 * Given a device, query the OpenCL API to detect the vendor and type of device and store it in our representation
 * \param dev The device to query
 * \return The encoded device information
 */
meta_cl_device_vendor metaOpenCLDetectDevice(cl_device_id dev);

/**
 * OpenCL wrapper to all dot product kernels
 * \param grid_size The number of workgroups in each dimension (NOT the global work size)
 * \param block_size The size of a workgroup in each dimension
 * \param data1 The first input array
 * \param data2 The second input array
 * \param array_size The X, Y, and Z sizes of data1 and data2
 * \param arr_start The X, Y, and Z start indicies for the dot product (to allow halo avoidance)
 * \param arr_end The X, Y, and Z end indicies for the dot product (to allow halo avoidance)
 * \param reduced_val The final global dot product value
 * \param type The data type to invoke the appropriate kernel for
 * \param async Whether the kernel should run asynchronously or block
 * \param event if a cl_event is provided pass it to the kernel enqueue call
 * \return An OpenCL error code if anything went wrong, CL_SUCCESS otherwise
 * \bug OpenCL return codes should not be binary OR'd, rather separately checked and the last error returned
 */
cl_int opencl_dotProd(size_t (*grid_size)[3], size_t (*block_size)[3],
		void * data1, void * data2, size_t (*array_size)[3],
		size_t (*arr_start)[3], size_t (*arr_end)[3], void * reduced_val,
		meta_type_id type, int async, cl_event * event);

/**
 * OpenCL wrapper to all reduction sum kernels
 * \param grid_size The number of workgroups in each dimension (NOT the global work size)
 * \param block_size The size of a workgroup in each dimension
 * \param data The array to sum across
 * \param array_size The X, Y, and Z sizes of data
 * \param arr_start The X, Y, and Z start indicies for the reduction sum (to allow halo avoidance)
 * \param arr_end The X, Y, and Z end indicies for the reduction sum (to allow halo avoidance)
 * \param reduced_val The final globally-reduced sum
 * \param type The data type to invoke the appropriate kernel for
 * \param async Whether the kernel should run asynchronously or block
 * \param event if a cl_event is provided pass it to the kernel enqueue call
 * \return An OpenCL error code if anything went wrong, CL_SUCCESS otherwise
 * \bug OpenCL return codes should not be binary OR'd, rather separately checked and the last error returned
 */
cl_int opencl_reduce(size_t (*grid_size)[3], size_t (*block_size)[3],
		void * data, size_t (*array_size)[3], size_t (*arr_start)[3],
		size_t (*arr_end)[3], void * reduced_val, meta_type_id type, int async,
		cl_event * event);
/**
 * OpenCL wrapper to transpose a 2D array
 * \param grid_size The number of workgroups in each dimension (NOT the global work size)
 * \param block_size The size of a workgroup in each dimension
 * \param indata The input untransposed 2D array
 * \param outdata The output transposed 2D array
 * \param arr_dim_xy the X and Y dimensions of indata, Z is ignored 
 * \param tran_dim_xy the X and Y dimensions of outdata, Z is ignored
 * \param type The data type to invoke the appropriate kernel for
 * \param async Whether the kernel should run asynchronously or block
 * \param event if a cl_event is provided pass it to the kernel enqueue call
 * \return An OpenCL error code if anything went wrong, CL_SUCCESS otherwise
 * \bug OpenCL return codes should not be binary OR'd, rather separately checked and the last error returned
 */
cl_int opencl_transpose_face(size_t (*grid_size)[3], size_t (*block_size)[3],
		void *indata, void *outdata, size_t (*arr_dim_xy)[3],
		size_t (*tran_dim_xy)[3], meta_type_id type, int async,
		cl_event * event);
/**
 * OpenCL wrapper for the face packing kernel
 * \param grid_size The number of workgroups to run in the X dimension, Y and Z are ignored (NOT the global worksize)
 * \param block_size The size of a workgroup in the X dimension, Y and Z are ignored
 * \param packed_buf The output packed array
 * \param buf The input full array
 * \param face The face/slab to extract, returned from make_slab_from_3d
 * \param remain_dim At each subsequent layer of the face struct, the *cummulative* size of all remaining descendant dimensions. Used for computing the per-thread read offset, typically automatically pre-computed in the backend-agnostic MetaMorph wrapper, meta_pack_face
 * \param type The type of data to run the kernel on
 * \param async whether to run the kernel asynchronously (1) or synchronously (0)
 * \param event_k1 the kernel event used for asynchronous calls and timing
 * \param event_c1 the clEnqueueWriteBuffer of face->size event used for asynchronous calls and timing
 * \param event_c2 the clEnqueueWriteBuffer of face->stride event used for asynchronous calls and timing
 * \param event_c3 the clEnqueueWriteBuffer of remain_dim to c_face_child_size event used for asynchronous calls and timing
 * \return either the result of enqueing the kernel if async or the result of clFinish if sync
 * \warning Implemented as a 1D kernel, Y and Z grid/block parameters will be ignored
 * \bug OpenCL return codes should not be binary OR'd, rather separately checked and the last error returned
 */
cl_int opencl_pack_face(size_t (*grid_size)[3], size_t (*block_size)[3],
		void *packed_buf, void *buf, meta_face *face,
		int *remain_dim, meta_type_id type, int async, cl_event * event_k1,
		cl_event * event_c1, cl_event *event_c2, cl_event *event_c3);
/**
 * OpenCL wrapper for the face unpacking kernel
 * \param grid_size The number of workgroups to run in the X dimension, Y and Z are ignored (NOT the global worksize)
 * \param block_size The size of a workgroup in the X dimension, Y and Z are ignored
 * \param packed_buf The input packed array
 * \param buf The output full array
 * \param face The face/slab to extract, returned from make_slab_from_3d
 * \param remain_dim At each subsequent layer of the face struct, the *cummulative* size of all remaining descendant dimensions. Used for computing the per-thread write offset, typically automatically pre-computed in the backend-agnostic MetaMorph wrapper, meta_pack_face
 * \param type The type of data to run the kernel on
 * \param async whether to run the kernel asynchronously (1) or synchronously (0)
 * \param event_k1 the kernel event used for asynchronous calls and timing
 * \param event_c1 the clEnqueueWriteBuffer of face->size event used for asynchronous calls and timing
 * \param event_c2 the clEnqueueWriteBuffer of face->stride event used for asynchronous calls and timing
 * \param event_c3 the clEnqueueWriteBuffer of remain_dim to c_face_child_size event used for asynchronous calls and timing
 * \return either the result of enqueuing the kernel if async or the result of clFinish if sync
 * \warning Implemented as a 1D kernel, Y and Z grid/block parameters will be ignored
 * \bug OpenCL return codes should not be binary OR'd, rather separately checked and the last error returned
 */
cl_int opencl_unpack_face(size_t (*grid_size)[3], size_t (*block_size)[3],
		void *packed_buf, void *buf, meta_face *face,
		int *remain_dim, meta_type_id type, int async, cl_event * event_k1,
		cl_event * event_c1, cl_event *event_c2, cl_event *event_c3);
/**
 * OpenCL wrapper for the 3D 7-point stencil averaging kernel
 * \param grid_size The number of workgroups to run in each dimension (NOT the global worksize)
 * \param block_size The size of a workgroup in each dimension
 * \param indata The array of input read values
 * \param outdata The array of output write values
 * \param array_size The size of the input and output arrays
 * \param arr_start The start index for writing in each dimension (for avoiding writes to halo cells)
 * \warning Assumes at least a one-thick halo region i.e. (arr_start[dim]-1 >= 0)
 * \param arr_end The end index for writing in each dimension (for avoiding writes to halo cells)
 * \warning Assumes at least a one-thick halo region i.e. (arr_end[dim]+1 <= array_size[dim]-1)
 * \param type The type of data to run the kernel on
 * \param async whether to run the kernel asynchronously (1) or synchronously (0)
 * \param event the kernel event used for asynchronous calls and timing
 * \return either the result of enqueuing the kernel if async or the result of clFinish if sync
 * \bug OpenCL return codes should not be binary OR'd, rather separately checked and the last error returned
 */
cl_int opencl_stencil_3d7p(size_t (*grid_size)[3], size_t (*block_size)[3],
		void * indata, void * outdata, size_t (*array_size)[3],
		size_t (*arr_start)[3], size_t (*arr_end)[3], meta_type_id type,
		int async, cl_event * event);
/**
 * OpenCL wrapper for the SPMV kernel for CSR sparse matrix format
 * \param grid_size The number of workgroups to run in the X dimension, Y and Z are ignored (NOT the global worksize)
 * \param block_size The size of a workgroup in each dimension
 * \param global_size The number of rows in the A matrix, one computed per work-item
 * \param csr_ap The row start and end index array
 * \param csr_aj The column index array
 * \param csr_ax The cell value array
 * \param x_loc The input vector to multiply A by
 * \param y_loc The output vector to sum into
 * \param type The type of data to run the kernel on
 * \param async whether to run the kernel asynchronously (1) or synchronously (0)
 * \param event the kernel event used for asynchronous calls and timing
 * \return either the result of enqueuing the kernel if async or the result of clFinish if sync
 * \warning, treated as a 1D kernel, Y and Z parameters ignored
 * \bug OpenCL return codes should not be binary OR'd, rather separately checked and the last error returned
 */
cl_int opencl_csr(size_t (*grid_size)[3], size_t (*block_size)[3], size_t global_size,
		void * csr_ap, void * csr_aj, void * csr_ax, void * x_loc, void * y_loc, 
		meta_type_id type, int async,
		// cl_event * wait, 
		cl_event * event);
/**
 * OpenCL wrapper for the cyclic redundancy check task kernel
 * \param dev_input The input buffer to perform the redundancy check on
 * \param page_size The length in bytes of each page
 * \param num_words TODO
 * \param numpages TODO
 * \param dev_output The result
 * \param type The type of data to run the kernel on
 * \param async whether to run the kernel asynchronously (1) or synchronously (0)
 * \param event the kernel event used for asynchronous calls and timing
 * \return either the result of enqueuing the kernel if async or the result of clFinish if sync
 * \warning all supported types are interpreted as binary data via the unsigned int kernel
 * \todo TODO Implement meta_typeless "type" for such kernels
 * \bug OpenCL return codes should not be binary OR'd, rather separately checked and the last error returned
 * \todo TODO Implement an NDRange kernel variant for better performance on non-FPGA platforms
 * \bug non-CL types need to be passed to the kernel through enforced-width cl_\<type\>s
 */
cl_int opencl_crc(void * dev_input, int page_size, int num_words, int numpages, void * dev_output, 
		meta_type_id type, int async,
		// cl_event * wait, 
		cl_event * event);


#if defined(__OPENCLCC__) || defined(__cplusplus)
}
#endif

#endif //METAMORPH_OPENCL_BACKEND_H
