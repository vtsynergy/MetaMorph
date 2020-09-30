/** \file
 * Emulatable portions of the OpenCL backend. Largely for use with
 * MetaCL-generated applications that don't wish to enforce a library dependency
 * on MetaMorph itself.
 */
#ifndef METAMORPH_OPENCL_EMULATABLE_H
#define METAMORPH_OPENCL_EMULATABLE_H

#ifndef METAMORPH_H
#include <metamorph.h>
#endif

#ifndef METAMORPH_OCL_DEFAULT_BLOCK_3D
/** Default 3D workgroup dimension */
#define METAMORPH_OCL_DEFAULT_BLOCK_3D                                         \
  { 16, 8, 1 }
#endif
#ifndef METAMORPH_OCL_DEFAULT_BLOCK_2D
/** Default 2D workgroup dimension */
#define METAMORPH_OCL_DEFAULT_BLOCK_2D                                         \
  { 16, 8 }
#endif
#ifndef METAMORPH_OCL_DEFAULT_BLOCK_1D
/** Default 1D workgroup dimension */
#define METAMORPH_OCL_DEFAULT_BLOCK_1D 16
#endif

#if defined(__OPENCLCC__) || defined(__cplusplus)
extern "C" {
#endif

/** This is a simple enum to store important details about the type of device
 * and which vendor is providing the implementation Currently ontly used to
 * check for Altera/IntelFPGA at runtime to load .aocx files rather than .cl
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

// Stub: make sure some device exists
void metaOpenCLFallback();

/**
 * \brief Load a specified OpenCL kernel implementation
 *
 * Attempts to load the OpenCL kernel implementation specified by filename with
 * a configurable search path. If the environment variable
 * METAMORPH_OCL_KERNEL_PATH is set (syntax like a regular path variable
 * \<dir1\>:\<dir2\>:...\<dirN\>), scan through those directories in order for
 * the specified filename. If not set or not found, then scan through the
 * compile-time configure directories. If still not found, emit a warning to
 * stderr and return a -1 program length
 * \param filename a pointer to a NULL-terminated string with the desired
 * filename
 * \warning the filename should be specified *without* any path information or
 * else it will be concatenated onto the search paths
 * \param progSrc The address of a character pointer in which to return the
 * address of the complete NULL-terminated string that is read in
 * \param foundFileDir The address of a character pointer in which to return
 * the address of a string containing the directory the file was located in,
 * after utilizing the METAMORPH_OCL_KERNEL_PATH. If NULL, the directory is
 * discarded.
 * \return The number of bytes read into progSrc, or -1 if the file is not found
 */
size_t metaOpenCLLoadProgramSource(const char *filename, const char **progSrc,
                                   const char **foundFileDir);

/**
 * Given a device, query the OpenCL API to detect the vendor and type of device
 * and store it in our representation
 * \param dev The device to query
 * \return The encoded device information
 */
meta_cl_device_vendor metaOpenCLDetectDevice(cl_device_id dev);

// share meta_context with with existing software
meta_int meta_get_state_OpenCL(cl_platform_id *platform, cl_device_id *device,
                            cl_context *context, cl_command_queue *queue);
meta_int meta_set_state_OpenCL(cl_platform_id platform, cl_device_id device,
                            cl_context context, cl_command_queue queue);
#if defined(__OPENCLCC__) || defined(__cplusplus)
}
#endif

#endif // METAMORPH_OPENCL_EMULATABLE_H
