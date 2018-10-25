/** OpenCL Back-End **/

#include "../../metamorph-backends/opencl-backend/mm_opencl_backend.h"

#if defined(__OPENCLCC__) || defined(__cplusplus)
extern "C" {
#endif

extern cl_context meta_context;
extern cl_command_queue meta_queue;
extern cl_device_id meta_device;

//Warning, none of these variables are threadsafe, so only one thread should ever
// perform the one-time scan!!
cl_uint num_platforms, num_devices;
cl_platform_id * platforms = NULL;
cl_device_id * devices = NULL;

typedef struct metaOpenCLStackNode {
	metaOpenCLStackFrame frame;
	struct metaOpenCLStackNode * next;
} metaOpenCLStackNode;

metaOpenCLStackNode * CLStack = NULL;

//Returns the first id in the device array of the desired device,
// or -1 if no such device is present.
int metaOpenCLGetDeviceID(char * desired, cl_device_id * devices,
		int numDevices) {
	char buff[128];
	int i;
	for (i = 0; i < numDevices; i++) {
		clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 128, (void *) buff, NULL);
		//printf("%s\n", buff);
		if (strcmp(desired, buff) == 0)
			return i;
	}
	return -1;
}

//Basic one-time scan of all platforms for all devices.
//This is not threadsafe, and I'm not sure we'll ever make it safe.
//If we need to though, it would likely be sufficient to CAS the num_platforms int or platforms pointer
// first to a hazard flag (like -1) to claim it, then to the actual pointer.
void metaOpenCLQueryDevices() {
	int i;
	num_platforms = 0, num_devices = 0;
	cl_uint temp_uint, temp_uint2;
	if (clGetPlatformIDs(0, NULL, &num_platforms) != CL_SUCCESS)
		printf("Failed to query platform count!\n");
	printf("Number of OpenCL Platforms: %d\n", num_platforms);

	platforms = (cl_platform_id *) malloc(
			sizeof(cl_platform_id) * (num_platforms + 1));

	if (clGetPlatformIDs(num_platforms, &platforms[0], NULL) != CL_SUCCESS)
		printf("Failed to get platform IDs\n");

	for (i = 0; i < num_platforms; i++) {
		temp_uint = 0;
		fprintf(stderr,
				"OCL DEBUG: clGetDeviceIDs Count query on platform[%d] has address[%x]!\n",
				i, &temp_uint);
		if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL,
				&temp_uint) != CL_SUCCESS)
			printf("Failed to query device count on platform %d!\n", i);
		num_devices += temp_uint;
	}
	printf("Number of Devices: %d\n", num_devices);

	devices = (cl_device_id *) malloc(sizeof(cl_device_id) * (num_devices + 1));
	temp_uint = 0;
	for (i = 0; i < num_platforms; i++) {
		fprintf(stderr,
				"OCL DEBUG: clGetDeviceIDs IDs query on platform[%d] has addresses[%x][%x]!\n",
				i, &devices[temp_uint], &temp_uint2);
		if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices,
				&devices[temp_uint], &temp_uint2) != CL_SUCCESS)
			printf("Failed to query device IDs on platform %d!\n", i);
		temp_uint += temp_uint2;
		temp_uint2 = 0;
	}

	//TODO figure out somewhere else to put this, like a terminating callback or something...
	//free(devices);
	//free(platforms);
}

size_t metaOpenCLLoadProgramSource(const char *filename, const char **progSrc) {
	FILE *f = fopen(filename, "r");
	fseek(f, 0, SEEK_END);
	size_t len = (size_t) ftell(f);
	*progSrc = (const char *) malloc(sizeof(char) * len);
	rewind(f);
	fread((void *) *progSrc, len, 1, f);
	fclose(f);
	return len;
}

cl_int metaOpenCLBuildSingleKernelProgram(metaOpenCLStackFrame * frame, cl_program prog, char const * prog_args) {
	if (prog_args == NULL) prog_args = "";
	cl_int ret = CL_SUCCESS;
	char * args = NULL;
	//FIXME Do these OpenCL build strings actually need "-D OPENCL" when building for WITH_INTELFPGA?
	if (getenv("METAMORPH_MODE") != NULL) {
		if (strcmp(getenv("METAMORPH_MODE"), "OpenCL") == 0) {
			size_t needed =
					snprintf(NULL, 0,
							"-I . -D TRANSPOSE_TILE_DIM=(%d) -D TRANSPOSE_TILE_BLOCK_ROWS=(%d) %s ",
							TRANSPOSE_TILE_DIM, TRANSPOSE_TILE_BLOCK_ROWS, prog_args);
			args = (char *) malloc(needed);
			snprintf(args, needed,
					"-I . -D TRANSPOSE_TILE_DIM=(%d) -D TRANSPOSE_TILE_BLOCK_ROWS=(%d) %s ",
					TRANSPOSE_TILE_DIM, TRANSPOSE_TILE_BLOCK_ROWS, prog_args);
			ret |= clBuildProgram(prog, 1,
					&(frame->device), args, NULL, NULL);
		} else if (strcmp(getenv("METAMORPH_MODE"), "OpenCL_DEBUG") == 0) {
			size_t needed =
					snprintf(NULL, 0,
							"-I . -D TRANSPOSE_TILE_DIM=(%d) -D TRANSPOSE_TILE_BLOCK_ROWS=(%d) -g -cl-opt-disable %s ",
							TRANSPOSE_TILE_DIM, TRANSPOSE_TILE_BLOCK_ROWS, prog_args);
			args = (char *) malloc(needed);
			snprintf(args, needed,
					"-I . -D TRANSPOSE_TILE_DIM=(%d) -D TRANSPOSE_TILE_BLOCK_ROWS=(%d) -g -cl-opt-disable %s ",
					TRANSPOSE_TILE_DIM, TRANSPOSE_TILE_BLOCK_ROWS, prog_args);
		}
	} else {
		//Do the same as if METAMORPH_MODE was set as OpenCL
		size_t needed =
				snprintf(NULL, 0,
						"-I . -D TRANSPOSE_TILE_DIM=(%d) -D TRANSPOSE_TILE_BLOCK_ROWS=(%d) %s ",
						TRANSPOSE_TILE_DIM, TRANSPOSE_TILE_BLOCK_ROWS, prog_args);
		args = (char *) malloc(needed);
		snprintf(args, needed,
				"-I . -D TRANSPOSE_TILE_DIM=(%d) -D TRANSPOSE_TILE_BLOCK_ROWS=(%d) %s ",
				TRANSPOSE_TILE_DIM, TRANSPOSE_TILE_BLOCK_ROWS, prog_args);
		//	ret |= clBuildProgram(frame->program_opencl_core, 1, &(frame->device), args, NULL, NULL);

	}
	ret |= clBuildProgram(prog, 1, &(frame->device), args,
			NULL, NULL);
	//Let us know if there's any errors in the build process
	if (ret != CL_SUCCESS) {
		fprintf(stderr, "Error in clBuildProgram: %d!\n", ret);
		ret = CL_SUCCESS;
	}
	//Stub to get build log
	size_t logsize = 0;
	clGetProgramBuildInfo(prog, frame->device,
			CL_PROGRAM_BUILD_LOG, 0, NULL, &logsize);
	char * log = (char *) malloc(sizeof(char) * (logsize + 1));
	clGetProgramBuildInfo(prog, frame->device,
			CL_PROGRAM_BUILD_LOG, logsize, log, NULL);
	if (logsize > 2) fprintf(stderr, "CL_PROGRAM_BUILD_LOG:\n%s", log);
	free(log);
	return ret;
}

cl_int metaOpenCLBuildProgram(metaOpenCLStackFrame * frame) {
	cl_int ret = CL_SUCCESS;
#if defined(WITH_INTELFPGA) && defined(OPENCL_SINGLE_KERNEL_PROGS)
	//will need a separate buffer for each kernel

#define ENSURE_SRC(name) if (frame->metaCLbinLen_##name == 0) \
	frame->metaCLbinLen_##name = metaOpenCLLoadProgramSource("lib/mm_opencl_intelfpga_backend_"##name".aocx", &(frame->metaCLbin_##name)); \
	} \
	frame->program_##name = clCreateProgramWithBinary(frame->context, 1, &(frame->device), &(frame->metaCLbinLen_##name), (const unsigned char **) &(frame->metaCLbin_##name), NULL, NULL);

	ENSURE_SRC(reduce_db);
	ENSURE_SRC(reduce_fl);
	ENSURE_SRC(reduce_ul);
	ENSURE_SRC(reduce_in);
	ENSURE_SRC(reduce_ui);
	ENSURE_SRC(dotProd_db);
	ENSURE_SRC(dotProd_fl);
	ENSURE_SRC(dotProd_ul);
	ENSURE_SRC(dotProd_in);
	ENSURE_SRC(dotProd_ui);
	ENSURE_SRC(transpose_2d_face_db);
	ENSURE_SRC(transpose_2d_face_fl);
	ENSURE_SRC(transpose_2d_face_ul);
	ENSURE_SRC(transpose_2d_face_in);
	ENSURE_SRC(transpose_2d_face_ui);
	ENSURE_SRC(pack_2d_face_db);
	ENSURE_SRC(pack_2d_face_fl);
	ENSURE_SRC(pack_2d_face_ul);
	ENSURE_SRC(pack_2d_face_in);
	ENSURE_SRC(pack_2d_face_ui);
	ENSURE_SRC(unpack_2d_face_db);
	ENSURE_SRC(unpack_2d_face_fl);
	ENSURE_SRC(unpack_2d_face_ul);
	ENSURE_SRC(unpack_2d_face_in);
	ENSURE_SRC(unpack_2d_face_ui);
	ENSURE_SRC(stencil_3d7p_db);
	ENSURE_SRC(stencil_3d7p_fl);
	ENSURE_SRC(stencil_3d7p_ul);
	ENSURE_SRC(stencil_3d7p_in);
	ENSURE_SRC(stencil_3d7p_ui);
	ENSURE_SRC(csr_db);
	ENSURE_SRC(csr_fl);
	ENSURE_SRC(csr_ul);
	ENSURE_SRC(csr_in);
	ENSURE_SRC(csr_ui);
//	ENSURE_SRC(crc_db);
//	ENSURE_SRC(crc_fl);
//	ENSURE_SRC(crc_ul);
//	ENSURE_SRC(crc_in);
	ENSURE_SRC(crc_ui);
#undef ENSURE_SRC
#elif defined(WITH_INTELFPGA) && !defined(OPENCL_SINGLE_KERNEL_PROGS)
	if (frame->metaCLProgLen == 0) {
		frame->metaCLProgLen = metaOpenCLLoadProgramSource("lib/mm_opencl_intelfpga_backend.aocx", &(frame->metaCLProgSrc));
	}
	frame->program_opencl_core=clCreateProgramWithBinary(frame->context,1,&(frame->device), &(frame->metaCLProgLen),(const unsigned char**) &(frame->metaCLProgSrc),NULL,NULL);
#elif !defined(WITH_INTELFPGA) && defined(OPENCL_SINGLE_KERNEL_PROGS)
	if (frame->metaCLProgLen == 0) {
		frame->metaCLProgLen = metaOpenCLLoadProgramSource("mm_opencl_backend.cl",&(frame->metaCLProgSrc));
	}
	//They use the same source with different defines
	frame->program_reduce_db = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
	frame->program_reduce_fl = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
	frame->program_reduce_ul = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
	frame->program_reduce_in = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
	frame->program_reduce_ui = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
	frame->program_dotProd_db = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
	frame->program_dotProd_fl = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
	frame->program_dotProd_ul = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
	frame->program_dotProd_in = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
	frame->program_dotProd_ui = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
	frame->program_transpose_2d_face_db = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
	frame->program_transpose_2d_face_fl = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
	frame->program_transpose_2d_face_ul = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
	frame->program_transpose_2d_face_in = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
	frame->program_transpose_2d_face_ui = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
	frame->program_pack_2d_face_db = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
	frame->program_pack_2d_face_fl = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
	frame->program_pack_2d_face_ul = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
	frame->program_pack_2d_face_in = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
	frame->program_pack_2d_face_ui = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
	frame->program_unpack_2d_face_db = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
	frame->program_unpack_2d_face_fl = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
	frame->program_unpack_2d_face_ul = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
	frame->program_unpack_2d_face_in = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
	frame->program_unpack_2d_face_ui = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
	frame->program_stencil_3d7p_db = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
	frame->program_stencil_3d7p_fl = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
	frame->program_stencil_3d7p_ul = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
	frame->program_stencil_3d7p_in = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
	frame->program_stencil_3d7p_ui = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
	frame->program_csr_db = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
	frame->program_csr_fl = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
	frame->program_csr_ul = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
	frame->program_csr_in = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
	frame->program_csr_ui = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
//	frame->program_crc_db = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
//	frame->program_crc_fl = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
//	frame->program_crc_ul = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
//	frame->program_crc_in = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
	frame->program_crc_ui = clCreateProgramWithSource(frame->context, 1,&(frame->metaCLProgSrc), &(frame->metaCLProgLen), NULL);
//TODO Implement (or can this be the same as OpenCL without IntelFPGA, since it's just reading the source and the defines come in during build
#else
	if (frame->metaCLProgLen == 0) {
		frame->metaCLProgLen = metaOpenCLLoadProgramSource("mm_opencl_backend.cl",
				&(frame->metaCLProgSrc));
	}
	frame->program_opencl_core = clCreateProgramWithSource(frame->context, 1,
			&(frame->metaCLProgSrc), &(frame->metaCLProgLen), &ret);
#endif

//TODO Support OPENCL_SINGLE_KERNEL_PROGS
#ifndef OPENCL_SINGLE_KERNEL_PROGS
	ret |= metaOpenCLBuildSingleKernelProgram(frame, frame->program_opencl_core, NULL);
	frame->kernel_reduce_db = clCreateKernel(frame->program_opencl_core,
			"kernel_reduce_db", &ret);
	frame->kernel_reduce_fl = clCreateKernel(frame->program_opencl_core,
			"kernel_reduce_fl", &ret);
	frame->kernel_reduce_ul = clCreateKernel(frame->program_opencl_core,
			"kernel_reduce_ul", &ret);
	frame->kernel_reduce_in = clCreateKernel(frame->program_opencl_core,
			"kernel_reduce_in", &ret);
	frame->kernel_reduce_ui = clCreateKernel(frame->program_opencl_core,
			"kernel_reduce_ui", &ret);
	frame->kernel_dotProd_db = clCreateKernel(frame->program_opencl_core,
			"kernel_dotProd_db", &ret);
	frame->kernel_dotProd_fl = clCreateKernel(frame->program_opencl_core,
			"kernel_dotProd_fl", &ret);
	frame->kernel_dotProd_ul = clCreateKernel(frame->program_opencl_core,
			"kernel_dotProd_ul", &ret);
	frame->kernel_dotProd_in = clCreateKernel(frame->program_opencl_core,
			"kernel_dotProd_in", &ret);
	frame->kernel_dotProd_ui = clCreateKernel(frame->program_opencl_core,
			"kernel_dotProd_ui", &ret);
	frame->kernel_transpose_2d_face_db = clCreateKernel(
			frame->program_opencl_core, "kernel_transpose_2d_face_db", &ret);
	frame->kernel_transpose_2d_face_fl = clCreateKernel(
			frame->program_opencl_core, "kernel_transpose_2d_face_fl", &ret);
	frame->kernel_transpose_2d_face_ul = clCreateKernel(
			frame->program_opencl_core, "kernel_transpose_2d_face_ul", &ret);
	frame->kernel_transpose_2d_face_in = clCreateKernel(
			frame->program_opencl_core, "kernel_transpose_2d_face_in", &ret);
	frame->kernel_transpose_2d_face_ui = clCreateKernel(
			frame->program_opencl_core, "kernel_transpose_2d_face_ui", &ret);
	frame->kernel_pack_2d_face_db = clCreateKernel(frame->program_opencl_core,
			"kernel_pack_2d_face_db", &ret);
	frame->kernel_pack_2d_face_fl = clCreateKernel(frame->program_opencl_core,
			"kernel_pack_2d_face_fl", &ret);
	frame->kernel_pack_2d_face_ul = clCreateKernel(frame->program_opencl_core,
			"kernel_pack_2d_face_ul", &ret);
	frame->kernel_pack_2d_face_in = clCreateKernel(frame->program_opencl_core,
			"kernel_pack_2d_face_in", &ret);
	frame->kernel_pack_2d_face_ui = clCreateKernel(frame->program_opencl_core,
			"kernel_pack_2d_face_ui", &ret);
	frame->kernel_unpack_2d_face_db = clCreateKernel(frame->program_opencl_core,
			"kernel_unpack_2d_face_db", &ret);
	frame->kernel_unpack_2d_face_fl = clCreateKernel(frame->program_opencl_core,
			"kernel_unpack_2d_face_fl", &ret);
	frame->kernel_unpack_2d_face_ul = clCreateKernel(frame->program_opencl_core,
			"kernel_unpack_2d_face_ul", &ret);
	frame->kernel_unpack_2d_face_in = clCreateKernel(frame->program_opencl_core,
			"kernel_unpack_2d_face_in", &ret);
	frame->kernel_unpack_2d_face_ui = clCreateKernel(frame->program_opencl_core,
			"kernel_unpack_2d_face_ui", &ret);
	frame->kernel_stencil_3d7p_db = clCreateKernel(frame->program_opencl_core,
			"kernel_stencil_3d7p_db", &ret);
	frame->kernel_stencil_3d7p_fl = clCreateKernel(frame->program_opencl_core,
			"kernel_stencil_3d7p_fl", &ret);
	frame->kernel_stencil_3d7p_ul = clCreateKernel(frame->program_opencl_core,
			"kernel_stencil_3d7p_ul", &ret);
	frame->kernel_stencil_3d7p_in = clCreateKernel(frame->program_opencl_core,
			"kernel_stencil_3d7p_in", &ret);
	frame->kernel_stencil_3d7p_ui = clCreateKernel(frame->program_opencl_core,
			"kernel_stencil_3d7p_ui", &ret);
	frame->kernel_csr_fl = clCreateKernel(frame->program_opencl_core,
			"kernel_csr_db", &ret);
	frame->kernel_csr_fl = clCreateKernel(frame->program_opencl_core,
			"kernel_csr_fl", &ret);
	frame->kernel_csr_fl = clCreateKernel(frame->program_opencl_core,
			"kernel_csr_ul", &ret);
	frame->kernel_csr_fl = clCreateKernel(frame->program_opencl_core,
			"kernel_csr_in", &ret);
	frame->kernel_csr_fl = clCreateKernel(frame->program_opencl_core,
			"kernel_csr_ui", &ret);
//	frame->kernel_crc_db = clCreateKernel(frame->program_opencl_core,
//			"kernel_crc_db", &ret);
//	frame->kernel_crc_fl = clCreateKernel(frame->program_opencl_core,
//			"kernel_crc_fl", &ret);
//	frame->kernel_crc_ul = clCreateKernel(frame->program_opencl_core,
//			"kernel_crc_ul", &ret);
//	frame->kernel_crc_in = clCreateKernel(frame->program_opencl_core,
//			"kernel_crc_in", &ret);
	frame->kernel_crc_ui = clCreateKernel(frame->program_opencl_core,
			"kernel_crc_ui", &ret);
#else
#define BUILD_PROG_AND_KERNEL(name, opts) ret |= metaOpenCLBuildSingleKernelProgram(frame, frame->program_##name, opts); \
	frame->kernel_##name = clCreateKernel(frame->program_##name, "kernel_"#name, &ret); \
        if (ret != CL_SUCCESS) fprintf(stderr, "Error in clCreateKernel for kernel_"#name" %d\n", ret);
//	ret |= metaOpenCLBuildSingleKernelProgram(frame, frame->program_reduce_db, "-D DOUBLE -D KERNEL_REDUCE");
//	frame->kernel_reduce_db = clCreateKernel(frame->program_opencl_reduce_db, "kernel_reduce_db", &ret);
	BUILD_PROG_AND_KERNEL(reduce_db, "-D SINGLE_KERNEL_PROGS -D DOUBLE -D KERNEL_REDUCE")
	BUILD_PROG_AND_KERNEL(reduce_fl, "-D SINGLE_KERNEL_PROGS -D FLOAT -D KERNEL_REDUCE")
	BUILD_PROG_AND_KERNEL(reduce_ul, "-D SINGLE_KERNEL_PROGS -D UNSIGNED_LONG -D KERNEL_REDUCE")
	BUILD_PROG_AND_KERNEL(reduce_in, "-D SINGLE_KERNEL_PROGS -D INTEGER -D KERNEL_REDUCE")
	BUILD_PROG_AND_KERNEL(reduce_ui, "-D SINGLE_KERNEL_PROGS -D UNSIGNED_INTEGER -D KERNEL_REDUCE")
	BUILD_PROG_AND_KERNEL(dotProd_db, "-D SINGLE_KERNEL_PROGS -D DOUBLE -D KERNEL_DOT_PROD")
	BUILD_PROG_AND_KERNEL(dotProd_fl, "-D SINGLE_KERNEL_PROGS -D FLOAT -D KERNEL_DOT_PROD")
	BUILD_PROG_AND_KERNEL(dotProd_ul, "-D SINGLE_KERNEL_PROGS -D UNSIGNED_LONG -D KERNEL_DOT_PROD")
	BUILD_PROG_AND_KERNEL(dotProd_in, "-D SINGLE_KERNEL_PROGS -D INTEGER -D KERNEL_DOT_PROD")
	BUILD_PROG_AND_KERNEL(dotProd_ui, "-D SINGLE_KERNEL_PROGS -D UNSIGNED_INTEGER -D KERNEL_DOT_PROD")
	BUILD_PROG_AND_KERNEL(transpose_2d_face_db, "-D SINGLE_KERNEL_PROGS -D DOUBLE -D KERNEL_TRANSPOSE")
	BUILD_PROG_AND_KERNEL(transpose_2d_face_fl, "-D SINGLE_KERNEL_PROGS -D FLOAT -D KERNEL_TRANSPOSE")
	BUILD_PROG_AND_KERNEL(transpose_2d_face_ul, "-D SINGLE_KERNEL_PROGS -D UNSIGNED_LONG -D KERNEL_TRANSPOSE")
	BUILD_PROG_AND_KERNEL(transpose_2d_face_in, "-D SINGLE_KERNEL_PROGS -D INTEGER -D KERNEL_TRANSPOSE")
	BUILD_PROG_AND_KERNEL(transpose_2d_face_ui, "-D SINGLE_KERNEL_PROGS -D UNSIGNED_INTEGER -D KERNEL_TRANSPOSE")
	BUILD_PROG_AND_KERNEL(pack_2d_face_db, "-D SINGLE_KERNEL_PROGS -D DOUBLE -D KERNEL_PACK")
	BUILD_PROG_AND_KERNEL(pack_2d_face_fl, "-D SINGLE_KERNEL_PROGS -D FLOAT -D KERNEL_PACK")
	BUILD_PROG_AND_KERNEL(pack_2d_face_ul, "-D SINGLE_KERNEL_PROGS -D UNSIGNED_LONG -D KERNEL_PACK")
	BUILD_PROG_AND_KERNEL(pack_2d_face_in, "-D SINGLE_KERNEL_PROGS -D INTEGER -D KERNEL_PACK")
	BUILD_PROG_AND_KERNEL(pack_2d_face_ui, "-D SINGLE_KERNEL_PROGS -D UNSIGNED_INTEGER -D KERNEL_PACK")
	BUILD_PROG_AND_KERNEL(unpack_2d_face_db, "-D SINGLE_KERNEL_PROGS -D DOUBLE -D KERNEL_UNPACK")
	BUILD_PROG_AND_KERNEL(unpack_2d_face_fl, "-D SINGLE_KERNEL_PROGS -D FLOAT -D KERNEL_UNPACK")
	BUILD_PROG_AND_KERNEL(unpack_2d_face_ul, "-D SINGLE_KERNEL_PROGS -D UNSIGNED_LONG -D KERNEL_UNPACK")
	BUILD_PROG_AND_KERNEL(unpack_2d_face_in, "-D SINGLE_KERNEL_PROGS -D INTEGER -D KERNEL_UNPACK")
	BUILD_PROG_AND_KERNEL(unpack_2d_face_ui, "-D SINGLE_KERNEL_PROGS -D UNSIGNED_INTEGER -D KERNEL_UNPACK")
	BUILD_PROG_AND_KERNEL(stencil_3d7p_db, "-D SINGLE_KERNEL_PROGS -D DOUBLE -D KERNEL_STENCIL")
	BUILD_PROG_AND_KERNEL(stencil_3d7p_fl, "-D SINGLE_KERNEL_PROGS -D FLOAT -D KERNEL_STENCIL")
	BUILD_PROG_AND_KERNEL(stencil_3d7p_ul, "-D SINGLE_KERNEL_PROGS -D UNSIGNED_LONG -D KERNEL_STENCIL")
	BUILD_PROG_AND_KERNEL(stencil_3d7p_in, "-D SINGLE_KERNEL_PROGS -D INTEGER -D KERNEL_STENCIL")
	BUILD_PROG_AND_KERNEL(stencil_3d7p_ui, "-D SINGLE_KERNEL_PROGS -D UNSIGNED_INTEGER -D KERNEL_STENCIL")
	BUILD_PROG_AND_KERNEL(csr_db, "-D SINGLE_KERNEL_PROGS -D DOUBLE -D KERNEL_CSR")
	BUILD_PROG_AND_KERNEL(csr_fl, "-D SINGLE_KERNEL_PROGS -D FLOAT -D KERNEL_CSR")
	BUILD_PROG_AND_KERNEL(csr_ul, "-D SINGLE_KERNEL_PROGS -D UNSIGNED_LONG -D KERNEL_CSR")
	BUILD_PROG_AND_KERNEL(csr_in, "-D SINGLE_KERNEL_PROGS -D INTEGER -D KERNEL_CSR")
	BUILD_PROG_AND_KERNEL(csr_ui, "-D SINGLE_KERNEL_PROGS -D UNSIGNED_INTEGER -D KERNEL_CSR")
//	BUILD_PROG_AND_KERNEL(crc_db, "-D SINGLE_KERNEL_PROGS -D DOUBLE -D KERNEL_CRC")
//	BUILD_PROG_AND_KERNEL(crc_fl, "-D SINGLE_KERNEL_PROGS -D FLOAT -D KERNEL_CRC")
//	BUILD_PROG_AND_KERNEL(crc_ul, "-D SINGLE_KERNEL_PROGS -D UNSIGNED_LONG -D KERNEL_CRC")
//	BUILD_PROG_AND_KERNEL(crc_in, "-D SINGLE_KERNEL_PROGS -D INTEGER -D KERNEL_CRC")
	BUILD_PROG_AND_KERNEL(crc_ui, "-D SINGLE_KERNEL_PROGS -D UNSIGNED_INTEGER -D KERNEL_CRC")
#endif
        //Reinitialize OpenCL add-on modules
        meta_reinitialize_modules(module_implements_opencl);
}

//This should not be exposed to the user, just the top and pop functions built on top of it
//for thread-safety, returns a new metaOpenCLStackNode, which is a direct copy
// of the top at some point in time.
// this way, the user can still use top, without having to manage hazard pointers themselves
//ASSUME HAZARD POINTERS ARE ALREADY SET FOR t BY THE CALLING METHOD
void copyStackNodeToFrame(metaOpenCLStackNode * t,
		metaOpenCLStackFrame ** frame) {

	//From here out, we have hazards
	//Copy all the parameters - REALLY HAZARDOUS

	*(*frame) = t->frame;

	//Top-level context info
/*
	(*frame)->platform = t->frame.platform;
	(*frame)->device = t->frame.device;
	(*frame)->context = t->frame.context;
	(*frame)->queue = t->frame.queue;
#if !defined(OPENCL_SINGLE_KERNEL_PROGS) || !defined(WITH_INTELFPGA)
	//copy just the regular buffer and length
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
#else
	//TODO copy all the buffers and lengths
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
	(*frame)->metaCLProgSrc = t->frame.metaCLProgSrc;
	(*frame)->metaCLProgLen = t->frame.metaCLProgLen;
#endif
#ifdef (OPENCL_SINGLE_KERNEL_PROGS)
	(*frame)->program_opencl_core = t->frame.program_opencl_core;
	//TODO copy just the regular buffer and length
#else
//TODO copy extra cl_programs
#endif

//TODO this should be acheivable with a memcpy
	//Kernels
	(*frame)->kernel_reduce_db = t->frame.kernel_reduce_db;
	(*frame)->kernel_reduce_fl = t->frame.kernel_reduce_fl;
	(*frame)->kernel_reduce_ul = t->frame.kernel_reduce_ul;
	(*frame)->kernel_reduce_in = t->frame.kernel_reduce_in;
	(*frame)->kernel_reduce_ui = t->frame.kernel_reduce_ui;
	(*frame)->kernel_dotProd_db = t->frame.kernel_dotProd_db;
	(*frame)->kernel_dotProd_fl = t->frame.kernel_dotProd_fl;
	(*frame)->kernel_dotProd_ul = t->frame.kernel_dotProd_ul;
	(*frame)->kernel_dotProd_in = t->frame.kernel_dotProd_in;
	(*frame)->kernel_dotProd_ui = t->frame.kernel_dotProd_ui;
	(*frame)->kernel_transpose_2d_face_db =
			t->frame.kernel_transpose_2d_face_db;
	(*frame)->kernel_transpose_2d_face_fl =
			t->frame.kernel_transpose_2d_face_fl;
	(*frame)->kernel_transpose_2d_face_ul =
			t->frame.kernel_transpose_2d_face_ul;
	(*frame)->kernel_transpose_2d_face_in =
			t->frame.kernel_transpose_2d_face_in;
	(*frame)->kernel_transpose_2d_face_ui =
			t->frame.kernel_transpose_2d_face_ui;
	(*frame)->kernel_pack_2d_face_db = t->frame.kernel_pack_2d_face_db;
	(*frame)->kernel_pack_2d_face_fl = t->frame.kernel_pack_2d_face_fl;
	(*frame)->kernel_pack_2d_face_ul = t->frame.kernel_pack_2d_face_ul;
	(*frame)->kernel_pack_2d_face_in = t->frame.kernel_pack_2d_face_in;
	(*frame)->kernel_pack_2d_face_ui = t->frame.kernel_pack_2d_face_ui;
	(*frame)->kernel_unpack_2d_face_db = t->frame.kernel_unpack_2d_face_db;
	(*frame)->kernel_unpack_2d_face_fl = t->frame.kernel_unpack_2d_face_fl;
	(*frame)->kernel_unpack_2d_face_ul = t->frame.kernel_unpack_2d_face_ul;
	(*frame)->kernel_unpack_2d_face_in = t->frame.kernel_unpack_2d_face_in;
	(*frame)->kernel_unpack_2d_face_ui = t->frame.kernel_unpack_2d_face_ui;
	(*frame)->kernel_stencil_3d7p_db = t->frame.kernel_stencil_3d7p_db;
	(*frame)->kernel_stencil_3d7p_fl = t->frame.kernel_stencil_3d7p_fl;
	(*frame)->kernel_stencil_3d7p_ul = t->frame.kernel_stencil_3d7p_ul;
	(*frame)->kernel_stencil_3d7p_in = t->frame.kernel_stencil_3d7p_in;
	(*frame)->kernel_stencil_3d7p_ui = t->frame.kernel_stencil_3d7p_ui;
	(*frame)->kernel_csr_db = t->frame.kernel_csr_db;
	(*frame)->kernel_csr_fl = t->frame.kernel_csr_fl;
	(*frame)->kernel_csr_ul = t->frame.kernel_csr_ul;
	(*frame)->kernel_csr_in = t->frame.kernel_csr_in;
	(*frame)->kernel_csr_ui = t->frame.kernel_csr_ui;
	(*frame)->kernel_crc_db = t->frame.kernel_crc_db;
	(*frame)->kernel_crc_fl = t->frame.kernel_crc_fl;
	(*frame)->kernel_crc_ul = t->frame.kernel_crc_ul;
	(*frame)->kernel_crc_in = t->frame.kernel_crc_in;
	(*frame)->kernel_crc_ui = t->frame.kernel_crc_ui;

	//Internal buffers
	(*frame)->constant_face_size = t->frame.constant_face_size;
	(*frame)->constant_face_stride = t->frame.constant_face_stride;
	(*frame)->constant_face_child_size = t->frame.constant_face_child_size;
	(*frame)->red_loc = t->frame.red_loc;
	//This should be the end of the hazards;
*/
}

//ASSUME HAZARD POINTERS ARE ALREADY SET FOR node BY THE CALLING METHOD
void copyStackFrameToNode(metaOpenCLStackFrame * f,
		metaOpenCLStackNode ** node) {

	(*node)->frame = (*f);
	//Top-level context info
/*
	(*node)->frame.platform = f->platform;
	(*node)->frame.device = f->device;
	(*node)->frame.context = f->context;
	(*node)->frame.queue = f->queue;
//TODO Support OPENCL_SINGLE_KERNEL_PROGS
	(*node)->frame.program_opencl_core = f->program_opencl_core;

//TODO use a memcpy for the rest
	//Kernels
	(*node)->frame.kernel_reduce_db = f->kernel_reduce_db;
	(*node)->frame.kernel_reduce_fl = f->kernel_reduce_fl;
	(*node)->frame.kernel_reduce_ul = f->kernel_reduce_ul;
	(*node)->frame.kernel_reduce_in = f->kernel_reduce_in;
	(*node)->frame.kernel_reduce_ui = f->kernel_reduce_ui;
	(*node)->frame.kernel_dotProd_db = f->kernel_dotProd_db;
	(*node)->frame.kernel_dotProd_fl = f->kernel_dotProd_fl;
	(*node)->frame.kernel_dotProd_ul = f->kernel_dotProd_ul;
	(*node)->frame.kernel_dotProd_in = f->kernel_dotProd_in;
	(*node)->frame.kernel_dotProd_ui = f->kernel_dotProd_ui;
	(*node)->frame.kernel_transpose_2d_face_db = f->kernel_transpose_2d_face_db;
	(*node)->frame.kernel_transpose_2d_face_fl = f->kernel_transpose_2d_face_fl;
	(*node)->frame.kernel_transpose_2d_face_ul = f->kernel_transpose_2d_face_ul;
	(*node)->frame.kernel_transpose_2d_face_in = f->kernel_transpose_2d_face_in;
	(*node)->frame.kernel_transpose_2d_face_ui = f->kernel_transpose_2d_face_ui;
	(*node)->frame.kernel_pack_2d_face_db = f->kernel_pack_2d_face_db;
	(*node)->frame.kernel_pack_2d_face_fl = f->kernel_pack_2d_face_fl;
	(*node)->frame.kernel_pack_2d_face_ul = f->kernel_pack_2d_face_ul;
	(*node)->frame.kernel_pack_2d_face_in = f->kernel_pack_2d_face_in;
	(*node)->frame.kernel_pack_2d_face_ui = f->kernel_pack_2d_face_ui;
	(*node)->frame.kernel_unpack_2d_face_db = f->kernel_unpack_2d_face_db;
	(*node)->frame.kernel_unpack_2d_face_fl = f->kernel_unpack_2d_face_fl;
	(*node)->frame.kernel_unpack_2d_face_ul = f->kernel_unpack_2d_face_ul;
	(*node)->frame.kernel_unpack_2d_face_in = f->kernel_unpack_2d_face_in;
	(*node)->frame.kernel_unpack_2d_face_ui = f->kernel_unpack_2d_face_ui;
	(*node)->frame.kernel_stencil_3d7p_db = f->kernel_stencil_3d7p_db;
	(*node)->frame.kernel_stencil_3d7p_fl = f->kernel_stencil_3d7p_fl;
	(*node)->frame.kernel_stencil_3d7p_ul = f->kernel_stencil_3d7p_ul;
	(*node)->frame.kernel_stencil_3d7p_in = f->kernel_stencil_3d7p_in;
	(*node)->frame.kernel_stencil_3d7p_ui = f->kernel_stencil_3d7p_ui;
	(*node)->frame.kernel_csr_db = f->kernel_csr_db;
	(*node)->frame.kernel_csr_fl = f->kernel_csr_fl;
	(*node)->frame.kernel_csr_ul = f->kernel_csr_ul;
	(*node)->frame.kernel_csr_in = f->kernel_csr_in;
	(*node)->frame.kernel_csr_ui = f->kernel_csr_ui;
	(*node)->frame.kernel_crc_db = f->kernel_crc_db;
	(*node)->frame.kernel_crc_fl = f->kernel_crc_fl;
	(*node)->frame.kernel_crc_ul = f->kernel_crc_ul;
	(*node)->frame.kernel_crc_in = f->kernel_crc_in;
	(*node)->frame.kernel_crc_ui = f->kernel_crc_ui;

	//Internal Buffers
	(*node)->frame.constant_face_size = f->constant_face_size;
	(*node)->frame.constant_face_stride = f->constant_face_stride;
	(*node)->frame.constant_face_child_size = f->constant_face_child_size;
	(*node)->frame.red_loc = f->red_loc;
*/
}

//For this push to be both exposed and safe from hazards, it must make a copy of the frame
// such that the user never has access to the pointer to the frame that's actually on the shared stack
// object
//This CAS-based push should be threadsafe
//WARNING assumes all parameters of metaOpenCLStackNode are set, except "next"
void metaOpenCLPushStackFrame(metaOpenCLStackFrame * frame) {
	//copy the frame, this is still the thread-private "allocated" state
	metaOpenCLStackNode * newNode = (metaOpenCLStackNode *) calloc(1,
			sizeof(metaOpenCLStackNode));
	copyStackFrameToNode(frame, &newNode);

	//grab the old node
	metaOpenCLStackNode * old = CLStack;
	//I think this is where hazards start..
	newNode->next = old;
	CLStack = newNode;

}

metaOpenCLStackFrame * metaOpenCLTopStackFrame() {
	metaOpenCLStackFrame * frame = (metaOpenCLStackFrame *) calloc(1,
			sizeof(metaOpenCLStackFrame));
	metaOpenCLStackNode * t = CLStack; //Hazards start
	//so set a hazard pointer
	//then copy the node
	copyStackNodeToFrame(t, &frame);
	//release the hazard pointer
	//so hazards are over, return the copy and exit
	return (frame);
}

metaOpenCLStackFrame * metaOpenCLPopStackFrame() {
	metaOpenCLStackFrame * frame = (metaOpenCLStackFrame *) calloc(1,
			sizeof(metaOpenCLStackFrame));
	metaOpenCLStackNode * t = CLStack; //Hazards start
	//so set a hazard pointer
	//then copy the node
	copyStackNodeToFrame(t, &frame);
	//and pull the node off the stack
	CLStack = t->next;
	//Do something with the memory
	//free(t);
	//release the hazard pointer
	//so hazards are over, return the copy and exit
	return (frame);
}

a_int meta_get_state_OpenCL(cl_platform_id * platform, cl_device_id * device,
		cl_context * context, cl_command_queue * queue) {
	metaOpenCLStackFrame * frame = metaOpenCLTopStackFrame();
	if (platform != NULL)
		*platform = frame->platform;
	if (device != NULL)
		*device = frame->device;
	if (context != NULL)
		*context = frame->context;
	if (queue != NULL)
		*queue = frame->queue;
}

a_int meta_set_state_OpenCL(cl_platform_id platform, cl_device_id device,
		cl_context context, cl_command_queue queue) {
	if (platform == NULL || device == NULL || context == NULL
			|| queue == NULL) {
		fprintf(stderr,
				"Error: meta_set_state_OpenCL requires a full frame specification!\n");
		return -1;
	}
	//Make a frame
	metaOpenCLStackFrame * frame = (metaOpenCLStackFrame *) calloc(1, 
			sizeof(metaOpenCLStackFrame));

	//copy the info into the frame
	frame->platform = platform;
	frame->device = device;
	frame->context = context;
	frame->queue = queue;

	//add the metamorph programs and kernels to it
	metaOpenCLBuildProgram(frame);

	//add the extra buffers needed for pack/unpack
	int zero = 0;
	frame->constant_face_size = clCreateBuffer(frame->context, CL_MEM_READ_ONLY,
			sizeof(cl_int) * METAMORPH_FACE_MAX_DEPTH, NULL, NULL);
	frame->constant_face_stride = clCreateBuffer(frame->context,
			CL_MEM_READ_ONLY, sizeof(cl_int) * METAMORPH_FACE_MAX_DEPTH, NULL,
			NULL);
	frame->constant_face_child_size = clCreateBuffer(frame->context,
			CL_MEM_READ_ONLY, sizeof(cl_int) * METAMORPH_FACE_MAX_DEPTH, NULL,
			NULL);
	frame->red_loc = clCreateBuffer(frame->context,
			CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_int), &zero,
			NULL);

	//push it onto the stack
	meta_context = frame->context;
	meta_queue = frame->queue;
	meta_device = frame->device;
	metaOpenCLPushStackFrame(frame);
	free(frame);

}

cl_int metaOpenCLInitStackFrame(metaOpenCLStackFrame ** frame, cl_int device) {
	int zero = 0;
	cl_int ret = CL_SUCCESS;
	//First, make sure we've run the one-time query to initialize the device array.
	//TODO, fix synchronization on the one-time device query.
	if (platforms == NULL || devices == NULL || (((long) platforms) == -1)) {
		//try to perform the scan, else wait while somebody else finishes it.
		metaOpenCLQueryDevices();
	}

	//Hack to allow choose device to set mode to OpenCL, but still pick a default
	if (device == -1)
		return metaOpenCLInitStackFrameDefault(frame);

	*frame = (metaOpenCLStackFrame *) calloc(1, sizeof(metaOpenCLStackFrame));
	//TODO use the device array to do reverse lookup to figure out which platform to attach to the frame.
	//TODO ensure the device array has been initialized by somebody (even if I have to do it) before executing this block
	//TODO implement an intelligent catch for if the device number is out of range

	//copy the chosen device from the array to the new frame
	(*frame)->device = devices[device];
	//reverse lookup the device's platform and add it to the frame
	clGetDeviceInfo((*frame)->device, CL_DEVICE_PLATFORM,
			sizeof(cl_platform_id), &((*frame)->platform), NULL);
	//create the context and add it to the frame
	(*frame)->context = clCreateContext(NULL, 1, &((*frame)->device), NULL,
			NULL, NULL);
	(*frame)->queue = clCreateCommandQueue((*frame)->context, (*frame)->device,
			CL_QUEUE_PROFILING_ENABLE, NULL);
	metaOpenCLBuildProgram((*frame));
	// Add this debug string if needed: -g -s\"./mm_opencl_backend.cl\"
	//Allocate any internal buffers necessary for kernel functions
	(*frame)->constant_face_size = clCreateBuffer((*frame)->context,
			CL_MEM_READ_ONLY, sizeof(cl_int) * METAMORPH_FACE_MAX_DEPTH, NULL,
			NULL);
	(*frame)->constant_face_stride = clCreateBuffer((*frame)->context,
			CL_MEM_READ_ONLY, sizeof(cl_int) * METAMORPH_FACE_MAX_DEPTH, NULL,
			NULL);
	(*frame)->constant_face_child_size = clCreateBuffer((*frame)->context,
			CL_MEM_READ_ONLY, sizeof(cl_int) * METAMORPH_FACE_MAX_DEPTH, NULL,
			NULL);
	(*frame)->red_loc = clCreateBuffer((*frame)->context,
			CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_int), &zero,
			NULL);

}

//calls all the necessary CLRelease* calls for frame members
//DOES NOT:
//	pop any stack nodes
//	free any stack nodes
//	free program source
//	implement any thread safety, frames should always be thread private, only the stack should be shared, and all franes must be copied to or from stack nodes using the hazard-aware copy methods.
//	 (more specifically, copying a frame to a node doesn't need to be hazard-aware, as the node cannot be shared unless copied inside the hazard-aware metaOpenCLPushStackFrame. Pop, Top, and copyStackNodeToFrame are all hazard aware and provide a thread-private copy back to the caller.)
cl_int metaOpenCLDestroyStackFrame(metaOpenCLStackFrame * frame) {

	//Release Kernels
	clReleaseKernel(frame->kernel_reduce_db);
	clReleaseKernel(frame->kernel_reduce_fl);
	clReleaseKernel(frame->kernel_reduce_ul);
	clReleaseKernel(frame->kernel_reduce_in);
	clReleaseKernel(frame->kernel_reduce_ui);
	clReleaseKernel(frame->kernel_dotProd_db);
	clReleaseKernel(frame->kernel_dotProd_fl);
	clReleaseKernel(frame->kernel_dotProd_ul);
	clReleaseKernel(frame->kernel_dotProd_in);
	clReleaseKernel(frame->kernel_dotProd_ui);
	clReleaseKernel(frame->kernel_transpose_2d_face_db);
	clReleaseKernel(frame->kernel_transpose_2d_face_fl);
	clReleaseKernel(frame->kernel_transpose_2d_face_ul);
	clReleaseKernel(frame->kernel_transpose_2d_face_in);
	clReleaseKernel(frame->kernel_transpose_2d_face_ui);
	clReleaseKernel(frame->kernel_pack_2d_face_db);
	clReleaseKernel(frame->kernel_pack_2d_face_fl);
	clReleaseKernel(frame->kernel_pack_2d_face_ul);
	clReleaseKernel(frame->kernel_pack_2d_face_in);
	clReleaseKernel(frame->kernel_pack_2d_face_ui);
	clReleaseKernel(frame->kernel_unpack_2d_face_db);
	clReleaseKernel(frame->kernel_unpack_2d_face_fl);
	clReleaseKernel(frame->kernel_unpack_2d_face_ul);
	clReleaseKernel(frame->kernel_unpack_2d_face_in);
	clReleaseKernel(frame->kernel_unpack_2d_face_ui);
	clReleaseKernel(frame->kernel_stencil_3d7p_db);
	clReleaseKernel(frame->kernel_stencil_3d7p_fl);
	clReleaseKernel(frame->kernel_stencil_3d7p_ul);
	clReleaseKernel(frame->kernel_stencil_3d7p_in);
	clReleaseKernel(frame->kernel_stencil_3d7p_ui);
	clReleaseKernel(frame->kernel_csr_db);
	clReleaseKernel(frame->kernel_csr_fl);
	clReleaseKernel(frame->kernel_csr_ul);
	clReleaseKernel(frame->kernel_csr_in);
	clReleaseKernel(frame->kernel_csr_ui);
//	clReleaseKernel(frame->kernel_crc_db);
//	clReleaseKernel(frame->kernel_crc_fl);
//	clReleaseKernel(frame->kernel_crc_ul);
//	clReleaseKernel(frame->kernel_crc_in);
	clReleaseKernel(frame->kernel_crc_ui);

	//Release Internal Buffers
	clReleaseMemObject(frame->constant_face_size);
	clReleaseMemObject(frame->constant_face_stride);
	clReleaseMemObject(frame->constant_face_child_size);
	clReleaseMemObject(frame->red_loc);

	//Release remaining context info
#ifndef OPENCL_SINGLE_KERNEL_PROGS
	clReleaseProgram(frame->program_opencl_core);
#else
	clReleaseProgram(frame->program_reduce_db);
	clReleaseProgram(frame->program_reduce_fl);
	clReleaseProgram(frame->program_reduce_ul);
	clReleaseProgram(frame->program_reduce_in);
	clReleaseProgram(frame->program_reduce_ui);
	clReleaseProgram(frame->program_dotProd_db);
	clReleaseProgram(frame->program_dotProd_fl);
	clReleaseProgram(frame->program_dotProd_ul);
	clReleaseProgram(frame->program_dotProd_in);
	clReleaseProgram(frame->program_dotProd_ui);
	clReleaseProgram(frame->program_transpose_2d_face_db);
	clReleaseProgram(frame->program_transpose_2d_face_fl);
	clReleaseProgram(frame->program_transpose_2d_face_ul);
	clReleaseProgram(frame->program_transpose_2d_face_in);
	clReleaseProgram(frame->program_transpose_2d_face_ui);
	clReleaseProgram(frame->program_pack_2d_face_db);
	clReleaseProgram(frame->program_pack_2d_face_fl);
	clReleaseProgram(frame->program_pack_2d_face_ul);
	clReleaseProgram(frame->program_pack_2d_face_in);
	clReleaseProgram(frame->program_pack_2d_face_ui);
	clReleaseProgram(frame->program_unpack_2d_face_db);
	clReleaseProgram(frame->program_unpack_2d_face_fl);
	clReleaseProgram(frame->program_unpack_2d_face_ul);
	clReleaseProgram(frame->program_unpack_2d_face_in);
	clReleaseProgram(frame->program_unpack_2d_face_ui);
	clReleaseProgram(frame->program_stencil_3d7p_db);
	clReleaseProgram(frame->program_stencil_3d7p_fl);
	clReleaseProgram(frame->program_stencil_3d7p_ul);
	clReleaseProgram(frame->program_stencil_3d7p_in);
	clReleaseProgram(frame->program_stencil_3d7p_ui);
	clReleaseProgram(frame->program_csr_db);
	clReleaseProgram(frame->program_csr_fl);
	clReleaseProgram(frame->program_csr_ul);
	clReleaseProgram(frame->program_csr_in);
	clReleaseProgram(frame->program_csr_ui);
//	clReleaseProgram(frame->program_crc_db);
//	clReleaseProgram(frame->program_crc_fl);
//	clReleaseProgram(frame->program_crc_ul);
//	clReleaseProgram(frame->program_crc_in);
	clReleaseProgram(frame->program_crc_ui);
	
#endif
	clReleaseCommandQueue(frame->queue);
	clReleaseContext(frame->context);

	//release the frame's program source
#if defined(OPENCL_SINGLE_KERNEL_PROGS) && defined(WITH_INTELFPGA)
	free((void *) frame->metaCLbin_reduce_db);
	frame->metaCLbinLen_reduce_db = 0;
	free((void *) frame->metaCLbin_reduce_fl);
	frame->metaCLbinLen_reduce_fl = 0;
	free((void *) frame->metaCLbin_reduce_ul);
	frame->metaCLbinLen_reduce_ul = 0;
	free((void *) frame->metaCLbin_reduce_in);
	frame->metaCLbinLen_reduce_in = 0;
	free((void *) frame->metaCLbin_reduce_ui);
	frame->metaCLbinLen_reduce_ui = 0;
	free((void *) frame->metaCLbin_dotProd_db);
	frame->metaCLbinLen_dotProd_db = 0;
	free((void *) frame->metaCLbin_dotProd_fl);
	frame->metaCLbinLen_dotProd_fl = 0;
	free((void *) frame->metaCLbin_dotProd_ul);
	frame->metaCLbinLen_dotProd_ul = 0;
	free((void *) frame->metaCLbin_dotProd_in);
	frame->metaCLbinLen_dotProd_in = 0;
	free((void *) frame->metaCLbin_dotProd_ui);
	frame->metaCLbinLen_dotProd_ui = 0;
	free((void *) frame->metaCLbin_transpose_2d_face_db);
	frame->metaCLbinLen_transpose_2d_face_db = 0;
	free((void *) frame->metaCLbin_transpose_2d_face_fl);
	frame->metaCLbinLen_transpose_2d_face_fl = 0;
	free((void *) frame->metaCLbin_transpose_2d_face_ul);
	frame->metaCLbinLen_transpose_2d_face_ul = 0;
	free((void *) frame->metaCLbin_transpose_2d_face_in);
	frame->metaCLbinLen_transpose_2d_face_in = 0;
	free((void *) frame->metaCLbin_transpose_2d_face_ui);
	frame->metaCLbinLen_transpose_2d_face_ui = 0;
	free((void *) frame->metaCLbin_pack_2d_face_db);
	frame->metaCLbinLen_pack_2d_face_db = 0;
	free((void *) frame->metaCLbin_pack_2d_face_fl);
	frame->metaCLbinLen_pack_2d_face_fl = 0;
	free((void *) frame->metaCLbin_pack_2d_face_ul);
	frame->metaCLbinLen_pack_2d_face_ul = 0;
	free((void *) frame->metaCLbin_pack_2d_face_in);
	frame->metaCLbinLen_pack_2d_face_in = 0;
	free((void *) frame->metaCLbin_pack_2d_face_ui);
	frame->metaCLbinLen_pack_2d_face_ui = 0;
	free((void *) frame->metaCLbin_unpack_2d_face_db);
	frame->metaCLbinLen_unpack_2d_face_db = 0;
	free((void *) frame->metaCLbin_unpack_2d_face_fl);
	frame->metaCLbinLen_unpack_2d_face_fl = 0;
	free((void *) frame->metaCLbin_unpack_2d_face_ul);
	frame->metaCLbinLen_unpack_2d_face_ul = 0;
	free((void *) frame->metaCLbin_unpack_2d_face_in);
	frame->metaCLbinLen_unpack_2d_face_in = 0;
	free((void *) frame->metaCLbin_unpack_2d_face_ui);
	frame->metaCLbinLen_unpack_2d_face_ui = 0;
	free((void *) frame->metaCLbin_stencil_3d7p_db);
	frame->metaCLbinLen_stencil_3d7p_db = 0;
	free((void *) frame->metaCLbin_stencil_3d7p_fl);
	frame->metaCLbinLen_stencil_3d7p_fl = 0;
	free((void *) frame->metaCLbin_stencil_3d7p_ul);
	frame->metaCLbinLen_stencil_3d7p_ul = 0;
	free((void *) frame->metaCLbin_stencil_3d7p_in);
	frame->metaCLbinLen_stencil_3d7p_in = 0;
	free((void *) frame->metaCLbin_stencil_3d7p_ui);
	frame->metaCLbinLen_stencil_3d7p_ui = 0;
	free((void *) frame->metaCLbin_csr_db);
	frame->metaCLbinLen_csr_db = 0;
	free((void *) frame->metaCLbin_csr_fl);
	frame->metaCLbinLen_csr_fl = 0;
	free((void *) frame->metaCLbin_csr_ul);
	frame->metaCLbinLen_csr_ul = 0;
	free((void *) frame->metaCLbin_csr_in);
	frame->metaCLbinLen_csr_in = 0;
	free((void *) frame->metaCLbin_csr_ui);
	frame->metaCLbinLen_csr_ui = 0;
//	free((void *) frame->metaCLbin_crc_db);
//	frame->metaCLbinLen_crc_db = 0;
//	free((void *) frame->metaCLbin_crc_fl);
//	frame->metaCLbinLen_crc_fl = 0;
//	free((void *) frame->metaCLbin_crc_ul);
//	frame->metaCLbinLen_crc_ul = 0;
//	free((void *) frame->metaCLbin_crc_in);
//	frame->metaCLbinLen_crc_in = 0;
	free((void *) frame->metaCLbin_crc_ui);
	frame->metaCLbinLen_crc_ui = 0;
#else
	free((void *) frame->metaCLProgSrc);
	frame->metaCLProgLen = 0;
#endif
}

//This is a fallback catchall to ensure some context is initialized
// iff the user calls an accelerator function in OpenCL mode before
// calling meta_set_acc in OpenCL mode.
//It will pick some valid OpenCL device, and emit a warning to stderr
// that all OpenCL calls until meta_set_acc will refer to this device
//It also implements environment-variable-controlled device selection
// via the "TARGET_DEVICE" string environemnt variable, which must match
// EXACTLY the device name reported to the OpenCL runtime.
//TODO implement a reasonable passthrough for any errors which OpenCL may throw.
cl_int metaOpenCLInitStackFrameDefault(metaOpenCLStackFrame ** frame) {
	cl_int ret = CL_SUCCESS;
	//First, make sure we've run the one-time query to initialize the device array
	if (platforms == NULL || devices == NULL || (((long) platforms) == -1)) {
		//try to perform the scan, else wait while somebody else finishes it.
		metaOpenCLQueryDevices();
	}

	//Simply print the names of all devices, to assist later environment-variable device selection
	fprintf(stderr, "WARNING: Automatic OpenCL device selection used!\n");
	fprintf(stderr, "\tThe following devices are identified in the system:\n");
	char buff[128];
	int i;
	for (i = 0; i < num_devices; i++) {
		clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 128, (void *) &buff[0],
				NULL);
		fprintf(stderr, "Device [%d]: \"%s\"\n", i, buff);
	}

	//This is how you pick a specific device using an environment variable

	int gpuID = -1;

	if (getenv("TARGET_DEVICE") != NULL) {
		gpuID = metaOpenCLGetDeviceID(getenv("TARGET_DEVICE"), &devices[0],
				num_devices);
		if (gpuID < 0)
			fprintf(stderr,
					"Device \"%s\" not found.\nDefaulting to first device found.\n",
					getenv("TARGET_DEVICE"));
	} else {
		fprintf(stderr,
				"Environment variable TARGET_DEVICE not set.\nDefaulting to first device found.\n");
	}

	gpuID = gpuID < 0 ? 0 : gpuID; //Ternary check to make sure gpuID is valid, if it's less than zero, default to zero, otherwise keep

	clGetDeviceInfo(devices[gpuID], CL_DEVICE_NAME, 128, (void *) &buff[0],
			NULL);
	fprintf(stderr, "Selected Device %d: %s\n", gpuID, buff);

	//Now that we've picked a reasonable default, fill in the details for the frame object
	metaOpenCLInitStackFrame(frame, gpuID);

	return (ret);
}

//  !this kernel works for 3D data only.
//  ! PHI1 and PHI2 are input arrays.
//  ! s* parameters are start values in each dimension.
//  ! e* parameters are end values in each dimension.
//  ! s* and e* are only necessary when the halo layers 
//  !   has different thickness along various directions.
//  ! i,j,k are the array dimensions
//  ! len_ is number of threads in a threadblock.
//  !      This can be computed in the kernel itself.
cl_int opencl_dotProd(size_t (*grid_size)[3], size_t (*block_size)[3],
		void * data1, void * data2, size_t (*array_size)[3],
		size_t (*arr_start)[3], size_t (*arr_end)[3], void * reduced_val,
		meta_type_id type, int async, cl_event * event) {
	cl_int ret;
	cl_kernel kern;
	cl_int smem_len;
	size_t grid[3];
	size_t block[3] = METAMORPH_OCL_DEFAULT_BLOCK_3D;
	int iters;

	//Allow for auto-selected grid/block size if either is not specified
	if (grid_size == NULL || block_size == NULL) {
		grid[0] = (((*arr_end)[0] - (*arr_start)[0] + (block[0])) / block[0])
				* block[0];
		grid[1] = (((*arr_end)[1] - (*arr_start)[1] + (block[1])) / block[1])
				* block[1];
		grid[2] = block[2];
		iters = (((*arr_end)[2] - (*arr_start)[2] + (block[2])) / block[2]);
	} else {
		grid[0] = (*grid_size)[0] * (*block_size)[0];
		grid[1] = (*grid_size)[1] * (*block_size)[1];
		grid[2] = (*block_size)[2];
		block[0] = (*block_size)[0];
		block[1] = (*block_size)[1];
		block[2] = (*block_size)[2];
		iters = (*grid_size)[2];
	}
	smem_len = block[0] * block[1] * block[2];
	//before enqueuing, get a copy of the top stack frame
	metaOpenCLStackFrame * frame = metaOpenCLTopStackFrame();

	switch (type) {
	case a_db:
		kern = frame->kernel_dotProd_db;
		break;

	case a_fl:
		kern = frame->kernel_dotProd_fl;
		break;

	case a_ul:
		kern = frame->kernel_dotProd_ul;
		break;

	case a_in:
		kern = frame->kernel_dotProd_in;
		break;

	case a_ui:
		kern = frame->kernel_dotProd_ui;
		break;

	default:
		fprintf(stderr,
				"Error: Function 'opencl_dotProd' not implemented for selected type!\n");
		return -1;
		break;

	}
	//printf("Grid: %d %d %d\n", grid[0], grid[1], grid[2]);
	//printf("Block: %d %d %d\n", block[0], block[1], block[2]);
	//printf("Size: %d %d %d\n", (*array_size)[0], (*array_size)[1], (*array_size)[2]);
	//printf("Start: %d %d %d\n", (*arr_start)[0], (*arr_start)[1], (*arr_start)[2]);
	//printf("End: %d %d %d\n", (*arr_end)[1], (*arr_end)[0], (*arr_end)[2]);
	//printf("SMEM: %d\n", smem_len);

	ret = clSetKernelArg(kern, 0, sizeof(cl_mem *), &data1);
	ret |= clSetKernelArg(kern, 1, sizeof(cl_mem *), &data2);
	ret |= clSetKernelArg(kern, 2, sizeof(cl_int), &(*array_size)[0]);
	ret |= clSetKernelArg(kern, 3, sizeof(cl_int), &(*array_size)[1]);
	ret |= clSetKernelArg(kern, 4, sizeof(cl_int), &(*array_size)[2]);
	ret |= clSetKernelArg(kern, 5, sizeof(cl_int), &(*arr_start)[0]);
	ret |= clSetKernelArg(kern, 6, sizeof(cl_int), &(*arr_start)[1]);
	ret |= clSetKernelArg(kern, 7, sizeof(cl_int), &(*arr_start)[2]);
	ret |= clSetKernelArg(kern, 8, sizeof(cl_int), &(*arr_end)[0]);
	ret |= clSetKernelArg(kern, 9, sizeof(cl_int), &(*arr_end)[1]);
	ret |= clSetKernelArg(kern, 10, sizeof(cl_int), &(*arr_end)[2]);
	ret |= clSetKernelArg(kern, 11, sizeof(cl_int), &iters);
	ret |= clSetKernelArg(kern, 12, sizeof(cl_mem *), &reduced_val);
	ret |= clSetKernelArg(kern, 13, sizeof(cl_int), &smem_len);
	switch (type) {
	case a_db:
		ret |= clSetKernelArg(kern, 14, smem_len * sizeof(cl_double), NULL);
		break;

	case a_fl:
		ret |= clSetKernelArg(kern, 14, smem_len * sizeof(cl_float), NULL);
		break;

	case a_ul:
		ret |= clSetKernelArg(kern, 14, smem_len * sizeof(cl_ulong), NULL);
		break;

	case a_in:
		ret |= clSetKernelArg(kern, 14, smem_len * sizeof(cl_int), NULL);
		break;

	case a_ui:
		ret |= clSetKernelArg(kern, 14, smem_len * sizeof(cl_uint), NULL);
		break;

		//Shouldn't be reachable, but cover our bases
	default:
		fprintf(stderr,
				"Error: unexpected type, cannot set shared memory size in 'opencl_dotProd'!\n");
	}

	ret |= clEnqueueNDRangeKernel(frame->queue, kern, 3, NULL, grid, block, 0,
			NULL, event);

	//TODO find a way to make explicit sync optional
	if (!async)
		ret |= clFinish(frame->queue);
	//printf("CHECK THIS! %d\n", ret);
	//free the copy of the top stack frame, DO NOT release it's members
	free(frame);

	return (ret);
}

cl_int opencl_reduce(size_t (*grid_size)[3], size_t (*block_size)[3],
		void * data, size_t (*array_size)[3], size_t (*arr_start)[3],
		size_t (*arr_end)[3], void * reduced_val, meta_type_id type, int async,
		cl_event * event) {
	cl_int ret;
	cl_kernel kern;
	cl_int smem_len;
	size_t grid[3];
	size_t block[3] = METAMORPH_OCL_DEFAULT_BLOCK_3D;
	int iters;
	//Allow for auto-selected grid/block size if either is not specified
	if (grid_size == NULL || block_size == NULL) {
		grid[0] = (((*arr_end)[0] - (*arr_start)[0] + (block[0])) / block[0])
				* block[0];
		grid[1] = (((*arr_end)[1] - (*arr_start)[1] + (block[1])) / block[1])
				* block[1];
		grid[2] = block[2];
		iters = (((*arr_end)[2] - (*arr_start)[2] + (block[2])) / block[2]);
	} else {
		grid[0] = (*grid_size)[0] * (*block_size)[0];
		grid[1] = (*grid_size)[1] * (*block_size)[1];
		grid[2] = (*block_size)[2];
		block[0] = (*block_size)[0];
		block[1] = (*block_size)[1];
		block[2] = (*block_size)[2];
		iters = (*grid_size)[2];
	}
	smem_len = block[0] * block[1] * block[2];
	//before enqueuing, get a copy of the top stack frame
	metaOpenCLStackFrame * frame = metaOpenCLTopStackFrame();

	switch (type) {
	case a_db:
		kern = frame->kernel_reduce_db;
		break;

	case a_fl:
		kern = frame->kernel_reduce_fl;
		break;

	case a_ul:
		kern = frame->kernel_reduce_ul;
		break;

	case a_in:
		kern = frame->kernel_reduce_in;
		break;

	case a_ui:
		kern = frame->kernel_reduce_ui;
		break;

	default:
		fprintf(stderr,
				"Error: Function 'opencl_reduce' not implemented for selected type!\n");
		return -1;
		break;

	}
	//printf("Grid: %d %d %d\n", grid[0], grid[1], grid[2]);
	//printf("Block: %d %d %d\n", block[0], block[1], block[2]);
	//printf("Size: %d %d %d\n", (*array_size)[0], (*array_size)[1], (*array_size)[2]);
	//printf("Start: %d %d %d\n", (*arr_start)[0], (*arr_start)[1], (*arr_start)[2]);
	//printf("End: %d %d %d\n", (*arr_end)[1], (*arr_end)[0], (*arr_end)[2]);
	//printf("SMEM: %d\n", smem_len);

	ret = clSetKernelArg(kern, 0, sizeof(cl_mem *), &data);
	ret |= clSetKernelArg(kern, 1, sizeof(cl_int), &(*array_size)[0]);
	ret |= clSetKernelArg(kern, 2, sizeof(cl_int), &(*array_size)[1]);
	ret |= clSetKernelArg(kern, 3, sizeof(cl_int), &(*array_size)[2]);
	ret |= clSetKernelArg(kern, 4, sizeof(cl_int), &(*arr_start)[0]);
	ret |= clSetKernelArg(kern, 5, sizeof(cl_int), &(*arr_start)[1]);
	ret |= clSetKernelArg(kern, 6, sizeof(cl_int), &(*arr_start)[2]);
	ret |= clSetKernelArg(kern, 7, sizeof(cl_int), &(*arr_end)[0]);
	ret |= clSetKernelArg(kern, 8, sizeof(cl_int), &(*arr_end)[1]);
	ret |= clSetKernelArg(kern, 9, sizeof(cl_int), &(*arr_end)[2]);
	ret |= clSetKernelArg(kern, 10, sizeof(cl_int), &iters);
	ret |= clSetKernelArg(kern, 11, sizeof(cl_mem *), &reduced_val);
	ret |= clSetKernelArg(kern, 12, sizeof(cl_int), &smem_len);
	switch (type) {
	case a_db:
		ret |= clSetKernelArg(kern, 13, smem_len * sizeof(cl_double), NULL);
		break;

	case a_fl:
		ret |= clSetKernelArg(kern, 13, smem_len * sizeof(cl_float), NULL);
		break;

	case a_ul:
		ret |= clSetKernelArg(kern, 13, smem_len * sizeof(cl_ulong), NULL);
		break;

	case a_in:
		ret |= clSetKernelArg(kern, 13, smem_len * sizeof(cl_int), NULL);
		break;

	case a_ui:
		ret |= clSetKernelArg(kern, 13, smem_len * sizeof(cl_uint), NULL);
		break;

		//Shouldn't be reachable, but cover our bases
	default:
		fprintf(stderr,
				"Error: unexpected type, cannot set shared memory size in 'opencl_reduce'!\n");
	}
	ret |= clEnqueueNDRangeKernel(frame->queue, kern, 3, NULL, grid, block, 0,
			NULL, event);

	//TODO find a way to make explicit sync optional
	if (!async)
		ret |= clFinish(frame->queue);
	//printf("CHECK THIS! %d\n", ret);
	//free the copy of the top stack frame, DO NOT release it's members
	free(frame);

	return (ret);
}

cl_int opencl_transpose_face(size_t (*grid_size)[3], size_t (*block_size)[3],
		void * indata, void *outdata, size_t (*arr_dim_xy)[3],
		size_t (*tran_dim_xy)[3], meta_type_id type, int async,
		cl_event * event) {
	cl_int ret;
	cl_kernel kern;
	cl_int smem_len;
	size_t grid[3];
	size_t block[3] = { TRANSPOSE_TILE_DIM, TRANSPOSE_TILE_BLOCK_ROWS, 1 };
	//cl_int smem_len = (*block_size)[0] * (*block_size)[1] * (*block_size)[2];
// TODO update to use user provided grid/block once multi-element per thread scaling is added
//	size_t grid[3] = {(*grid_size)[0]*(*block_size)[0], (*grid_size)[1]*(*block_size)[1], (*block_size)[2]};
//	size_t block[3] = {(*block_size)[0], (*block_size)[1], (*block_size)[2]};\
	//FIXME: make this smart enough to rescale the threadblock (and thus shared memory - e.g. bank conflicts) w.r.t. double vs. float
	if (grid_size == NULL || block_size == NULL) {
		//FIXME: reconcile TILE_DIM/BLOCK_ROWS
		grid[0] = (((*tran_dim_xy)[0] + block[0] - 1) / block[0]) * block[0];
		grid[1] = (((*tran_dim_xy)[1] + block[1] - 1) / block[1]) * block[1];
		grid[2] = 1;
	} else {
		grid[0] = (*grid_size)[0] * (*block_size)[0];
		grid[1] = (*grid_size)[1] * (*block_size)[1];
		grid[2] = (*block_size)[2];
		block[0] = (*block_size)[0];
		block[1] = (*block_size)[1];
		block[2] = (*block_size)[2];
	}
	//The +1 here is to avoid bank conflicts with 32 floats or 16 doubles and is required by the kernel logic
	smem_len = (block[0] + 1) * block[1] * block[2];
	//TODO as the frame grows larger with more kernels, this overhead will start to add up
	// Need a better (safe) way of accessing the stack for kernel launches
	//before enqueuing, get a copy of the top stack frame
	metaOpenCLStackFrame * frame = metaOpenCLTopStackFrame();

	switch (type) {
	case a_db:
		kern = frame->kernel_transpose_2d_face_db;
		break;

	case a_fl:
		kern = frame->kernel_transpose_2d_face_fl;
		break;

	case a_ul:
		kern = frame->kernel_transpose_2d_face_ul;
		break;

	case a_in:
		kern = frame->kernel_transpose_2d_face_in;
		break;

	case a_ui:
		kern = frame->kernel_transpose_2d_face_ui;
		break;

	default:
		fprintf(stderr,
				"Error: Function 'opencl_transpose_face' not implemented for selected type!\n");
		return -1;
		break;

	}
	ret = clSetKernelArg(kern, 0, sizeof(cl_mem *), &outdata);
	ret |= clSetKernelArg(kern, 1, sizeof(cl_mem *), &indata);
	ret |= clSetKernelArg(kern, 2, sizeof(cl_int), &(*arr_dim_xy)[0]);
	ret |= clSetKernelArg(kern, 3, sizeof(cl_int), &(*arr_dim_xy)[1]);
	ret |= clSetKernelArg(kern, 4, sizeof(cl_int), &(*tran_dim_xy)[0]);
	ret |= clSetKernelArg(kern, 5, sizeof(cl_int), &(*tran_dim_xy)[1]);
	switch (type) {
	case a_db:
		ret |= clSetKernelArg(kern, 6, smem_len * sizeof(cl_double), NULL);
		break;

	case a_fl:
		ret |= clSetKernelArg(kern, 6, smem_len * sizeof(cl_float), NULL);
		break;

	case a_ul:
		ret |= clSetKernelArg(kern, 6, smem_len * sizeof(cl_ulong), NULL);
		break;

	case a_in:
		ret |= clSetKernelArg(kern, 6, smem_len * sizeof(cl_int), NULL);
		break;

	case a_ui:
		ret |= clSetKernelArg(kern, 6, smem_len * sizeof(cl_uint), NULL);
		break;

		//Shouldn't be reachable, but cover our bases
	default:
		fprintf(stderr,
				"Error: unexpected type, cannot set shared memory size in 'opencl_transpose_face'!\n");
	}
	ret |= clEnqueueNDRangeKernel(frame->queue, kern, 2, NULL, grid, block, 0,
			NULL, event);

	//TODO find a way to make explicit sync optional
	if (!async)
		ret |= clFinish(frame->queue);
	//free the copy of the top stack frame, DO NOT release it's members
	free(frame);

	return (ret);
}

cl_int opencl_pack_face(size_t (*grid_size)[3], size_t (*block_size)[3],
		void *packed_buf, void *buf, meta_face *face,
		int *remain_dim, meta_type_id type, int async, cl_event * event_k1,
		cl_event * event_c1, cl_event *event_c2, cl_event *event_c3) {
	cl_int ret;
	cl_kernel kern;
	cl_int size = face->size[0] * face->size[1] * face->size[2];
	cl_int smem_size;
	size_t grid[3];
	size_t block[3] = { 256, 1, 1 };
	//before enqueuing, get a copy of the top stack frame
	metaOpenCLStackFrame * frame = metaOpenCLTopStackFrame();

	//copy required pieces of the face struct into constant memory
	ret = clEnqueueWriteBuffer(frame->queue, frame->constant_face_size,
			((async) ? CL_FALSE : CL_TRUE), 0, sizeof(cl_int) * face->count,
			face->size, 0, NULL, event_c1);
	ret |= clEnqueueWriteBuffer(frame->queue, frame->constant_face_stride,
			((async) ? CL_FALSE : CL_TRUE), 0, sizeof(cl_int) * face->count,
			face->stride, 0, NULL, event_c2);
	ret |= clEnqueueWriteBuffer(frame->queue, frame->constant_face_child_size,
			((async) ? CL_FALSE : CL_TRUE), 0, sizeof(cl_int) * face->count,
			remain_dim, 0, NULL, event_c3);
//TODO update to use user-provided grid/block once multi-element per thread scaling is added
//	size_t grid[3] = {(*grid_size)[0]*(*block_size)[0], (*grid_size)[1]*(*block_size)[1], (*block_size)[2]};
//	size_t block[3] = {(*block_size)[0], (*block_size)[1], (*block_size)[2]};
	if (grid_size == NULL || block_size == NULL) {
		grid[0] = ((size + block[0] - 1) / block[0]) * block[0];
		grid[1] = 1;
		grid[2] = 1;
	} else {
		//This is a workaround for some non-determinism that was observed when allowing fully-arbitrary spec of grid/block
		if ((*block_size)[1] != 1 || (*block_size)[2] != 1
				|| (*grid_size)[1] != 1 || (*grid_size)[2])
			fprintf(stderr,
					"WARNING: Pack requires 1D block and grid, ignoring y/z params!\n");
		grid[0] = (*grid_size)[0] * (*block_size)[0];
		grid[1] = 1;
		grid[2] = 1;
		//grid[1] = (*grid_size)[1]*(*block_size)[1];
		//grid[2] = (*block_size)[2];
		block[0] = (*block_size)[0];
		block[1] = 1;
		block[2] = 1;
		//block[1] = (*block_size)[1];
		//block[2] = (*block_size)[2];
	}
	smem_size = face->count * block[0] * sizeof(int);
//TODO Timing needs to be made consistent with CUDA (ie the event should return time for copying to constant memory and the kernel

	switch (type) {
	case a_db:
		kern = frame->kernel_pack_2d_face_db;
		break;

	case a_fl:
		kern = frame->kernel_pack_2d_face_fl;
		break;

	case a_ul:
		kern = frame->kernel_pack_2d_face_ul;
		break;

	case a_in:
		kern = frame->kernel_pack_2d_face_in;
		break;

	case a_ui:
		kern = frame->kernel_pack_2d_face_ui;
		break;

	default:
		fprintf(stderr,
				"Error: Function 'opencl_pack_face' not implemented for selected type!\n");
		return -1;
		break;

	}
	//printf("Grid: %d %d %d\n", grid[0], grid[1], grid[2]);
	//printf("Block: %d %d %d\n", block[0], block[1], block[2]);
	//printf("Size: %d %d %d\n", (*array_size)[0], (*array_size)[1], (*array_size)[2]);
	//printf("Start: %d %d %d\n", (*arr_start)[0], (*arr_start)[1], (*arr_start)[2]);
	//printf("End: %d %d %d\n", (*arr_end)[1], (*arr_end)[0], (*arr_end)[2]);
	//printf("SMEM: %d\n", smem_len);

	ret = clSetKernelArg(kern, 0, sizeof(cl_mem *), &packed_buf);
	ret |= clSetKernelArg(kern, 1, sizeof(cl_mem *), &buf);
	ret |= clSetKernelArg(kern, 2, sizeof(cl_int), &size);
	ret |= clSetKernelArg(kern, 3, sizeof(cl_int), &(face->start));
	ret |= clSetKernelArg(kern, 4, sizeof(cl_int), &(face->count));
	ret |= clSetKernelArg(kern, 5, sizeof(cl_mem *),
			&frame->constant_face_size);
	ret |= clSetKernelArg(kern, 6, sizeof(cl_mem *),
			&frame->constant_face_stride);
	ret |= clSetKernelArg(kern, 7, sizeof(cl_mem *),
			&frame->constant_face_child_size);
	ret |= clSetKernelArg(kern, 8, smem_size, NULL);
	ret |= clEnqueueNDRangeKernel(frame->queue, kern, 1, NULL, grid, block, 0,
			NULL, event_k1);

	//TODO find a way to make explicit sync optional
	if (!async)
		ret |= clFinish(frame->queue);
	//printf("CHECK THIS! %d\n", ret);
	//free the copy of the top stack frame, DO NOT release it's members
	free(frame);

	return (ret);
}

cl_int opencl_unpack_face(size_t (*grid_size)[3], size_t (*block_size)[3],
		void *packed_buf, void *buf, meta_face *face,
		int *remain_dim, meta_type_id type, int async, cl_event * event_k1,
		cl_event * event_c1, cl_event *event_c2, cl_event *event_c3) {

	cl_int ret;
	cl_kernel kern;
	cl_int size = face->size[0] * face->size[1] * face->size[2];
	cl_int smem_size;
	size_t grid[3];
	size_t block[3] = { 256, 1, 1 };
	//before enqueuing, get a copy of the top stack frame
	metaOpenCLStackFrame * frame = metaOpenCLTopStackFrame();

	//copy required pieces of the face struct into constant memory
	ret = clEnqueueWriteBuffer(frame->queue, frame->constant_face_size,
			((async) ? CL_FALSE : CL_TRUE), 0, sizeof(cl_int) * face->count,
			face->size, 0, NULL, event_c1);
	ret |= clEnqueueWriteBuffer(frame->queue, frame->constant_face_stride,
			((async) ? CL_FALSE : CL_TRUE), 0, sizeof(cl_int) * face->count,
			face->stride, 0, NULL, event_c2);
	ret |= clEnqueueWriteBuffer(frame->queue, frame->constant_face_child_size,
			((async) ? CL_FALSE : CL_TRUE), 0, sizeof(cl_int) * face->count,
			remain_dim, 0, NULL, event_c3);
//TODO update to use user-provided grid/block once multi-element per thread scaling is added
//	size_t grid[3] = {(*grid_size)[0]*(*block_size)[0], (*grid_size)[1]*(*block_size)[1], (*block_size)[2]};
//	size_t block[3] = {(*block_size)[0], (*block_size)[1], (*block_size)[2]};
	if (grid_size == NULL || block_size == NULL) {
		grid[0] = ((size + block[0] - 1) / block[0]) * block[0];
		grid[1] = 1;
		grid[2] = 1;
	} else {
		//This is a workaround for some non-determinism that was observed when allowing fully-arbitrary spec of grid/block
		if ((*block_size)[1] != 1 || (*block_size)[2] != 1
				|| (*grid_size)[1] != 1 || (*grid_size)[2])
			fprintf(stderr,
					"WARNING: Unpack requires 1D block and grid, ignoring y/z params!\n");
		grid[0] = (*grid_size)[0] * (*block_size)[0];
		grid[1] = 1;
		grid[2] = 1;
		//grid[1] = (*grid_size)[1]*(*block_size)[1];
		//grid[2] = (*block_size)[2];
		block[0] = (*block_size)[0];
		block[1] = 1;
		block[2] = 1;
		//block[1] = (*block_size)[1];
		//block[2] = (*block_size)[2];
	}
	smem_size = face->count * block[0] * sizeof(int);
//TODO Timing needs to be made consistent with CUDA (ie the event should return time for copying to constant memory and the kernel

	switch (type) {
	case a_db:
		kern = frame->kernel_unpack_2d_face_db;
		break;

	case a_fl:
		kern = frame->kernel_unpack_2d_face_fl;
		break;

	case a_ul:
		kern = frame->kernel_unpack_2d_face_ul;
		break;

	case a_in:
		kern = frame->kernel_unpack_2d_face_in;
		break;

	case a_ui:
		kern = frame->kernel_unpack_2d_face_ui;
		break;

	default:
		fprintf(stderr,
				"Error: Function 'opencl_unpack_face' not implemented for selected type!\n");
		return -1;
		break;

	}
	//printf("Grid: %d %d %d\n", grid[0], grid[1], grid[2]);
	//printf("Block: %d %d %d\n", block[0], block[1], block[2]);
	//printf("Size: %d %d %d\n", (*array_size)[0], (*array_size)[1], (*array_size)[2]);
	//printf("Start: %d %d %d\n", (*arr_start)[0], (*arr_start)[1], (*arr_start)[2]);
	//printf("End: %d %d %d\n", (*arr_end)[1], (*arr_end)[0], (*arr_end)[2]);
	//printf("SMEM: %d\n", smem_len);

	ret = clSetKernelArg(kern, 0, sizeof(cl_mem *), &packed_buf);
	ret |= clSetKernelArg(kern, 1, sizeof(cl_mem *), &buf);
	ret |= clSetKernelArg(kern, 2, sizeof(cl_int), &size);
	ret |= clSetKernelArg(kern, 3, sizeof(cl_int), &(face->start));
	ret |= clSetKernelArg(kern, 4, sizeof(cl_int), &(face->count));
	ret |= clSetKernelArg(kern, 5, sizeof(cl_mem *),
			&frame->constant_face_size);
	ret |= clSetKernelArg(kern, 6, sizeof(cl_mem *),
			&frame->constant_face_stride);
	ret |= clSetKernelArg(kern, 7, sizeof(cl_mem *),
			&frame->constant_face_child_size);
	ret |= clSetKernelArg(kern, 8, smem_size, NULL);
	ret |= clEnqueueNDRangeKernel(frame->queue, kern, 1, NULL, grid, block, 0,
			NULL, event_k1);

	//TODO find a way to make explicit sync optional
	if (!async)
		ret |= clFinish(frame->queue);
	//printf("CHECK THIS! %d\n", ret);
	//free the copy of the top stack frame, DO NOT release it's members
	free(frame);

	return (ret);
}

cl_int opencl_stencil_3d7p(size_t (*grid_size)[3], size_t (*block_size)[3],
		void * indata, void * outdata, size_t (*array_size)[3],
		size_t (*arr_start)[3], size_t (*arr_end)[3], meta_type_id type,
		int async, cl_event * event) {
	cl_int ret = CL_SUCCESS;
	cl_kernel kern;
	cl_int smem_len;
	size_t grid[3] = { 1, 1, 1 };
	size_t block[3] = { 128, 2, 1 };
	//size_t block[3] = METAMORPH_OCL_DEFAULT_BLOCK;
	int iters;
	//Allow for auto-selected grid/block size if either is not specified
	if (grid_size == NULL || block_size == NULL) {
		//do not subtract 1 from blocx for the case when end == start
		grid[0] = (((*arr_end)[0] - (*arr_start)[0] + (block[0])) / block[0]);
		grid[1] = (((*arr_end)[1] - (*arr_start)[1] + (block[1])) / block[1]);
		iters = (((*arr_end)[2] - (*arr_start)[2] + (block[2])) / block[2]);
	} else {
//		grid[0] = (*grid_size)[0];
//		grid[1] = (*grid_size)[1];
//		block[0] = (*block_size)[0];
//		block[1] = (*block_size)[1];
//		block[2] = (*block_size)[2];
//		iters = (*grid_size)[2];
		grid[0] = (*grid_size)[0] * (*block_size)[0];
		grid[1] = (*grid_size)[1] * (*block_size)[1];
		grid[2] = (*block_size)[2];
		block[0] = (*block_size)[0];
		block[1] = (*block_size)[1];
		block[2] = (*block_size)[2];
		iters = (*grid_size)[2];
	}
	//smem_len = (block[0]+2) * (block[1]+2) * block[2];
	smem_len = 0;
#if WITH_INTELFPGA
//TODO check why the FPGA backend uses non-zero smem_len
	smem_len = (block[0] + 2) * (block[1] + 2) * block[2];
#endif
	metaOpenCLStackFrame * frame = metaOpenCLTopStackFrame();

	switch (type) {
	case a_db:
		kern = frame->kernel_stencil_3d7p_db;
		break;

	case a_fl:
		kern = frame->kernel_stencil_3d7p_fl;
		break;

	case a_ul:
		kern = frame->kernel_stencil_3d7p_ul;
		break;

	case a_in:
		kern = frame->kernel_stencil_3d7p_in;
		break;

	case a_ui:
		kern = frame->kernel_stencil_3d7p_ui;
		break;

	default:
		fprintf(stderr,
				"Error: Function 'opencl_stencil_3d7p' not implemented for selected type!\n");
		return -1;
		break;
	}

	ret = clSetKernelArg(kern, 0, sizeof(cl_mem *), &indata);
	ret |= clSetKernelArg(kern, 1, sizeof(cl_mem *), &outdata);
	ret |= clSetKernelArg(kern, 2, sizeof(cl_int), &(*array_size)[0]);
	ret |= clSetKernelArg(kern, 3, sizeof(cl_int), &(*array_size)[1]);
	ret |= clSetKernelArg(kern, 4, sizeof(cl_int), &(*array_size)[2]);
	ret |= clSetKernelArg(kern, 5, sizeof(cl_int), &(*arr_start)[0]);
	ret |= clSetKernelArg(kern, 6, sizeof(cl_int), &(*arr_start)[1]);
	ret |= clSetKernelArg(kern, 7, sizeof(cl_int), &(*arr_start)[2]);
	ret |= clSetKernelArg(kern, 8, sizeof(cl_int), &(*arr_end)[0]);
	ret |= clSetKernelArg(kern, 9, sizeof(cl_int), &(*arr_end)[1]);
	ret |= clSetKernelArg(kern, 10, sizeof(cl_int), &(*arr_end)[2]);
	ret |= clSetKernelArg(kern, 11, sizeof(cl_int), &iters);
	ret |= clSetKernelArg(kern, 12, sizeof(cl_int), &smem_len);
	switch (type) {
	case a_db:
		ret |= clSetKernelArg(kern, 13, smem_len * sizeof(cl_double), NULL);
		break;

	case a_fl:
		ret |= clSetKernelArg(kern, 13, smem_len * sizeof(cl_float), NULL);
		break;

	case a_ul:
		ret |= clSetKernelArg(kern, 13, smem_len * sizeof(cl_ulong), NULL);
		break;

	case a_in:
		ret |= clSetKernelArg(kern, 13, smem_len * sizeof(cl_int), NULL);
		break;

	case a_ui:
		ret |= clSetKernelArg(kern, 13, smem_len * sizeof(cl_uint), NULL);
		break;

		//Shouldn't be reachable, but cover our bases
	default:
		fprintf(stderr,
				"Error: unexpected type, cannot set shared memory size in 'opencl_stencil_3d7p'!\n");
	}
	ret |= clEnqueueNDRangeKernel(frame->queue, kern, 3, NULL, grid, block, 0,
			NULL, event);

	if (!async)
		ret |= clFinish(frame->queue);
	free(frame);

	return (ret);

}

cl_int opencl_csr(size_t global_size, size_t local_size,
		void * csr_ap, void * csr_aj, void * csr_ax, void * x_loc, void * y_loc, 
		meta_type_id type, int async,
		// cl_event * wait, 
		cl_event * event) {

	cl_int ret = CL_SUCCESS;
	cl_kernel kern;
	
	size_t grid = 1;
	size_t block = METAMORPH_OCL_DEFAULT_BLOCK_1D;
	
	//TODO does this assume the OpenCL model or CUDA model for grid size?
	//FIXME it appears to use OpenCL model whereas everyone else uses CUDA
	if (global_size == NULL || local_size == NULL) {
		//TODO fallback scaling
	} else {
		grid = global_size;
		block = local_size;
	}
	
	metaOpenCLStackFrame * frame = metaOpenCLTopStackFrame();
	switch (type) {
	case a_db:
		kern = frame->kernel_csr_db;
		break;
	case a_fl:
		kern = frame->kernel_csr_fl;
		break;
	case a_ul:
		kern = frame->kernel_csr_ul;
		break;
	case a_in:
		kern = frame->kernel_csr_in;
		break;
	case a_ui:
		kern = frame->kernel_csr_ui;
		break;
	default:
		fprintf(stderr,
				"Error: Function 'opencl_csr' not implemented for selected type!\n");
		return -1;
		break;
	}

	ret = clSetKernelArg(kern, 0, sizeof(int), &global_size);
	ret |= clSetKernelArg(kern, 1, sizeof(cl_mem *), &csr_ap);
	ret |= clSetKernelArg(kern, 2, sizeof(cl_mem *), &csr_aj);
	ret |= clSetKernelArg(kern, 3, sizeof(cl_mem *), &csr_ax);
	ret |= clSetKernelArg(kern, 4, sizeof(cl_mem *), &x_loc);
	ret |= clSetKernelArg(kern, 5, sizeof(cl_mem *), &y_loc);
	
	//ret = clEnqueueNDRangeKernel(frame->queue, kern, 1, NULL, &grid, &block, 1, wait, event);
	ret |= clEnqueueNDRangeKernel(frame->queue, kern, 1, NULL, &grid, &block, 0, NULL, event);
	if (!async)
		ret |= clFinish(frame->queue);
	free(frame);

	return (ret);

}


cl_int opencl_crc(size_t global_size, size_t local_size,
		void * dev_input, int page_size, int num_words, int numpages, void * dev_output, 
		meta_type_id type, int async,
		// cl_event * wait, 
		cl_event * event) {

	cl_int ret = CL_SUCCESS;
	cl_kernel kern;

//TODO it doesn't do anything with size since it uses a task, either enforce it or remove it	
//TODO I doubt there is any reason for non-FPGA platforms to use a task
//TODO Since it operates on binary data, having it typed is sort of nonsense	
	metaOpenCLStackFrame * frame = metaOpenCLTopStackFrame();
	switch (type) {
	case a_db:
		kern = frame->kernel_crc_ui;
		break;
	case a_fl:
		kern = frame->kernel_crc_ui;
		break;
	case a_ul:
		kern = frame->kernel_crc_ui;
		break;
	case a_in:
		kern = frame->kernel_crc_ui;
		break;
	case a_ui:
		kern = frame->kernel_crc_ui;
		break;
	default:
		fprintf(stderr,
				"Error: Function 'opencl_csr' not implemented for selected type!\n");
		return -1;
		break;
	}

	ret = clSetKernelArg(kern, 0, sizeof(cl_mem *), &dev_input);
	ret |= clSetKernelArg(kern, 1, sizeof(cl_uint), &page_size);
	ret |= clSetKernelArg(kern, 2, sizeof(cl_uint), &num_words);
	ret |= clSetKernelArg(kern, 3, sizeof(cl_uint), &numpages);
	ret |= clSetKernelArg(kern, 4, sizeof(cl_mem *), &dev_output);
	
	//ret = clEnqueueTask(frame->queue, kern, 1, wait, event);
	ret = clEnqueueTask(frame->queue, kern, 1, NULL, event);
	
	if (!async)
		ret |= clFinish(frame->queue);
	free(frame);

	return (ret);

}

#if defined(__OPENCLCC__) || defined(__cplusplus)
}
#endif

