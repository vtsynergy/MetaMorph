/*
 * The workhorse pass-through of the library. This file implements
 *  control over which modes are compiled in, and generic wrappers
 *  for all functions which must be passed through to a specific
 *  mode's backend "core" implementation.
 *
 * Should support targeting a specific class of device, a
 *  specific accelerator model (CUDA/OpenCL/OpenMP/...), and
 *  "choose best", which would be similar to Tom's CoreTSAR.
 *
 * For now, Generic, CUDA, and OpenCL are supported.
 * Generic simply uses environment variable "METAMORPH_MODE" to select
 *  a mode at runtime.
 * OpenCL mode also supports selecting the "-1st" device, which
 *  forces it to refer to environment variable "TARGET_DEVICE" to attempt
 *  to string match the name of an OpenCL device, defaulting to the zeroth
 *  if no match is identified.
 *
 * TODO For OpenCL context switching, should we implement checks to ensure cl_mem
 *  regions are created w.r.t. the current context? Should it throw an error,
 *  or attempt to go through the stack and execute the call w.r.t. the correct
 *  context for the cl_mem pointer? What if a kernel launch mixes cl_mems from
 *  multiple contexts? TODO
 */

#include "metamorph.h"

//Globally-set mode
meta_preferred_mode run_mode = metaModePreferGeneric;


#ifdef WITH_OPENCL
cl_context meta_context = NULL;
cl_command_queue meta_queue = NULL;
cl_device_id meta_device = NULL;

//All this does is wrap calling metaOpenCLInitStackFrameDefault
// and setting meta_context and meta_queue appropriately
void metaOpenCLFallBack() {
	metaOpenCLStackFrame * frame;
	metaOpenCLInitStackFrameDefault(&frame);
	meta_context = frame->context;
	meta_queue = frame->queue;
	meta_device = frame->device;
	metaOpenCLPushStackFrame(frame);
	free(frame); //This is safe, it's just a copy of what should now be the bottom of the stack
}
#endif

//Unexposed convenience function to get the byte width of a selected type
size_t get_atype_size(meta_type_id type) {
	switch (type) {
		case a_db:
			return sizeof(double);
		break;

		case a_fl:
			return sizeof(float);
		break;

		case a_ul:
			return sizeof(unsigned long);
		break;

		case a_in:
			return sizeof(int);
		break;
		
		case a_ui:
			return sizeof(unsigned int);
		break;

		default:
			fprintf(stderr, "Error: no size retreivable for selected type [%d]!\n", type);
			return (size_t)-1;
		break;
	}
}

//TODO - Validate attempted allocation sizes against maximum supported by the device
// particularly for OpenCL (AMD 7970 doesn't support 512 512 512 reduce size
a_err meta_alloc(void ** ptr, size_t size) {
	//TODO: should always set ret to a value
	a_err ret;
	switch(run_mode) {
		default:
		case metaModePreferGeneric:
			//TODO implement generic (runtime choice) allocation
			break;

		#ifdef WITH_CUDA
		case metaModePreferCUDA:
			ret = cudaMalloc(ptr, size);
			break;
		#endif

		#ifdef WITH_OPENCL
		case metaModePreferOpenCL:
			//Make sure some context exists..
			if (meta_context == NULL) metaOpenCLFallBack();
			*ptr = (void *) clCreateBuffer(meta_context, CL_MEM_READ_WRITE, size, NULL, (cl_int *)&ret);
			break;
		#endif


		#ifdef WITH_OPENMP
		case metaModePreferOpenMP:
			*ptr = (void *) malloc(size);
			if(*ptr != NULL)
				ret = 0; // success
			break;
		#endif
	}
	return (ret);
}

//TODO implement a way for this to trigger destroying an OpenCL stack frame
// iff all cl_mems in the frame's context, as well as the frame members themselves
// have been released.
a_err meta_free(void * ptr) {
	//TODO: should always set ret to a value
	a_err ret;
	switch (run_mode) {
		default:
		case metaModePreferGeneric:
			//TODO implement generic (runtime choice) free
			break;

		#ifdef WITH_CUDA
		case metaModePreferCUDA:
			ret = cudaFree(ptr);
			break;
		#endif

		#ifdef WTIH_OPENCL
		case metaModePreferOpenCL:
			//Make sure some context exists..
			if (meta_context == NULL) metaOpenCLFallBack();
			ret = clReleaseMemObject((cl_mem)ptr);
			break;
		#endif

		#ifdef WITH_OPENMP
		case metaModePreferOpenMP:
			free(ptr);
			break;
		#endif
	}
	return (ret);
}

//Simplified wrapper for sync copies
//a_err meta_copy_h2d(void * dst, void * src, size_t size) {
//	return meta_copy_h2d(dst, src, size, true);
//}
//Workhorse for both sync and async variants
a_err meta_copy_h2d(void * dst, void * src, size_t size, a_bool async) {
	return meta_copy_h2d_cb(dst, src, size, async, (meta_callback*)NULL, NULL);
}
a_err meta_copy_h2d_cb(void * dst, void * src, size_t size, a_bool async, meta_callback *call, void *call_pl) {
	a_err ret;
	#ifdef WITH_TIMERS
	metaTimerQueueFrame * frame = (metaTimerQueueFrame *)malloc (sizeof(metaTimerQueueFrame));
	frame->mode = run_mode;
	frame->size = size;
	#endif
	switch (run_mode) {
		default:
		case metaModePreferGeneric:
			//TODO implement generic (runtime choice) H2D copy
			break;

		#ifdef WITH_CUDA
		case metaModePreferCUDA:
			#ifdef WITH_TIMERS
			cudaEventCreate(&(frame->event.cuda[0]));
			cudaEventRecord(frame->event.cuda[0], 0);
			#endif
			if (async) {
				ret = cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, 0);
			} else {
				ret = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
			}
			#ifdef WITH_TIMERS
			cudaEventCreate(&(frame->event.cuda[1]));
			cudaEventRecord(frame->event.cuda[1], 0);
			#endif
			//If a callback is provided, register it immediately after the transfer
			if ((void*)call != NULL && call_pl != NULL) cudaStreamAddCallback(0, call->cudaCallback, call_pl, 0);
			break;
		#endif

		#ifdef WITH_OPENCL
		case metaModePreferOpenCL:
			//Make sure some context exists..
			if (meta_context == NULL) metaOpenCLFallBack();
			#ifdef WITH_TIMERS
			ret = clEnqueueWriteBuffer(meta_queue, (cl_mem) dst, ((async) ? CL_FALSE : CL_TRUE), 0, size, src, 0, NULL, &(frame->event.opencl));
			//If timers exist, use their event to add the callback
			if ((void*)call != NULL && call_pl != NULL) clSetEventCallback(frame->event.opencl, CL_COMPLETE, call->openclCallback, call_pl);
			#else
			//If timers don't exist, get the event via a locally-scoped event to add to the callback
			cl_event cb_event;
			ret = clEnqueueWriteBuffer(meta_queue, (cl_mem) dst, ((async) ? CL_FALSE : CL_TRUE), 0, size, src, 0, NULL, &cb_event);
			if ((void*)call != NULL && call_pl != NULL) clSetEventCallback(cb_event, CL_COMPLETE, call->openclCallback, call_pl);
			 #endif
			break;
		#endif

		#ifdef WITH_OPENMP
		case metaModePreferOpenMP:
			#ifdef WITH_TIMERS
			frame->event.openmp[0]= omp_get_wtime();
			#endif
			memcpy(dst, src, size);
			ret = 0;
			#ifdef WITH_TIMERS
			frame->event.openmp[1]= omp_get_wtime();
			#endif
			break;
		#endif
	}
	#ifdef WITH_TIMERS
	metaTimerEnqueue(frame, &(metaBuiltinQueues[c_H2D]));
	#endif
	return (ret);
}

//Simplified wrapper for sync copies
//a_err meta_copy_d2h(void * dst, void * src, size_t) {
//	return meta_copy_d2h(dst, src, size, true);
//}
//Workhorse for both sync and async copies
a_err meta_copy_d2h(void *dst, void *src, size_t size, a_bool async) {
	return meta_copy_d2h_cb(dst, src, size, async, (meta_callback*)NULL, NULL);
}
a_err meta_copy_d2h_cb(void * dst, void * src, size_t size, a_bool async, meta_callback *call, void *call_pl) {
	a_err ret;
	#ifdef WITH_TIMERS
	metaTimerQueueFrame * frame = (metaTimerQueueFrame*)malloc (sizeof(metaTimerQueueFrame));
	frame->mode = run_mode;
	frame->size = size;
	#endif
	switch (run_mode) {
		default:
		case metaModePreferGeneric:
			//TODO implement generic (runtime choice) H2D copy
			break;

		#ifdef WITH_CUDA
		case metaModePreferCUDA:
			#ifdef WITH_TIMERS
			cudaEventCreate(&(frame->event.cuda[0]));
			cudaEventRecord(frame->event.cuda[0], 0);
			#endif
			if (async) {
				ret = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, 0);
			} else {
				ret = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
			}
			#ifdef WITH_TIMERS
			cudaEventCreate(&(frame->event.cuda[1]));
			cudaEventRecord(frame->event.cuda[1], 0);
			#endif
			//If a callback is provided, register it immediately after the transfer
			if ((void*)call != NULL && call_pl != NULL) cudaStreamAddCallback(0, call->cudaCallback, call_pl, 0);
			break;
		#endif

		#ifdef WITH_OPENCL
		case metaModePreferOpenCL:
			//Make sure some context exists..
			if (meta_context == NULL) metaOpenCLFallBack();
			#ifdef WITH_TIMERS
			ret = clEnqueueReadBuffer(meta_queue, (cl_mem) src, ((async) ? CL_FALSE : CL_TRUE), 0, size, dst, 0, NULL, &(frame->event.opencl));
			//If timers exist, use their event to add the callback
			if ((void*)call != NULL && call_pl != NULL) clSetEventCallback(frame->event.opencl, CL_COMPLETE, call->openclCallback, call_pl);
			#else
			//If timers don't exist, get the event via a locally-scoped event to add to the callback
			cl_event cb_event;
			ret = clEnqueueReadBuffer(meta_queue, (cl_mem) src, ((async) ? CL_FALSE : CL_TRUE), 0, size, dst, 0, NULL, &cb_event);
			if ((void*)call != NULL && call_pl != NULL) clSetEventCallback(cb_event, CL_COMPLETE, call->openclCallback, call_pl);
			#endif
			break;
		#endif

		#ifdef WITH_OPENMP
		case metaModePreferOpenMP:
			#ifdef WITH_TIMERS
			frame->event.openmp[0]= omp_get_wtime();
			#endif
			memcpy(dst, src, size);
			ret = 0;
			#ifdef WITH_TIMERS
			frame->event.openmp[1]= omp_get_wtime();
			#endif
			break;
		#endif
	}
	#ifdef WITH_TIMERS
	metaTimerEnqueue(frame, &(metaBuiltinQueues[c_D2H]));
	#endif
	return (ret);
}

//Simplified wrapper for sync copies
//a_err meta_copy_d2d(void * dst, void * src, size_t size) {
//	return (dst, src, size, true);
//}
//Workhorse for both sync and async copies
a_err meta_copy_d2d(void *dst, void *src, size_t size, a_bool async) {
	return meta_copy_d2d_cb(dst, src, size, async, (meta_callback*)NULL, NULL);
}
a_err meta_copy_d2d_cb(void * dst, void * src, size_t size, a_bool async, meta_callback *call, void *call_pl) {
	a_err ret;
	#ifdef WITH_TIMERS
	metaTimerQueueFrame * frame = (metaTimerQueueFrame*)malloc (sizeof(metaTimerQueueFrame));
	frame->mode = run_mode;
	frame->size = size;
	#endif
	switch (run_mode) {
		default:
		case metaModePreferGeneric:
			//TODO implement generic (runtime choice) H2D copy
			break;

		#ifdef WITH_CUDA
		case metaModePreferCUDA:
			#ifdef WITH_TIMERS
			cudaEventCreate(&(frame->event.cuda[0]));
			cudaEventRecord(frame->event.cuda[0], 0);
			#endif
			if (async) {
				ret = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, 0);
			} else {
				ret = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
			}
			#ifdef WITH_TIMERS
			cudaEventCreate(&(frame->event.cuda[1]));
			cudaEventRecord(frame->event.cuda[1], 0);
			#endif
			//If a callback is provided, register it immediately after the transfer
			if ((void*)call != NULL && call_pl != NULL) cudaStreamAddCallback(0, call->cudaCallback, call_pl, 0);
			break;
		#endif

		#ifdef WITH_OPENCL
		case metaModePreferOpenCL:
			//Make sure some context exists..
			if (meta_context == NULL) metaOpenCLFallBack();
			#ifdef WITH_TIMERS
			ret = clEnqueueCopyBuffer(meta_queue, (cl_mem) src, (cl_mem) dst, 0, 0, size, 0, NULL, &(frame->event.opencl));
			//If timers exist, use their event to add the callback
			if ((void*)call != NULL && call_pl != NULL) clSetEventCallback(frame->event.opencl, CL_COMPLETE, call->openclCallback, call_pl);
			#else
			//If timers don't exist, get the event via a locally-scoped event to add to the callback
			cl_event cb_event;
			ret = clEnqueueCopyBuffer(meta_queue, (cl_mem) src, (cl_mem) dst, 0, 0, size, 0, NULL, &cb_event);
			if ((void*)call != NULL && call_pl != NULL) clSetEventCallback(cb_event, CL_COMPLETE, call->openclCallback, call_pl);
			#endif
			//clEnqueueCopyBuffer is by default async, so clFinish
			if (!async) clFinish(meta_queue);
			break;
		#endif

		#ifdef WITH_OPENMP
		case metaModePreferOpenMP:
			#ifdef WITH_TIMERS
			frame->event.openmp[0]= omp_get_wtime();
			#endif
			memcpy(dst, src, size);
			ret = 0;
			#ifdef WITH_TIMERS
			frame->event.openmp[1]= omp_get_wtime();
			#endif
			break;
		#endif
	}
	#ifdef WITH_TIMERS
	metaTimerEnqueue(frame, &(metaBuiltinQueues[c_D2D]));
	#endif
	return (ret);
}

//In CUDA choosing an accelerator is optional, make sure the OpenCL version
// implements some reasonable default iff choose_accel is not called before the
// first OpenCL command
//TODO make this compatible with OpenCL initialization
//TODO unpack OpenCL platform from the uint's high short, and the device from the low short
a_err choose_accel(int accel, meta_preferred_mode mode) {
	a_err ret;
	run_mode = mode;
	switch(run_mode) {
		default:
		case metaModePreferGeneric:
			//TODO support generic (auto-config) runtime selection
			//TODO support "METAMORPH_MODE" environment variable.
			if (getenv("METAMORPH_MODE") != NULL) {
				#ifdef WITH_CUDA
				if (strcmp(getenv("METAMORPH_MODE"), "CUDA") == 0) return choose_accel(accel, metaModePreferCUDA);
				#endif

				#ifdef WITH_OPENCL
				if (strcmp(getenv("METAMORPH_MODE"), "OpenCL") == 0 || strcmp(getenv("METAMORPH_MODE"), "OpenCL_DEBUG") == 0) return choose_accel(accel, metaModePreferOpenCL);
				#endif

				#ifdef WITH_OPENMP
				if (strcmp(getenv("METAMORPH_MODE"), "OpenMP") == 0) return choose_accel(accel, metaModePreferOpenMP);
				#endif

				#ifdef WITH_CORETSAR
				if (strcmp(getenv("METAMORPH_MODE"), "CoreTsar") == 0) {
					fprintf(stderr, "CoreTsar mode not yet supported!\n");
					//TODO implement whatever's required to get CoreTsar going...

				}
				#endif

				fprintf(stderr, "Error: METAMORPH_MODE=\"%s\" not supported with specified compiler definitions.\n", getenv("METAMORPH_MODE"));
			}
			fprintf(stderr, "Generic Mode only supported with \"METAMORPH_MODE\" environment variable set to one of:\n");
			#ifdef WITH_CUDA
			fprintf(stderr, "\"CUDA\"\n");
			#endif
			#ifdef WITH_OPENCL
			fprintf(stderr, "\"OpenCL\" (or \"OpenCL_DEBUG\")\n");
			#endif
			#ifdef WITH_OPENMP
			fprintf(stderr, "\"OpenMP\"\n");
			#endif
			#ifdef WITH_CORETSAR
			//TODO - Left as a stub to remind us that Generic Mode was intended to be used for
			// CoreTsar auto configuration across devices/modes
			fprintf(stderr, "\"CoreTsar\"\n");
			#endif
			break;

		#ifdef WITH_CUDA
		case metaModePreferCUDA:
			ret = cudaSetDevice(accel);
			printf("CUDA Mode selected with device: %d\n", accel);
			//TODO add a statement printing the device's name
			break;
		#endif

		#ifdef WITH_OPENCL
		case metaModePreferOpenCL:
			{metaOpenCLStackFrame * frame;
				metaOpenCLInitStackFrame(&frame, (cl_int) accel); //no hazards, frames are thread-private
				//make sure this library knows what the opencl library is using internally..
				meta_context = frame->context;
				meta_queue = frame->queue;
				meta_device = frame->device;

				metaOpenCLPushStackFrame(frame); //no hazards, HPs are internally managed when copying the frame to a new stack node before pushing.

				//Now it's safe to free the frame
				// But not to destroy it, as we shouldn't release the frame members
				free(frame);
				//If users request it, a full set of contexts could be pre-initialized..
				// but we can lessen overhead by only eagerly initializing one.
				//fprintf(stderr, "OpenCL Mode not yet implemented!\n");
			}
			break;
		#endif

		#ifdef WITH_OPENMP
		case metaModePreferOpenMP:
			printf("OpenMP Mode selected\n");
			break;
		#endif
	}
	return(ret);
}

//TODO make this compatible with OpenCL device querying
//TODO pack OpenCL platform into the uint's high short, and the device into the low short
a_err get_accel(int * accel, meta_preferred_mode * mode) {
	a_err ret;
	switch(run_mode) {
		default:
		case metaModePreferGeneric:
			//TODO implement a generic response for which device was runtime selected
			//fprintf(stderr, "Generic Device Query not yet implemented!\n");
			*mode = metaModePreferGeneric;
			break;

		#ifdef WITH_CUDA
		case metaModePreferCUDA:
			ret = cudaGetDevice(accel);
			*mode = metaModePreferCUDA;
			break;
		#endif

		#ifdef WITH_OPENCL
		case metaModePreferOpenCL:
			//Make sure some context exists..
			if (meta_context == NULL) metaOpenCLFallBack();
			//TODO implement appropriate response for OpenCL device number
			//TODO implement this based on metaOpenCLTopStackFrame
			//fprintf(stderr, "OpenCL Device Query not yet implemented!\n");
			*mode = metaModePreferOpenCL;
			break;
		#endif

		#ifdef WITH_OPENMP
		case metaModePreferOpenMP:
			*mode = metaModePreferOpenMP;
			break;
		#endif
	}
	return(ret);
}

//Stub: checks that grid_size is a valid multiple of block_size, and that
// both obey the bounds specified for the device/execution model currently in use.
// For now CUDA and OpenMP are NOOP
//For now it does not directly modify the work size at all
// nor does it return the ideal worksize in any way
// it just produces a STDERR comment stating which variable is out of bounds
// what the bound is, and what the variable is currently.
a_err meta_validate_worksize(a_dim3 * grid_size, a_dim3 * block_size) {
	a_err ret;
	switch(run_mode) {
		default:
		case metaModePreferGeneric:
			//TODO the only reason we should still be here is CoreTsar
			// Don't worry about doing anything until when/if we add that
			break;

		#ifdef WITH_CUDA
		case metaModePreferCUDA:
			//TODO implement whatever bounds checking is needed by CUDA
			return 0;
			break;
		#endif

		#ifdef WITH_OPENCL
		case metaModePreferOpenCL:
			//Make sure some context exists..
			if (meta_context == NULL) metaOpenCLFallBack();
			size_t max_wg_size, max_wg_dim_sizes[3];
			ret = clGetDeviceInfo(meta_device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_wg_size, NULL);
			ret |= clGetDeviceInfo(meta_device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t)*3, &max_wg_dim_sizes, NULL);
			if ((*block_size)[0] * (*block_size)[1] * (*block_size)[2] > max_wg_size)
				{fprintf(stderr, "Error: Maximum block volume is: %lu\nRequested block volume of: %lu (%lu * %lu * %lu) not supported!\n", max_wg_size, (*block_size)[0] * (*block_size)[1] * (*block_size)[2], (*block_size)[0], (*block_size)[1], (*block_size)[2]); ret |= -1;}
			
			if ((*block_size)[0] > max_wg_dim_sizes[0]) {fprintf(stderr, "Error: Maximum block size for dimension 0 is: %lu\nRequested 0th dimension size of: %lu not supported\n!", max_wg_dim_sizes[0], (*block_size)[0]); ret |= -1;}
			if ((*block_size)[1] > max_wg_dim_sizes[1]) {fprintf(stderr, "Error: Maximum block size for dimension 1 is: %lu\nRequested 1st dimension size of: %lu not supported\n!", max_wg_dim_sizes[1], (*block_size)[1]); ret |= -1;}
			if ((*block_size)[2] > max_wg_dim_sizes[2]) {fprintf(stderr, "Error: Maximum block size for dimension 2 is: %lu\nRequested 2nd dimension size of: %lu not supported\n!", max_wg_dim_sizes[2], (*block_size)[2]); ret |= -1;}
			return(ret);
		break;
		#endif

		#ifdef WITH_OPENMP
		case metaModePreferOpenMP:
			//TODO implement any bounds checking OpenMP may need
			return 0;
			break;
		#endif
	}
	return(ret);
}

//Flush does exactly what it sounds like - forces any
// outstanding work to be finished before returning
//For now it handles flushing async kernels, and finishing outstanding
// transfers from the MPI plugin
//It could easily handle timer dumping as well, but that might not be ideal
a_err meta_flush() {
	switch(run_mode) {
		default:
		case metaModePreferGeneric:
			//TODO implement a generic flush?
			break;

		#ifdef WITH_CUDA
		case metaModePreferCUDA:
			cudaThreadSynchronize();
			break;
		#endif

		#ifdef WITH_OPENCL
		case metaModePreferOpenCL:
			//Make sure some context exists..
			if (meta_context == NULL) metaOpenCLFallBack();
			clFinish(meta_queue);
			break;
		#endif

		#ifdef WITH_OPENMP
		case metaModePreferOpenMP:
			#pragma omp barrier // synchronize threads
			break;
		#endif
	}
	//Flush all outstanding MPI work
	// We do this after flushing the GPUs as any packing will be finished
	#ifdef WITH_MPI
	finish_mpi_requests();
	#endif
	printf("\n");	
}

//Simple wrapper for synchronous kernel
//a_err meta_dotProd(a_dim3 * grid_size, a_dim3 * block_size, a_double * data1, a_double * data2, a_dim3 * array_size, a_dim3 * array_start, a_dim3 * array_end, a_double * reduction_var) {
//	return meta_dotProd(grid_size, block_size, data1, data2, array_size, array_start, array_end, reduction_var, true);
//}
//Workhorse for both sync and async dot products
a_err meta_dotProd(a_dim3 * grid_size, a_dim3 * block_size, void * data1, void * data2, a_dim3 * array_size, a_dim3 * array_start, a_dim3 * array_end, void * reduction_var, meta_type_id type, a_bool async) {
	return meta_dotProd_cb(grid_size, block_size, data1, data2, array_size, array_start, array_end, reduction_var, type, async, (meta_callback*)NULL, NULL);
}
a_err meta_dotProd_cb(a_dim3 * grid_size, a_dim3 * block_size, void * data1, void * data2, a_dim3 * array_size, a_dim3 * array_start, a_dim3 * array_end, void * reduction_var, meta_type_id type, a_bool async, meta_callback *call, void *call_pl) {
	a_err ret;

	//FIXME? Consider adding a compiler flag "UNCHECKED_EXPLICIT" to streamline out sanity checks like this
	//Before we do anything, sanity check the start/end/size
	if (array_start == NULL || array_end == NULL || array_size == NULL) {
		fprintf(stderr, "ERROR in meta_dotProd: array_start=[%p], array_end=[%p], or array_size=[%p] is NULL!\n", array_start, array_end, array_size);
		return -1;
	}
	int i;
	for (i = 0; i < 3; i ++) {
		if ((*array_start)[i] < 0 || (*array_end)[i] < 0) {
			fprintf(stderr, "ERROR in meta_dotProd: array_start[%d]=[%ld] or array_end[%d]=[%ld] is negative!\n", i, (*array_start)[i], i, (*array_end)[i]);
			return -1;
		}
		if ((*array_size)[i] < 1) {
			fprintf(stderr, "ERROR in meta_dotProd: array_size[%d]=[%ld] must be >=1!\n", i, (*array_size)[i]);
			return -1;
		}
		if ((*array_start)[i] > (*array_end)[i]) {
			fprintf(stderr, "ERROR in meta_dotProd: array_start[%d]=[%ld] is after array_end[%d]=[%ld]!\n", i, (*array_start)[i], i, (*array_end)[i]);
			return -1;
		}
		if ((*array_end)[i] >= (*array_size)[i]) {
			fprintf(stderr, "ERROR in meta_dotProd: array_end[%d]=[%ld] is bigger than array_size[%d]=[%ld]!\n", i, (*array_end)[i], i, (*array_size)[i]);
			return -1;
		}

	}
	//Ensure the block is all powers of two
	// do not fail if not, but rescale and emit a warning
	if (grid_size != NULL && block_size != NULL) {
		int flag = 0;
		size_t new_block[3];
		size_t new_grid[3];
		for (i = 0; i < 3; i++) {
			new_block[i] = (*block_size)[i];
			new_grid[i] = (*grid_size)[i];
			//Bit-twiddle our way to the next-highest power of 2, from: (checked 2015.01.06)
			//http://graphics.standford.edu/~seander/bithacks.html#RoundUpPowerOf2
			new_block[i]--;
			new_block[i] |= new_block[i] >> 1;
			new_block[i] |= new_block[i] >> 2;
			new_block[i] |= new_block[i] >> 4;
			new_block[i] |= new_block[i] >> 8;
			new_block[i] |= new_block[i] >> 16;
			new_block[i]++;
			if (new_block[i] != (*block_size)[i]) {
				flag = 1; //Trip the flag to emit a warning
				new_grid[i] = ((*block_size)[i]*(*grid_size)[i]-1+new_block[i])/new_block[i];
			}
		}
		if (flag) {
			fprintf(stderr, "WARNING in meta_dotProd: block_size={%ld, %ld, %ld} must be all powers of two!\n\tRescaled grid_size={%ld, %ld, %ld}, block_size={%ld, %ld, %ld} to\n\tnew_grid={%ld, %ld, %ld}, new_block={%ld, %ld, %ld}\n", (*block_size)[0], (*block_size)[1], (*block_size)[2], (*grid_size)[0], (*grid_size)[1], (*grid_size)[2], (*block_size)[0], (*block_size)[1], (*block_size)[2], new_grid[0], new_grid[1], new_grid[2], new_block[0], new_block[1], new_block[2]);
			(*grid_size)[0] = new_grid[0];
			(*grid_size)[1] = new_grid[1];
			(*grid_size)[2] = new_grid[2];
			(*block_size)[0] = new_block[0];
			(*block_size)[1] = new_block[1];
			(*block_size)[2] = new_block[2];
		}
	}

	#ifdef WITH_TIMERS
	metaTimerQueueFrame * frame = (metaTimerQueueFrame*)malloc (sizeof(metaTimerQueueFrame));
	frame->mode = run_mode;
	frame->size = (*array_size)[0]*(*array_size)[1]*(*array_size)[2]*get_atype_size(type);
	#endif
	switch(run_mode) {
		default:
		case metaModePreferGeneric:
			//TODO implement a generic reduce
			break;

		#ifdef WITH_CUDA
		case metaModePreferCUDA: {
						#ifdef WITH_TIMERS
						  ret = (a_err) cuda_dotProd(grid_size, block_size, data1, data2, array_size, array_start, array_end, reduction_var, type, async, &(frame->event.cuda));
						#else
						  ret = (a_err) cuda_dotProd(grid_size, block_size, data1, data2, array_size, array_start, array_end, reduction_var, type, async, NULL);
						#endif
			//If a callback is provided, register it immediately after returning from enqueuing the kernel
			if ((void*)call != NULL && call_pl != NULL) cudaStreamAddCallback(0, call->cudaCallback, call_pl, 0);
						  break;
					  }
		#endif

		#ifdef WITH_OPENCL
		case metaModePreferOpenCL:
					  //Make sure some context exists..
					  if (meta_context == NULL) metaOpenCLFallBack();
					  #ifdef WITH_TIMERS
					  ret = (a_err) opencl_dotProd(grid_size, block_size, data1, data2, array_size, array_start, array_end, reduction_var, type, async, &(frame->event.opencl));
			//If timers exist, use their event to add the callback
			if ((void*)call != NULL && call_pl != NULL) clSetEventCallback(frame->event.opencl, CL_COMPLETE, call->openclCallback, call_pl);
					  #else
			//If timers don't exist, get the event via a locally-scoped event to add to the callback
			cl_event cb_event;
					  ret = (a_err) opencl_dotProd(grid_size, block_size, data1, data2, array_size, array_start, array_end, reduction_var, type, async, &cb_event);
			if ((void*)call != NULL && call_pl != NULL) clSetEventCallback(cb_event, CL_COMPLETE, call->openclCallback, call_pl);
					  #endif
					  break;
		#endif

		#ifdef WITH_OPENMP
		case metaModePreferOpenMP:
					#ifdef WITH_TIMERS
					frame->event.openmp[0]= omp_get_wtime();
					#endif
					ret = omp_dotProd(grid_size, block_size, data1, data2, array_size, array_start,  array_end, reduction_var, type, async);
					#ifdef WITH_TIMERS
					frame->event.openmp[1]= omp_get_wtime();
					#endif
					 break;
		#endif
	}
	#ifdef WITH_TIMERS
	metaTimerEnqueue(frame, &(metaBuiltinQueues[k_dotProd]));
	#endif
	return(ret);
}

//Simplified wrapper for synchronous kernel
//a_err meta_reduce(a_dim3 * grid_size, a_dim3 * block_size, a_double * data, a_dim3 * array_size, a_dim3 * array_start, a_dim3 * array_end, a_double * reduction_var) {
//	return meta_reduce(grid_size, block_size, data, array_size, array_start, array_end, reduction_var, true);
//}
//Workhorse for both sync and async reductions
a_err meta_reduce(a_dim3 * grid_size, a_dim3 * block_size, void * data, a_dim3 * array_size, a_dim3 * array_start, a_dim3 * array_end, void * reduction_var, meta_type_id type, a_bool async) {
	return meta_reduce_cb(grid_size, block_size, data, array_size, array_start, array_end, reduction_var, type, async, (meta_callback*)NULL, NULL);
}
a_err meta_reduce_cb(a_dim3 * grid_size, a_dim3 * block_size, void * data, a_dim3 * array_size, a_dim3 * array_start, a_dim3 * array_end, void * reduction_var, meta_type_id type, a_bool async, meta_callback *call, void *call_pl) {
	a_err ret;

	//FIXME? Consider adding a compiler flag "UNCHECKED_EXPLICIT" to streamline out sanity checks like this
	//Before we do anything, sanity check the start/end/size
	if (array_start == NULL || array_end == NULL || array_size == NULL) {
		fprintf(stderr, "ERROR in meta_reduce: array_start=[%p], array_end=[%p], or array_size=[%p] is NULL!\n", array_start, array_end, array_size);
		return -1;
	}
	int i;
	for (i = 0; i < 3; i ++) {
		if ((*array_start)[i] < 0 || (*array_end)[i] < 0) {
			fprintf(stderr, "ERROR in meta_reduce: array_start[%d]=[%ld] or array_end[%d]=[%ld] is negative!\n", i, (*array_start)[i], i, (*array_end)[i]);
			return -1;
		}
		if ((*array_size)[i] < 1) {
			fprintf(stderr, "ERROR in meta_reduce: array_size[%d]=[%ld] must be >=1!\n", i, (*array_size)[i]);
			return -1;
		}
		if ((*array_start)[i] > (*array_end)[i]) {
			fprintf(stderr, "ERROR in meta_reduce: array_start[%d]=[%ld] is after array_end[%d]=[%ld]!\n", i, (*array_start)[i], i, (*array_end)[i]);
			return -1;
		}
		if ((*array_end)[i] >= (*array_size)[i]) {
			fprintf(stderr, "ERROR in meta_reduce: array_end[%d]=[%ld] is bigger than array_size[%d]=[%ld]!\n", i, (*array_end)[i], i, (*array_size)[i]);
			return -1;
		}

	}
	//Ensure the block is all powers of two
	// do not fail if not, but rescale and emit a warning
	if (grid_size != NULL && block_size != NULL) {
		int flag = 0;
		size_t new_block[3];
		size_t new_grid[3];
		for (i = 0; i < 3; i++) {
			new_block[i] = (*block_size)[i];
			new_grid[i] = (*grid_size)[i];
			//Bit-twiddle our way to the next-highest power of 2, from: (checked 2015.01.06)
			//http://graphics.standford.edu/~seander/bithacks.html#RoundUpPowerOf2
			new_block[i]--;
			new_block[i] |= new_block[i] >> 1;
			new_block[i] |= new_block[i] >> 2;
			new_block[i] |= new_block[i] >> 4;
			new_block[i] |= new_block[i] >> 8;
			new_block[i] |= new_block[i] >> 16;
			new_block[i]++;
			if (new_block[i] != (*block_size)[i]) {
				flag = 1; //Trip the flag to emit a warning
				new_grid[i] = ((*block_size)[i]*(*grid_size)[i]-1+new_block[i])/new_block[i];
			}
		}
		if (flag) {
			fprintf(stderr, "WARNING in meta_reduce: block_size={%ld, %ld, %ld} must be all powers of two!\n\tRescaled grid_size={%ld, %ld, %ld}, block_size={%ld, %ld, %ld} to\n\tnew_grid={%ld, %ld, %ld}, new_block={%ld, %ld, %ld}\n", (*block_size)[0], (*block_size)[1], (*block_size)[2], (*grid_size)[0], (*grid_size)[1], (*grid_size)[2], (*block_size)[0], (*block_size)[1], (*block_size)[2], new_grid[0], new_grid[1], new_grid[2], new_block[0], new_block[1], new_block[2]);
			(*grid_size)[0] = new_grid[0];
			(*grid_size)[1] = new_grid[1];
			(*grid_size)[2] = new_grid[2];
			(*block_size)[0] = new_block[0];
			(*block_size)[1] = new_block[1];
			(*block_size)[2] = new_block[2];
		}
	}

	#ifdef WITH_TIMERS
	metaTimerQueueFrame * frame = (metaTimerQueueFrame*)malloc (sizeof(metaTimerQueueFrame));
	frame->mode = run_mode;
	frame->size = (*array_size)[0]*(*array_size)[1]*(*array_size)[2]*get_atype_size(type);
	#endif
	switch(run_mode) {
		default:
		case metaModePreferGeneric:
			//TODO implement a generic reduce
			break;

		#ifdef WITH_CUDA
		case metaModePreferCUDA: {
						#ifdef WITH_TIMERS
						  ret = (a_err) cuda_reduce(grid_size, block_size, data, array_size, array_start, array_end, reduction_var, type, async, &(frame->event.cuda));
						#else
						  ret = (a_err) cuda_reduce(grid_size, block_size, data, array_size, array_start, array_end, reduction_var, type, async, NULL);
						#endif
			//If a callback is provided, register it immediately after returning from enqueuing the kernel
			if ((void*)call != NULL && call_pl != NULL) cudaStreamAddCallback(0, call->cudaCallback, call_pl, 0);
						  break;
					  }
		#endif

		#ifdef WITH_OPENCL
		case metaModePreferOpenCL:
					  //Make sure some context exists..
					  if (meta_context == NULL) metaOpenCLFallBack();
					  #ifdef WITH_TIMERS
					  ret = (a_err) opencl_reduce(grid_size, block_size, data, array_size, array_start, array_end, reduction_var, type, async, &(frame->event.opencl));
			//If timers exist, use their event to add the callback
			if ((void*)call != NULL && call_pl != NULL) clSetEventCallback(frame->event.opencl, CL_COMPLETE, call->openclCallback, call_pl);
					  #else
			//If timers don't exist, get the event via a locally-scoped event to add to the callback
			cl_event cb_event;
					  ret = (a_err) opencl_reduce(grid_size, block_size, data, array_size, array_start, array_end, reduction_var, type, async, &cb_event);
			if ((void*)call != NULL && call_pl != NULL) clSetEventCallback(cb_event, CL_COMPLETE, call->openclCallback, call_pl);
					  #endif
					  break;
		#endif

		#ifdef WITH_OPENMP
		case metaModePreferOpenMP:
					#ifdef WITH_TIMERS
					frame->event.openmp[0]= omp_get_wtime();
					#endif
					ret = omp_reduce(grid_size, block_size, data, array_size, array_start,  array_end, reduction_var, type, async);
					#ifdef WITH_TIMERS
					frame->event.openmp[1]= omp_get_wtime();
					#endif
					break;
		#endif
	}
	#ifdef WITH_TIMERS
	metaTimerEnqueue(frame, &(metaBuiltinQueues[k_reduce]));
	#endif
	return(ret);
}

meta_2d_face_indexed * meta_get_face_index(int s, int c, int *si, int *st) {
	//Unlike Kaixi's, we return a pointer copy, to ease Fortran implementation
	meta_2d_face_indexed * face = (meta_2d_face_indexed*)malloc(sizeof(meta_2d_face_indexed));
	//We create our own copy of size and stride arrays to prevent
	// issues if the user unexpectedly reuses or frees the original pointer
	size_t sz = sizeof(int)*c;
	face->size = (int*)malloc(sz);
	face->stride = (int*)malloc(sz);
	memcpy((void*) face->size, (const void *)si, sz);
	memcpy((void*) face->stride, (const void *)st, sz);

	face->start = s;
	face->count = c;
}

//Simple deallocator for an meta_2d_face_indexed type
// Assumes face, face->size, and ->stride are unfreed
//This is the only way a user should release a face returned
// from meta_get_face_index, and should not be used
// if the face was assembled by hand.
int meta_free_face_index(meta_2d_face_indexed * face) {
	free(face->size);
	free(face->stride);
	free(face);
}

a_err meta_transpose_2d_face(a_dim3 * grid_size, a_dim3 * block_size, void *indata, void *outdata, a_dim3 * arr_dim_xy, a_dim3 * tran_dim_xy, meta_type_id type, a_bool async) {
	return meta_transpose_2d_face_cb(grid_size, block_size, indata, outdata, arr_dim_xy, tran_dim_xy, type, async, (meta_callback*)NULL, NULL);
}
a_err meta_transpose_2d_face_cb(a_dim3 * grid_size, a_dim3 * block_size, void *indata, void *outdata, a_dim3 * arr_dim_xy, a_dim3 * tran_dim_xy, meta_type_id type, a_bool async, meta_callback *call, void *call_pl) {
	a_err ret;
	//FIXME? Consider adding a compiler flag "UNCHECKED_EXPLICIT" to streamline out sanity checks like this
	//Before we do anything, sanity check that trans_dim_xy fits inside arr_dim_xy
	if (arr_dim_xy == NULL || tran_dim_xy == NULL) {
		fprintf(stderr, "ERROR in meta_transpose_2d_face: arr_dim_xy=[%p] or tran_dim_xy=[%p] is NULL!\n", arr_dim_xy, tran_dim_xy);
		return -1;
	}
	int i;
	for (i = 0; i < 2; i ++) {
		if ((*arr_dim_xy)[i] < 1 || (*tran_dim_xy)[i] < 1) {
			fprintf(stderr, "ERROR in meta_transpose_2d_face: arr_dim_xy[%d]=[%ld] and tran_dim_xy[%d]=[%ld] must be >=1!\n", i, (*arr_dim_xy)[i], i, (*tran_dim_xy)[i]);
			return -1;
		}
		if ((*arr_dim_xy)[i] < (*tran_dim_xy)[i]) {
			fprintf(stderr, "ERROR in meta_transpose_2d_face: tran_dim_xy[%d]=[%ld] must be <= arr_dim_xy[%d]=[%ld]!\n", i, (*tran_dim_xy)[i], i, (*arr_dim_xy)[i]);
			return -1;
		}

	}
	#ifdef WITH_TIMERS
	metaTimerQueueFrame * frame = (metaTimerQueueFrame*)malloc (sizeof(metaTimerQueueFrame));
	frame->mode = run_mode;
	frame->size = (*tran_dim_xy)[0]*(*tran_dim_xy)[1]*get_atype_size(type);
	#endif
	switch(run_mode) {
		default:
		case metaModePreferGeneric:
			//TODO implement a generic reduce
			break;

		#ifdef WITH_CUDA
		case metaModePreferCUDA: {
						#ifdef WITH_TIMERS
						  ret = (a_err) cuda_transpose_2d_face(grid_size, block_size, indata, outdata, arr_dim_xy, tran_dim_xy, type, async, &(frame->event.cuda));
						#else
						  ret = (a_err) cuda_transpose_2d_face(grid_size, block_size, indata, outdata, arr_dim_xy, tran_dim_xy, type, async, NULL);
						#endif
			//If a callback is provided, register it immediately after returning from enqueuing the kernel
			if ((void*)call != NULL && call_pl != NULL) cudaStreamAddCallback(0, call->cudaCallback, call_pl, 0);
						  break;
					  }
		#endif

		#ifdef WITH_OPENCL
		case metaModePreferOpenCL:
					  //Make sure some context exists..
					  if (meta_context == NULL) metaOpenCLFallBack();
					  #ifdef WITH_TIMERS
					  ret = (a_err) opencl_transpose_2d_face(grid_size, block_size, indata, outdata, arr_dim_xy, tran_dim_xy, type, async, &(frame->event.opencl));
			//If timers exist, use their event to add the callback
			if ((void*)call != NULL && call_pl != NULL) clSetEventCallback(frame->event.opencl, CL_COMPLETE, call->openclCallback, call_pl);
					  #else
			//If timers don't exist, get the event via a locally-scoped event to add to the callback
			cl_event cb_event;
					  ret = (a_err) opencl_transpose_2d_face(grid_size, block_size, indata, outdata, arr_dim_xy, tran_dim_xy, type, async, &cb_event);
			if ((void*)call != NULL && call_pl != NULL) clSetEventCallback(cb_event, CL_COMPLETE, call->openclCallback, call_pl);
					  #endif
					  break;
		#endif

		#ifdef WITH_OPENMP
		case metaModePreferOpenMP:
					  #ifdef WITH_TIMERS
					  frame->event.openmp[0]= omp_get_wtime();
					  #endif
					  ret = omp_transpose_2d_face(grid_size, block_size, indata, outdata, arr_dim_xy, tran_dim_xy, type, async);
					  #ifdef WITH_TIMERS
					  frame->event.openmp[1]= omp_get_wtime();
					  #endif
					  break;

		#endif
	}
	#ifdef WITH_TIMERS
	metaTimerEnqueue(frame, &(metaBuiltinQueues[k_transpose_2d_face]));
	#endif
	return(ret);
}
a_err meta_pack_2d_face(a_dim3 * grid_size, a_dim3 * block_size, void *packed_buf, void *buf, meta_2d_face_indexed *face, meta_type_id type, a_bool async) {
	return meta_pack_2d_face_cb(grid_size, block_size, packed_buf, buf, face, type, async, (meta_callback*)NULL, NULL);
} 
a_err meta_pack_2d_face_cb(a_dim3 * grid_size, a_dim3 * block_size, void *packed_buf, void *buf, meta_2d_face_indexed *face, meta_type_id type, a_bool async, meta_callback *call, void *call_pl) {
	a_err ret;
	//FIXME? Consider adding a compiler flag "UNCHECKED_EXPLICIT" to streamline out sanity checks like this
	//Before we do anything, sanity check that the face is set up
	if (face == NULL) {
		fprintf(stderr, "ERROR in meta_pack_2d_face: face=[%p] is NULL!\n", face);
		return -1;
	}
	if (face->size == NULL || face->stride == NULL) {
		fprintf(stderr, "ERROR in meta_pack_2d_face: face->size=[%p] or face->stride=[%p] is NULL!\n", face->size, face->stride);
		return -1;
	}
	#ifdef WITH_TIMERS
	//TODO: Add another queue for copies into constant memory
	//TODO: Hoist copies into constant memory out of the cores to here
	metaTimerQueueFrame * frame_k1 = (metaTimerQueueFrame*)malloc (sizeof(metaTimerQueueFrame));
	metaTimerQueueFrame * frame_c1 = (metaTimerQueueFrame*)malloc (sizeof(metaTimerQueueFrame));
	metaTimerQueueFrame * frame_c2 = (metaTimerQueueFrame*)malloc (sizeof(metaTimerQueueFrame));
	metaTimerQueueFrame * frame_c3 = (metaTimerQueueFrame*)malloc (sizeof(metaTimerQueueFrame));
	frame_k1->mode = run_mode;
	frame_c1->mode = run_mode;
	frame_c2->mode = run_mode;
	frame_c3->mode = run_mode;
//	frame->size = (*dim_xy)[0]*(*dim_xy)[1]*get_atype_size(type);
	#endif

	//figure out what the aggregate size of all descendant branches are
	int *remain_dim = (int *)malloc(sizeof(int)*face->count);
	int i,j;
	remain_dim[face->count-1] = 1;
	//This loop is backwards from Kaixi's to compute the child size in O(n)
	// rather than O(n^2) by recognizing that the size of nodes higher in the tree
	// is just the size of a child multiplied by the number of children, applied
	// upwards from the leaves
	for (i = face->count-2; i >= 0; i--) {
		remain_dim[i] = remain_dim[i+1]*face->size[i+1];
	//	printf("Remain_dim[%d]: %d\n", i, remain_dim[i]);
	}

//	for(i = 0; i < face->count; i++){
//		remain_dim[i] = 1;
//		for(j=i+1; j < face->count; j++) {
//			remain_dim[i] *=face->size[j];
//		}
//			printf("Remain_dim[%d]: %d\n", i, remain_dim[i]);
//	}	
	
	switch(run_mode) {
		default:
		case metaModePreferGeneric:
			//TODO implement a generic reduce
			break;

		#ifdef WITH_CUDA
		case metaModePreferCUDA: {
						#ifdef WITH_TIMERS
						  ret = (a_err) cuda_pack_2d_face(grid_size, block_size, packed_buf, buf, face, remain_dim, type, async, &(frame_k1->event.cuda), &(frame_c1->event.cuda), &(frame_c2->event.cuda), &(frame_c3->event.cuda));
						#else
						  ret = (a_err) cuda_pack_2d_face(grid_size, block_size, packed_buf, buf, face, remain_dim, type, async, NULL, NULL, NULL, NULL);
						#endif
			//If a callback is provided, register it immediately after returning from enqueuing the kernel
			if ((void*)call != NULL && call_pl != NULL) cudaStreamAddCallback(0, call->cudaCallback, call_pl, 0);
						  break;
					  }
		#endif

		#ifdef WITH_OPENCL
		case metaModePreferOpenCL:
					  //Make sure some context exists..
					  if (meta_context == NULL) metaOpenCLFallBack();
					  #ifdef WITH_TIMERS
					  ret = (a_err) opencl_pack_2d_face(grid_size, block_size, packed_buf, buf, face, remain_dim, type, async, &(frame_k1->event.opencl), &(frame_c1->event.opencl), &(frame_c2->event.opencl), &(frame_c3->event.opencl));
			//If timers exist, use their event to add the callback
			if ((void*)call != NULL && call_pl != NULL) clSetEventCallback(frame_k1->event.opencl, CL_COMPLETE, call->openclCallback, call_pl);
					  #else
			//If timers don't exist, get the event via a locally-scoped event to add to the callback
			cl_event cb_event;
					  ret = (a_err) opencl_pack_2d_face(grid_size, block_size, packed_buf, buf, face, remain_dim, type, async, NULL, NULL, NULL, &cb_event);
			if ((void*)call != NULL && call_pl != NULL) clSetEventCallback(cb_event, CL_COMPLETE, call->openclCallback, call_pl);
					  #endif
					  break;
		#endif

		#ifdef WITH_OPENMP
		case metaModePreferOpenMP:
						#ifdef WITH_TIMERS
						frame_k1->event.openmp[0]= omp_get_wtime();
						#endif
						ret = omp_pack_2d_face(grid_size, block_size, packed_buf, buf, face, remain_dim, type, async);
						#ifdef WITH_TIMERS
						frame_k1->event.openmp[1]= omp_get_wtime();
						frame_c1->event.openmp[0] = 0.0;
						frame_c1->event.openmp[1] = 0.0;
						frame_c2->event.openmp[0] = 0.0;
						frame_c2->event.openmp[1] = 0.0;
						frame_c3->event.openmp[0] = 0.0;
						frame_c3->event.openmp[1] = 0.0;
						#endif
					  break;
		#endif
	}
	#ifdef WITH_TIMERS
	//TODO Add queue c_H2Dc for copies into constant memory
	metaTimerEnqueue(frame_k1, &(metaBuiltinQueues[k_pack_2d_face]));
	metaTimerEnqueue(frame_c1, &(metaBuiltinQueues[c_H2Dc]));
	metaTimerEnqueue(frame_c2, &(metaBuiltinQueues[c_H2Dc]));
	metaTimerEnqueue(frame_c3, &(metaBuiltinQueues[c_H2Dc]));
	#endif
	return(ret);
}

//TODO fix frame->size to reflect face size
a_err meta_unpack_2d_face(a_dim3 * grid_size, a_dim3 * block_size, void *packed_buf, void *buf, meta_2d_face_indexed *face, meta_type_id type, a_bool async) {
	return meta_unpack_2d_face_cb(grid_size, block_size, packed_buf, buf, face, type, async, (meta_callback*)NULL, NULL);
}
a_err meta_unpack_2d_face_cb(a_dim3 * grid_size, a_dim3 * block_size, void *packed_buf, void *buf, meta_2d_face_indexed *face, meta_type_id type, a_bool async, meta_callback *call, void *call_pl) {
	a_err ret;
	//FIXME? Consider adding a compiler flag "UNCHECKED_EXPLICIT" to streamline out sanity checks like this
	//Before we do anything, sanity check that the face is set up
	if (face == NULL) {
		fprintf(stderr, "ERROR in meta_unpack_2d_face: face=[%p] is NULL!\n", face);
		return -1;
	}
	if (face->size == NULL || face->stride == NULL) {
		fprintf(stderr, "ERROR in meta_unpack_2d_face: face->size=[%p] or face->stride=[%p] is NULL!\n", face->size, face->stride);
		return -1;
	}
	#ifdef WITH_TIMERS
	//TODO: Add another queue for copies into constant memory
	//TODO: Hoist copies into constant memory out of the cores to here
	metaTimerQueueFrame * frame_k1 = (metaTimerQueueFrame*)malloc (sizeof(metaTimerQueueFrame));
	metaTimerQueueFrame * frame_c1 = (metaTimerQueueFrame*)malloc (sizeof(metaTimerQueueFrame));
	metaTimerQueueFrame * frame_c2 = (metaTimerQueueFrame*)malloc (sizeof(metaTimerQueueFrame));
	metaTimerQueueFrame * frame_c3 = (metaTimerQueueFrame*)malloc (sizeof(metaTimerQueueFrame));
	frame_k1->mode = run_mode;
	frame_c1->mode = run_mode;
	frame_c2->mode = run_mode;
	frame_c3->mode = run_mode;
//	frame->size = (*dim_xy)[0]*(*dim_xy)[1]*get_atype_size(type);
	#endif

	//figure out what the aggregate size of all descendant branches are
	int *remain_dim = (int *)malloc(sizeof(int)*face->count);
	int i;
	remain_dim[face->count-1] = 1;
	//This loop is backwards from Kaixi's to compute the child size in O(n)
	// rather than O(n^2) by recognizing that the size of nodes higher in the tree
	// is just the size of a child multiplied by the number of children, applied
	// upwards from the leaves
	for (i = face->count-2; i >= 0; i--) {
		remain_dim[i] = remain_dim[i+1]*face->size[i+1];
	}	
	
	switch(run_mode) {
		default:
		case metaModePreferGeneric:
			//TODO implement a generic reduce
			break;

		#ifdef WITH_CUDA
		case metaModePreferCUDA: {
						#ifdef WITH_TIMERS
						  ret = (a_err) cuda_unpack_2d_face(grid_size, block_size, packed_buf, buf, face, remain_dim, type, async, &(frame_k1->event.cuda), &(frame_c1->event.cuda), &(frame_c2->event.cuda), &(frame_c3->event.cuda));
						#else
						  ret = (a_err) cuda_unpack_2d_face(grid_size, block_size, packed_buf, buf, face, remain_dim, type, async, NULL, NULL, NULL, NULL);
						#endif
			//If a callback is provided, register it immediately after returning from enqueuing the kernel
			if ((void*)call != NULL && call_pl != NULL) cudaStreamAddCallback(0, call->cudaCallback, call_pl, 0);
						  break;
					  }
		#endif

		#ifdef WITH_OPENCL
		case metaModePreferOpenCL:
					  //Make sure some context exists..
					  if (meta_context == NULL) metaOpenCLFallBack();
					  #ifdef WITH_TIMERS
					  ret = (a_err) opencl_unpack_2d_face(grid_size, block_size, packed_buf, buf, face, remain_dim, type, async, &(frame_k1->event.opencl), &(frame_c1->event.opencl), &(frame_c2->event.opencl), &(frame_c3->event.opencl));
			//If timers exist, use their event to add the callback
			if ((void*)call != NULL && call_pl != NULL) clSetEventCallback(frame_k1->event.opencl, CL_COMPLETE, call->openclCallback, call_pl);
					  #else
			//If timers don't exist, get the event via a locally-scoped event to add to the callback
			cl_event cb_event;
					  ret = (a_err) opencl_unpack_2d_face(grid_size, block_size, packed_buf, buf, face, remain_dim, type, async, NULL, NULL, NULL, &cb_event);
			if ((void*)call != NULL && call_pl != NULL) clSetEventCallback(cb_event, CL_COMPLETE, call->openclCallback, call_pl);
					  #endif
					  break;
		#endif

		#ifdef WITH_OPENMP
		case metaModePreferOpenMP:
						#ifdef WITH_TIMERS
						frame_k1->event.openmp[0]= omp_get_wtime();;
						#endif
						ret = omp_unpack_2d_face(grid_size, block_size, packed_buf, buf, face, remain_dim, type, async);
						#ifdef WITH_TIMERS
						frame_k1->event.openmp[1]= omp_get_wtime();
						frame_c1->event.openmp[0] = 0.0;
						frame_c1->event.openmp[1] = 0.0;
						frame_c2->event.openmp[0] = 0.0;
						frame_c2->event.openmp[1] = 0.0;
						frame_c3->event.openmp[0] = 0.0;
						frame_c3->event.openmp[1] = 0.0;
						#endif
					  break;
		#endif
	}
	#ifdef WITH_TIMERS
	//TODO Add queue c_H2Dc for copies into constant memory
	metaTimerEnqueue(frame_k1, &(metaBuiltinQueues[k_pack_2d_face]));
	metaTimerEnqueue(frame_c1, &(metaBuiltinQueues[c_H2Dc]));
	metaTimerEnqueue(frame_c2, &(metaBuiltinQueues[c_H2Dc]));
	metaTimerEnqueue(frame_c3, &(metaBuiltinQueues[c_H2Dc]));
	#endif
	return(ret);
}

