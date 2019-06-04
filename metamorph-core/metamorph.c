/*
 * The workhorse pass-through of the library. This file implements
 *  control over which modes are compiled in, and generic wrappers
 *  for all functions which must be passed through to a specific
 *  mode's backend implementation.
 *
 * Should support targeting a specific class of device, a
 *  specific accelerator model (CUDA/OpenCL/OpenMP/...), and
 *  "choose best", which would be similar to Tom's CoreTSAR.
 *
 * For now, Generic, OpenMP, CUDA, and OpenCL are supported.
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
#ifdef ALIGNED_MEMORY
#include "xmmintrin.h"
#endif

#define CHKERR(err, str)\
if ( err != CL_SUCCESS)\
{\
        fprintf(stderr, "Error in executing \"%s\", %d\n", str, err);  \
        exit(EXIT_FAILURE);\
}


//Globally-set mode
meta_preferred_mode run_mode = metaModePreferGeneric;

//Module Management
//The global list always maintains a sentinel
struct a_module_record_listelem;
typedef struct a_module_record_listelem {
  a_module_record * rec;
  struct a_module_record_listelem * next;
} a_module_list;
a_module_list global_modules = {NULL, NULL};
//Returns true if the provided record is already in the list
//TODO Make Hazard-aware
a_bool __module_registered(a_module_record * record) {
  if (record == NULL) return false;
  if (global_modules.next == NULL) return false;
  a_module_list * elem;
  for (elem = global_modules.next; elem != NULL; elem = elem->next) {
    if (elem->rec == record) return true;
  }
  return false;
}
//Only adds a module to our records, does not initialize it
// returns true iff added, false otherwise
//TODO Make Hazard-aware
a_bool __record_module(a_module_record * record) {
  if (record == NULL) return false;
  if (__module_registered(record) != false) return false;
  a_module_list * new_elem = (a_module_list *)malloc(sizeof(a_module_list));
  new_elem->rec = record;
  new_elem->next = NULL;
  //begin hazards
  while (new_elem->next != global_modules.next) {
    new_elem->next = global_modules.next;
  }
  //TODO atomic CAS the pointer in
  global_modules.next = new_elem;
  return true; //TODO return the eventual true from CAS
}
//Only removes a module from our records, does not deinitialize it
// returns true iff removed, false otherwise
//TODO Make Hazard-aware
a_bool __remove_module(a_module_record * record) {
  if (record == NULL) return false;
  a_module_list * curr = global_modules.next, * prev = &global_modules;
  //begin hazards
  while(prev->next == curr && curr != NULL) {
    //TODO Ensure curr w/ atomic CAS
    if (curr->rec == record) {
      //TODO Remove w/ atomic CAS
      prev->next = curr->next;
      free(curr); //Just the list wrapper
      return true;
    }
    curr = curr->next;
    prev = prev->next;
  }
  return false;
}

//Lookup all modules with a matching implementation signature
//\param retRecords a pointer to a storage buffer for matched record pointers
//\param szRetRecords the size of the retRecords buffer
//\param signature the signature to match on
//\param matchAny if true, will match if the record and signature have any matching flags (binary OR), otherwise will only match exactly (binary AND)
// Will return the number of matched modules, regardless of how many were actually storable in retRecords
int lookup_implementing_modules(a_module_record ** retRecords, size_t szRetRecords, a_module_implements_backend signature, a_bool matchAny) {
  int matches = 0, retIdx= 0;
  a_module_list * elem = global_modules.next;
  for (; elem != NULL; elem = elem->next) {
    if (elem->rec != NULL && (elem->rec->implements == signature || (matchAny && (elem->rec->implements & signature)))) {
      if (retRecords != NULL && retIdx < szRetRecords) {
        retRecords[retIdx] = elem->rec;
        retIdx++;
      }
      matches++;
    }
  }
  return matches;
}
//Allow add-on modules to pass a pointer to their registration function to
// metamorph-core, so that the core can then initialize and exchange the appropriate internal data structs
a_err meta_register_module(a_module_record * (*module_registry_func)(a_module_record * record)) {
  //Do nothing if they don't specify a pointer
  if (module_registry_func == NULL) return -1;
  //Each module contains a referency to it's own registry object
  //We can check if it's registered by calling the registry function with a NULL record
  a_module_record * existing = (*module_registry_func)(NULL);
  //if there is an existing record, make sure we know about it
  if (existing != NULL) {
    __record_module(existing); //Don't need to check, __record_module will check before insertion
    //But don't call the registration function with the existing record, because that will de-register it
    return 0;
  }
  //It has not already been registered

  
  //TODO add this registry object to a list of currently-registered modules;
  a_module_record * new_record = (a_module_record *)calloc(sizeof(a_module_record), 1);
  a_module_record * returned = (*module_registry_func)(new_record);
  //TODO make this threadsafe
  //If we trust that the registration function will only accept one registry, then we just have to check that the one returned is the same as the one we created. otherwise release the new one
  if (returned == NULL) {
    __record_module(new_record); //shouldn't be able to return false, because we should block recording records that aren't our own;
    //Only initialize if it's a new registration, do nothing if it's a re-register
    //It's been explicitly registered, immediately initialize it
    if (new_record->module_init != NULL) (*new_record->module_init)();
  } else if (new_record == returned) {
    //Somewhere between checking whether the module was registered and trying to register it, someone else has already done so, just release our new record and move on
    //TODO if we want to support re-registration we will change this mechanism
    free(new_record);
  } //There should not be an else unless we support multiple registrations, in which case returned will belong to the module's previous registration
  
  return 0;
}

//Explicitly remove an external module from metamorph-core, causing it to trigger
// any deconstruction code necessary to clean up the module
a_err meta_deregister_module(a_module_record * (*module_registry_func)(a_module_record * record)) {
  //If they gave us NULL, nothing to do
  if (module_registry_func == NULL) return -1;
  //Get the registration from the module
  a_module_record * existing = (*module_registry_func)(NULL);
  //If the module doesn't think it's registered, we can't do anything with it
  if (existing == NULL) return 0;

  //If it does exist, deinitialize it, thenremove our record of it
  //Deinit has to occure before the module forgets its record so that the
  // module can inform MetaMorph when it is finish via the init flag
  if (existing->module_deinit != NULL) {
    while (existing->initialized) { //The module is responsible for updating its registration when it has completed deinitialization
      //Loop in case there are multiple stacked initializations that need to be handled
      (*existing->module_deinit)(); //It has been deinitialized
    }
  }
  //TODO make threadsafe
  //Tell the module to forget itself
  a_module_record * returned = (*module_registry_func)(existing);
  //Check that the module indeed thinks it's deregistered (to distinguish from the multiple-registration case that also immediately returns the same record)
  a_module_record * double_check = (*module_registry_func)(NULL);
  if (returned == existing && double_check == NULL) { //The module has forgotten its registration
    //Then deinitialize the module
    __remove_module(existing); //We have no record of it outside this function call
    free(existing); //Record memory released and this function's reference invalidated, we are done
  }
  return 0;
}

//Reinitialize all modules with a given implements_backend mask
a_err meta_reinitialize_modules(a_module_implements_backend module_type) {
  if (global_modules.next == NULL) return -1;
  a_module_list * elem;
  for (elem = global_modules.next; elem != NULL; elem = elem->next) {
    if (elem->rec != NULL && (elem->rec->implements & module_type) && elem->rec->module_init != NULL) (*elem->rec->module_init)();
  }
  return 0;
}

#ifdef WITH_OPENCL
cl_context meta_context = NULL;
cl_command_queue meta_queue = NULL;
cl_device_id meta_device = NULL;

/*metaTimerQueueFrame * hold_ex_frame = (metaTimerQueueFrame *)malloc (sizeof(metaTimerQueueFrame));
metaTimerQueueFrame * hold_wr_frame = (metaTimerQueueFrame *)malloc (sizeof(metaTimerQueueFrame));
*/
//All this does is wrap calling metaOpenCLInitStackFrameDefault
// and setting meta_context and meta_queue appropriately
void metaOpenCLFallBack() {
	metaOpenCLStackFrame * frame;
	metaOpenCLInitStackFrameDefault(&frame);
	meta_context = frame->context;
	meta_queue = frame->queue;
	meta_device = frame->device;
	metaOpenCLPushStackFrame(frame);
	free(frame); //This is safe, it's just a copy of what should now be
				 // the bottom of the stack
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
		fprintf(stderr, "Error: no size retreivable for selected type [%d]!\n",
				type);
		return (size_t) - 1;
		break;
	}
}

//TODO - Validate attempted allocation sizes against maximum supported by the device
// particularly for OpenCL (AMD 7970 doesn't support 512 512 512 grid size)
a_err meta_alloc(void ** ptr, size_t size) {
	//TODO: should always set ret to a value
	a_err ret;
	switch (run_mode) {
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
		{
		//Make sure some context exists..
		if (meta_context == NULL) metaOpenCLFallBack();
		*ptr = (void *) clCreateBuffer(meta_context, CL_MEM_READ_WRITE, size, NULL, (cl_int *)&ret);
		}
		break;
#endif

#ifdef WITH_OPENMP
		case metaModePreferOpenMP:
#ifdef ALIGNED_MEMORY
		*ptr = (void *) _mm_malloc(size, ALIGNED_MEMORY_PAGE);
#else
		*ptr = (void *) malloc(size);
#endif
		if(*ptr != NULL)
		ret = 0; // success
		break;
#endif
	}
	return (ret);
}

//TODO implement a way for this to trigger destroying an OpenCL stack frame
// if all cl_mems in the frame's context, as well as the frame members themselves
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
		{
		//Make sure some context exists..
		if (meta_context == NULL) metaOpenCLFallBack();
		ret = clReleaseMemObject((cl_mem)ptr);
		}
		break;
#endif

#ifdef WITH_OPENMP
		case metaModePreferOpenMP:
#ifdef ALIGNED_MEMORY
		_mm_free(ptr);
#else
		free(ptr);
#endif
		break;
#endif
	}
	return (ret);
}


#ifdef WITH_TIMERS
#ifdef WITH_OPENCL
//getting a pointer to specific event  
a_err meta_get_event(char * qname, char * ename, cl_event ** e)
{
	a_err ret;
	metaTimerQueueFrame * frame = (metaTimerQueueFrame*) malloc(sizeof(metaTimerQueueFrame));

	if(strcmp(qname,"c_D2H") == 0)
		ret = cl_get_event_node(&(metaBuiltinQueues[c_D2H]), ename, &frame);
	else if(strcmp(qname,"c_H2D") == 0)
		ret = cl_get_event_node(&(metaBuiltinQueues[c_H2D]), ename, &frame);
	else if(strcmp(qname,"c_D2D") == 0)
		ret = cl_get_event_node(&(metaBuiltinQueues[c_D2D]), ename, &frame);
	else if(strcmp(qname,"k_csr") == 0)
		ret = cl_get_event_node(&(metaBuiltinQueues[k_csr]), ename, &frame);
        else if(strcmp(qname,"k_crc") == 0)
                ret = cl_get_event_node(&(metaBuiltinQueues[k_crc]), ename, &frame);
	else
		printf("Event queue '%s' is not recognized...\n",qname);	

/*	if(frame == NULL)
		printf("event node search failed ..\n");
	else
		printf("This is 'MetaMorph C core', event '%s' retrieved succesfully\n",frame->name);
*/

/*
cl_ulong start_time;
size_t return_bytes;
clGetEventProfilingInfo(frame->event.opencl,CL_PROFILING_COMMAND_QUEUED,sizeof(cl_ulong),&start_time,&return_bytes);
printf("check profiling event is correct (MM side) %lu\n",start_time);
*/	
	(*e) =&(frame->event.opencl);
	return(ret);
}
#endif // WITH_OPENCL
#endif // WITH_TIMERS



//Simplified wrapper for sync copies
//a_err meta_copy_h2d(void * dst, void * src, size_t size) {
//	return meta_copy_h2d(dst, src, size, true);
//}
//Workhorse for both sync and async variants
a_err meta_copy_h2d(void * dst, void * src, size_t size, a_bool async) {//, char * event_name, cl_event * wait) {
	return meta_copy_h2d_cb(dst, src, size, async,
	// event_name, wait,
	 (meta_callback*) NULL, NULL);
}
a_err meta_copy_h2d_cb(void * dst, void * src, size_t size, a_bool async,
		// char * event_name, cl_event * wait,
		meta_callback *call, void *call_pl) {
	a_err ret;
#ifdef WITH_TIMERS
	metaTimerQueueFrame * frame = (metaTimerQueueFrame *)malloc (sizeof(metaTimerQueueFrame));
	frame->mode = run_mode;
	frame->size = size;
	//frame->name = event_name;
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
		{
		//Make sure some context exists..
		if (meta_context == NULL) metaOpenCLFallBack();
#ifdef WITH_TIMERS
		//if(wait != NULL)
		//{
		//	ret = clEnqueueWriteBuffer(meta_queue, (cl_mem) dst, ((async) ? CL_FALSE : CL_TRUE), 0, size, src, 1, wait, &(frame->event.opencl));
		//	CHKERR(ret, "Failed to write to source array!");
		//}
		//else
		//{
			ret = clEnqueueWriteBuffer(meta_queue, (cl_mem) dst, ((async) ? CL_FALSE : CL_TRUE), 0, size, src, 0, NULL, &(frame->event.opencl));
//			CHKERR(ret, "Failed to write to source array!");
//		}
		//If timers exist, use their event to add the callback
		if ((void*)call != NULL && call_pl != NULL) clSetEventCallback(frame->event.opencl, CL_COMPLETE, call->openclCallback, call_pl);
#else
		//If timers don't exist, get the event via a locally-scoped event to add to the callback
		cl_event cb_event;
		ret = clEnqueueWriteBuffer(meta_queue, (cl_mem) dst, ((async) ? CL_FALSE : CL_TRUE), 0, size, src, 0, NULL, &cb_event);
		if ((void*)call != NULL && call_pl != NULL) clSetEventCallback(cb_event, CL_COMPLETE, call->openclCallback, call_pl);
#endif
		}
		break;
#endif

#ifdef WITH_OPENMP
		case metaModePreferOpenMP:
#ifdef WITH_TIMERS
		{	struct timeval start, end;
			gettimeofday(&start, NULL);
			//		frame->event.openmp[0]= omp_get_wtime();
#endif
		memcpy(dst, src, size);
		ret = 0;
#ifdef WITH_TIMERS
		gettimeofday(&end, NULL);
		frame->event.openmp[0] = (start.tv_usec*0.000001)+start.tv_sec;
		frame->event.openmp[1] = (end.tv_usec*0.000001)+end.tv_sec;}
	//		frame->event.openmp[1]= omp_get_wtime();
#endif
		break;
#endif
	}

/*
cl_ulong start_time;
size_t return_bytes;
clGetEventProfilingInfo(frame->event.opencl,CL_PROFILING_COMMAND_QUEUED,sizeof(cl_ulong),&start_time,&return_bytes);
printf("check profiling event '%s' (MM side) %lu\n",frame->name,start_time);
*/
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
a_err meta_copy_d2h(void *dst, void *src, size_t size, a_bool async) {//, char * event_name, cl_event * wait) {
	return meta_copy_d2h_cb(dst, src, size, async,
	// event_name, wait,
	 (meta_callback*) NULL, NULL);
}
a_err meta_copy_d2h_cb(void * dst, void * src, size_t size, a_bool async,
		// char * event_name, cl_event * wait,
		meta_callback *call, void *call_pl) {
	a_err ret;
#ifdef WITH_TIMERS
	metaTimerQueueFrame * frame = (metaTimerQueueFrame*)malloc (sizeof(metaTimerQueueFrame));
	frame->mode = run_mode;
	frame->size = size;
	//frame->name = event_name;
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
		{
		//Make sure some context exists..
		if (meta_context == NULL) metaOpenCLFallBack();
#ifdef WITH_TIMERS
		//if(wait != NULL)
		//{
		//	ret = clEnqueueReadBuffer(meta_queue, (cl_mem) src, ((async) ? CL_FALSE : CL_TRUE), 0, size, dst, 1, wait, &(frame->event.opencl));
			//ret = clEnqueueReadBuffer(meta_queue, (cl_mem) src, ((async) ? CL_FALSE : CL_TRUE), 0, size, dst, 0, NULL, &(frame->event.opencl));
		//}
		//else
		//{
			ret = clEnqueueReadBuffer(meta_queue, (cl_mem) src, ((async) ? CL_FALSE : CL_TRUE), 0, size, dst, 0, NULL, &(frame->event.opencl));
		//}

		//If timers exist, use their event to add the callback
		if ((void*)call != NULL && call_pl != NULL) clSetEventCallback(frame->event.opencl, CL_COMPLETE, call->openclCallback, call_pl);
#else
		//If timers don't exist, get the event via a locally-scoped event to add to the callback
		cl_event cb_event;
		ret = clEnqueueReadBuffer(meta_queue, (cl_mem) src, ((async) ? CL_FALSE : CL_TRUE), 0, size, dst, 0, NULL, &cb_event);
		if ((void*)call != NULL && call_pl != NULL) clSetEventCallback(cb_event, CL_COMPLETE, call->openclCallback, call_pl);
#endif
		}
		break;
#endif

#ifdef WITH_OPENMP
		case metaModePreferOpenMP:
#ifdef WITH_TIMERS
		{	struct timeval start, end;
			gettimeofday(&start, NULL);
			//		frame->event.openmp[0]= omp_get_wtime();
#endif
		memcpy(dst, src, size);
		ret = 0;
#ifdef WITH_TIMERS
		gettimeofday(&end, NULL);
		frame->event.openmp[0] = (start.tv_usec*0.000001)+start.tv_sec;
		frame->event.openmp[1] = (end.tv_usec*0.000001)+end.tv_sec;}
	//		frame->event.openmp[1]= omp_get_wtime();
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
a_err meta_copy_d2d(void *dst, void *src, size_t size, a_bool async) {//, char * event_name, cl_event * wait) {
	return meta_copy_d2d_cb(dst, src, size, async,
	// event_name, wait, 
	(meta_callback*) NULL, NULL);
}
a_err meta_copy_d2d_cb(void * dst, void * src, size_t size, a_bool async,
		// char * event_name, cl_event * wait,
		meta_callback *call, void *call_pl) {
	a_err ret;
#ifdef WITH_TIMERS
	metaTimerQueueFrame * frame = (metaTimerQueueFrame*)malloc (sizeof(metaTimerQueueFrame));
	frame->mode = run_mode;
	frame->size = size;
	//frame->name = event_name;
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
		{
		//Make sure some context exists..
		if (meta_context == NULL) metaOpenCLFallBack();
#ifdef WITH_TIMERS
		//if(wait != NULL)
		//{
		//	ret = clEnqueueCopyBuffer(meta_queue, (cl_mem) src, (cl_mem) dst, 0, 0, size, 1, wait, &(frame->event.opencl));
		//}
		//else
		//{
			ret = clEnqueueCopyBuffer(meta_queue, (cl_mem) src, (cl_mem) dst, 0, 0, size, 0, NULL, &(frame->event.opencl));
		//}
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
		}
		break;
#endif

#ifdef WITH_OPENMP
		case metaModePreferOpenMP:
#ifdef WITH_TIMERS
		{	struct timeval start, end;
			gettimeofday(&start, NULL);
			//		frame->event.openmp[0]= omp_get_wtime();
#endif
			//memcpy(dst, src, size);
			omp_copy_d2d(dst, src, size, async);
			ret = 0;
#ifdef WITH_TIMERS
			gettimeofday(&end, NULL);
			frame->event.openmp[0] = (start.tv_usec*0.000001)+start.tv_sec;
			frame->event.openmp[1] = (end.tv_usec*0.000001)+end.tv_sec;}
		//		frame->event.openmp[1]= omp_get_wtime();
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
// implements some reasonable default if meta_set_acc is not called before the
// first OpenCL command
//TODO make this compatible with OpenCL initialization
//TODO unpack OpenCL platform from the uint's high short, and the device from the low short
a_err meta_set_acc(int accel, meta_preferred_mode mode) {
	a_err ret;
	run_mode = mode;
	switch (run_mode) {
	default:
	case metaModePreferGeneric:
		//TODO support generic (auto-config) runtime selection
		//TODO support "METAMORPH_MODE" environment variable.
		if (getenv("METAMORPH_MODE") != NULL) {
#ifdef WITH_CUDA
			if (strcmp(getenv("METAMORPH_MODE"), "CUDA") == 0) return meta_set_acc(accel, metaModePreferCUDA);
#endif

#ifdef WITH_OPENCL
			if (strcmp(getenv("METAMORPH_MODE"), "OpenCL") == 0 || strcmp(getenv("METAMORPH_MODE"), "OpenCL_DEBUG") == 0) return meta_set_acc(accel, metaModePreferOpenCL);
#endif

#ifdef WITH_OPENMP
			if (strcmp(getenv("METAMORPH_MODE"), "OpenMP") == 0) return meta_set_acc(accel, metaModePreferOpenMP);
#endif

#ifdef WITH_CORETSAR
			if (strcmp(getenv("METAMORPH_MODE"), "CoreTsar") == 0) {
				fprintf(stderr, "CoreTsar mode not yet supported!\n");
				//TODO implement whatever's required to get CoreTsar going...

			}
#endif

			fprintf(stderr,
					"Error: METAMORPH_MODE=\"%s\" not supported with specified compiler definitions.\n",
					getenv("METAMORPH_MODE"));
		}
		fprintf(stderr,
				"Generic Mode only supported with \"METAMORPH_MODE\" environment variable set to one of:\n");
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
		{	metaOpenCLStackFrame * frame;
			metaOpenCLInitStackFrame(&frame, (cl_int) accel); //no hazards, frames are thread-private
			metaOpenCLPushStackFrame(frame);//no hazards, HPs are internally managed when copying the frame to a new stack node before pushing.

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
	return (ret);
}

//TODO make this compatible with OpenCL device querying
//TODO pack OpenCL platform into the uint's high short, and the device into the low short
a_err meta_get_acc(int * accel, meta_preferred_mode * mode) {
	a_err ret;
	switch (run_mode) {
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
		{
		//Make sure some context exists..
		if (meta_context == NULL) metaOpenCLFallBack();
		//TODO implement appropriate response for OpenCL device number
		//TODO implement this based on metaOpenCLTopStackFrame
		//fprintf(stderr, "OpenCL Device Query not yet implemented!\n");
		*mode = metaModePreferOpenCL;
		}
		break;
#endif

#ifdef WITH_OPENMP
		case metaModePreferOpenMP:
		*mode = metaModePreferOpenMP;
		break;
#endif
	}
	return (ret);
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
	switch (run_mode) {
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
		{
		//Make sure some context exists..
		if (meta_context == NULL) metaOpenCLFallBack();
		size_t max_wg_size, max_wg_dim_sizes[3];
		ret = clGetDeviceInfo(meta_device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_wg_size, NULL);
		ret |= clGetDeviceInfo(meta_device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t)*3, &max_wg_dim_sizes, NULL);
		if ((*block_size)[0] * (*block_size)[1] * (*block_size)[2] > max_wg_size)
		{	fprintf(stderr, "Error: Maximum block volume is: %lu\nRequested block volume of: %lu (%lu * %lu * %lu) not supported!\n", max_wg_size, (*block_size)[0] * (*block_size)[1] * (*block_size)[2], (*block_size)[0], (*block_size)[1], (*block_size)[2]); ret |= -1;}

		if ((*block_size)[0] > max_wg_dim_sizes[0]) {fprintf(stderr, "Error: Maximum block size for dimension 0 is: %lu\nRequested 0th dimension size of: %lu not supported\n!", max_wg_dim_sizes[0], (*block_size)[0]); ret |= -1;}
		if ((*block_size)[1] > max_wg_dim_sizes[1]) {fprintf(stderr, "Error: Maximum block size for dimension 1 is: %lu\nRequested 1st dimension size of: %lu not supported\n!", max_wg_dim_sizes[1], (*block_size)[1]); ret |= -1;}
		if ((*block_size)[2] > max_wg_dim_sizes[2]) {fprintf(stderr, "Error: Maximum block size for dimension 2 is: %lu\nRequested 2nd dimension size of: %lu not supported\n!", max_wg_dim_sizes[2], (*block_size)[2]); ret |= -1;}
		return(ret);
		}
		break;
#endif

#ifdef WITH_OPENMP
		case metaModePreferOpenMP:
		//TODO implement any bounds checking OpenMP may need
		return 0;
		break;
#endif
	}
	return (ret);
}

//Flush does exactly what it sounds like - forces any
// outstanding work to be finished before returning
//For now it handles flushing async kernels, and finishing outstanding
// transfers from the MPI plugin
//It could easily handle timer dumping as well, but that might not be ideal
a_err meta_flush() {
	switch (run_mode) {
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
		{
		//Make sure some context exists..
		if (meta_context == NULL) metaOpenCLFallBack();
		clFinish(meta_queue);
		}
		break;
#endif

#ifdef WITH_OPENMP
		case metaModePreferOpenMP: {
#pragma omp barrier // synchronize threads
}
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
a_err meta_dotProd(a_dim3 * grid_size, a_dim3 * block_size, void * data1,
		void * data2, a_dim3 * array_size, a_dim3 * array_start,
		a_dim3 * array_end, void * reduction_var, meta_type_id type,
		a_bool async) {
	return meta_dotProd_cb(grid_size, block_size, data1, data2, array_size,
			array_start, array_end, reduction_var, type, async,
			(meta_callback*) NULL, NULL);
}
a_err meta_dotProd_cb(a_dim3 * grid_size, a_dim3 * block_size, void * data1,
		void * data2, a_dim3 * array_size, a_dim3 * array_start,
		a_dim3 * array_end, void * reduction_var, meta_type_id type,
		a_bool async, meta_callback *call, void *call_pl) {
	a_err ret;

	//FIXME? Consider adding a compiler flag "UNCHECKED_EXPLICIT" to streamline out sanity checks like this
	//Before we do anything, sanity check the start/end/size
	if (array_start == NULL || array_end == NULL || array_size == NULL) {
		fprintf(stderr,
				"ERROR in meta_dotProd: array_start=[%p], array_end=[%p], or array_size=[%p] is NULL!\n",
				array_start, array_end, array_size);
		return -1;
	}
	int i;
	for (i = 0; i < 3; i++) {
		if ((*array_start)[i] < 0 || (*array_end)[i] < 0) {
			fprintf(stderr,
					"ERROR in meta_dotProd: array_start[%d]=[%ld] or array_end[%d]=[%ld] is negative!\n",
					i, (*array_start)[i], i, (*array_end)[i]);
			return -1;
		}
		if ((*array_size)[i] < 1) {
			fprintf(stderr,
					"ERROR in meta_dotProd: array_size[%d]=[%ld] must be >=1!\n",
					i, (*array_size)[i]);
			return -1;
		}
		if ((*array_start)[i] > (*array_end)[i]) {
			fprintf(stderr,
					"ERROR in meta_dotProd: array_start[%d]=[%ld] is after array_end[%d]=[%ld]!\n",
					i, (*array_start)[i], i, (*array_end)[i]);
			return -1;
		}
		if ((*array_end)[i] >= (*array_size)[i]) {
			fprintf(stderr,
					"ERROR in meta_dotProd: array_end[%d]=[%ld] is bigger than array_size[%d]=[%ld]!\n",
					i, (*array_end)[i], i, (*array_size)[i]);
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
				new_grid[i] = ((*block_size)[i] * (*grid_size)[i] - 1
						+ new_block[i]) / new_block[i];
			}
		}
		if (flag) {
			fprintf(stderr,
					"WARNING in meta_dotProd: block_size={%ld, %ld, %ld} must be all powers of two!\n\tRescaled grid_size={%ld, %ld, %ld}, block_size={%ld, %ld, %ld} to\n\tnew_grid={%ld, %ld, %ld}, new_block={%ld, %ld, %ld}\n",
					(*block_size)[0], (*block_size)[1], (*block_size)[2],
					(*grid_size)[0], (*grid_size)[1], (*grid_size)[2],
					(*block_size)[0], (*block_size)[1], (*block_size)[2],
					new_grid[0], new_grid[1], new_grid[2], new_block[0],
					new_block[1], new_block[2]);
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
	switch (run_mode) {
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
		{
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
		}
		break;
#endif

#ifdef WITH_OPENMP
		case metaModePreferOpenMP:
#ifdef WITH_TIMERS
		{	struct timeval start, end;
			gettimeofday(&start, NULL);
			//		frame->event.openmp[0]= omp_get_wtime();
#endif
			ret = omp_dotProd(grid_size, block_size, data1, data2, array_size, array_start, array_end, reduction_var, type, async);
#ifdef WITH_TIMERS
			gettimeofday(&end, NULL);
			frame->event.openmp[0] = (start.tv_usec*0.000001)+start.tv_sec;
			frame->event.openmp[1] = (end.tv_usec*0.000001)+end.tv_sec;}
		//		frame->event.openmp[1]= omp_get_wtime();
#endif
		break;
#endif
	}
#ifdef WITH_TIMERS
	metaTimerEnqueue(frame, &(metaBuiltinQueues[k_dotProd]));
#endif
	return (ret);
}

//Simplified wrapper for synchronous kernel
//a_err meta_reduce(a_dim3 * grid_size, a_dim3 * block_size, a_double * data, a_dim3 * array_size, a_dim3 * array_start, a_dim3 * array_end, a_double * reduction_var) {
//	return meta_reduce(grid_size, block_size, data, array_size, array_start, array_end, reduction_var, true);
//}
//Workhorse for both sync and async reductions
a_err meta_reduce(a_dim3 * grid_size, a_dim3 * block_size, void * data,
		a_dim3 * array_size, a_dim3 * array_start, a_dim3 * array_end,
		void * reduction_var, meta_type_id type, a_bool async) {
	return meta_reduce_cb(grid_size, block_size, data, array_size, array_start,
			array_end, reduction_var, type, async, (meta_callback*) NULL, NULL);
}
a_err meta_reduce_cb(a_dim3 * grid_size, a_dim3 * block_size, void * data,
		a_dim3 * array_size, a_dim3 * array_start, a_dim3 * array_end,
		void * reduction_var, meta_type_id type, a_bool async,
		meta_callback *call, void *call_pl) {
	a_err ret;

	//FIXME? Consider adding a compiler flag "UNCHECKED_EXPLICIT" to streamline out sanity checks like this
	//Before we do anything, sanity check the start/end/size
	if (array_start == NULL || array_end == NULL || array_size == NULL) {
		fprintf(stderr,
				"ERROR in meta_reduce: array_start=[%p], array_end=[%p], or array_size=[%p] is NULL!\n",
				array_start, array_end, array_size);
		return -1;
	}
	int i;
	for (i = 0; i < 3; i++) {
		if ((*array_start)[i] < 0 || (*array_end)[i] < 0) {
			fprintf(stderr,
					"ERROR in meta_reduce: array_start[%d]=[%ld] or array_end[%d]=[%ld] is negative!\n",
					i, (*array_start)[i], i, (*array_end)[i]);
			return -1;
		}
		if ((*array_size)[i] < 1) {
			fprintf(stderr,
					"ERROR in meta_reduce: array_size[%d]=[%ld] must be >=1!\n",
					i, (*array_size)[i]);
			return -1;
		}
		if ((*array_start)[i] > (*array_end)[i]) {
			fprintf(stderr,
					"ERROR in meta_reduce: array_start[%d]=[%ld] is after array_end[%d]=[%ld]!\n",
					i, (*array_start)[i], i, (*array_end)[i]);
			return -1;
		}
		if ((*array_end)[i] >= (*array_size)[i]) {
			fprintf(stderr,
					"ERROR in meta_reduce: array_end[%d]=[%ld] is bigger than array_size[%d]=[%ld]!\n",
					i, (*array_end)[i], i, (*array_size)[i]);
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
				new_grid[i] = ((*block_size)[i] * (*grid_size)[i] - 1
						+ new_block[i]) / new_block[i];
			}
		}
		if (flag) {
			fprintf(stderr,
					"WARNING in meta_reduce: block_size={%ld, %ld, %ld} must be all powers of two!\n\tRescaled grid_size={%ld, %ld, %ld}, block_size={%ld, %ld, %ld} to\n\tnew_grid={%ld, %ld, %ld}, new_block={%ld, %ld, %ld}\n",
					(*block_size)[0], (*block_size)[1], (*block_size)[2],
					(*grid_size)[0], (*grid_size)[1], (*grid_size)[2],
					(*block_size)[0], (*block_size)[1], (*block_size)[2],
					new_grid[0], new_grid[1], new_grid[2], new_block[0],
					new_block[1], new_block[2]);
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
	switch (run_mode) {
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
		{
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
		}
		break;
#endif

#ifdef WITH_OPENMP
		case metaModePreferOpenMP:
#ifdef WITH_TIMERS
		{	struct timeval start, end;
			gettimeofday(&start, NULL);
			//		frame->event.openmp[0]= omp_get_wtime();
#endif
		ret = omp_reduce(grid_size, block_size, data, array_size, array_start, array_end, reduction_var, type, async);
#ifdef WITH_TIMERS
		gettimeofday(&end, NULL);
		frame->event.openmp[0] = (start.tv_usec*0.000001)+start.tv_sec;
		frame->event.openmp[1] = (end.tv_usec*0.000001)+end.tv_sec;}
	//		frame->event.openmp[1]= omp_get_wtime();
#endif
		break;
#endif
	}
#ifdef WITH_TIMERS
	metaTimerEnqueue(frame, &(metaBuiltinQueues[k_reduce]));
#endif
	return (ret);
}

meta_face * meta_get_face(int s, int c, int *si, int *st) {
	//Unlike Kaixi's, we return a pointer copy, to ease Fortran implementation
	meta_face * face = (meta_face*) malloc(
			sizeof(meta_face));
	//We create our own copy of size and stride arrays to prevent
	// issues if the user unexpectedly reuses or frees the original pointer
	size_t sz = sizeof(int) * c;
	face->size = (int*) malloc(sz);
	face->stride = (int*) malloc(sz);
	memcpy((void*) face->size, (const void *) si, sz);
	memcpy((void*) face->stride, (const void *) st, sz);

	face->start = s;
	face->count = c;
}

//Simple deallocator for an meta_face type
// Assumes face, face->size, and ->stride are unfreed
//This is the only way a user should release a face returned
// from meta_get_face_index, and should not be used
// if the face was assembled by hand.
int meta_free_face(meta_face * face) {
	free(face->size);
	free(face->stride);
	free(face);
}

meta_face * make_slab_from_3d(int face, int ni, int nj, int nk, int thickness) {
	meta_face * ret = (meta_face*) malloc(sizeof(meta_face));
	ret->count = 3;
	ret->size = (int*) malloc(sizeof(int) * 3);
	ret->stride = (int*) malloc(sizeof(int) * 3);
	//all even faces start at the origin, all others start at some offset
	// defined by the dimensions of the prism
	if (face & 1) {
		if (face == 1)
			ret->start = ni * nj * (nk - thickness);
		if (face == 3)
			ret->start = ni - thickness;
		if (face == 5)
			ret->start = ni * (nj - thickness);
	} else
		ret->start = 0;
	ret->size[0] = nk, ret->size[1] = nj, ret->size[2] = ni;
	if (face < 2)
		ret->size[0] = thickness;
	if (face > 3)
		ret->size[1] = thickness;
	if (face > 1 && face < 4)
		ret->size[2] = thickness;
	ret->stride[0] = ni * nj, ret->stride[1] = ni, ret->stride[2] = 1;
	printf(
			"Generated Face:\n\tcount: %d\n\tstart: %d\n\tsize: %d %d %d\n\tstride: %d %d %d\n",
			ret->count, ret->start, ret->size[0], ret->size[1], ret->size[2],
			ret->stride[0], ret->stride[1], ret->stride[2]);
	return ret;
}

//Workhorse for both sync and async transpose
a_err meta_transpose_face(a_dim3 * grid_size, a_dim3 * block_size,
		void *indata, void *outdata, a_dim3 * arr_dim_xy, a_dim3 * tran_dim_xy,
		meta_type_id type, a_bool async) {
	return meta_transpose_face_cb(grid_size, block_size, indata, outdata,
			arr_dim_xy, tran_dim_xy, type, async, (meta_callback*) NULL, NULL);
}
a_err meta_transpose_face_cb(a_dim3 * grid_size, a_dim3 * block_size,
		void *indata, void *outdata, a_dim3 * arr_dim_xy, a_dim3 * tran_dim_xy,
		meta_type_id type, a_bool async, meta_callback *call, void *call_pl) {
	a_err ret;
	//FIXME? Consider adding a compiler flag "UNCHECKED_EXPLICIT" to streamline out sanity checks like this
	//Before we do anything, sanity check that trans_dim_xy fits inside arr_dim_xy
	if (arr_dim_xy == NULL || tran_dim_xy == NULL) {
		fprintf(stderr,
				"ERROR in meta_transpose_face: arr_dim_xy=[%p] or tran_dim_xy=[%p] is NULL!\n",
				arr_dim_xy, tran_dim_xy);
		return -1;
	}
	int i;
	for (i = 0; i < 2; i++) {
		if ((*arr_dim_xy)[i] < 1 || (*tran_dim_xy)[i] < 1) {
			fprintf(stderr,
					"ERROR in meta_transpose_face: arr_dim_xy[%d]=[%ld] and tran_dim_xy[%d]=[%ld] must be >=1!\n",
					i, (*arr_dim_xy)[i], i, (*tran_dim_xy)[i]);
			return -1;
		}
		if ((*arr_dim_xy)[i] < (*tran_dim_xy)[i]) {
			fprintf(stderr,
					"ERROR in meta_transpose_face: tran_dim_xy[%d]=[%ld] must be <= arr_dim_xy[%d]=[%ld]!\n",
					i, (*tran_dim_xy)[i], i, (*arr_dim_xy)[i]);
			return -1;
		}

	}
#ifdef WITH_TIMERS
	metaTimerQueueFrame * frame = (metaTimerQueueFrame*)malloc (sizeof(metaTimerQueueFrame));
	frame->mode = run_mode;
	frame->size = (*tran_dim_xy)[0]*(*tran_dim_xy)[1]*get_atype_size(type);
#endif
	switch (run_mode) {
	default:
	case metaModePreferGeneric:
		//TODO implement a generic reduce
		break;

#ifdef WITH_CUDA
		case metaModePreferCUDA: {
#ifdef WITH_TIMERS
			ret = (a_err) cuda_transpose_face(grid_size, block_size, indata, outdata, arr_dim_xy, tran_dim_xy, type, async, &(frame->event.cuda));
#else
			ret = (a_err) cuda_transpose_face(grid_size, block_size, indata, outdata, arr_dim_xy, tran_dim_xy, type, async, NULL);
#endif
			//If a callback is provided, register it immediately after returning from enqueuing the kernel
			if ((void*)call != NULL && call_pl != NULL) cudaStreamAddCallback(0, call->cudaCallback, call_pl, 0);
			break;
		}
#endif

#ifdef WITH_OPENCL
		case metaModePreferOpenCL:
		{
#ifdef WITH_TIMERS
		ret = (a_err) opencl_transpose_face(grid_size, block_size, indata, outdata, arr_dim_xy, tran_dim_xy, type, async, &(frame->event.opencl));
		//If timers exist, use their event to add the callback
		if ((void*)call != NULL && call_pl != NULL) clSetEventCallback(frame->event.opencl, CL_COMPLETE, call->openclCallback, call_pl);
#else
		//If timers don't exist, get the event via a locally-scoped event to add to the callback
		cl_event cb_event;
		ret = (a_err) opencl_transpose_face(grid_size, block_size, indata, outdata, arr_dim_xy, tran_dim_xy, type, async, &cb_event);
		if ((void*)call != NULL && call_pl != NULL) clSetEventCallback(cb_event, CL_COMPLETE, call->openclCallback, call_pl);
#endif
		}
		break;
#endif

#ifdef WITH_OPENMP
		case metaModePreferOpenMP:
#ifdef WITH_TIMERS
		{	struct timeval start, end;
			gettimeofday(&start, NULL);
			//		frame->event.openmp[0]= omp_get_wtime();
#endif
		ret = omp_transpose_face(grid_size, block_size, indata, outdata, arr_dim_xy, tran_dim_xy, type, async);
#ifdef WITH_TIMERS
		gettimeofday(&end, NULL);
		frame->event.openmp[0] = (start.tv_usec*0.000001)+start.tv_sec;
		frame->event.openmp[1] = (end.tv_usec*0.000001)+end.tv_sec;}
	//		frame->event.openmp[1]= omp_get_wtime();
#endif
		break;

#endif
	}
#ifdef WITH_TIMERS
	metaTimerEnqueue(frame, &(metaBuiltinQueues[k_transpose_2d_face]));
#endif
	return (ret);
}

//Workhorse for both sync and async pack
a_err meta_pack_face(a_dim3 * grid_size, a_dim3 * block_size,
		void *packed_buf, void *buf, meta_face *face,
		meta_type_id type, a_bool async) {
	return meta_pack_face_cb(grid_size, block_size, packed_buf, buf, face,
			type, async, (meta_callback*) NULL, NULL);
}
a_err meta_pack_face_cb(a_dim3 * grid_size, a_dim3 * block_size,
		void *packed_buf, void *buf, meta_face *face,
		meta_type_id type, a_bool async, meta_callback *call, void *call_pl) {
	a_err ret;
	//FIXME? Consider adding a compiler flag "UNCHECKED_EXPLICIT" to streamline out sanity checks like this
	//Before we do anything, sanity check that the face is set up
	if (face == NULL) {
		fprintf(stderr, "ERROR in meta_pack_face: face=[%p] is NULL!\n",
				face);
		return -1;
	}
	if (face->size == NULL || face->stride == NULL) {
		fprintf(stderr,
				"ERROR in meta_pack_face: face->size=[%p] or face->stride=[%p] is NULL!\n",
				face->size, face->stride);
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
	frame_k1->size = get_atype_size(type)*face->size[0]*face->size[1]*face->size[2];
	frame_c1->size = get_atype_size(type)*3;
	frame_c2->size = get_atype_size(type)*3;
	frame_c3->size = get_atype_size(type)*3;
//	frame->size = (*dim_xy)[0]*(*dim_xy)[1]*get_atype_size(type);
#endif

	//figure out what the aggregate size of all descendant branches are
	int *remain_dim = (int *) malloc(sizeof(int) * face->count);
	int i, j;
	remain_dim[face->count - 1] = 1;
	//This loop is backwards from Kaixi's to compute the child size in O(n)
	// rather than O(n^2) by recognizing that the size of nodes higher in the tree
	// is just the size of a child multiplied by the number of children, applied
	// upwards from the leaves
	for (i = face->count - 2; i >= 0; i--) {
		remain_dim[i] = remain_dim[i + 1] * face->size[i + 1];
		//	printf("Remain_dim[%d]: %d\n", i, remain_dim[i]);
	}

//	for(i = 0; i < face->count; i++){
//		remain_dim[i] = 1;
//		for(j=i+1; j < face->count; j++) {
//			remain_dim[i] *=face->size[j];
//		}
//			printf("Remain_dim[%d]: %d\n", i, remain_dim[i]);
//	}	

	switch (run_mode) {
	default:
	case metaModePreferGeneric:
		//TODO implement a generic reduce
		break;

#ifdef WITH_CUDA
		case metaModePreferCUDA: {
#ifdef WITH_TIMERS
			ret = (a_err) cuda_pack_face(grid_size, block_size, packed_buf, buf, face, remain_dim, type, async, &(frame_k1->event.cuda), &(frame_c1->event.cuda), &(frame_c2->event.cuda), &(frame_c3->event.cuda));
#else
			ret = (a_err) cuda_pack_face(grid_size, block_size, packed_buf, buf, face, remain_dim, type, async, NULL, NULL, NULL, NULL);
#endif
			//If a callback is provided, register it immediately after returning from enqueuing the kernel
			if ((void*)call != NULL && call_pl != NULL) cudaStreamAddCallback(0, call->cudaCallback, call_pl, 0);
			break;
		}
#endif

#ifdef WITH_OPENCL
		case metaModePreferOpenCL:
		{
#ifdef WITH_TIMERS
		ret = (a_err) opencl_pack_face(grid_size, block_size, packed_buf, buf, face, remain_dim, type, async, &(frame_k1->event.opencl), &(frame_c1->event.opencl), &(frame_c2->event.opencl), &(frame_c3->event.opencl));
		//If timers exist, use their event to add the callback
		if ((void*)call != NULL && call_pl != NULL) clSetEventCallback(frame_k1->event.opencl, CL_COMPLETE, call->openclCallback, call_pl);
#else
		//If timers don't exist, get the event via a locally-scoped event to add to the callback
		cl_event cb_event;
		ret = (a_err) opencl_pack_face(grid_size, block_size, packed_buf, buf, face, remain_dim, type, async, &cb_event, NULL, NULL, NULL);
		if ((void*)call != NULL && call_pl != NULL) clSetEventCallback(cb_event, CL_COMPLETE, call->openclCallback, call_pl);
#endif
		}
		break;
#endif

#ifdef WITH_OPENMP
		case metaModePreferOpenMP:
#ifdef WITH_TIMERS
		{	struct timeval start, end;
			gettimeofday(&start, NULL);
			//		frame_k1->event.openmp[0]= omp_get_wtime();
#endif
		ret = omp_pack_face(grid_size, block_size, packed_buf, buf, face, remain_dim, type, async);
#ifdef WITH_TIMERS
		gettimeofday(&end, NULL);
		frame_k1->event.openmp[0] = (start.tv_usec*0.000001)+start.tv_sec;
		frame_k1->event.openmp[1] = (end.tv_usec*0.000001)+end.tv_sec;}
	//		frame_k1->event.openmp[1]= omp_get_wtime();
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
	return (ret);
}

//Workhorse for both sync and async unpack
//TODO fix frame->size to reflect face size
a_err meta_unpack_face(a_dim3 * grid_size, a_dim3 * block_size,
		void *packed_buf, void *buf, meta_face *face,
		meta_type_id type, a_bool async) {
	return meta_unpack_face_cb(grid_size, block_size, packed_buf, buf, face,
			type, async, (meta_callback*) NULL, NULL);
}
a_err meta_unpack_face_cb(a_dim3 * grid_size, a_dim3 * block_size,
		void *packed_buf, void *buf, meta_face *face,
		meta_type_id type, a_bool async, meta_callback *call, void *call_pl) {
	a_err ret;
	//FIXME? Consider adding a compiler flag "UNCHECKED_EXPLICIT" to streamline out sanity checks like this
	//Before we do anything, sanity check that the face is set up
	if (face == NULL) {
		fprintf(stderr, "ERROR in meta_unpack_face: face=[%p] is NULL!\n",
				face);
		return -1;
	}
	if (face->size == NULL || face->stride == NULL) {
		fprintf(stderr,
				"ERROR in meta_unpack_face: face->size=[%p] or face->stride=[%p] is NULL!\n",
				face->size, face->stride);
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
	frame_k1->size = get_atype_size(type)*face->size[0]*face->size[1]*face->size[2];
	frame_c1->size = get_atype_size(type)*3;
	frame_c2->size = get_atype_size(type)*3;
	frame_c3->size = get_atype_size(type)*3;
//	frame->size = (*dim_xy)[0]*(*dim_xy)[1]*get_atype_size(type);
#endif

	//figure out what the aggregate size of all descendant branches are
	int *remain_dim = (int *) malloc(sizeof(int) * face->count);
	int i;
	remain_dim[face->count - 1] = 1;
	//This loop is backwards from Kaixi's to compute the child size in O(n)
	// rather than O(n^2) by recognizing that the size of nodes higher in the tree
	// is just the size of a child multiplied by the number of children, applied
	// upwards from the leaves
	for (i = face->count - 2; i >= 0; i--) {
		remain_dim[i] = remain_dim[i + 1] * face->size[i + 1];
	}

	switch (run_mode) {
	default:
	case metaModePreferGeneric:
		//TODO implement a generic reduce
		break;

#ifdef WITH_CUDA
		case metaModePreferCUDA: {
#ifdef WITH_TIMERS
			ret = (a_err) cuda_unpack_face(grid_size, block_size, packed_buf, buf, face, remain_dim, type, async, &(frame_k1->event.cuda), &(frame_c1->event.cuda), &(frame_c2->event.cuda), &(frame_c3->event.cuda));
#else
			ret = (a_err) cuda_unpack_face(grid_size, block_size, packed_buf, buf, face, remain_dim, type, async, NULL, NULL, NULL, NULL);
#endif
			//If a callback is provided, register it immediately after returning from enqueuing the kernel
			if ((void*)call != NULL && call_pl != NULL) cudaStreamAddCallback(0, call->cudaCallback, call_pl, 0);
			break;
		}
#endif

#ifdef WITH_OPENCL
		case metaModePreferOpenCL:
		{
#ifdef WITH_TIMERS
		ret = (a_err) opencl_unpack_face(grid_size, block_size, packed_buf, buf, face, remain_dim, type, async, &(frame_k1->event.opencl), &(frame_c1->event.opencl), &(frame_c2->event.opencl), &(frame_c3->event.opencl));
		//If timers exist, use their event to add the callback
		if ((void*)call != NULL && call_pl != NULL) clSetEventCallback(frame_k1->event.opencl, CL_COMPLETE, call->openclCallback, call_pl);
#else
		//If timers don't exist, get the event via a locally-scoped event to add to the callback
		cl_event cb_event;
		ret = (a_err) opencl_unpack_face(grid_size, block_size, packed_buf, buf, face, remain_dim, type, async, &cb_event, NULL, NULL, NULL);
		if ((void*)call != NULL && call_pl != NULL) clSetEventCallback(cb_event, CL_COMPLETE, call->openclCallback, call_pl);
#endif
		}
		break;
#endif

#ifdef WITH_OPENMP
		case metaModePreferOpenMP:
#ifdef WITH_TIMERS
		{	struct timeval start, end;
			gettimeofday(&start, NULL);
			//		frame_k1->event.openmp[0]= omp_get_wtime();
#endif
		ret = omp_unpack_face(grid_size, block_size, packed_buf, buf, face, remain_dim, type, async);
#ifdef WITH_TIMERS
		gettimeofday(&end, NULL);
		frame_k1->event.openmp[0] = (start.tv_usec*0.000001)+start.tv_sec;
		frame_k1->event.openmp[1] = (end.tv_usec*0.000001)+end.tv_sec;}
	//		frame->event.openmp[1]= omp_get_wtime();
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
	metaTimerEnqueue(frame_k1, &(metaBuiltinQueues[k_unpack_2d_face]));
	metaTimerEnqueue(frame_c1, &(metaBuiltinQueues[c_H2Dc]));
	metaTimerEnqueue(frame_c2, &(metaBuiltinQueues[c_H2Dc]));
	metaTimerEnqueue(frame_c3, &(metaBuiltinQueues[c_H2Dc]));
#endif
	return (ret);
}

//Workhorse for both sync and async stencil
a_err meta_stencil_3d7p(a_dim3 * grid_size, a_dim3 * block_size, void *indata,
		void *outdata, a_dim3 * array_size, a_dim3 * array_start,
		a_dim3 * array_end, meta_type_id type, a_bool async) {
	return meta_stencil_3d7p_cb(grid_size, block_size, indata, outdata,
			array_size, array_start, array_end, type, async,
			(meta_callback*) NULL, NULL);
}
a_err meta_stencil_3d7p_cb(a_dim3 * grid_size, a_dim3 * block_size,
		void *indata, void *outdata, a_dim3 * array_size, a_dim3 * array_start,
		a_dim3 * array_end, meta_type_id type, a_bool async,
		meta_callback *call, void *call_pl) {
	a_err ret;

	//FIXME? Consider adding a compiler flag "UNCHECKED_EXPLICIT" to streamline out sanity checks like this
	//Before we do anything, sanity check the start/end/size
	if (array_start == NULL || array_end == NULL || array_size == NULL) {
		fprintf(stderr,
				"ERROR in meta_stencil: array_start=[%p], array_end=[%p], or array_size=[%p] is NULL!\n",
				array_start, array_end, array_size);
		return -1;
	}
	int i;
	for (i = 0; i < 3; i++) {
		if ((*array_start)[i] < 0 || (*array_end)[i] < 0) {
			fprintf(stderr,
					"ERROR in meta_stencil: array_start[%d]=[%ld] or array_end[%d]=[%ld] is negative!\n",
					i, (*array_start)[i], i, (*array_end)[i]);
			return -1;
		}
		if ((*array_size)[i] < 1) {
			fprintf(stderr,
					"ERROR in meta_stencil: array_size[%d]=[%ld] must be >=1!\n",
					i, (*array_size)[i]);
			return -1;
		}
		if ((*array_start)[i] > (*array_end)[i]) {
			fprintf(stderr,
					"ERROR in meta_stencil: array_start[%d]=[%ld] is after array_end[%d]=[%ld]!\n",
					i, (*array_start)[i], i, (*array_end)[i]);
			return -1;
		}
		if ((*array_end)[i] >= (*array_size)[i]) {
			fprintf(stderr,
					"ERROR in meta_stencil: array_end[%d]=[%ld] is bigger than array_size[%d]=[%ld]!\n",
					i, (*array_end)[i], i, (*array_size)[i]);
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
				new_grid[i] = ((*block_size)[i] * (*grid_size)[i] - 1
						+ new_block[i]) / new_block[i];
			}
		}
		if (flag) {
			fprintf(stderr,
					"WARNING in meta_stencil: block_size={%ld, %ld, %ld} must be all powers of two!\n\tRescaled grid_size={%ld, %ld, %ld}, block_size={%ld, %ld, %ld} to\n\tnew_grid={%ld, %ld, %ld}, new_block={%ld, %ld, %ld}\n",
					(*block_size)[0], (*block_size)[1], (*block_size)[2],
					(*grid_size)[0], (*grid_size)[1], (*grid_size)[2],
					(*block_size)[0], (*block_size)[1], (*block_size)[2],
					new_grid[0], new_grid[1], new_grid[2], new_block[0],
					new_block[1], new_block[2]);
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
	switch (run_mode) {
	default:
	case metaModePreferGeneric:
		//TODO implement a generic reduce
		break;

#ifdef WITH_CUDA
		case metaModePreferCUDA: {
#ifdef WITH_TIMERS
			ret = (a_err) cuda_stencil_3d7p(grid_size, block_size, indata, outdata, array_size, array_start, array_end, type, async, &(frame->event.cuda));
#else
			ret = (a_err) cuda_stencil_3d7p(grid_size, block_size, indata, outdata, array_size, array_start, array_end, type, async, NULL);
#endif
			//If a callback is provided, register it immediately after returning from enqueuing the kernel
			if ((void*)call != NULL && call_pl != NULL) cudaStreamAddCallback(0, call->cudaCallback, call_pl, 0);
			break;
		}
#endif

#ifdef WITH_OPENCL
		case metaModePreferOpenCL:
		{
#ifdef WITH_TIMERS
		ret = (a_err) opencl_stencil_3d7p(grid_size, block_size, indata, outdata, array_size, array_start, array_end, type, async, &(frame->event.opencl));
		//If timers exist, use their event to add the callback
		if ((void*)call != NULL && call_pl != NULL) clSetEventCallback(frame->event.opencl, CL_COMPLETE, call->openclCallback, call_pl);
#else
		//If timers don't exist, get the event via a locally-scoped event to add to the callback
		cl_event cb_event;
		ret = (a_err) opencl_stencil_3d7p(grid_size, block_size, indata, outdata, array_size, array_start, array_end, type, async, &cb_event);

		if ((void*)call != NULL && call_pl != NULL) clSetEventCallback(cb_event, CL_COMPLETE, call->openclCallback, call_pl);
#endif
		}
		break;
#endif

#ifdef WITH_OPENMP
		case metaModePreferOpenMP:
#ifdef WITH_TIMERS
		{	struct timeval start, end;
			gettimeofday(&start, NULL);
			//		frame->event.openmp[0]= omp_get_wtime();
#endif
			ret = omp_stencil_3d7p(grid_size, block_size, indata, outdata, array_size, array_start, array_end, type, async);
#ifdef WITH_TIMERS
			gettimeofday(&end, NULL);
			frame->event.openmp[0] = (start.tv_usec*0.000001)+start.tv_sec;
			frame->event.openmp[1] = (end.tv_usec*0.000001)+end.tv_sec;}
		//		frame->event.openmp[1]= omp_get_wtime();
#endif
		break;
#endif
	}
#ifdef WITH_TIMERS
	metaTimerEnqueue(frame, &(metaBuiltinQueues[k_stencil_3d7p]));
#endif
	return (ret);
}
///////////////////////
//SPMV API (meta_csr)//
///////////////////////
a_err meta_csr(a_dim3 * grid_size, a_dim3 * block_size, size_t global_size, void * csr_ap, void * csr_aj, void * csr_ax, void * x_loc, void * y_loc, 
		size_t wg_size, meta_type_id type, a_bool async) {//, cl_event * wait) {
	return meta_csr_cb(grid_size, block_size, global_size, csr_ap, csr_aj, csr_ax, x_loc, y_loc,
			wg_size, type, async,
			// wait,
			(meta_callback*) NULL, NULL);
}
a_err meta_csr_cb(a_dim3 * grid_size, a_dim3 * block_size, size_t global_size, void * csr_ap, void * csr_aj, void * csr_ax, void * x_loc, void * y_loc,
		size_t wg_size, meta_type_id type, a_bool async,
		// cl_event * wait,
		meta_callback *call, void *call_pl) {
	a_err ret;
#ifdef WITH_TIMERS
	metaTimerQueueFrame * frame = (metaTimerQueueFrame*)malloc (sizeof(metaTimerQueueFrame));
	frame->mode = run_mode;
	frame->size = wg_size*get_atype_size(type);
	frame->name = "CSR";
#endif

	switch (run_mode) {
	default:
	case metaModePreferGeneric:
		//TODO implement a generic reduce
		break;

#ifdef WITH_CUDA
		case metaModePreferCUDA:
		fprintf(stderr, "CUDA CSR Not yet supported\n"); 
		break;
#endif

#ifdef WITH_OPENCL
		case metaModePreferOpenCL:
		{
#ifdef WITH_TIMERS
		ret = (a_err) opencl_csr(grid_size, block_size, global_size, csr_ap, csr_aj, csr_ax, x_loc, y_loc, type, async,
		// wait,
		 &(frame->event.opencl));
		//If timers exist, use their event to add the callback
		if ((void*)call != NULL && call_pl != NULL) clSetEventCallback(frame->event.opencl, CL_COMPLETE, call->openclCallback, call_pl);
#else
		//If timers don't exist, get the event via a locally-scoped event to add to the callback
		cl_event cb_event;
		ret = (a_err) opencl_csr(grid_size, block_size, global_size, csr_ap, csr_aj, csr_ax, x_loc, y_loc, type, async,
		// wait,
		 &cb_event);

		if ((void*)call != NULL && call_pl != NULL) clSetEventCallback(cb_event, CL_COMPLETE, call->openclCallback, call_pl);
#endif
		}
		break;
#endif

#ifdef WITH_OPENMP
		case metaModePreferOpenMP:
		fprintf(stderr, "OpenMP CSR Not yet supported");
		break;
#endif
	}
#ifdef WITH_TIMERS
	metaTimerEnqueue(frame, &(metaBuiltinQueues[k_csr]));
#endif
	return (ret);
}

///////////////////////
//CRC API (meta_crc)//
///////////////////////
a_err meta_crc(a_dim3 * grid_size, a_dim3 * block_size, void * dev_input, int page_size, int num_words, int numpages, void * dev_output, 
		meta_type_id type, a_bool async) {//, cl_event * wait) {
	return meta_crc_cb(grid_size, block_size, dev_input, page_size, num_words, numpages, dev_output,
			type, async,// wait,
			(meta_callback*) NULL, NULL);
}
a_err meta_crc_cb(a_dim3 * grid_size, a_dim3 * block_size, void * dev_input, int page_size, int num_words, int numpages, void * dev_output,
		meta_type_id type, a_bool async,
		// cl_event * wait,
		meta_callback *call, void *call_pl) {
	a_err ret;
#ifdef WITH_TIMERS
	metaTimerQueueFrame * frame = (metaTimerQueueFrame*)malloc (sizeof(metaTimerQueueFrame));
	frame->mode = run_mode;
	frame->size = (*block_size)[0]*get_atype_size(type);
	frame->name = "CRC";
#endif

	switch (run_mode) {
	default:
	case metaModePreferGeneric:
		//TODO implement a generic reduce
		break;

#ifdef WITH_CUDA
		case metaModePreferCUDA: 
		fprintf(stderr, "CUDA CRC Not yet supported\n");
		break;
#endif

#ifdef WITH_OPENCL
		case metaModePreferOpenCL:
		{
#ifdef WITH_TIMERS
		ret = (a_err) opencl_crc(dev_input, page_size, num_words, numpages, dev_output, type, async,
		// wait,
		 &(frame->event.opencl));
		
		if ((void*)call != NULL && call_pl != NULL) clSetEventCallback(frame->event.opencl, CL_COMPLETE, call->openclCallback, call_pl);
#else
		cl_event cb_event;
		ret = (a_err) opencl_crc(dev_input, page_size, num_words, numpages, dev_output, type, async,
		// wait,
		 &cb_event);

		if ((void*)call != NULL && call_pl != NULL) clSetEventCallback(cb_event, CL_COMPLETE, call->openclCallback, call_pl);
#endif
		}
		break;
#endif

#ifdef WITH_OPENMP
		case metaModePreferOpenMP:
		fprintf(stderr, "OpenMP CRC Not yet supported");
		break;
#endif
	}
#ifdef WITH_TIMERS
	metaTimerEnqueue(frame, &(metaBuiltinQueues[k_crc]));
#endif
	return (ret);
}
