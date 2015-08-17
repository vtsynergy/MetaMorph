#include "metamorph_timers.h"

metaTimerQueue metaBuiltinQueues[queue_count];


//Take a copy of the frame and shove it on to the selected queue
// Do nothing else with the frame, the caller should handle releasing it
a_err metaTimerEnqueue(metaTimerQueueFrame * frame, metaTimerQueue * queue) {
	//allocate a new node - still in the thread-private allocated state
	metaTimerQueueNode * newnode =(metaTimerQueueNode*) malloc(sizeof(metaTimerQueueNode));
	//initialize the new node
//	printf("ENQUEUE %x\n", newnode);
	newnode->event = frame->event;
	newnode->mode = frame->mode;
	newnode->size = frame->size;
	newnode->next = NULL;
	metaTimerQueueNode *t, *n;
	while(1) {
		t = queue->tail;
		//Set a hazard pointer for tail, and check it
		if (queue->tail != t) continue;
		n = t->next;
		if (queue->tail != t) continue;
		if (n != NULL) {
			__sync_bool_compare_and_swap(&(queue->tail), t, n);
			continue;
		}
		if (__sync_bool_compare_and_swap(&(t->next), NULL, newnode)) break;
	}
	__sync_bool_compare_and_swap(&(queue->tail), t, newnode);	
}

//Take the copy of the front frame on the queue, and then remove it from the queue
// Do nothing else with the frame's copy, the caller should allocate and free it.
a_err metaTimerDequeue(metaTimerQueueFrame ** frame, metaTimerQueue * queue) {
	//TODO add a check to make sure the caller actually allocated the frame
	//TODO make this dequeue hazard-aware 
	int count = 0;
	metaTimerQueueNode *h, *t, *n;
	while(1) { //keep attempting til the dequeue is uncontended
		h = queue->head;
//		printf("DEQUEUE %d %x\n", count, h);
		count++;
		//Add hazard pointer for h, check if head is still h
		if (queue->head != h) continue;
		t = queue->tail;
		n = h->next;
		//add hazard pointer for n
		if (queue->head != h) continue;
		if (n == NULL) return -1; //empty status
		if (h == t) {
			__sync_bool_compare_and_swap(&(queue->tail), t, n);
			continue;
		}
		//Copy the node's data to the caller-allocated frame
		(*frame)->event = n->event;
		(*frame)->mode = n->mode;
		(*frame)->size = n->size;
		if (__sync_bool_compare_and_swap(&(queue->head), h, n)) break;
	}
	//Need to make this a hazard-aware retire
//	free(h);
	return 0; //success
}

//Prepare the environment for timing, should be called by the first METAMORPH Runtime
// Library Call, and should never be called again. If called a second time before
// metaTimersFinish is called, will be a silent NOOP.
a_err metaTimersInit() {
	//Each builtin queue needs one of these pairs
	// try to use the enum ints/names to minimize changes if new members
	// are added to the enum

	//Init the device-to-host copy queue
	metaBuiltinQueues[c_D2H].head = metaBuiltinQueues[c_D2H].tail = (metaTimerQueueNode*) malloc(sizeof(metaTimerQueueNode));
	metaBuiltinQueues[c_D2H].head->mode = metaModeUnset;
	metaBuiltinQueues[c_D2H].head->next = NULL;
	metaBuiltinQueues[c_D2H].name = "Device to Host transfer"; 

	//Init the host-to-device copy queue
	metaBuiltinQueues[c_H2D].head = metaBuiltinQueues[c_H2D].tail = (metaTimerQueueNode*) malloc(sizeof(metaTimerQueueNode));

	metaBuiltinQueues[c_H2D].head->mode = metaModeUnset;
	metaBuiltinQueues[c_H2D].head->next = NULL;
	metaBuiltinQueues[c_H2D].name = "Host to Device transfer";

	//Init the host-to-host copy queue
	metaBuiltinQueues[c_H2H].head = metaBuiltinQueues[c_H2H].tail = (metaTimerQueueNode*) malloc(sizeof(metaTimerQueueNode));
	metaBuiltinQueues[c_H2H].head->mode = metaModeUnset;
	metaBuiltinQueues[c_H2H].head->next = NULL;
	metaBuiltinQueues[c_H2H].name = "Host to Host transfer";

	//Init the device-to-device copy queue
	metaBuiltinQueues[c_D2D].head = metaBuiltinQueues[c_D2D].tail = (metaTimerQueueNode*) malloc(sizeof(metaTimerQueueNode));
	metaBuiltinQueues[c_D2D].head->mode = metaModeUnset;
	metaBuiltinQueues[c_D2D].head->next = NULL;
	metaBuiltinQueues[c_D2D].name = "Device to Device transfer";

	//Init the host-to-device (constant) copy queue
	metaBuiltinQueues[c_H2Dc].head = metaBuiltinQueues[c_H2Dc].tail = (metaTimerQueueNode*) malloc(sizeof(metaTimerQueueNode));
	metaBuiltinQueues[c_H2Dc].head->mode = metaModeUnset;
	metaBuiltinQueues[c_H2Dc].head->next = NULL;
	metaBuiltinQueues[c_H2Dc].name = "Host to Constant transfer";

	//Init the Reduction kernel queue
	metaBuiltinQueues[k_reduce].head = metaBuiltinQueues[k_reduce].tail = (metaTimerQueueNode*) malloc(sizeof(metaTimerQueueNode));
	metaBuiltinQueues[k_reduce].head->mode = metaModeUnset;
	metaBuiltinQueues[k_reduce].head->next = NULL;
	metaBuiltinQueues[k_reduce].name = "Reduction kernel call";

	//Init the Dot Product kernel queue
	metaBuiltinQueues[k_dotProd].head = metaBuiltinQueues[k_dotProd].tail = (metaTimerQueueNode*) malloc(sizeof(metaTimerQueueNode));
	metaBuiltinQueues[k_dotProd].head->mode = metaModeUnset;
	metaBuiltinQueues[k_dotProd].head->next = NULL;
	metaBuiltinQueues[k_dotProd].name = "Dot Product kernel call";

	//Init the Dot Product kernel queue
	metaBuiltinQueues[k_transpose_2d_face].head = metaBuiltinQueues[k_transpose_2d_face].tail = (metaTimerQueueNode*) malloc(sizeof(metaTimerQueueNode));
	metaBuiltinQueues[k_transpose_2d_face].head->mode = metaModeUnset;
	metaBuiltinQueues[k_transpose_2d_face].head->next = NULL;
	metaBuiltinQueues[k_transpose_2d_face].name = "Transpose 2DFace kernel call";

	//Init the Dot Product kernel queue
	metaBuiltinQueues[k_pack_2d_face].head = metaBuiltinQueues[k_pack_2d_face].tail = (metaTimerQueueNode*) malloc(sizeof(metaTimerQueueNode));
	metaBuiltinQueues[k_pack_2d_face].head->mode = metaModeUnset;
	metaBuiltinQueues[k_pack_2d_face].head->next = NULL;
	metaBuiltinQueues[k_pack_2d_face].name = "Pack 2DFace kernel call";

	//Init the Dot Product kernel queue
	metaBuiltinQueues[k_unpack_2d_face].head = metaBuiltinQueues[k_unpack_2d_face].tail = (metaTimerQueueNode*) malloc(sizeof(metaTimerQueueNode));
	metaBuiltinQueues[k_unpack_2d_face].head->mode = metaModeUnset;
	metaBuiltinQueues[k_unpack_2d_face].head->next = NULL;
	metaBuiltinQueues[k_unpack_2d_face].name = "Unpack 2DFace kernel call";
}

//Workhorse that loops over a queue until it receives an empty signal
// Performs work according to what METAMORPH_TIMER_LEVEL is passed in.
//TODO figure out how to handle encountered events which have not completed
// (do we put them back on the queue? register a callback? force the command_queue to finish?
void flushWorker(metaTimerQueue * queue, int level) {
	metaTimerQueueFrame * frame = (metaTimerQueueFrame*) malloc(sizeof(metaTimerQueueFrame));
	int val;
	unsigned long start, end, count = 0;
	size_t size = 0;
	float time = 0.0f, temp_t = 0.0f;
	while((val =metaTimerDequeue(&frame, queue)) != -1) {
		//use one loop to do everything
//		printf("JUST CHECKING %d\n", val);
		if (level >= 1) {
			if (frame->mode == metaModePreferGeneric) {
				//TODO add some generic stuff
			}
			#ifdef WITH_CUDA
			else if (frame->mode == metaModePreferCUDA) {
				//TODO add a check to cudaEventQuery to make sure frame->event.cuda[1] is finished
				cudaEventElapsedTime(&temp_t, frame->event.cuda[0], frame->event.cuda[1]);	
			}
			#endif
			#ifdef WITH_OPENCL
			else if (frame->mode == metaModePreferOpenCL) {
				//TODO add a check via clGetEventInfo to make sure the event has completed
				clGetEventProfilingInfo(frame->event.opencl, CL_PROFILING_COMMAND_START, sizeof(unsigned long), &start, NULL);
				clGetEventProfilingInfo(frame->event.opencl, CL_PROFILING_COMMAND_END, sizeof(unsigned long), &end, NULL);
				//TODO does this accumulate need any extra checks?
				temp_t = (end-start)*0.000001;
				
			}
			#endif
			#ifdef WITH_OPENMP
			else if (frame->mode == metaModePreferOpenMP) {
				temp_t = (float) ((frame->event.openmp[1] - frame->event.openmp[0]) * 1000.0);
			}
	 		#endif 
			//Aggregate times/bandwidth across all 
		}		
		if (level >= 2) {
			//Individual call times/bandwidths
			//TODO come up with a reasonable, generic bandwidth calculation.
			fprintf(stderr, "\t%s [%lu] on [%lu]bytes took [%f]ms, with [%f]MB/s approximate bandwidth.\n", queue->name, count, frame->size, temp_t, (frame->size > 0 && temp_t > 0) ? frame->size*.001/temp_t : 0.0);
		}

		if (level == 3) {
			//Really verbose stuff, like block/grid size
		}
		time += temp_t;
		size += frame->size;
		count++;
		//Eating the node for level 0 is inherent in the while
	}
	if (level > 0 && time > 0)fprintf(stderr, "All %ss took [%f]ms, with [%f]MB/s approximate average bandwidth.\n", queue->name, time, (size > 0 && time > 0) ? size*.001/time : 0.0);
}

//Flush out properly-formatted timers/bandwidths/diagnostics
// Clears the timer queues entirely, and performs some (potentially intensive)
// aggregation of statistics, so should be used sparingly. If the application
// consists of large-scale sequential phases it might be appropriate to flush
// between phases, or if a sufficient number of library calls are performed to
// cause significant memory overhead (unlikely except in RAM-starved systems).
// Otherwise the metaTimersFlush() call inherent in metaTimersFinish should
// be sufficient.
a_err metaTimersFlush() {
	//Basically, just run through all the builtin queues,
	// dequeuing each element and tallying it up
	//This is where we need METAMORPH_TIMER_LEVEL
	int i;
	if  (getenv("METAMORPH_TIMER_LEVEL") != NULL) {
		if (strcmp(getenv("METAMORPH_TIMER_LEVEL"), "0") == 0) {
			for(i = 0; i < queue_count; i++) flushWorker(&metaBuiltinQueues[i], 0);
			//Just eat the nodes, don't do anything with them
		}

		else if (strcmp(getenv("METAMORPH_TIMER_LEVEL"), "1") == 0) {
			fprintf(stderr, "***TIMER LEVEL 1 FLUSH START***\n");
			for(i = 0; i < queue_count; i++) flushWorker(&metaBuiltinQueues[i], 1);
			fprintf(stderr, "***TIMER LEVEL 1 FLUSH END***\n");
			//Aggregate just averages
		}

		else if (strcmp(getenv("METAMORPH_TIMER_LEVEL"), "2") == 0) {
			fprintf(stderr, "***TIMER LEVEL 2 FLUSH START***\n");
			for(i = 0; i < queue_count; i++) flushWorker(&metaBuiltinQueues[i], 2);
			fprintf(stderr, "***TIMER LEVEL 2 FLUSH END***\n");
			//Display all individual nodes, with bandwidth (for transfers)
		}

		else if (strcmp(getenv("METAMORPH_TIMER_LEVEL"), "3") == 0) {
			for(i = 0; i < queue_count; i++) flushWorker(&metaBuiltinQueues[i], 3);
			//Display full diagnostics
			// This should include grid and block sizes
			// offsets, device and mode, and once-only dumps at the
			// beginning (like system specs)
		}

		else {
			fprintf(stderr, "Error: METAMORPH_TIMER_LEVEL=\"%s\" not supported!\n",
				getenv("METAMORPH_TIMER_LEVEL"));
		} 
	} else {
		for(i = 0; i < queue_count; i++) flushWorker(&metaBuiltinQueues[i], 0);
		//no mode is specified, just silently eat the nodes.
	}
}


//Safely flush timer stats to the output stream
a_err metaTimersFinish() {

	//first, make sure everything is flushed.
	metaTimersFlush();
	
	//then remove all reference points to these timers
	// (so that another thread can potentially spin up a separate new set..)
	//TODO timer cleanup
}


//TODO expose a way for users to generate their own timer queues
// Will likely require overloaded funciton headers for each call which
// take a queue list/count struct..