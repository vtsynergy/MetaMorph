/*
 * Exclusively for the core mechanics of timer implementations
 * Don't need to expose too many things to direct users of the profiing API
 */

//TODO Now that we have moved to an expliit enqueue call, design an API for a timer to reside on multiple queues, including *user* queues

/** The top-level user APIs **/
#ifndef METAMORPH_PROFILING_H
#define METAMORPH_PROFILING_H

//Until the dyn pointers are declared in their own header (last) these have to be declared before definition so that the metamorph.h and metamorph_timers.h headers work whichever order they show up in
struct meta_timer;
typedef struct meta_timer meta_timer;
//A convenience structure to tie logically-named queues to possibly-changing
// indices in the Queue array. This way we can add a new type of timer
// (for a kernel or copy or whatever) and potentially change indices without
// needing to change a bunch of hardcoded constants.
typedef enum metaProfilingBuiltinQueueType {
	//Host to Device copies into __constant memory (cudaMemcpyToSymbol and CL_MEM_READ_ONLY)
	c_D2H, c_H2D, c_H2H, c_D2D, c_H2Dc,
	k_reduce,
	k_dotProd,
	k_transpose_2d_face,
	k_pack_2d_face,
	k_unpack_2d_face,
	k_stencil_3d7p,
	k_csr,
	k_crc,
	
	//Special value used just to determine the size of the enum automagically
	// only works if we don't set explicit values for anything, and let it start from 0
	queue_count
}metaProfilingBuiltinQueueType;
//Include metamorph.h to grab the properly-defined enum for modes
#ifndef METAMORPH_H
#include "metamorph.h"
#endif
#include <stdlib.h>

struct meta_timer {
  char const * name;
  meta_event event;
//Hijack meta_preferred_mode enum to advise the user of the frame/node how to interpret event
  meta_preferred_mode mode;
  size_t size;
//TODO add level 3 items
};

//TODO refactor code to use a frame internally, so that the frame can be changed
// without having to modify the QueueNode to match
typedef struct metaTimerQueueNode {
	meta_timer timer;
//	char const * name;
//	meta_event event;
//Hijack meta_preferred_mode enum to advise the user of the frame/node how to interpret event
//	meta_preferred_mode mode;
//	size_t size;
	struct metaTimerQueueNode * next;
} metaTimerQueueNode;

typedef struct metaTimerQueue {
	const char * name;
	metaTimerQueueNode * head, *tail;
} metaTimerQueue;

a_err metaProfilingCreateTimer(meta_timer **, meta_preferred_mode, size_t);
a_err metaProfilingEnqueueTimer(meta_timer, metaProfilingBuiltinQueueType);
//a_err metaProfilingDestroyTimer(meta_timer *);

#ifdef DEPRECATED
a_err cl_get_event_node(metaTimerQueue * queue, char * ename,  metaTimerQueueFrame ** frame);
#endif //DEPRECATED
__attribute__((constructor(104))) a_err metaTimersInit();
a_err metaTimersFlush();
__attribute__((destructor(104))) a_err metaTimersFinish();

extern metaTimerQueue metaBuiltinQueues[];
//Expose fortran versions now that fortran is in core capability
int meta_timers_init_c_();
int meta_timers_flush_c_();
int meta_timers_finish_c_();
#endif //METAMORPH_PROFILING_H
