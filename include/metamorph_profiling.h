/** \file
 * Exposed functions and types for the MetaMorph event-based profiling plugin
 * Exclusively for the core mechanics of timer implementations
 * Don't need to expose too many things to direct users of the profiing API
 */

// TODO Now that we have moved to an expliit enqueue call, design an API for a
// timer to reside on multiple queues, including *user* queues

/** The top-level user APIs **/
#ifndef METAMORPH_PROFILING_H
#define METAMORPH_PROFILING_H

// Until the dyn pointers are declared in their own header (last) these have to
// be declared before definition so that the metamorph.h and metamorph_timers.h
// headers work whichever order they show up in struct meta_timer; typedef struct
// meta_timer meta_timer; A convenience structure to tie logically-named queues
// to possibly-changing
// indices in the Queue array. This way we can add a new type of timer
// (for a kernel or copy or whatever) and potentially change indices without
// needing to change a bunch of hardcoded constants.
typedef enum metaProfilingBuiltinQueueType {
  // Host to Device copies into __constant memory (cudaMemcpyToSymbol and
  // CL_MEM_READ_ONLY)
  c_D2H,
  c_H2D,
  c_H2H,
  c_D2D,
  c_H2Dc,
  k_reduce,
  k_dotProd,
  k_transpose_2d_face,
  k_pack_2d_face,
  k_unpack_2d_face,
  k_stencil_3d7p,
  k_csr,
  k_crc,

  // Special value used just to determine the size of the enum automagically
  // only works if we don't set explicit values for anything, and let it start
  // from 0
  queue_count
} metaProfilingBuiltinQueueType;
// Include metamorph.h to grab the properly-defined enum for modes
#ifndef METAMORPH_H
#include "metamorph.h"
#endif
#include <stdlib.h>

/**
 * Store a single timing object (typically a single kernel run or a single
 * transfer)
 */
typedef struct meta_timer {
  /** Record the name of the timing object, if relevant */
  char const *name;
  /** Timers are event-based, which abstract the various backend' event
   * mechanisms, timer is associated with one */
  meta_event event;
  /** What global mode MetaMorph was running in when the timer was created */
  meta_preferred_mode mode;
  /** The size of whatever object is being timed (size of transfer, volume of
   * the computed region, etc. */
  size_t size;
  //\todo add level 3 items
} meta_timer;

/** All timing objects are maintained on separate queues, inside one of these
 * nodes */
typedef struct metaTimerQueueNode {
  /** A complete timing object, copied from upper layers */
  meta_timer timer;
  /** Pointer to the next timer in the queue */
  struct metaTimerQueueNode *next;
} metaTimerQueueNode;

/** Each queue is named for reporting results, and needs to maintain its head
 * and tail pointers */
typedef struct metaTimerQueue {
  /** The human-readable name for this queue, for printing results */
  const char *name;
  /** The dequeue end of the queue **/
  metaTimerQueueNode *head;
  /** The enqueue end of the queue **/
  metaTimerQueueNode *tail;
} metaTimerQueue;

/**
 * Create a timer, initializing an event on the underlying backend currently in
 * use
 * \param ret_timer Address in which to return the allocated and initialized
 * timer
 * \param mode The mode to initialize the timer in, typically whatever the
 * current global mode is
 * \param size The size of the event being timed (transfer bytes, kernel volume,
 * etc.)
 * \return -1 if the return pointer is NULL, otherwise 0
 */
a_err metaProfilingCreateTimer(meta_timer **ret_timer, meta_preferred_mode mode,
                               size_t size);
/**
 * Take a copy of the frame and insert it on to the selected queue.
 * Do nothing else with the frame, the caller should handle releasing it.
 * \param timer The timer to record
 * \param type Which of the builtin queue types to record it on
 * \return -1 if the queue is invalid, 0 otherwise
 * \todo FIXME, implement a real set of error return codes
 * \todo Make Hazard aware
 */
a_err metaProfilingEnqueueTimer(meta_timer timer,
                                metaProfilingBuiltinQueueType type);

#ifdef DEPRECATED
a_err cl_get_event_node(metaTimerQueue *queue, char *ename,
                        metaTimerQueueFrame **frame);
#endif // DEPRECATED
/**
 * Prepare the environment for timing, should be called by the first METAMORPH
 * Runtime Library Call, and should never be called again. If called a second
 * time before metaTimersFinish is called, will be a silent NOOP.
 */
__attribute__((constructor(104))) void metaTimersInit();
/**
 * Flush out properly-formatted timers/bandwidths/diagnostics
 * Clears the timer queues entirely, and performs some (potentially intensive)
 * aggregation of statistics, so should be used sparingly. If the application
 * consists of large-scale sequential phases it might be appropriate to flush
 * between phases, or if a sufficient number of library calls are performed to
 * cause significant memory overhead (unlikely except in RAM-starved systems).
 * Otherwise the metaTimersFlush() call inherent in metaTimersFinish should
 * be sufficient.
 * \return hardcoded to zero
 * \todo Implement proper error codes
 */
a_err metaTimersFlush();
/**
 * Flush and destroy the profiling infrastructure
 * \return hardcoded to zero
 * \todo Implement proper error codes
 * \bug FIXME: Doesn't currently destroy anything, it's just a wrapper around
 * flush that then sets the initialization state to false
 */
a_err metaTimersFinish();

/** Wrap the initializer for Fortran ISO_C_BINDINGS */
void meta_timers_init_c_();
/** Wrap the flush for Fortran ISO_C_BINDINGS */
void meta_timers_flush_c_();
/** Wrap the finalizer for Fortran ISO_C_BINDINGS */
void meta_timers_finish_c_();
#endif // METAMORPH_PROFILING_H
