#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/** OpenMP Back-End **/
#ifndef METAMORPH_OPENMP_BACKEND_H
#define METAMORPH_OPENMP_BACKEND_H

#ifndef METAMORPH_H
#include "metamorph.h"
#endif

typedef struct timeval openmpEvent;

#ifdef __OPENMPCC__
extern "C" {
#endif
a_err metaOpenMPAlloc(void **, size_t);
a_err metaOpenMPFree(void * ptr);
a_err metaOpenMPWrite(void *, void *, size_t, a_bool, meta_callback *, meta_event *);
a_err metaOpenMPRead(void *, void *, size_t, a_bool, meta_callback *, meta_event *);
a_err metaOpenMPDevCopy(void *, void *, size_t, a_bool, meta_callback *, meta_event *);
a_err metaOpenMPFlush();
a_err metaOpenMPCreateEvent(void**);
a_err metaOpenMPDestroyEvent(void*);
a_err metaOpenMPRegisterCallback(meta_callback *);

  a_err openmp_dotProd(size_t (*)[3], size_t (*)[3], void *, void *, size_t (*)[3], size_t (*)[3], size_t (*)[3], void *, meta_type_id, int, meta_callback *, meta_event *);
a_err openmp_reduce(size_t (*)[3], size_t (*)[3], void *, size_t (*)[3], size_t (*)[3], size_t (*)[3], void *, meta_type_id, int, meta_callback *, meta_event *);
a_err openmp_transpose_face(size_t (*)[3], size_t (*)[3], void *, void *, size_t (*)[3], size_t (*)[3], meta_type_id, int, meta_callback *, meta_event *);
a_err openmp_pack_face(size_t (*)[3], size_t (*)[3], void *, void *, meta_face *, int *, meta_type_id, int, meta_callback *, meta_event *, meta_event *, meta_event *, meta_event *);
a_err openmp_unpack_face(size_t (*)[3], size_t (*)[3], void *, void *, meta_face *, int *, meta_type_id, int, meta_callback *, meta_event *, meta_event *, meta_event *, meta_event *);

a_err openmp_stencil_3d7p(size_t (*)[3], size_t (*)[3], void *, void *, size_t (*)[3], size_t (*)[3], size_t (*)[3], meta_type_id, int, meta_callback *, meta_event *);

int omp_copy_d2d(void *dst, void *src, size_t size, int async);

#ifdef __OPENMPCC__
}
#endif

#endif //METAMORPH_OPENMP_BACKEND_H
