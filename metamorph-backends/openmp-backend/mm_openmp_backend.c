/** OpenMP Back-End **/
#include <sys/time.h>
#include "../../metamorph-backends/openmp-backend/mm_openmp_backend.h"

extern struct profiling_dyn_ptrs profiling_symbols;
//#define	COLLAPSE
//#define USE_AVX

//TODO figure out how to use templates with the  intrinsics

#define BLOCK 	8
#define CX		256
#define CY		32
#define CZ		1

#include <x86intrin.h>

#ifdef USE_AVX
#include <x86intrin.h>

inline double hadd_pd(__m256d a) {
	__m256d s1 = _mm256_hadd_pd(a,a);
	__m128d s1_h = _mm256_extractf128_pd(s1,1);
	__m128d result = _mm_add_pd(_mm256_castpd256_pd128(s1),s1_h);
	return _mm_cvtsd_f64(result);
}

inline float hadd_ps(__m256 a) {
	__m256 s1 = _mm256_hadd_ps(a,a);
	__m256 s2 = _mm256_hadd_ps(s1,s1);
	__m128 s2_h = _mm256_extractf128_ps(s2,1);
	__m128 result = _mm_add_ss(_mm256_castps256_ps128(s2),s2_h);

	return _mm_cvtss_f32(result);
}

#endif

a_err metaOpenMPAlloc(void ** ptr, size_t size) {
  a_err ret = 0;
#ifdef ALIGNED_MEMORY
  *ptr = (void *) _mm_malloc(size, ALIGNED_MEMORY_PAGE);
#else
  *ptr = (void *) malloc(size);
#endif
  if(*ptr == NULL) ret = -1;
  return ret;
}
a_err  metaOpenMPFree(void * ptr) {
#ifdef ALIGNED_MEMORY
  _mm_free(ptr);
#else
  free(ptr);
#endif
  return 0;
}
a_err metaOpenMPWrite(void * dst, void * src, size_t size, a_bool async, meta_callback * call, meta_event * ret_event) {
  a_err ret = 0;
  openmpEvent * events = NULL;
  if (ret_event != NULL && ret_event->mode == metaModePreferOpenMP && ret_event->event_pl != NULL) events = ((openmpEvent *)ret_event->event_pl);
  meta_timer * timer = NULL;
  if (profiling_symbols.metaProfilingCreateTimer != NULL) {
    (*(profiling_symbols.metaProfilingCreateTimer))(&timer, metaModePreferOpenMP, size);
    if (events == NULL) {
      events = ((openmpEvent *)timer->event.event_pl);
    } else {
      //FIXME: are we leaking a created openmpEvent here since the profiling function calls create?
      //metaOpenMPDestroyEvent(frame->event.event_pl);
      timer->event = *ret_event;
    }
  }
  if (events != NULL) {
    gettimeofday(&(events[0]), NULL);
  }
  //FIXME: Implement async
  memcpy(dst, src, size);
  if (events != NULL) {
    gettimeofday(&(events[1]), NULL);
  }
  //FIXME: Implement async "callback"
  if (call != NULL) {
    (call->callback_func)(call);
  }
  ret = 0;
    if (profiling_symbols.metaProfilingEnqueueTimer != NULL) (*(profiling_symbols.metaProfilingEnqueueTimer))(*timer, c_H2D);
  return ret;
}
a_err metaOpenMPRead(void * dst, void * src, size_t size, a_bool async, meta_callback * call, meta_event * ret_event) {
  a_err ret = 0;
  openmpEvent * events = NULL;
  if (ret_event != NULL && ret_event->mode == metaModePreferOpenMP && ret_event->event_pl != NULL) events = ((openmpEvent *)ret_event->event_pl);
  meta_timer * timer = NULL;
  if (profiling_symbols.metaProfilingCreateTimer != NULL) {
    (*(profiling_symbols.metaProfilingCreateTimer))(&timer, metaModePreferOpenMP, size);
    if (events == NULL) {
      events = ((openmpEvent *)timer->event.event_pl);
    } else {
      //FIXME: are we leaking a created openmpEvent here since the profiling function calls create?
      //metaOpenMPDestroyEvent(frame->event.event_pl);
      timer->event = *ret_event;
    }
  }
  if (events != NULL) {
    gettimeofday(&(events[0]), NULL);
  }
  //FIXME: Implement async
  memcpy(dst, src, size);
  //FIXME: Implement async "callback"
  if (events != NULL) {
    gettimeofday(&(events[1]), NULL);
  }
  if (call != NULL) {
    (call->callback_func)(call);
  }
  ret = 0;
    if (profiling_symbols.metaProfilingEnqueueTimer != NULL) (*(profiling_symbols.metaProfilingEnqueueTimer))(*timer, c_D2H);
  return ret;
}
a_err metaOpenMPDevCopy(void * dst, void * src, size_t size, a_bool async, meta_callback * call, meta_event * ret_event) {
  a_err ret = 0;
  openmpEvent * events = NULL;
  if (ret_event != NULL && ret_event->mode == metaModePreferOpenMP && ret_event->event_pl != NULL) events = ((openmpEvent *)ret_event->event_pl);
  meta_timer * timer = NULL;
  if (profiling_symbols.metaProfilingCreateTimer != NULL) {
    (*(profiling_symbols.metaProfilingCreateTimer))(&timer, metaModePreferOpenMP, size);
    if (events == NULL) {
      events = ((openmpEvent *)timer->event.event_pl);
    } else {
      //FIXME: are we leaking a created openmpEvent here since the profiling function calls create?
      //metaOpenMPDestroyEvent(frame->event.event_pl);
      timer->event = *ret_event;
    }
  }
  if (events != NULL) {
    gettimeofday(&(events[0]), NULL);
  }
  //FIXME: Implement async
			//memcpy(dst, src, size);
	int i;
	//int num_t = omp_get _num_threads();
	int num_b = size / sizeof(unsigned long);

#pragma omp parallel for
#pragma ivdep
	//#pragma vector nontemporal (dst)
	for (i = 0; i < num_b; i++) {
		//memcpy((unsigned char *) dst+i, (unsigned char *) src+i, size);
		*((unsigned long *) dst + i) = *((unsigned long *) src + i);
	}
  //FIXME: Implement async "callback"
  if (events != NULL) {
    gettimeofday(&(events[1]), NULL);
  }
  if (call != NULL) {
    (call->callback_func)(call);
  }
  ret = 0;
    if (profiling_symbols.metaProfilingEnqueueTimer != NULL) (*(profiling_symbols.metaProfilingEnqueueTimer))(*timer, c_D2D);
  return ret;
}
a_err metaOpenMPFlush() {
  a_err ret = 0;
//FIXME: When the OpenMP backend actually supports async (via futures or OpenMP 4.0, whatever, make this a proper flush like the other backends
#pragma omp barrier
  return ret;
}
a_err metaOpenMPCreateEvent(void ** ret_event) {
  a_err ret = 0;
  if (ret_event != NULL) {
    *ret_event = calloc(2, sizeof(openmpEvent));
  }
  else ret = -1;
  return ret;
}
a_err metaOpenMPDestroyEvent(void * event) {
  a_err ret = 0;
  if (event != NULL) {
    free(event);
  }
  else ret = -1;
  return ret;
}

a_err metaOpenMPRegisterCallback(meta_callback * call) {
  //FIXME: Implement async "callback"
  if (call != NULL) {
    (call->callback_func)(call);
  }
}

// Kernels
static void omp_dotProd_kernel_db(double * __restrict__ data1,
		double * __restrict__ data2, size_t (*array_size)[3],
		size_t (*arr_start)[3], size_t (*arr_end)[3], double * reduction_var) {
	int ni, nj, nk;
	double sum = 0;

	ni = (*array_size)[0];
	nj = (*array_size)[1];
	nk = (*array_size)[2];

#ifdef COLLAPSE
	int n;
	n = ((* arr_end)[0] - (* arr_start)[0] + 1) *
	((* arr_end)[1] - (* arr_start)[1] + 1) *
	((* arr_end)[2] - (* arr_start)[2] + 1);

	if( n == ni*nj*nk )	// original 3d grid
	{
		int i;
#ifdef USE_AVX
#pragma omp parallel shared(n, data1, data2)  private(i) reduction(+: sum)
		{
			__m256d sum1 = _mm256_setzero_pd();
			__m256d sum2 = _mm256_setzero_pd();
			__m256d sum3 = _mm256_setzero_pd();
			__m256d sum4 = _mm256_setzero_pd();
			__m256d x4, y4;
#pragma omp for
			for (i = 0; i < n; i += 16) {
				x4 = _mm256_loadu_pd(&data1[i]);
				y4 = _mm256_loadu_pd(&data2[i]);
				sum1 = _mm256_add_pd(_mm256_mul_pd(x4,y4),sum1);
				x4 = _mm256_loadu_pd(&data1[i+4]);
				y4 = _mm256_loadu_pd(&data2[i+4]);
				sum2 = _mm256_add_pd(_mm256_mul_pd(x4,y4),sum2);
				x4 = _mm256_loadu_pd(&data1[i+8]);
				y4 = _mm256_loadu_pd(&data2[i+8]);
				sum3 = _mm256_add_pd(_mm256_mul_pd(x4,y4),sum3);
				x4 = _mm256_loadu_pd(&data1[i+12]);
				y4 = _mm256_loadu_pd(&data2[i+12]);
				sum4 = _mm256_add_pd(_mm256_mul_pd(x4,y4),sum4);
			}
			sum += hadd_pd(_mm256_add_pd(_mm256_add_pd(sum1,sum2),_mm256_add_pd(sum3,sum4)));
		}
#else
#pragma omp parallel for shared(n, data1, data2)  private(i) reduction(+: sum)
		for (i = 0; i < n; i++) {
			sum += data1[i] * data2[i];
		}
#endif
	}
	else	// 3d sub-grid
#endif
	{
#pragma omp parallel shared(ni, nj, nk, data1, data2, sum)
		{
			int i, j, k;
			double psum = 0;
			//double *d1, *d2;

#ifdef COLLAPSE
#pragma omp for collapse(2)
#else
#pragma omp for
#endif
			for (k = (*arr_start)[2]; k <= (*arr_end)[2]; k++) {
				for (j = (*arr_start)[1]; j <= (*arr_end)[1]; j++) {
					//d1 = &data1[j*ni+k*ni*nj];
					//d2 = &data2[j*ni+k*ni*nj];
					for (i = (*arr_start)[0]; i <= (*arr_end)[0]; i++) {
						int x;
						x = i + j * ni + k * ni * nj;
						psum += data1[x] * data2[x];
						//psum += d1[i] * d2[i];
					}
				}
			}

#pragma omp critical
			{
				sum += psum;

			}
		}
	} //endif if( n == ni*nj*nk )

	*reduction_var += sum;
}

static void omp_dotProd_kernel_fl(float * __restrict__ data1,
		float * __restrict__ data2, size_t (*array_size)[3],
		size_t (*arr_start)[3], size_t (*arr_end)[3], float * reduction_var) {
	float sum = 0;
	int ni, nj, nk;

	//data1 = __builtin_assume_aligned(data1, 32);
	//data2= __builtin_assume_aligned(data2, 32);

	ni = (*array_size)[0];
	nj = (*array_size)[1];
	nk = (*array_size)[2];

#ifdef COLLAPSE
	int n;
	n = ((* arr_end)[0] - (* arr_start)[0] + 1) *
	((* arr_end)[1] - (* arr_start)[1] + 1) *
	((* arr_end)[2] - (* arr_start)[2] + 1);

	if( n == ni*nj*nk )	// original 3d grid
	{
		int i;
#ifdef USE_AVX
#pragma omp parallel shared(n, data1, data2)  private(i) reduction(+: sum)
		{
			__m256 sum1 = _mm256_setzero_ps();
			__m256 sum2 = _mm256_setzero_ps();
			__m256 sum3 = _mm256_setzero_ps();
			__m256 sum4 = _mm256_setzero_ps();
			__m256 x8, y8;
#pragma omp for
			for (i = 0; i < n; i += 32) {
				x8 = _mm256_loadu_ps(&data1[i]);
				y8 = _mm256_loadu_ps(&data2[i]);
				sum1 = _mm256_add_ps(_mm256_mul_ps(x8,y8),sum1);
				x8 = _mm256_loadu_ps(&data1[i+8]);
				y8 = _mm256_loadu_ps(&data2[i+8]);
				sum2 = _mm256_add_ps(_mm256_mul_ps(x8,y8),sum2);
				x8 = _mm256_loadu_ps(&data1[i+16]);
				y8 = _mm256_loadu_ps(&data2[i+16]);
				sum3 = _mm256_add_ps(_mm256_mul_ps(x8,y8),sum3);
				x8 = _mm256_loadu_ps(&data1[i+24]);
				y8 = _mm256_loadu_ps(&data2[i+24]);
				sum4 = _mm256_add_ps(_mm256_mul_ps(x8,y8),sum4);
			}
			sum += hadd_ps(_mm256_add_ps(_mm256_add_ps(sum1,sum2),_mm256_add_ps(sum3,sum4)));
		}
#else
#pragma omp parallel for shared(n, data1, data2)  private(i) reduction(+: sum)
		for (i = 0; i < n; i++) {
			sum += data1[i] * data2[i];
		}
#endif
	}
	else	// 3d sub-grid
#endif
	{
#pragma omp parallel shared(ni, nj, nk, data1, data2, sum)
		{
			int i, j, k;
			float psum = 0;
#ifdef COLLAPSE
#pragma omp for collapse(2)
#else
#pragma omp for
#endif
			for (k = (*arr_start)[2]; k <= (*arr_end)[2]; k++) {
				for (j = (*arr_start)[1]; j <= (*arr_end)[1]; j++) {
					for (i = (*arr_start)[0]; i <= (*arr_end)[0]; i++) {
						int x;
						x = i + j * ni + k * ni * nj;
						psum += data1[x] * data2[x];
					}
				}
			}

#pragma omp critical
			{
				sum += psum;

			}
		}
	} //endif if( n == ni*nj*nk )

	*reduction_var += sum;
}

static void omp_dotProd_kernel_ul(unsigned long * __restrict__ data1,
		unsigned long * __restrict__ data2, size_t (*array_size)[3],
		size_t (*arr_start)[3], size_t (*arr_end)[3],
		unsigned long * reduction_var) {
	unsigned long sum = 0;
	int ni, nj, nk;

	ni = (*array_size)[0];
	nj = (*array_size)[1];
	nk = (*array_size)[2];

#ifdef COLLAPSE
	int n;
	n = ((* arr_end)[0] - (* arr_start)[0] + 1) *
	((* arr_end)[1] - (* arr_start)[1] + 1) *
	((* arr_end)[2] - (* arr_start)[2] + 1);

	if( n == ni*nj*nk )	// original 3d grid
	{
		int i;
#ifdef USE_AVX
#pragma omp parallel shared(n, data1, data2)  private(i) reduction(+: sum)
		{
#pragma omp parallel for
			for (i = 0; i < n; i++) {
				sum += data1[i] * data2[i];
			}
		}
#else
#pragma omp parallel for shared(n, data1, data2)  private(i) reduction(+: sum)
		for (i = 0; i < n; i++) {
			sum += data1[i] * data2[i];
		}
#endif
	}
	else	// 3d sub-grid
#endif
	{
#pragma omp parallel shared(ni, nj, nk, data1, data2, sum)
		{
			int i, j, k;
			unsigned long psum = 0;
#ifdef COLLAPSE
#pragma omp for collapse(2)
#else
#pragma omp for
#endif
			for (k = (*arr_start)[2]; k <= (*arr_end)[2]; k++) {
				for (j = (*arr_start)[1]; j <= (*arr_end)[1]; j++) {
					for (i = (*arr_start)[0]; i <= (*arr_end)[0]; i++) {
						int x;
						x = i + j * ni + k * ni * nj;
						psum += data1[x] * data2[x];
					}
				}
			}

#pragma omp critical
			{
				sum += psum;

			}
		}
	} //endif if( n == ni*nj*nk )

	*reduction_var += sum;
}

static void omp_dotProd_kernel_in(int * __restrict__ data1,
		int * __restrict__ data2, size_t (*array_size)[3],
		size_t (*arr_start)[3], size_t (*arr_end)[3], int * reduction_var) {
	int sum = 0;
	int ni, nj, nk;

	ni = (*array_size)[0];
	nj = (*array_size)[1];
	nk = (*array_size)[2];

#ifdef COLLAPSE
	int n;
	n = ((* arr_end)[0] - (* arr_start)[0] + 1) *
	((* arr_end)[1] - (* arr_start)[1] + 1) *
	((* arr_end)[2] - (* arr_start)[2] + 1);

	if( n == ni*nj*nk )	// original 3d grid
	{
		int i;
#ifdef USE_AVX
#pragma omp parallel shared(n, data1, data2)  private(i) reduction(+: sum)
		{
#pragma omp parallel for
			for (i = 0; i < n; i++) {
				sum += data1[i] * data2[i];
			}
		}
#else
#pragma omp parallel for shared(n, data1, data2)  private(i) reduction(+: sum)
		for (i = 0; i < n; i++) {
			sum += data1[i] * data2[i];
		}
#endif
	}
	else	// 3d sub-grid
#endif
	{
#pragma omp parallel shared(ni, nj, nk, data1, data2, sum)
		{
			int i, j, k;
			int psum = 0;
#ifdef COLLAPSE
#pragma omp for collapse(2)
#else
#pragma omp for
#endif
			for (k = (*arr_start)[2]; k <= (*arr_end)[2]; k++) {
				for (j = (*arr_start)[1]; j <= (*arr_end)[1]; j++) {
					for (i = (*arr_start)[0]; i <= (*arr_end)[0]; i++) {
						int x;
						x = i + j * ni + k * ni * nj;
						psum += data1[x] * data2[x];
					}
				}
			}

#pragma omp critical
			{
				sum += psum;

			}
		}
	} //endif if( n == ni*nj*nk )

	*reduction_var += sum;
}

static void omp_dotProd_kernel_ui(unsigned int * __restrict__ data1,
		unsigned int * __restrict__ data2, size_t (*array_size)[3],
		size_t (*arr_start)[3], size_t (*arr_end)[3],
		unsigned int * reduction_var) {
	unsigned int sum = 0;
	int ni, nj, nk;

	ni = (*array_size)[0];
	nj = (*array_size)[1];
	nk = (*array_size)[2];

#ifdef COLLAPSE
	int n;
	n = ((* arr_end)[0] - (* arr_start)[0] + 1) *
	((* arr_end)[1] - (* arr_start)[1] + 1) *
	((* arr_end)[2] - (* arr_start)[2] + 1);

	if( n == ni*nj*nk )	// original 3d grid
	{
		int i;
#ifdef USE_AVX
#pragma omp parallel shared(n, data1, data2)  private(i) reduction(+: sum)
		{
#pragma omp parallel for
			for (i = 0; i < n; i++) {
				sum += data1[i] * data2[i];
			}
		}
#else
#pragma omp parallel for shared(n, data1, data2)  private(i) reduction(+: sum)
		for (i = 0; i < n; i++) {
			sum += data1[i] * data2[i];
		}
#endif
	}
	else	// 3d sub-grid
#endif
	{
#pragma omp parallel shared(ni, nj, nk, data1, data2, sum)
		{
			int i, j, k;
			unsigned int psum = 0;
#ifdef COLLAPSE
#pragma omp for collapse(2)
#else
#pragma omp for
#endif
			for (k = (*arr_start)[2]; k <= (*arr_end)[2]; k++) {
				for (j = (*arr_start)[1]; j <= (*arr_end)[1]; j++) {
					for (i = (*arr_start)[0]; i <= (*arr_end)[0]; i++) {
						int x;
						x = i + j * ni + k * ni * nj;
						psum += data1[x] * data2[x];
					}
				}
			}

#pragma omp critical
			{
				sum += psum;

			}
		}
	} //endif if( n == ni*nj*nk )

	*reduction_var += sum;
}

//reduce
static void omp_reduce_kernel_db(double * __restrict__ data,
		size_t (*array_size)[3], size_t (*arr_start)[3], size_t (*arr_end)[3],
		double * reduction_var) {
	int ni, nj, nk;
	double sum = 0;

	ni = (*array_size)[0];
	nj = (*array_size)[1];
	nk = (*array_size)[2];

#ifdef COLLAPSE
	int n;
	n = ((* arr_end)[0] - (* arr_start)[0] + 1) *
	((* arr_end)[1] - (* arr_start)[1] + 1) *
	((* arr_end)[2] - (* arr_start)[2] + 1);

	if( n == ni*nj*nk )	// original 3d grid
	{
		int i;
#ifdef USE_AVX
#pragma omp parallel shared(n, data)  private(i) reduction(+: sum)
		{
#pragma omp for
			for (i = 0; i < n; i++) {
				sum += data[i];
			}
		}
#else
#pragma omp parallel for shared(n, data)  private(i) reduction(+: sum)
		for (i = 0; i < n; i++) {
			sum += data[i];
		}
#endif
	}
	else	// 3d sub-grid
#endif
	{
#pragma omp parallel shared(ni, nj, nk, data, sum)
		{
			int i, j, k;
			double psum = 0;
#ifdef COLLAPSE
#pragma omp for collapse(2)
#else
#pragma omp for
#endif
			for (k = (*arr_start)[2]; k <= (*arr_end)[2]; k++) {
				for (j = (*arr_start)[1]; j <= (*arr_end)[1]; j++) {
					for (i = (*arr_start)[0]; i <= (*arr_end)[0]; i++) {
						int x;
						x = i + j * ni + k * ni * nj;
						psum += data[x];
					}
				}
			}

#pragma omp critical
			{
				sum += psum;

			}
		}
	} //endif if( n == ni*nj*nk )

	*reduction_var += sum;
}

static void omp_reduce_kernel_fl(float *__restrict__ data,
		size_t (*array_size)[3], size_t (*arr_start)[3], size_t (*arr_end)[3],
		float * reduction_var) {
	int ni, nj, nk;
	float sum = 0;

	ni = (*array_size)[0];
	nj = (*array_size)[1];
	nk = (*array_size)[2];

#ifdef COLLAPSE
	int n;
	n = ((* arr_end)[0] - (* arr_start)[0] + 1) *
	((* arr_end)[1] - (* arr_start)[1] + 1) *
	((* arr_end)[2] - (* arr_start)[2] + 1);

	if( n == ni*nj*nk )	// original 3d grid
	{
		int i;
#ifdef USE_AVX
#pragma omp parallel shared(n, data)  private(i) reduction(+: sum)
		{
#pragma omp for
			for (i = 0; i < n; i++) {
				sum += data[i];
			}
		}
#else
#pragma omp parallel for shared(n, data)  private(i) reduction(+: sum)
		for (i = 0; i < n; i++) {
			sum += data[i];
		}
#endif
	}
	else	// 3d sub-grid
#endif
	{
#pragma omp parallel shared(ni, nj, nk, data, sum)
		{
			int i, j, k;
			float psum = 0;
#ifdef COLLAPSE
#pragma omp for collapse(2)
#else
#pragma omp for
#endif
			for (k = (*arr_start)[2]; k <= (*arr_end)[2]; k++) {
				for (j = (*arr_start)[1]; j <= (*arr_end)[1]; j++) {
					for (i = (*arr_start)[0]; i <= (*arr_end)[0]; i++) {
						int x;
						x = i + j * ni + k * ni * nj;
						psum += data[x];
					}
				}
			}

#pragma omp critical
			{
				sum += psum;

			}
		}
	} //endif if( n == ni*nj*nk )

	*reduction_var += sum;
}

static void omp_reduce_kernel_ul(unsigned long * __restrict__ data,
		size_t (*array_size)[3], size_t (*arr_start)[3], size_t (*arr_end)[3],
		unsigned long * reduction_var) {
	int ni, nj, nk;
	unsigned long sum = 0;

	ni = (*array_size)[0];
	nj = (*array_size)[1];
	nk = (*array_size)[2];

#ifdef COLLAPSE
	int n;
	n = ((* arr_end)[0] - (* arr_start)[0] + 1) *
	((* arr_end)[1] - (* arr_start)[1] + 1) *
	((* arr_end)[2] - (* arr_start)[2] + 1);

	if( n == ni*nj*nk )	// original 3d grid
	{
		int i;
#ifdef USE_AVX
#pragma omp parallel shared(n, data)  private(i) reduction(+: sum)
		{
#pragma omp for
			for (i = 0; i < n; i++) {
				sum += data[i];
			}
		}
#else
#pragma omp parallel for shared(n, data)  private(i) reduction(+: sum)
		for (i = 0; i < n; i++) {
			sum += data[i];
		}
#endif
	}
	else	// 3d sub-grid
#endif
	{
#pragma omp parallel shared(ni, nj, nk, data, sum)
		{
			int i, j, k;
			unsigned long psum = 0;
#ifdef COLLAPSE
#pragma omp for collapse(2)
#else
#pragma omp for
#endif
			for (k = (*arr_start)[2]; k <= (*arr_end)[2]; k++) {
				for (j = (*arr_start)[1]; j <= (*arr_end)[1]; j++) {
					for (i = (*arr_start)[0]; i <= (*arr_end)[0]; i++) {
						int x;
						x = i + j * ni + k * ni * nj;
						psum += data[x];
					}
				}
			}

#pragma omp critical
			{
				sum += psum;

			}
		}
	} //endif if( n == ni*nj*nk )

	*reduction_var += sum;
}

static void omp_reduce_kernel_in(int * __restrict__ data,
		size_t (*array_size)[3], size_t (*arr_start)[3], size_t (*arr_end)[3],
		int * reduction_var) {
	int ni, nj, nk;
	int sum = 0;

	ni = (*array_size)[0];
	nj = (*array_size)[1];
	nk = (*array_size)[2];

#ifdef COLLAPSE
	int n;
	n = ((* arr_end)[0] - (* arr_start)[0] + 1) *
	((* arr_end)[1] - (* arr_start)[1] + 1) *
	((* arr_end)[2] - (* arr_start)[2] + 1);

	if( n == ni*nj*nk )	// original 3d grid
	{
		int i;
#ifdef USE_AVX
#pragma omp parallel shared(n, data)  private(i) reduction(+: sum)
		{
#pragma omp for
			for (i = 0; i < n; i++) {
				sum += data[i];
			}
		}
#else
#pragma omp parallel for shared(n, data)  private(i) reduction(+: sum)
		for (i = 0; i < n; i++) {
			sum += data[i];
		}
#endif
	}
	else	// 3d sub-grid
#endif
	{
#pragma omp parallel shared(ni, nj, nk, data, sum)
		{
			int i, j, k;
			int psum = 0;
#ifdef COLLAPSE
#pragma omp for collapse(2)
#else
#pragma omp for
#endif
			for (k = (*arr_start)[2]; k <= (*arr_end)[2]; k++) {
				for (j = (*arr_start)[1]; j <= (*arr_end)[1]; j++) {
					for (i = (*arr_start)[0]; i <= (*arr_end)[0]; i++) {
						int x;
						x = i + j * ni + k * ni * nj;
						psum += data[x];
					}
				}
			}

#pragma omp critical
			{
				sum += psum;

			}
		}
	} //endif if( n == ni*nj*nk )

	*reduction_var += sum;
}

static void omp_reduce_kernel_ui(unsigned int * __restrict__ data,
		size_t (*array_size)[3], size_t (*arr_start)[3], size_t (*arr_end)[3],
		unsigned int * reduction_var) {
	int ni, nj, nk;
	unsigned int sum = 0;

	ni = (*array_size)[0];
	nj = (*array_size)[1];
	nk = (*array_size)[2];

#ifdef COLLAPSE
	int n;
	n = ((* arr_end)[0] - (* arr_start)[0] + 1) *
	((* arr_end)[1] - (* arr_start)[1] + 1) *
	((* arr_end)[2] - (* arr_start)[2] + 1);

	if( n == ni*nj*nk )	// original 3d grid
	{
		int i;
#ifdef USE_AVX
#pragma omp parallel shared(n, data)  private(i) reduction(+: sum)
		{
#pragma omp for
			for (i = 0; i < n; i++) {
				sum += data[i];
			}
		}
#else
#pragma omp parallel for shared(n, data)  private(i) reduction(+: sum)
		for (i = 0; i < n; i++) {
			sum += data[i];
		}
#endif
	}
	else	// 3d sub-grid
#endif
	{
#pragma omp parallel shared(ni, nj, nk, data, sum)
		{
			int i, j, k;
			unsigned int psum = 0;
#ifdef COLLAPSE
#pragma omp for collapse(2)
#else
#pragma omp for
#endif
			for (k = (*arr_start)[2]; k <= (*arr_end)[2]; k++) {
				for (j = (*arr_start)[1]; j <= (*arr_end)[1]; j++) {
					for (i = (*arr_start)[0]; i <= (*arr_end)[0]; i++) {
						int x;
						x = i + j * ni + k * ni * nj;
						psum += data[x];
					}
				}
			}

#pragma omp critical
			{
				sum += psum;

			}
		}
	} //endif if( n == ni*nj*nk )

	*reduction_var += sum;
}

static void omp_transpose_face_kernel_db(double * __restrict__ indata,
		double * __restrict__ outdata, size_t (*arr_dim_xy)[3],
		size_t (*tran_dim_xy)[3]) {
	int ni, nj;
	unsigned int sum = 0;

	ni = (*arr_dim_xy)[0];
	nj = (*arr_dim_xy)[1];

#pragma omp parallel shared(ni, nj, indata, outdata)
	{
		int i, j, ii, jj;

		//#pragma omp for schedule(static, 1)
#pragma omp for schedule(dynamic, 1) nowait
		for (j = 0; j < nj; j += BLOCK)
		{
			for (i = 0; i < ni; i += BLOCK)
			{
				int iimax = (ni < i + BLOCK ? ni : i + BLOCK);
				int jjmax = (nj < j + BLOCK ? nj : j + BLOCK);

				for (jj = j; jj < jjmax; jj++) {
					for (ii = i; ii < iimax; ii++) {
						outdata[jj + ii * nj] = indata[ii + jj * ni];
					}
				}
			}
		}
	}

}

static void omp_transpose_face_kernel_fl(float * __restrict__ indata,
		float * __restrict__ outdata, size_t (*arr_dim_xy)[3],
		size_t (*tran_dim_xy)[3]) {
	int ni, nj;
	unsigned int sum = 0;

	ni = (*arr_dim_xy)[0];
	nj = (*arr_dim_xy)[1];

#pragma omp parallel shared(ni, nj, indata, outdata)
	{
		int i, j, ii, jj;

		//#pragma omp for schedule(static, 1)
#pragma omp for schedule(dynamic, 1) nowait
		for (j = 0; j < nj; j += BLOCK)
		{
			for (i = 0; i < ni; i += BLOCK)
			{
				int iimax = (ni < i + BLOCK ? ni : i + BLOCK);
				int jjmax = (nj < j + BLOCK ? nj : j + BLOCK);

				for (jj = j; jj < jjmax; jj++) {
					for (ii = i; ii < iimax; ii++) {
						outdata[jj + ii * nj] = indata[ii + jj * ni];
					}
				}
			}
		}
	}
}

static void omp_transpose_face_kernel_ul(unsigned long * __restrict__ indata,
		unsigned long * __restrict__ outdata, size_t (*arr_dim_xy)[3],
		size_t (*tran_dim_xy)[3]) {

	int ni, nj;
	unsigned int sum = 0;

	ni = (*arr_dim_xy)[0];
	nj = (*arr_dim_xy)[1];

#pragma omp parallel shared(ni, nj, indata, outdata)
	{
		int i, j, ii, jj;

		//#pragma omp for schedule(static, 1)
#pragma omp for schedule(dynamic, 1) nowait
		for (j = 0; j < nj; j += BLOCK)
		{
			for (i = 0; i < ni; i += BLOCK)
			{
				int iimax = (ni < i + BLOCK ? ni : i + BLOCK);
				int jjmax = (nj < j + BLOCK ? nj : j + BLOCK);

				for (jj = j; jj < jjmax; jj++) {
					for (ii = i; ii < iimax; ii++) {
						outdata[jj + ii * nj] = indata[ii + jj * ni];
					}
				}
			}
		}
	}
}

static void omp_transpose_face_kernel_in(int * __restrict__ indata,
		int * __restrict__ outdata, size_t (*arr_dim_xy)[3],
		size_t (*tran_dim_xy)[3]) {
	int ni, nj;
	unsigned int sum = 0;

	ni = (*arr_dim_xy)[0];
	nj = (*arr_dim_xy)[1];

#pragma omp parallel shared(ni, nj, indata, outdata)
	{
		int i, j, ii, jj;

		//#pragma omp for schedule(static, 1)
#pragma omp for schedule(dynamic, 1) nowait
		for (j = 0; j < nj; j += BLOCK)
		{
			for (i = 0; i < ni; i += BLOCK)
			{
				int iimax = (ni < i + BLOCK ? ni : i + BLOCK);
				int jjmax = (nj < j + BLOCK ? nj : j + BLOCK);

				for (jj = j; jj < jjmax; jj++) {
					for (ii = i; ii < iimax; ii++) {
						outdata[jj + ii * nj] = indata[ii + jj * ni];
					}
				}
			}
		}
	}

}

static void omp_transpose_face_kernel_ui(unsigned int * __restrict__ indata,
		unsigned int * __restrict__ outdata, size_t (*arr_dim_xy)[3],
		size_t (*tran_dim_xy)[3]) {
	int ni, nj;
	unsigned int sum = 0;

	ni = (*arr_dim_xy)[0];
	nj = (*arr_dim_xy)[1];

#pragma omp parallel shared(ni, nj, indata, outdata)
	{
		int i, j, ii, jj;

		//#pragma omp for schedule(static, 1)
#pragma omp for schedule(dynamic, 1) nowait
		for (j = 0; j < nj; j += BLOCK)
		{
			for (i = 0; i < ni; i += BLOCK)
			{
				int iimax = (ni < i + BLOCK ? ni : i + BLOCK);
				int jjmax = (nj < j + BLOCK ? nj : j + BLOCK);

				for (jj = j; jj < jjmax; jj++) {
					for (ii = i; ii < iimax; ii++) {
						outdata[jj + ii * nj] = indata[ii + jj * ni];
					}
				}
			}
		}
	}
}

static int get_pack_index(int idx, meta_face *face,
		int * __restrict__ remain_dim) {
	int pos;
	int i, j, k, l;
	int a[METAMORPH_FACE_MAX_DEPTH];

	for (i = 0; i < face->count; i++)
		a[i] = 0;

	for (i = 0; i < face->count; i++) {
		k = 0;
		for (j = 0; j < i; j++) {
			k += a[j] * remain_dim[j];
		}
		l = remain_dim[i];
		for (j = 0; j < face->size[i]; j++) {
			if (idx - k < l)
				break;
			else
				l += remain_dim[i];
		}
		a[i] = j;
	}
	pos = face->start;
	for (i = 0; i < face->count; i++) {
		pos += a[i] * face->stride[i];
	}
	return pos;
}

void omp_pack_face_kernel_db(double * __restrict__ packed_buf,
		double * __restrict__ buf, meta_face *face, int *remain_dim) {
	int size = face->size[0] * face->size[1] * face->size[2];

#pragma omp parallel shared(size, packed_buf, buf, face, remain_dim)
	{
		int idx;
#pragma omp for schedule(dynamic,16) nowait
		for (idx = 0; idx < size; idx++)
			packed_buf[idx] = buf[get_pack_index(idx, face, remain_dim)];
	}
}

void omp_pack_face_kernel_fl(float * __restrict__ packed_buf,
		float * __restrict__ buf, meta_face *face, int *remain_dim) {
	int size = face->size[0] * face->size[1] * face->size[2];

#pragma omp parallel shared(size, packed_buf, buf, face, remain_dim)
	{
		int idx;
#pragma omp for schedule(dynamic,16) nowait
		for (idx = 0; idx < size; idx++)
			packed_buf[idx] = buf[get_pack_index(idx, face, remain_dim)];
	}
}

void omp_pack_face_kernel_ul(unsigned long * __restrict__ packed_buf,
		unsigned long * __restrict__ buf, meta_face *face,
		int *remain_dim) {
	int size = face->size[0] * face->size[1] * face->size[2];

#pragma omp parallel shared(size, packed_buf, buf, face, remain_dim)
	{
		int idx;
#pragma omp for schedule(dynamic,16) nowait
		for (idx = 0; idx < size; idx++)
			packed_buf[idx] = buf[get_pack_index(idx, face, remain_dim)];
	}
}

void omp_pack_face_kernel_in(int * __restrict__ packed_buf,
		int * __restrict__ buf, meta_face *face, int *remain_dim) {
	int size = face->size[0] * face->size[1] * face->size[2];

#pragma omp parallel shared(size, packed_buf, buf, face, remain_dim)
	{
		int idx;
#pragma omp for schedule(dynamic,16) nowait
		for (idx = 0; idx < size; idx++)
			packed_buf[idx] = buf[get_pack_index(idx, face, remain_dim)];
	}
}

void omp_pack_face_kernel_ui(unsigned int * __restrict__ packed_buf,
		unsigned int * __restrict__ buf, meta_face *face,
		int *remain_dim) {
	int size = face->size[0] * face->size[1] * face->size[2];

#pragma omp parallel shared(size, packed_buf, buf, face, remain_dim)
	{
		int idx;
#pragma omp for schedule(dynamic,16) nowait
		for (idx = 0; idx < size; idx++)
			packed_buf[idx] = buf[get_pack_index(idx, face, remain_dim)];
	}
}

void omp_unpack_face_kernel_db(double * __restrict__ packed_buf,
		double * __restrict__ buf, meta_face *face, int *remain_dim) {
	int size = face->size[0] * face->size[1] * face->size[2];

#pragma omp parallel shared(size, packed_buf, buf, face, remain_dim)
	{
		int idx;
#pragma omp for schedule(dynamic,16) nowait
		for (idx = 0; idx < size; idx++)
			buf[get_pack_index(idx, face, remain_dim)] = packed_buf[idx];
	}
}

void omp_unpack_face_kernel_fl(float * __restrict__ packed_buf,
		float * __restrict__ buf, meta_face *face, int *remain_dim) {
	int size = face->size[0] * face->size[1] * face->size[2];

#pragma omp parallel shared(size, packed_buf, buf, face, remain_dim)
	{
		int idx;
#pragma omp for schedule(dynamic,16) nowait
		for (idx = 0; idx < size; idx++)
			buf[get_pack_index(idx, face, remain_dim)] = packed_buf[idx];
	}
}

void omp_unpack_face_kernel_ul(unsigned long * __restrict__ packed_buf,
		unsigned long * __restrict__ buf, meta_face *face,
		int *remain_dim) {
	int size = face->size[0] * face->size[1] * face->size[2];

#pragma omp parallel shared(size, packed_buf, buf, face, remain_dim)
	{
		int idx;
#pragma omp for schedule(dynamic,16) nowait
		for (idx = 0; idx < size; idx++)
			buf[get_pack_index(idx, face, remain_dim)] = packed_buf[idx];
	}
}

void omp_unpack_face_kernel_in(int * __restrict__ packed_buf,
		int * __restrict__ buf, meta_face *face, int *remain_dim) {
	int size = face->size[0] * face->size[1] * face->size[2];

#pragma omp parallel shared(size, packed_buf, buf, face, remain_dim)
	{
		int idx;
#pragma omp for schedule(dynamic,16) nowait
		for (idx = 0; idx < size; idx++)
			buf[get_pack_index(idx, face, remain_dim)] = packed_buf[idx];
	}
}

void omp_unpack_face_kernel_ui(unsigned int * __restrict__ packed_buf,
		unsigned int * __restrict__ buf, meta_face *face,
		int *remain_dim) {
	int size = face->size[0] * face->size[1] * face->size[2];

#pragma omp parallel shared(size, packed_buf, buf, face, remain_dim)
	{
		int idx;
#pragma omp for schedule(dynamic,16) nowait
		for (idx = 0; idx < size; idx++)
			buf[get_pack_index(idx, face, remain_dim)] = packed_buf[idx];
	}
}

#if 1
void omp_stencil_3d7p_kernel_db(double * __restrict__ indata,
		double * __restrict__ outdata, size_t (*array_size)[3],
		size_t (*arr_start)[3], size_t (*arr_end)[3]) {
	int ni, nj, nk;

	ni = (*array_size)[0];
	nj = (*array_size)[1];
	nk = (*array_size)[2];

#pragma omp parallel shared(ni, nj, nk, indata, outdata)
	{
		int i, j, k;
		//double *in, *out;
#pragma omp for collapse(2) schedule(static, 1) nowait
		//#pragma omp for
		for (k = (*arr_start)[2] + 1; k < (*arr_end)[2]; k++) {
			for (j = (*arr_start)[1] + 1; j < (*arr_end)[1]; j++) {
				//in = &indata[j*ni+k*ni*nj];
				//out = &outdata[j*ni+k*ni*nj];
				//#pragma unroll (8)
				//#pragma prefetch indata:_MM_HINT_T2:8,outdata:_MM_HINT_NTA
				//#pragma loop count (256)
#pragma ivdep
#pragma vector nontemporal (outdata)
				for (i = (*arr_start)[0] + 1; i < (*arr_end)[0]; i++) {
					outdata[i + j * ni + k * ni * nj] = (indata[i + j * ni
							+ (k - 1) * ni * nj]
							+ indata[(i - 1) + j * ni + k * ni * nj]
							+ indata[i + (j - 1) * ni + k * ni * nj]
							+ indata[i + j * ni + k * ni * nj]
							+ indata[i + (j + 1) * ni + k * ni * nj]
							+ indata[(i + 1) + j * ni + k * ni * nj]
							+ indata[i + j * ni + (k + 1) * ni * nj])
							/ (double) 7.0;
					//out[i] = ( in[i-ni*nj] + in[i-1] + in[i-ni] + in[i] +
					//			in[i+ni] + in[i+1] + in[i+ni*nj] )
					//			/ (double) 7.0;
				}
			}
		}
	}
}
#elif 0
void omp_stencil_3d7p_kernel_db(double * indata, double * outdata, size_t (* array_size)[3], size_t (* arr_start)[3], size_t (* arr_end)[3])
{
	int ni, nj, nk;

	ni = (* array_size)[0];
	nj = (* array_size)[1];
	nk = (* array_size)[2];

#pragma omp parallel shared(ni, nj, nk, indata, outdata)
	{
		int i, j, k, ii, jj, kk;
#ifdef COLLAPSE
#pragma omp for collapse(2)
#else
#pragma omp for
		//#pragma omp for schedule(static, 1)
		//#pragma omp for schedule(dynamic, 1) nowait
#endif
		for (k = (* arr_start)[2]+1; k < (* arr_end)[2]; k+= CZ) {
			for (j = (* arr_start)[1]+1; j < (* arr_end)[1]; j+= CY) {
				for (i = (* arr_start)[0]+1; i < (* arr_end)[0]; i+=CX) {
					int kkmax = ((* arr_end)[2] < k+CZ ? (* arr_end)[2] : k+CZ);
					int jjmax = ((* arr_end)[1] < j+CY ? (* arr_end)[1] : j+CY);
					int iimax = ((* arr_end)[0] < i+CX ? (* arr_end)[0] : i+CX);
					for (kk = k; kk < kkmax; kk++) {
						for (jj = j; jj < jjmax; jj++) {
							for (ii = i; ii < iimax; ii++) {
								outdata[ii+jj*ni+kk*ni*nj] = ( indata[ii+jj*ni+ (kk-1)*ni*nj] + indata[(ii-1)+jj*ni+kk*ni*nj] + indata[ii+(jj-1)*ni+kk*ni*nj] +
										indata[ii+jj*ni+kk*ni*nj] + indata[ii+(jj+1)*ni+kk*ni*nj] + indata[(ii+1)+jj*ni+kk*ni*nj] +
										indata[ii+jj*ni+(kk+1)*ni*nj] ) / (double) 7.0;
							}
						}
					}
				}
			}
		}
	}
}
#elif 0
void omp_stencil_3d7p_kernel_db(double * indata, double * outdata, size_t (* array_size)[3], size_t (* arr_start)[3], size_t (* arr_end)[3])
{
	int ni, nj, nk;

	ni = (* array_size)[0];
	nj = (* array_size)[1];
	nk = (* array_size)[2];

#pragma omp parallel shared(ni, nj, nk, indata, outdata)
	{
		int i, j, k, jj, kk;
#ifdef COLLAPSE
#pragma omp for collapse(2)
#else
		//#pragma omp for
		//#pragma omp for schedule(static, 1)
#pragma omp for schedule(dynamic, 1) nowait
#endif
		for (k = (* arr_start)[2]+1; k < (* arr_end)[2]; k+= CZ) {
			for (j = (* arr_start)[1]+1; j < (* arr_end)[1]; j+= CY) {
				int kkmax = ((* arr_end)[2] < k+CZ ? (* arr_end)[2] : k+CZ);
				int jjmax = ((* arr_end)[1] < j+CY ? (* arr_end)[1] : j+CY);
				for (kk = k; kk < kkmax; kk++) {
					for (jj = j; jj < jjmax; jj++) {
#pragma ivdep
#pragma vector nontemporal (outdata)
						for (i = (* arr_start)[0]+1; i < (* arr_end)[0]; i++) {
							outdata[i+jj*ni+kk*ni*nj] = ( indata[i+jj*ni+ (kk-1)*ni*nj] + indata[(i-1)+jj*ni+kk*ni*nj] + indata[i+(jj-1)*ni+kk*ni*nj] +
									indata[i+jj*ni+kk*ni*nj] + indata[i+(jj+1)*ni+kk*ni*nj] + indata[(i+1)+jj*ni+kk*ni*nj] +
									indata[i+jj*ni+(kk+1)*ni*nj] ) / (double) 7.0;
						}
					}
				}
			}
		}
	}
}
#elif 0
void omp_stencil_3d7p_kernel_db(double * indata, double * outdata, size_t (* array_size)[3], size_t (* arr_start)[3], size_t (* arr_end)[3])
{
	int ni, nj, nk;

	ni = (* array_size)[0];
	nj = (* array_size)[1];
	nk = (* array_size)[2];

#pragma omp parallel shared(ni, nj, nk, indata, outdata)
	{
		int i, j, k, ii, jj, kk;
#ifdef COLLAPSE
#pragma omp for collapse(2)
#else
#pragma omp for
		//#pragma omp for schedule(static, 1)
		//#pragma omp for schedule(dynamic, 1)
#endif
		for (k = (* arr_start)[2]+1; k < (* arr_end)[2]; k++) {
			for (j = (* arr_start)[1]+1; j < (* arr_end)[1]; j+= CY) {
				for (i = (* arr_start)[0]+1; i < (* arr_end)[0]; i+=CX) {
					int jjmax = ((* arr_end)[1] < j+CY ? (* arr_end)[1] : j+CY);
					int iimax = ((* arr_end)[0] < i+CX ? (* arr_end)[0] : i+CX);
					for (jj = j; jj < jjmax; jj++) {
						for (ii = i; ii < iimax; ii++) {
							outdata[ii+jj*ni+k*ni*nj] = ( indata[ii+jj*ni+ (k-1)*ni*nj] + indata[(ii-1)+jj*ni+kk*ni*nj] + indata[ii+(jj-1)*ni+k*ni*nj] +
									indata[ii+jj*ni+k*ni*nj] + indata[ii+(jj+1)*ni+k*ni*nj] + indata[(ii+1)+jj*ni+k*ni*nj] +
									indata[ii+jj*ni+(k+1)*ni*nj] ) / (double) 7.0;
						}
					}
				}
			}
		}
	}
}
#else
void omp_stencil_3d7p_kernel_db(double * __restrict__ indata, double * __restrict__ outdata, size_t (* array_size)[3], size_t (* arr_start)[3], size_t (* arr_end)[3])
{
	int ni, nj, nk;

	ni = (* array_size)[0];
	nj = (* array_size)[1];
	nk = (* array_size)[2];

#pragma omp parallel shared(ni, nj, nk, indata, outdata)
	{
		int i, j, k;
		int c;
		double r0, rx1, rx2;

#ifdef COLLAPSE
#pragma omp for collapse(2)
#else
#pragma omp for collapse(2) schedule(static, 1) nowait
		//#pragma omp for
#endif
		for (k = (* arr_start)[2]+1; k < (* arr_end)[2]; k++) {
			for (j = (* arr_start)[1]+1; j < (* arr_end)[1]; j++) {
				//#pragma Loop_Optimize Ivdep No_Unroll
				//#pragma unroll(8)
				i = (* arr_start)[0]+1;
				c = i+j*ni+k*ni*nj;
				r0 = indata[c];
				rx1 = indata[c-1];
				rx2 = indata[c+1];
				for (; i < (* arr_end)[0]; i++) {
					outdata[i+j*ni+k*ni*nj] = ( indata[i+j*ni+ (k-1)*ni*nj] + rx1 + indata[i+(j-1)*ni+k*ni*nj] +
							r0 + indata[i+(j+1)*ni+k*ni*nj] + indata[(i+1)+j*ni+k*ni*nj] +
							rx2 ) / (double) 7.0;
					c++;
					r0 = rx2;
					rx1 = r0;
					rx2 = indata[c+1];
				}
			}
		}
	}
}
#endif

void omp_stencil_3d7p_kernel_fl(float * indata, float * outdata,
		size_t (*array_size)[3], size_t (*arr_start)[3], size_t (*arr_end)[3]) {
	int ni, nj, nk;

	ni = (*array_size)[0];
	nj = (*array_size)[1];
	nk = (*array_size)[2];

#pragma omp parallel shared(ni, nj, nk, indata, outdata)
	{
		int i, j, k;
#pragma omp for collapse(2) schedule(static, 1) nowait
		//#pragma omp for
		for (k = (*arr_start)[2] + 1; k < (*arr_end)[2]; k++) {
			for (j = (*arr_start)[1] + 1; j < (*arr_end)[1]; j++) {
				//#pragma unroll (8)
				//#pragma prefetch indata:_MM_HINT_T2:8,outdata:_MM_HINT_NTA
				//#pragma loop count (256)
#pragma ivdep
#pragma vector nontemporal (outdata)
				for (i = (*arr_start)[0] + 1; i < (*arr_end)[0]; i++) {
					outdata[i + j * ni + k * ni * nj] = (indata[i + j * ni
							+ (k - 1) * ni * nj]
							+ indata[(i - 1) + j * ni + k * ni * nj]
							+ indata[i + (j - 1) * ni + k * ni * nj]
							+ indata[i + j * ni + k * ni * nj]
							+ indata[i + (j + 1) * ni + k * ni * nj]
							+ indata[(i + 1) + j * ni + k * ni * nj]
							+ indata[i + j * ni + (k + 1) * ni * nj])
							/ (float) 7.0;
				}
			}
		}
	}
}

void omp_stencil_3d7p_kernel_ul(unsigned long * indata, unsigned long * outdata,
		size_t (*array_size)[3], size_t (*arr_start)[3], size_t (*arr_end)[3]) {
	int ni, nj, nk;

	ni = (*array_size)[0];
	nj = (*array_size)[1];
	nk = (*array_size)[2];

#pragma omp parallel shared(ni, nj, nk, indata, outdata)
	{
		int i, j, k;
#pragma omp for collapse(2) schedule(static, 1) nowait
		//#pragma omp for
		for (k = (*arr_start)[2] + 1; k < (*arr_end)[2]; k++) {
			for (j = (*arr_start)[1] + 1; j < (*arr_end)[1]; j++) {
				//#pragma unroll (8)
				//#pragma prefetch indata:_MM_HINT_T2:8,outdata:_MM_HINT_NTA
				//#pragma loop count (256)
#pragma ivdep
#pragma vector nontemporal (outdata)
				for (i = (*arr_start)[0] + 1; i < (*arr_end)[0]; i++) {
					outdata[i + j * ni + k * ni * nj] = (indata[i + j * ni
							+ (k - 1) * ni * nj]
							+ indata[(i - 1) + j * ni + k * ni * nj]
							+ indata[i + (j - 1) * ni + k * ni * nj]
							+ indata[i + j * ni + k * ni * nj]
							+ indata[i + (j + 1) * ni + k * ni * nj]
							+ indata[(i + 1) + j * ni + k * ni * nj]
							+ indata[i + j * ni + (k + 1) * ni * nj])
							/ (unsigned long) 7;
				}
			}
		}
	}
}

void omp_stencil_3d7p_kernel_in(int * indata, int * outdata,
		size_t (*array_size)[3], size_t (*arr_start)[3], size_t (*arr_end)[3]) {
	int ni, nj, nk;

	ni = (*array_size)[0];
	nj = (*array_size)[1];
	nk = (*array_size)[2];

#pragma omp parallel shared(ni, nj, nk, indata, outdata)
	{
		int i, j, k;
#pragma omp for collapse(2) schedule(static, 1) nowait
		//#pragma omp for
		for (k = (*arr_start)[2] + 1; k < (*arr_end)[2]; k++) {
			for (j = (*arr_start)[1] + 1; j < (*arr_end)[1]; j++) {
				//#pragma unroll (8)
				//#pragma prefetch indata:_MM_HINT_T2:8,outdata:_MM_HINT_NTA
				//#pragma loop count (256)
#pragma ivdep
#pragma vector nontemporal (outdata)
				for (i = (*arr_start)[0] + 1; i < (*arr_end)[0]; i++) {
					outdata[i + j * ni + k * ni * nj] = (indata[i + j * ni
							+ (k - 1) * ni * nj]
							+ indata[(i - 1) + j * ni + k * ni * nj]
							+ indata[i + (j - 1) * ni + k * ni * nj]
							+ indata[i + j * ni + k * ni * nj]
							+ indata[i + (j + 1) * ni + k * ni * nj]
							+ indata[(i + 1) + j * ni + k * ni * nj]
							+ indata[i + j * ni + (k + 1) * ni * nj]) / (int) 7;
				}
			}
		}
	}
}

void omp_stencil_3d7p_kernel_ui(unsigned int * indata, unsigned int * outdata,
		size_t (*array_size)[3], size_t (*arr_start)[3], size_t (*arr_end)[3]) {
	int ni, nj, nk;

	ni = (*array_size)[0];
	nj = (*array_size)[1];
	nk = (*array_size)[2];

#pragma omp parallel shared(ni, nj, nk, indata, outdata)
	{
		int i, j, k;
#pragma omp for collapse(2) schedule(static, 1) nowait
		//#pragma omp for
		for (k = (*arr_start)[2] + 1; k < (*arr_end)[2]; k++) {
			for (j = (*arr_start)[1] + 1; j < (*arr_end)[1]; j++) {
				//#pragma unroll (8)
				//#pragma prefetch indata:_MM_HINT_T2:8,outdata:_MM_HINT_NTA
				//#pragma loop count (256)
#pragma ivdep
#pragma vector nontemporal (outdata)
				for (i = (*arr_start)[0] + 1; i < (*arr_end)[0]; i++) {
					outdata[i + j * ni + k * ni * nj] = (indata[i + j * ni
							+ (k - 1) * ni * nj]
							+ indata[(i - 1) + j * ni + k * ni * nj]
							+ indata[i + (j - 1) * ni + k * ni * nj]
							+ indata[i + j * ni + k * ni * nj]
							+ indata[i + (j + 1) * ni + k * ni * nj]
							+ indata[(i + 1) + j * ni + k * ni * nj]
							+ indata[i + j * ni + (k + 1) * ni * nj])
							/ (unsigned int) 7;
				}
			}
		}
	}
}

//wrappers
a_err openmp_dotProd(size_t (* grid_size)[3], size_t (* block_size)[3], void * data1, void * data2, size_t (* array_size)[3], size_t (* arr_start)[3], size_t (* arr_end)[3], void * reduction_var, meta_type_id type, int async, meta_callback * call, meta_event * ret_event) {
	int ret = 0; //Success
openmpEvent * events;
  if (ret_event != NULL && ret_event->mode == metaModePreferOpenMP && ret_event->event_pl != NULL) events = ((openmpEvent *)ret_event->event_pl);
  meta_timer * timer = NULL;
  if (profiling_symbols.metaProfilingCreateTimer != NULL) {
    (*(profiling_symbols.metaProfilingCreateTimer))(&timer, metaModePreferOpenMP, (*array_size)[0]*(*array_size)[1]*(*array_size)[2]*get_atype_size(type));
    if (events == NULL) {
      events = ((openmpEvent *)timer->event.event_pl);
    } else {
      //FIXME: are we leaking a created openmpEvent here since the profiling function calls create?
      //metaOpenMPDestroyEvent(frame->event.event_pl);
      timer->event = *ret_event;
    }
  }
  if (events != NULL) {
    gettimeofday(&(events[0]), NULL);
  }

	// ignore grid_size, block_size, async

	switch (type) {
	case a_db:
		omp_dotProd_kernel_db((double*) data1, (double*) data2, array_size,
				arr_start, arr_end, (double*) reduction_var);
		break;

	case a_fl:
		omp_dotProd_kernel_fl((float*) data1, (float*) data2, array_size,
				arr_start, arr_end, (float*) reduction_var);
		break;

	case a_ul:
		omp_dotProd_kernel_ul((unsigned long*) data1, (unsigned long*) data2,
				array_size, arr_start, arr_end, (unsigned long*) reduction_var);
		break;

	case a_in:
		omp_dotProd_kernel_in((int*) data1, (int*) data2, array_size, arr_start,
				arr_end, (int*) reduction_var);
		break;

	case a_ui:
		omp_dotProd_kernel_ui((unsigned int*) data1, (unsigned int*) data2,
				array_size, arr_start, arr_end, (unsigned int*) reduction_var);
		break;

	default:
		fprintf(stderr,
				"Error: Function 'omp_dotProd' not implemented for selected type!\n");
		break;
	}
  if (events != NULL) {
    gettimeofday(&(events[1]), NULL);
  }
  //FIXME: Implement async "callback"
  if (call != NULL) {
    (call->callback_func)(call);
  }
    if (profiling_symbols.metaProfilingEnqueueTimer != NULL) (*(profiling_symbols.metaProfilingEnqueueTimer))(*timer, k_dotProd);
	return (ret);
}

a_err openmp_reduce(size_t (* grid_size)[3], size_t (* block_size)[3], void * data, size_t (* array_size)[3], size_t (* arr_start)[3], size_t (* arr_end)[3], void * reduction_var, meta_type_id type, int async, meta_callback * call, meta_event * ret_event) {
	int ret = 0; //Success
openmpEvent * events;
  if (ret_event != NULL && ret_event->mode == metaModePreferOpenMP && ret_event->event_pl != NULL) events = ((openmpEvent *)ret_event->event_pl);
  meta_timer * timer = NULL;
  if (profiling_symbols.metaProfilingCreateTimer != NULL) {
    (*(profiling_symbols.metaProfilingCreateTimer))(&timer, metaModePreferOpenMP, (*array_size)[0]*(*array_size)[1]*(*array_size)[2]*get_atype_size(type));
    if (events == NULL) {
      events = ((openmpEvent *)timer->event.event_pl);
    } else {
      //FIXME: are we leaking a created openmpEvent here since the profiling function calls create?
      //metaOpenMPDestroyEvent(frame->event.event_pl);
      timer->event = *ret_event;
    }
  }
  if (events != NULL) {
    gettimeofday(&(events[0]), NULL);
  }

	// ignore grid_size, block_size, async

	switch (type) {
	case a_db:
		omp_reduce_kernel_db((double*) data, array_size, arr_start, arr_end,
				(double*) reduction_var);
		break;

	case a_fl:
		omp_reduce_kernel_fl((float*) data, array_size, arr_start, arr_end,
				(float*) reduction_var);
		break;

	case a_ul:
		omp_reduce_kernel_ul((unsigned long*) data, array_size, arr_start,
				arr_end, (unsigned long*) reduction_var);
		break;

	case a_in:
		omp_reduce_kernel_in((int*) data, array_size, arr_start, arr_end,
				(int*) reduction_var);
		break;

	case a_ui:
		omp_reduce_kernel_ui((unsigned int*) data, array_size, arr_start,
				arr_end, (unsigned int*) reduction_var);
		break;

	default:
		fprintf(stderr,
				"Error: Function 'omp_reduce' not implemented for selected type!\n");
		break;
	}

  if (events != NULL) {
    gettimeofday(&(events[1]), NULL);
  }
  //FIXME: Implement async "callback"
  if (call != NULL) {
    (call->callback_func)(call);
  }
    if (profiling_symbols.metaProfilingEnqueueTimer != NULL) (*(profiling_symbols.metaProfilingEnqueueTimer))(*timer, k_reduce);
	return (ret);
}

a_err openmp_transpose_face(size_t (* grid_size)[3], size_t (* block_size)[3], void * indata, void * outdata, size_t (* arr_dim_xy)[3], size_t (* tran_dim_xy)[3], meta_type_id type, int async, meta_callback * call, meta_event * ret_event) {
	int ret = 0; //Success
openmpEvent * events;
  if (ret_event != NULL && ret_event->mode == metaModePreferOpenMP && ret_event->event_pl != NULL) events = ((openmpEvent *)ret_event->event_pl);
  meta_timer * timer = NULL;
  if (profiling_symbols.metaProfilingCreateTimer != NULL) {
    (*(profiling_symbols.metaProfilingCreateTimer))(&timer, metaModePreferOpenMP, (*tran_dim_xy)[0]*(*tran_dim_xy)[1]*get_atype_size(type));
    if (events == NULL) {
      events = ((openmpEvent *)timer->event.event_pl);
    } else {
      //FIXME: are we leaking a created openmpEvent here since the profiling function calls create?
      //metaOpenMPDestroyEvent(frame->event.event_pl);
      timer->event = *ret_event;
    }
  }
  if (events != NULL) {
    gettimeofday(&(events[0]), NULL);
  }

	// ignore grid_size, block_size, async

	switch (type) {
	case a_db:
		omp_transpose_face_kernel_db((double*) indata, (double*) outdata,
				arr_dim_xy, tran_dim_xy);
		break;

	case a_fl:
		omp_transpose_face_kernel_fl((float*) indata, (float*) outdata,
				arr_dim_xy, tran_dim_xy);
		break;

	case a_ul:
		omp_transpose_face_kernel_ul((unsigned long*) indata,
				(unsigned long*) outdata, arr_dim_xy, tran_dim_xy);
		break;

	case a_in:
		omp_transpose_face_kernel_in((int*) indata, (int*) outdata,
				arr_dim_xy, tran_dim_xy);
		break;

	case a_ui:
		omp_transpose_face_kernel_ui((unsigned int*) indata,
				(unsigned int*) outdata, arr_dim_xy, tran_dim_xy);
		break;

	default:
		fprintf(stderr,
				"Error: Function 'omp_transpose_face' not implemented for selected type!\n");
		break;
	}
  if (events != NULL) {
    gettimeofday(&(events[1]), NULL);
  }
  //FIXME: Implement async "callback"
  if (call != NULL) {
    (call->callback_func)(call);
  }
    if (profiling_symbols.metaProfilingEnqueueTimer != NULL) (*(profiling_symbols.metaProfilingEnqueueTimer))(*timer, k_transpose_2d_face);
	return (ret);
}

a_err openmp_pack_face(size_t (* grid_size)[3], size_t (* block_size)[3], void * packed_buf, void * buf, meta_face * face, int * remain_dim, meta_type_id type, int async, meta_callback * call, meta_event * ret_event_k1, meta_event * ret_event_c1, meta_event * ret_event_c2, meta_event * ret_event_c3) {
	int ret = 0; //Success
openmpEvent * events_k1, * events_c1, * events_c2, * events_c3;
  if (ret_event_k1 != NULL && ret_event_k1->mode == metaModePreferOpenMP && ret_event_k1->event_pl != NULL) events_k1 = ((openmpEvent *)ret_event_k1->event_pl);
  if (ret_event_c1 != NULL && ret_event_c1->mode == metaModePreferOpenMP && ret_event_c1->event_pl != NULL) events_c1 = ((openmpEvent *)ret_event_c1->event_pl);
  if (ret_event_c2 != NULL && ret_event_c2->mode == metaModePreferOpenMP && ret_event_c2->event_pl != NULL) events_c2 = ((openmpEvent *)ret_event_c2->event_pl);
  if (ret_event_c3 != NULL && ret_event_c3->mode == metaModePreferOpenMP && ret_event_c3->event_pl != NULL) events_c3 = ((openmpEvent *)ret_event_c3->event_pl);
  meta_timer * timer_k1 = NULL, * timer_c1 = NULL, * timer_c2 = NULL, * timer_c3 = NULL;
  if (profiling_symbols.metaProfilingCreateTimer != NULL) {
    (*(profiling_symbols.metaProfilingCreateTimer))(&timer_k1, metaModePreferOpenMP, get_atype_size(type)*face->size[0]*face->size[1]*face->size[2]);
    (*(profiling_symbols.metaProfilingCreateTimer))(&timer_c1, metaModePreferOpenMP, get_atype_size(type)*3);
    (*(profiling_symbols.metaProfilingCreateTimer))(&timer_c2, metaModePreferOpenMP, get_atype_size(type)*3);
    (*(profiling_symbols.metaProfilingCreateTimer))(&timer_c3, metaModePreferOpenMP, get_atype_size(type)*3);
    if (events_k1 == NULL) {
      events_k1 = ((openmpEvent *)timer_k1->event.event_pl);
    } else {
      //FIXME: are we leaking a created openmpEvent here since the profiling function calls create?
      //metaOpenMPDestroyEvent(frame->event.event_pl);
      timer_k1->event = *ret_event_k1;
    }
    if (events_c1 == NULL) {
      events_c1 = ((openmpEvent *)timer_c1->event.event_pl);
    } else {
      //FIXME: are we leaking a created openmpEvent here since the profiling function calls create?
      //metaOpenMPDestroyEvent(frame->event.event_pl);
      timer_c1->event = *ret_event_c1;
    }
    if (events_c2 == NULL) {
      events_c2 = ((openmpEvent *)timer_c2->event.event_pl);
    } else {
      //FIXME: are we leaking a created openmpEvent here since the profiling function calls create?
      //metaOpenMPDestroyEvent(frame->event.event_pl);
      timer_c2->event = *ret_event_c2;
    }
    if (events_c3 == NULL) {
      events_c3 = ((openmpEvent *)timer_c3->event.event_pl);
    } else {
      //FIXME: are we leaking a created openmpEvent here since the profiling function calls create?
      //metaOpenMPDestroyEvent(frame->event.event_pl);
      timer_c3->event = *ret_event_c3;
    }
  }
  //The constant copies are treated as a zero-time event
  if (events_c1 != NULL) {
    gettimeofday(&(events_c1[0]), NULL);
    events_c1[1] = events_c1[0];
  }
  if (events_c2 != NULL) {
    gettimeofday(&(events_c2[0]), NULL);
    events_c2[1] = events_c2[0];
  }
  if (events_c3 != NULL) {
    gettimeofday(&(events_c3[0]), NULL);
    events_c3[1] = events_c3[0];
  }
  if (events_k1 != NULL) {
    gettimeofday(&(events_k1[0]), NULL);
  }

	// ignore grid_size, BLOCK_size, async

	switch (type) {
	case a_db:
		omp_pack_face_kernel_db((double*) packed_buf, (double*) buf, face,
				remain_dim);
		break;

	case a_fl:
		omp_pack_face_kernel_fl((float*) packed_buf, (float*) buf, face,
				remain_dim);
		break;

	case a_ul:
		omp_pack_face_kernel_ul((unsigned long*) packed_buf,
				(unsigned long*) buf, face, remain_dim);
		break;

	case a_in:
		omp_pack_face_kernel_in((int*) packed_buf, (int*) buf, face,
				remain_dim);
		break;

	case a_ui:
		omp_pack_face_kernel_ui((unsigned int*) packed_buf,
				(unsigned int*) buf, face, remain_dim);
		break;

	default:
		fprintf(stderr,
				"Error: Function 'omp_transpose_face' not implemented for selected type!\n");
		break;
	}
  if (events_k1 != NULL) {
    gettimeofday(&(events_k1[1]), NULL);
  }
  //FIXME: Implement async "callback"
  if (call != NULL) {
    (call->callback_func)(call);
  }
    if (profiling_symbols.metaProfilingEnqueueTimer != NULL) {
  (*(profiling_symbols.metaProfilingEnqueueTimer))(*timer_k1, k_pack_2d_face);
  (*(profiling_symbols.metaProfilingEnqueueTimer))(*timer_c1, c_H2Dc);
  (*(profiling_symbols.metaProfilingEnqueueTimer))(*timer_c2, c_H2Dc);
  (*(profiling_symbols.metaProfilingEnqueueTimer))(*timer_c3, c_H2Dc);
}
  //FIXME: Implement sync point here
	return (ret);
}

a_err openmp_unpack_face(size_t (* grid_zie)[3], size_t (* block_size)[3], void * packed_buf, void * buf, meta_face * face, int * remain_dim, meta_type_id type, int async, meta_callback * call, meta_event * ret_event_k1, meta_event * ret_event_c1, meta_event * ret_event_c2, meta_event * ret_event_c3) {
	int ret = 0; //Success
openmpEvent * events_k1, * events_c1, * events_c2, * events_c3;
  if (ret_event_k1 != NULL && ret_event_k1->mode == metaModePreferOpenMP && ret_event_k1->event_pl != NULL) events_k1 = ((openmpEvent *)ret_event_k1->event_pl);
  if (ret_event_c1 != NULL && ret_event_c1->mode == metaModePreferOpenMP && ret_event_c1->event_pl != NULL) events_c1 = ((openmpEvent *)ret_event_c1->event_pl);
  if (ret_event_c2 != NULL && ret_event_c2->mode == metaModePreferOpenMP && ret_event_c2->event_pl != NULL) events_c2 = ((openmpEvent *)ret_event_c2->event_pl);
  if (ret_event_c3 != NULL && ret_event_c3->mode == metaModePreferOpenMP && ret_event_c3->event_pl != NULL) events_c3 = ((openmpEvent *)ret_event_c3->event_pl);
  meta_timer * timer_k1 = NULL, * timer_c1 = NULL, * timer_c2 = NULL, * timer_c3 = NULL;
  if (profiling_symbols.metaProfilingCreateTimer != NULL) {
    (*(profiling_symbols.metaProfilingCreateTimer))(&timer_k1, metaModePreferOpenMP, get_atype_size(type)*face->size[0]*face->size[1]*face->size[2]);
    (*(profiling_symbols.metaProfilingCreateTimer))(&timer_c1, metaModePreferOpenMP, get_atype_size(type)*3);
    (*(profiling_symbols.metaProfilingCreateTimer))(&timer_c2, metaModePreferOpenMP, get_atype_size(type)*3);
    (*(profiling_symbols.metaProfilingCreateTimer))(&timer_c3, metaModePreferOpenMP, get_atype_size(type)*3);
    if (events_k1 == NULL) {
      events_k1 = ((openmpEvent *)timer_k1->event.event_pl);
    } else {
      //FIXME: are we leaking a created openmpEvent here since the profiling function calls create?
      //metaOpenMPDestroyEvent(frame->event.event_pl);
      timer_k1->event = *ret_event_k1;
    }
    if (events_c1 == NULL) {
      events_c1 = ((openmpEvent *)timer_c1->event.event_pl);
    } else {
      //FIXME: are we leaking a created openmpEvent here since the profiling function calls create?
      //metaOpenMPDestroyEvent(frame->event.event_pl);
      timer_c1->event = *ret_event_c1;
    }
    if (events_c2 == NULL) {
      events_c2 = ((openmpEvent *)timer_c2->event.event_pl);
    } else {
      //FIXME: are we leaking a created openmpEvent here since the profiling function calls create?
      //metaOpenMPDestroyEvent(frame->event.event_pl);
      timer_c2->event = *ret_event_c2;
    }
    if (events_c3 == NULL) {
      events_c3 = ((openmpEvent *)timer_c3->event.event_pl);
    } else {
      //FIXME: are we leaking a created openmpEvent here since the profiling function calls create?
      //metaOpenMPDestroyEvent(frame->event.event_pl);
      timer_c3->event = *ret_event_c3;
    }
  }
  //The constant copies are treated as a zero-time event
  if (events_c1 != NULL) {
    gettimeofday(&(events_c1[0]), NULL);
    events_c1[1] = events_c1[0];
  }
  if (events_c2 != NULL) {
    gettimeofday(&(events_c2[0]), NULL);
    events_c2[1] = events_c2[0];
  }
  if (events_c3 != NULL) {
    gettimeofday(&(events_c3[0]), NULL);
    events_c3[1] = events_c3[0];
  }
  if (events_k1 != NULL) {
    gettimeofday(&(events_k1[0]), NULL);
  }
	// ignore grid_size, BLOCK_size, async

	switch (type) {
	case a_db:
		omp_unpack_face_kernel_db((double*) packed_buf, (double*) buf, face,
				remain_dim);
		break;

	case a_fl:
		omp_unpack_face_kernel_fl((float*) packed_buf, (float*) buf, face,
				remain_dim);
		break;

	case a_ul:
		omp_unpack_face_kernel_ul((unsigned long*) packed_buf,
				(unsigned long*) buf, face, remain_dim);
		break;

	case a_in:
		omp_unpack_face_kernel_in((int*) packed_buf, (int*) buf, face,
				remain_dim);
		break;

	case a_ui:
		omp_unpack_face_kernel_ui((unsigned int*) packed_buf,
				(unsigned int*) buf, face, remain_dim);
		break;

	default:
		fprintf(stderr,
				"Error: Function 'omp_transpose_face' not implemented for selected type!\n");
		break;
	}
  if (events_k1 != NULL) {
    gettimeofday(&(events_k1[1]), NULL);
  }
  //FIXME: Implement async "callback"
  if (call != NULL) {
    (call->callback_func)(call);
  }
    if (profiling_symbols.metaProfilingEnqueueTimer != NULL) {
  (*(profiling_symbols.metaProfilingEnqueueTimer))(*timer_k1, k_unpack_2d_face);
  (*(profiling_symbols.metaProfilingEnqueueTimer))(*timer_c1, c_H2Dc);
  (*(profiling_symbols.metaProfilingEnqueueTimer))(*timer_c2, c_H2Dc);
  (*(profiling_symbols.metaProfilingEnqueueTimer))(*timer_c3, c_H2Dc);
}
  //FIXME: Implement sync point here
	return (ret);
}

a_err openmp_stencil_3d7p(size_t (* grid_size)[3], size_t (* block_size)[3], void * indata, void * outdata, size_t (* array_size)[3], size_t (* arr_start)[3], size_t (* arr_end)[3], meta_type_id type, int async, meta_callback * call, meta_event * ret_event) {
	int ret = 0; //Success
openmpEvent * events;
  if (ret_event != NULL && ret_event->mode == metaModePreferOpenMP && ret_event->event_pl != NULL) events = ((openmpEvent *)ret_event->event_pl);
  meta_timer * timer = NULL;
  if (profiling_symbols.metaProfilingCreateTimer != NULL) {
    (*(profiling_symbols.metaProfilingCreateTimer))(&timer, metaModePreferOpenMP, (*array_size)[0]*(*array_size)[1]*(*array_size)[2]*get_atype_size(type));
    if (events == NULL) {
      events = ((openmpEvent *)timer->event.event_pl);
    } else {
      //FIXME: are we leaking a created openmpEvent here since the profiling function calls create?
      //metaOpenMPDestroyEvent(frame->event.event_pl);
      timer->event = *ret_event;
    }
  }
  if (events != NULL) {
    gettimeofday(&(events[0]), NULL);
  }

	// ignore grid_size, block_size, async

	switch (type) {
	case a_db:
		omp_stencil_3d7p_kernel_db((double*) indata, (double*) outdata,
				array_size, arr_start, arr_end);
		break;

	case a_fl:
		omp_stencil_3d7p_kernel_fl((float*) indata, (float*) outdata,
				array_size, arr_start, arr_end);
		break;

	case a_ul:
		omp_stencil_3d7p_kernel_ul((unsigned long*) indata,
				(unsigned long*) outdata, array_size, arr_start, arr_end);
		break;

	case a_in:
		omp_stencil_3d7p_kernel_in((int*) indata, (int*) outdata, array_size,
				arr_start, arr_end);
		break;

	case a_ui:
		omp_stencil_3d7p_kernel_ui((unsigned int*) indata,
				(unsigned int*) outdata, array_size, arr_start, arr_end);
		break;

	default:
		fprintf(stderr,
				"Error: Function 'omp_stencil_3d7p' not implemented for selected type!\n");
		break;
	}
  if (events != NULL) {
    gettimeofday(&(events[1]), NULL);
  }
  //FIXME: Implement async "callback"
  if (call != NULL) {
    (call->callback_func)(call);
  }
    if (profiling_symbols.metaProfilingEnqueueTimer != NULL) (*(profiling_symbols.metaProfilingEnqueueTimer))(*timer, k_stencil_3d7p);
	return (ret);

}
