/** OpenMP Back-End: MIC customization **/
#include "../../metamorph-backends/openmp-backend/metamorph_openmp.h"

//#define	COLLAPSE
//#define USE_AVX

// TODO figure out how to use templates with the  intrinsics

#define BLOCK 16
#define CX 256
#define CY 32
#define CZ 1

#include <x86intrin.h>

#ifdef USE_AVX
#include <x86intrin.h>

inline double hadd_pd(__m256d a) {
  __m256d s1 = _mm256_hadd_pd(a, a);
  __m128d s1_h = _mm256_extractf128_pd(s1, 1);
  __m128d result = _mm_add_pd(_mm256_castpd256_pd128(s1), s1_h);
  return _mm_cvtsd_f64(result);
}

inline float hadd_ps(__m256 a) {
  __m256 s1 = _mm256_hadd_ps(a, a);
  __m256 s2 = _mm256_hadd_ps(s1, s1);
  __m128 s2_h = _mm256_extractf128_ps(s2, 1);
  __m128 result = _mm_add_ss(_mm256_castps256_ps128(s2), s2_h);

  return _mm_cvtss_f32(result);
}

#endif

// Kernels
static void omp_dotProd_kernel_db(double *__restrict__ data1,
                                  double *__restrict__ data2,
                                  size_t (*array_size)[3],
                                  size_t (*arr_start)[3], size_t (*arr_end)[3],
                                  double *reduction_var) {
  int ni, nj, nk;
  double sum = 0;

  ni = (*array_size)[0];
  nj = (*array_size)[1];
  nk = (*array_size)[2];

#ifdef COLLAPSE
  int n;
  n = ((*arr_end)[0] - (*arr_start)[0] + 1) *
      ((*arr_end)[1] - (*arr_start)[1] + 1) *
      ((*arr_end)[2] - (*arr_start)[2] + 1);

  if (n == ni * nj * nk) // original 3d grid
  {
    int i;
#ifdef USE_AVX
#pragma omp parallel shared(n, data1, data2) private(i) reduction(+ : sum)
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
        sum1 = _mm256_add_pd(_mm256_mul_pd(x4, y4), sum1);
        x4 = _mm256_loadu_pd(&data1[i + 4]);
        y4 = _mm256_loadu_pd(&data2[i + 4]);
        sum2 = _mm256_add_pd(_mm256_mul_pd(x4, y4), sum2);
        x4 = _mm256_loadu_pd(&data1[i + 8]);
        y4 = _mm256_loadu_pd(&data2[i + 8]);
        sum3 = _mm256_add_pd(_mm256_mul_pd(x4, y4), sum3);
        x4 = _mm256_loadu_pd(&data1[i + 12]);
        y4 = _mm256_loadu_pd(&data2[i + 12]);
        sum4 = _mm256_add_pd(_mm256_mul_pd(x4, y4), sum4);
      }
      sum += hadd_pd(
          _mm256_add_pd(_mm256_add_pd(sum1, sum2), _mm256_add_pd(sum3, sum4)));
    }
#else
#pragma omp parallel for shared(n, data1, data2) private(i) reduction(+ : sum)
    for (i = 0; i < n; i++) {
      sum += data1[i] * data2[i];
    }
#endif
  } else // 3d sub-grid
#endif
  {
#pragma omp parallel shared(ni, nj, nk, data1, data2, sum) num_threads(60)
    {
      int i, j, k;
      double psum = 0;
      // double *d1, *d2;
#pragma omp for collapse(2)
      //#pragma omp for
      for (k = (*arr_start)[2]; k <= (*arr_end)[2]; k++) {
        for (j = (*arr_start)[1]; j <= (*arr_end)[1]; j++) {
          // d1 = &data1[j*ni+k*ni*nj];
          // d2 = &data2[j*ni+k*ni*nj];
          for (i = (*arr_start)[0]; i <= (*arr_end)[0]; i++) {
            int x;
            x = i + j * ni + k * ni * nj;
            psum += data1[x] * data2[x];
            // psum += d1[i] * d2[i];
          }
        }
      }

#pragma omp critical
      { sum += psum; }
    }
  } // endif if( n == ni*nj*nk )

  *reduction_var += sum;
}

static void omp_dotProd_kernel_fl(float *__restrict__ data1,
                                  float *__restrict__ data2,
                                  size_t (*array_size)[3],
                                  size_t (*arr_start)[3], size_t (*arr_end)[3],
                                  float *reduction_var) {
  float sum = 0;
  int ni, nj, nk;

  // data1 = __builtin_assume_aligned(data1, 32);
  // data2= __builtin_assume_aligned(data2, 32);

  ni = (*array_size)[0];
  nj = (*array_size)[1];
  nk = (*array_size)[2];

#ifdef COLLAPSE
  int n;
  n = ((*arr_end)[0] - (*arr_start)[0] + 1) *
      ((*arr_end)[1] - (*arr_start)[1] + 1) *
      ((*arr_end)[2] - (*arr_start)[2] + 1);

  if (n == ni * nj * nk) // original 3d grid
  {
    int i;
#ifdef USE_AVX
#pragma omp parallel shared(n, data1, data2) private(i) reduction(+ : sum)
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
        sum1 = _mm256_add_ps(_mm256_mul_ps(x8, y8), sum1);
        x8 = _mm256_loadu_ps(&data1[i + 8]);
        y8 = _mm256_loadu_ps(&data2[i + 8]);
        sum2 = _mm256_add_ps(_mm256_mul_ps(x8, y8), sum2);
        x8 = _mm256_loadu_ps(&data1[i + 16]);
        y8 = _mm256_loadu_ps(&data2[i + 16]);
        sum3 = _mm256_add_ps(_mm256_mul_ps(x8, y8), sum3);
        x8 = _mm256_loadu_ps(&data1[i + 24]);
        y8 = _mm256_loadu_ps(&data2[i + 24]);
        sum4 = _mm256_add_ps(_mm256_mul_ps(x8, y8), sum4);
      }
      sum += hadd_ps(
          _mm256_add_ps(_mm256_add_ps(sum1, sum2), _mm256_add_ps(sum3, sum4)));
    }
#else
#pragma omp parallel for shared(n, data1, data2) private(i) reduction(+ : sum)
    for (i = 0; i < n; i++) {
      sum += data1[i] * data2[i];
    }
#endif
  } else // 3d sub-grid
#endif
  {
#pragma omp parallel shared(ni, nj, nk, data1, data2, sum)
    {
      int i, j, k;
      float psum = 0;
#pragma omp for collapse(2)
      //#pragma omp for
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
      { sum += psum; }
    }
  } // endif if( n == ni*nj*nk )

  *reduction_var += sum;
}

static void omp_dotProd_kernel_ul(unsigned long *__restrict__ data1,
                                  unsigned long *__restrict__ data2,
                                  size_t (*array_size)[3],
                                  size_t (*arr_start)[3], size_t (*arr_end)[3],
                                  unsigned long *reduction_var) {
  unsigned long sum = 0;
  int ni, nj, nk;

  ni = (*array_size)[0];
  nj = (*array_size)[1];
  nk = (*array_size)[2];

#ifdef COLLAPSE
  int n;
  n = ((*arr_end)[0] - (*arr_start)[0] + 1) *
      ((*arr_end)[1] - (*arr_start)[1] + 1) *
      ((*arr_end)[2] - (*arr_start)[2] + 1);

  if (n == ni * nj * nk) // original 3d grid
  {
    int i;
#ifdef USE_AVX
#pragma omp parallel shared(n, data1, data2) private(i) reduction(+ : sum)
    {
#pragma omp parallel for
      for (i = 0; i < n; i++) {
        sum += data1[i] * data2[i];
      }
    }
#else
#pragma omp parallel for shared(n, data1, data2) private(i) reduction(+ : sum)
    for (i = 0; i < n; i++) {
      sum += data1[i] * data2[i];
    }
#endif
  } else // 3d sub-grid
#endif
  {
#pragma omp parallel shared(ni, nj, nk, data1, data2, sum)
    {
      int i, j, k;
      unsigned long psum = 0;
#pragma omp for collapse(2)
      //#pragma omp for
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
      { sum += psum; }
    }
  } // endif if( n == ni*nj*nk )

  *reduction_var += sum;
}

static void omp_dotProd_kernel_in(int *__restrict__ data1,
                                  int *__restrict__ data2,
                                  size_t (*array_size)[3],
                                  size_t (*arr_start)[3], size_t (*arr_end)[3],
                                  int *reduction_var) {
  int sum = 0;
  int ni, nj, nk;

  ni = (*array_size)[0];
  nj = (*array_size)[1];
  nk = (*array_size)[2];

#ifdef COLLAPSE
  int n;
  n = ((*arr_end)[0] - (*arr_start)[0] + 1) *
      ((*arr_end)[1] - (*arr_start)[1] + 1) *
      ((*arr_end)[2] - (*arr_start)[2] + 1);

  if (n == ni * nj * nk) // original 3d grid
  {
    int i;
#ifdef USE_AVX
#pragma omp parallel shared(n, data1, data2) private(i) reduction(+ : sum)
    {
#pragma omp parallel for
      for (i = 0; i < n; i++) {
        sum += data1[i] * data2[i];
      }
    }
#else
#pragma omp parallel for shared(n, data1, data2) private(i) reduction(+ : sum)
    for (i = 0; i < n; i++) {
      sum += data1[i] * data2[i];
    }
#endif
  } else // 3d sub-grid
#endif
  {
#pragma omp parallel shared(ni, nj, nk, data1, data2, sum)
    {
      int i, j, k;
      int psum = 0;
#pragma omp for collapse(2)
      //#pragma omp for
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
      { sum += psum; }
    }
  } // endif if( n == ni*nj*nk )

  *reduction_var += sum;
}

static void omp_dotProd_kernel_ui(unsigned int *__restrict__ data1,
                                  unsigned int *__restrict__ data2,
                                  size_t (*array_size)[3],
                                  size_t (*arr_start)[3], size_t (*arr_end)[3],
                                  unsigned int *reduction_var) {
  unsigned int sum = 0;
  int ni, nj, nk;

  ni = (*array_size)[0];
  nj = (*array_size)[1];
  nk = (*array_size)[2];

#ifdef COLLAPSE
  int n;
  n = ((*arr_end)[0] - (*arr_start)[0] + 1) *
      ((*arr_end)[1] - (*arr_start)[1] + 1) *
      ((*arr_end)[2] - (*arr_start)[2] + 1);

  if (n == ni * nj * nk) // original 3d grid
  {
    int i;
#ifdef USE_AVX
#pragma omp parallel shared(n, data1, data2) private(i) reduction(+ : sum)
    {
#pragma omp parallel for
      for (i = 0; i < n; i++) {
        sum += data1[i] * data2[i];
      }
    }
#else
#pragma omp parallel for shared(n, data1, data2) private(i) reduction(+ : sum)
    for (i = 0; i < n; i++) {
      sum += data1[i] * data2[i];
    }
#endif
  } else // 3d sub-grid
#endif
  {
#pragma omp parallel shared(ni, nj, nk, data1, data2, sum)
    {
      int i, j, k;
      unsigned int psum = 0;
#pragma omp for collapse(2)
      //#pragma omp for
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
      { sum += psum; }
    }
  } // endif if( n == ni*nj*nk )

  *reduction_var += sum;
}

// reduce
static void omp_reduce_kernel_db(double *__restrict__ data,
                                 size_t (*array_size)[3],
                                 size_t (*arr_start)[3], size_t (*arr_end)[3],
                                 double *reduction_var) {
  int ni, nj, nk;
  double sum = 0;

  ni = (*array_size)[0];
  nj = (*array_size)[1];
  nk = (*array_size)[2];

#ifdef COLLAPSE
  int n;
  n = ((*arr_end)[0] - (*arr_start)[0] + 1) *
      ((*arr_end)[1] - (*arr_start)[1] + 1) *
      ((*arr_end)[2] - (*arr_start)[2] + 1);

  if (n == ni * nj * nk) // original 3d grid
  {
    int i;
#ifdef USE_AVX
#pragma omp parallel shared(n, data) private(i) reduction(+ : sum)
    {
#pragma omp for
      for (i = 0; i < n; i++) {
        sum += data[i];
      }
    }
#else
#pragma omp parallel for shared(n, data) private(i) reduction(+ : sum)
    for (i = 0; i < n; i++) {
      sum += data[i];
    }
#endif
  } else // 3d sub-grid
#endif
  {
#pragma omp parallel shared(ni, nj, nk, data, sum)
    {
      int i, j, k;
      double psum = 0;
#pragma omp for collapse(2)
      //#pragma omp for
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
      { sum += psum; }
    }
  } // endif if( n == ni*nj*nk )

  *reduction_var += sum;
}

static void omp_reduce_kernel_fl(float *__restrict__ data,
                                 size_t (*array_size)[3],
                                 size_t (*arr_start)[3], size_t (*arr_end)[3],
                                 float *reduction_var) {
  int ni, nj, nk;
  float sum = 0;

  ni = (*array_size)[0];
  nj = (*array_size)[1];
  nk = (*array_size)[2];

#ifdef COLLAPSE
  int n;
  n = ((*arr_end)[0] - (*arr_start)[0] + 1) *
      ((*arr_end)[1] - (*arr_start)[1] + 1) *
      ((*arr_end)[2] - (*arr_start)[2] + 1);

  if (n == ni * nj * nk) // original 3d grid
  {
    int i;
#ifdef USE_AVX
#pragma omp parallel shared(n, data) private(i) reduction(+ : sum)
    {
#pragma omp for
      for (i = 0; i < n; i++) {
        sum += data[i];
      }
    }
#else
#pragma omp parallel for shared(n, data) private(i) reduction(+ : sum)
    for (i = 0; i < n; i++) {
      sum += data[i];
    }
#endif
  } else // 3d sub-grid
#endif
  {
#pragma omp parallel shared(ni, nj, nk, data, sum)
    {
      int i, j, k;
      float psum = 0;
#pragma omp for collapse(2)
      //#pragma omp for
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
      { sum += psum; }
    }
  } // endif if( n == ni*nj*nk )

  *reduction_var += sum;
}

static void omp_reduce_kernel_ul(unsigned long *__restrict__ data,
                                 size_t (*array_size)[3],
                                 size_t (*arr_start)[3], size_t (*arr_end)[3],
                                 unsigned long *reduction_var) {
  int ni, nj, nk;
  unsigned long sum = 0;

  ni = (*array_size)[0];
  nj = (*array_size)[1];
  nk = (*array_size)[2];

#ifdef COLLAPSE
  int n;
  n = ((*arr_end)[0] - (*arr_start)[0] + 1) *
      ((*arr_end)[1] - (*arr_start)[1] + 1) *
      ((*arr_end)[2] - (*arr_start)[2] + 1);

  if (n == ni * nj * nk) // original 3d grid
  {
    int i;
#ifdef USE_AVX
#pragma omp parallel shared(n, data) private(i) reduction(+ : sum)
    {
#pragma omp for
      for (i = 0; i < n; i++) {
        sum += data[i];
      }
    }
#else
#pragma omp parallel for shared(n, data) private(i) reduction(+ : sum)
    for (i = 0; i < n; i++) {
      sum += data[i];
    }
#endif
  } else // 3d sub-grid
#endif
  {
#pragma omp parallel shared(ni, nj, nk, data, sum)
    {
      int i, j, k;
      unsigned long psum = 0;
#pragma omp for collapse(2)
      //#pragma omp for
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
      { sum += psum; }
    }
  } // endif if( n == ni*nj*nk )

  *reduction_var += sum;
}

static void omp_reduce_kernel_in(int *__restrict__ data,
                                 size_t (*array_size)[3],
                                 size_t (*arr_start)[3], size_t (*arr_end)[3],
                                 int *reduction_var) {
  int ni, nj, nk;
  int sum = 0;

  ni = (*array_size)[0];
  nj = (*array_size)[1];
  nk = (*array_size)[2];

#ifdef COLLAPSE
  int n;
  n = ((*arr_end)[0] - (*arr_start)[0] + 1) *
      ((*arr_end)[1] - (*arr_start)[1] + 1) *
      ((*arr_end)[2] - (*arr_start)[2] + 1);

  if (n == ni * nj * nk) // original 3d grid
  {
    int i;
#ifdef USE_AVX
#pragma omp parallel shared(n, data) private(i) reduction(+ : sum)
    {
#pragma omp for
      for (i = 0; i < n; i++) {
        sum += data[i];
      }
    }
#else
#pragma omp parallel for shared(n, data) private(i) reduction(+ : sum)
    for (i = 0; i < n; i++) {
      sum += data[i];
    }
#endif
  } else // 3d sub-grid
#endif
  {
#pragma omp parallel shared(ni, nj, nk, data, sum)
    {
      int i, j, k;
      int psum = 0;
#pragma omp for collapse(2)
      //#pragma omp for
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
      { sum += psum; }
    }
  } // endif if( n == ni*nj*nk )

  *reduction_var += sum;
}

static void omp_reduce_kernel_ui(unsigned int *__restrict__ data,
                                 size_t (*array_size)[3],
                                 size_t (*arr_start)[3], size_t (*arr_end)[3],
                                 unsigned int *reduction_var) {
  int ni, nj, nk;
  unsigned int sum = 0;

  ni = (*array_size)[0];
  nj = (*array_size)[1];
  nk = (*array_size)[2];

#ifdef COLLAPSE
  int n;
  n = ((*arr_end)[0] - (*arr_start)[0] + 1) *
      ((*arr_end)[1] - (*arr_start)[1] + 1) *
      ((*arr_end)[2] - (*arr_start)[2] + 1);

  if (n == ni * nj * nk) // original 3d grid
  {
    int i;
#ifdef USE_AVX
#pragma omp parallel shared(n, data) private(i) reduction(+ : sum)
    {
#pragma omp for
      for (i = 0; i < n; i++) {
        sum += data[i];
      }
    }
#else
#pragma omp parallel for shared(n, data) private(i) reduction(+ : sum)
    for (i = 0; i < n; i++) {
      sum += data[i];
    }
#endif
  } else // 3d sub-grid
#endif
  {
#pragma omp parallel shared(ni, nj, nk, data, sum)
    {
      int i, j, k;
      unsigned int psum = 0;
#pragma omp for collapse(2)
      //#pragma omp for
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
      { sum += psum; }
    }
  } // endif if( n == ni*nj*nk )

  *reduction_var += sum;
}

static void omp_transpose_face_kernel_db(double *__restrict__ indata,
                                         double *__restrict__ outdata,
                                         size_t (*arr_dim_xy)[3],
                                         size_t (*tran_dim_xy)[3]) {
  int ni, nj;
  unsigned int sum = 0;

  ni = (*arr_dim_xy)[0];
  nj = (*arr_dim_xy)[1];

#pragma omp parallel shared(ni, nj, indata, outdata)
  {
    int i, j, ii, jj;

#pragma omp for schedule(static) nowait collapse(2)
    //#pragma omp for schedule(static) nowait
    //#pragma omp for schedule(dynamic, 1) nowait
    for (j = 0; j < nj; j += BLOCK) {
      for (i = 0; i < ni; i += BLOCK) {
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

static void omp_transpose_face_kernel_fl(float *__restrict__ indata,
                                         float *__restrict__ outdata,
                                         size_t (*arr_dim_xy)[3],
                                         size_t (*tran_dim_xy)[3]) {
  int ni, nj;
  unsigned int sum = 0;

  ni = (*arr_dim_xy)[0];
  nj = (*arr_dim_xy)[1];

#pragma omp parallel shared(ni, nj, indata, outdata)
  {
    int i, j, ii, jj;

#pragma omp for schedule(static) nowait collapse(2)
    //#pragma omp for schedule(static) nowait
    //#pragma omp for schedule(dynamic, 1) nowait
    for (j = 0; j < nj; j += BLOCK) {
      for (i = 0; i < ni; i += BLOCK) {
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

static void omp_transpose_face_kernel_ul(unsigned long *__restrict__ indata,
                                         unsigned long *__restrict__ outdata,
                                         size_t (*arr_dim_xy)[3],
                                         size_t (*tran_dim_xy)[3]) {

  int ni, nj;
  unsigned int sum = 0;

  ni = (*arr_dim_xy)[0];
  nj = (*arr_dim_xy)[1];

#pragma omp parallel shared(ni, nj, indata, outdata)
  {
    int i, j, ii, jj;

#pragma omp for schedule(static) nowait collapse(2)
    //#pragma omp for schedule(static) nowait
    //#pragma omp for schedule(dynamic, 1) nowait
    for (j = 0; j < nj; j += BLOCK) {
      for (i = 0; i < ni; i += BLOCK) {
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

static void omp_transpose_face_kernel_in(int *__restrict__ indata,
                                         int *__restrict__ outdata,
                                         size_t (*arr_dim_xy)[3],
                                         size_t (*tran_dim_xy)[3]) {
  int ni, nj;
  unsigned int sum = 0;

  ni = (*arr_dim_xy)[0];
  nj = (*arr_dim_xy)[1];

#pragma omp parallel shared(ni, nj, indata, outdata)
  {
    int i, j, ii, jj;

#pragma omp for schedule(static) nowait collapse(2)
    //#pragma omp for schedule(static) nowait
    //#pragma omp for schedule(dynamic, 1) nowait
    for (j = 0; j < nj; j += BLOCK) {
      for (i = 0; i < ni; i += BLOCK) {
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

static void omp_transpose_face_kernel_ui(unsigned int *__restrict__ indata,
                                         unsigned int *__restrict__ outdata,
                                         size_t (*arr_dim_xy)[3],
                                         size_t (*tran_dim_xy)[3]) {
  int ni, nj;
  unsigned int sum = 0;

  ni = (*arr_dim_xy)[0];
  nj = (*arr_dim_xy)[1];

#pragma omp parallel shared(ni, nj, indata, outdata)
  {
    int i, j, ii, jj;

#pragma omp for schedule(static) nowait collapse(2)
    //#pragma omp for schedule(static) nowait
    //#pragma omp for schedule(dynamic, 1) nowait
    for (j = 0; j < nj; j += BLOCK) {
      for (i = 0; i < ni; i += BLOCK) {
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
                          int *__restrict__ remain_dim) {
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

void omp_pack_face_kernel_db(double *__restrict__ packed_buf,
                             double *__restrict__ buf, meta_face *face,
                             int *remain_dim) {
  int size = face->size[0] * face->size[1] * face->size[2];

#pragma omp parallel shared(size, packed_buf, buf, face, remain_dim)
  {
    int idx;
#pragma omp for schedule(static) nowait
    //#pragma omp for schedule(dynamic,16) nowait
    for (idx = 0; idx < size; idx++)
      packed_buf[idx] = buf[get_pack_index(idx, face, remain_dim)];
  }
}

void omp_pack_face_kernel_fl(float *__restrict__ packed_buf,
                             float *__restrict__ buf, meta_face *face,
                             int *remain_dim) {
  int size = face->size[0] * face->size[1] * face->size[2];

#pragma omp parallel shared(size, packed_buf, buf, face, remain_dim)
  {
    int idx;
#pragma omp for schedule(static) nowait
    //#pragma omp for schedule(dynamic,16) nowait
    for (idx = 0; idx < size; idx++)
      packed_buf[idx] = buf[get_pack_index(idx, face, remain_dim)];
  }
}

void omp_pack_face_kernel_ul(unsigned long *__restrict__ packed_buf,
                             unsigned long *__restrict__ buf, meta_face *face,
                             int *remain_dim) {
  int size = face->size[0] * face->size[1] * face->size[2];

#pragma omp parallel shared(size, packed_buf, buf, face, remain_dim)
  {
    int idx;
#pragma omp for schedule(static) nowait
    //#pragma omp for schedule(dynamic,16) nowait
    for (idx = 0; idx < size; idx++)
      packed_buf[idx] = buf[get_pack_index(idx, face, remain_dim)];
  }
}

void omp_pack_face_kernel_in(int *__restrict__ packed_buf,
                             int *__restrict__ buf, meta_face *face,
                             int *remain_dim) {
  int size = face->size[0] * face->size[1] * face->size[2];

#pragma omp parallel shared(size, packed_buf, buf, face, remain_dim)
  {
    int idx;
#pragma omp for schedule(static) nowait
    //#pragma omp for schedule(dynamic,16) nowait
    for (idx = 0; idx < size; idx++)
      packed_buf[idx] = buf[get_pack_index(idx, face, remain_dim)];
  }
}

void omp_pack_face_kernel_ui(unsigned int *__restrict__ packed_buf,
                             unsigned int *__restrict__ buf, meta_face *face,
                             int *remain_dim) {
  int size = face->size[0] * face->size[1] * face->size[2];

#pragma omp parallel shared(size, packed_buf, buf, face, remain_dim)
  {
    int idx;
#pragma omp for schedule(static) nowait
    //#pragma omp for schedule(dynamic,16) nowait
    for (idx = 0; idx < size; idx++)
      packed_buf[idx] = buf[get_pack_index(idx, face, remain_dim)];
  }
}

void omp_unpack_face_kernel_db(double *__restrict__ packed_buf,
                               double *__restrict__ buf, meta_face *face,
                               int *remain_dim) {
  int size = face->size[0] * face->size[1] * face->size[2];

#pragma omp parallel shared(size, packed_buf, buf, face, remain_dim)
  {
    int idx;
#pragma omp for schedule(static) nowait
    //#pragma omp for schedule(dynamic,16) nowait
    for (idx = 0; idx < size; idx++)
      buf[get_pack_index(idx, face, remain_dim)] = packed_buf[idx];
  }
}

void omp_unpack_face_kernel_fl(float *__restrict__ packed_buf,
                               float *__restrict__ buf, meta_face *face,
                               int *remain_dim) {
  int size = face->size[0] * face->size[1] * face->size[2];

#pragma omp parallel shared(size, packed_buf, buf, face, remain_dim)
  {
    int idx;
#pragma omp for schedule(static) nowait
    //#pragma omp for schedule(dynamic,16) nowait
    for (idx = 0; idx < size; idx++)
      buf[get_pack_index(idx, face, remain_dim)] = packed_buf[idx];
  }
}

void omp_unpack_face_kernel_ul(unsigned long *__restrict__ packed_buf,
                               unsigned long *__restrict__ buf, meta_face *face,
                               int *remain_dim) {
  int size = face->size[0] * face->size[1] * face->size[2];

#pragma omp parallel shared(size, packed_buf, buf, face, remain_dim)
  {
    int idx;
#pragma omp for schedule(static) nowait
    //#pragma omp for schedule(dynamic,16) nowait
    for (idx = 0; idx < size; idx++)
      buf[get_pack_index(idx, face, remain_dim)] = packed_buf[idx];
  }
}

void omp_unpack_face_kernel_in(int *__restrict__ packed_buf,
                               int *__restrict__ buf, meta_face *face,
                               int *remain_dim) {
  int size = face->size[0] * face->size[1] * face->size[2];

#pragma omp parallel shared(size, packed_buf, buf, face, remain_dim)
  {
    int idx;
#pragma omp for schedule(static) nowait
    //#pragma omp for schedule(dynamic,16) nowait
    for (idx = 0; idx < size; idx++)
      buf[get_pack_index(idx, face, remain_dim)] = packed_buf[idx];
  }
}

void omp_unpack_face_kernel_ui(unsigned int *__restrict__ packed_buf,
                               unsigned int *__restrict__ buf, meta_face *face,
                               int *remain_dim) {
  int size = face->size[0] * face->size[1] * face->size[2];

#pragma omp parallel shared(size, packed_buf, buf, face, remain_dim)
  {
    int idx;
#pragma omp for schedule(static) nowait
    //#pragma omp for schedule(dynamic,16) nowait
    for (idx = 0; idx < size; idx++)
      buf[get_pack_index(idx, face, remain_dim)] = packed_buf[idx];
  }
}

#if 1
void omp_stencil_3d7p_kernel_db(double *__restrict__ indata,
                                double *__restrict__ outdata,
                                size_t (*array_size)[3], size_t (*arr_start)[3],
                                size_t (*arr_end)[3]) {
  int ni, nj, nk;

  ni = (*array_size)[0];
  nj = (*array_size)[1];
  nk = (*array_size)[2];

#pragma omp parallel shared(ni, nj, nk, indata, outdata)
  {
    int i, j, k;
    // double *in, *out;
#pragma omp for collapse(2) schedule(static) nowait
    //#pragma omp for
    for (k = (*arr_start)[2] + 1; k < (*arr_end)[2]; k++) {
      for (j = (*arr_start)[1] + 1; j < (*arr_end)[1]; j++) {
        // in = &indata[j*ni+k*ni*nj];
        // out = &outdata[j*ni+k*ni*nj];
        //#pragma unroll (8)
        //#pragma prefetch indata:_MM_HINT_T2:8,outdata:_MM_HINT_NTA
        //#pragma loop count (256)
#pragma ivdep
#pragma vector nontemporal(outdata)
        for (i = (*arr_start)[0] + 1; i < (*arr_end)[0]; i++) {
          outdata[i + j * ni + k * ni * nj] =
              (indata[i + j * ni + (k - 1) * ni * nj] +
               indata[(i - 1) + j * ni + k * ni * nj] +
               indata[i + (j - 1) * ni + k * ni * nj] +
               indata[i + j * ni + k * ni * nj] +
               indata[i + (j + 1) * ni + k * ni * nj] +
               indata[(i + 1) + j * ni + k * ni * nj] +
               indata[i + j * ni + (k + 1) * ni * nj]) /
              (double)7.0;
          // out[i] = ( in[i-ni*nj] + in[i-1] + in[i-ni] + in[i] +
          //			in[i+ni] + in[i+1] + in[i+ni*nj] )
          //			/ (double) 7.0;
        }
      }
    }
  }
}
#elif 0
void omp_stencil_3d7p_kernel_db(double *indata, double *outdata,
                                size_t (*array_size)[3], size_t (*arr_start)[3],
                                size_t (*arr_end)[3]) {
  int ni, nj, nk;

  ni = (*array_size)[0];
  nj = (*array_size)[1];
  nk = (*array_size)[2];

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
    for (k = (*arr_start)[2] + 1; k < (*arr_end)[2]; k += CZ) {
      for (j = (*arr_start)[1] + 1; j < (*arr_end)[1]; j += CY) {
        for (i = (*arr_start)[0] + 1; i < (*arr_end)[0]; i += CX) {
          int kkmax = ((*arr_end)[2] < k + CZ ? (*arr_end)[2] : k + CZ);
          int jjmax = ((*arr_end)[1] < j + CY ? (*arr_end)[1] : j + CY);
          int iimax = ((*arr_end)[0] < i + CX ? (*arr_end)[0] : i + CX);
          for (kk = k; kk < kkmax; kk++) {
            for (jj = j; jj < jjmax; jj++) {
              for (ii = i; ii < iimax; ii++) {
                outdata[ii + jj * ni + kk * ni * nj] =
                    (indata[ii + jj * ni + (kk - 1) * ni * nj] +
                     indata[(ii - 1) + jj * ni + kk * ni * nj] +
                     indata[ii + (jj - 1) * ni + kk * ni * nj] +
                     indata[ii + jj * ni + kk * ni * nj] +
                     indata[ii + (jj + 1) * ni + kk * ni * nj] +
                     indata[(ii + 1) + jj * ni + kk * ni * nj] +
                     indata[ii + jj * ni + (kk + 1) * ni * nj]) /
                    (double)7.0;
              }
            }
          }
        }
      }
    }
  }
}
#elif 0
void omp_stencil_3d7p_kernel_db(double *indata, double *outdata,
                                size_t (*array_size)[3], size_t (*arr_start)[3],
                                size_t (*arr_end)[3]) {
  int ni, nj, nk;

  ni = (*array_size)[0];
  nj = (*array_size)[1];
  nk = (*array_size)[2];

#pragma omp parallel shared(ni, nj, nk, indata, outdata)
  {
    int i, j, k, jj, kk;

#pragma omp for collapse(2) schedule(static) nowait
    for (k = (*arr_start)[2] + 1; k < (*arr_end)[2]; k += CZ) {
      for (j = (*arr_start)[1] + 1; j < (*arr_end)[1]; j += CY) {
        int kkmax = ((*arr_end)[2] < k + CZ ? (*arr_end)[2] : k + CZ);
        int jjmax = ((*arr_end)[1] < j + CY ? (*arr_end)[1] : j + CY);
        for (kk = k; kk < kkmax; kk++) {
          for (jj = j; jj < jjmax; jj++) {
#pragma ivdep
#pragma vector nontemporal(outdata)
            for (i = (*arr_start)[0] + 1; i < (*arr_end)[0]; i++) {
              outdata[i + jj * ni + kk * ni * nj] =
                  (indata[i + jj * ni + (kk - 1) * ni * nj] +
                   indata[(i - 1) + jj * ni + kk * ni * nj] +
                   indata[i + (jj - 1) * ni + kk * ni * nj] +
                   indata[i + jj * ni + kk * ni * nj] +
                   indata[i + (jj + 1) * ni + kk * ni * nj] +
                   indata[(i + 1) + jj * ni + kk * ni * nj] +
                   indata[i + jj * ni + (kk + 1) * ni * nj]) /
                  (double)7.0;
            }
          }
        }
      }
    }
  }
}
#elif 0
void omp_stencil_3d7p_kernel_db(double *indata, double *outdata,
                                size_t (*array_size)[3], size_t (*arr_start)[3],
                                size_t (*arr_end)[3]) {
  int ni, nj, nk;

  ni = (*array_size)[0];
  nj = (*array_size)[1];
  nk = (*array_size)[2];

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
    for (k = (*arr_start)[2] + 1; k < (*arr_end)[2]; k++) {
      for (j = (*arr_start)[1] + 1; j < (*arr_end)[1]; j += CY) {
        for (i = (*arr_start)[0] + 1; i < (*arr_end)[0]; i += CX) {
          int jjmax = ((*arr_end)[1] < j + CY ? (*arr_end)[1] : j + CY);
          int iimax = ((*arr_end)[0] < i + CX ? (*arr_end)[0] : i + CX);
          for (jj = j; jj < jjmax; jj++) {
            for (ii = i; ii < iimax; ii++) {
              outdata[ii + jj * ni + k * ni * nj] =
                  (indata[ii + jj * ni + (k - 1) * ni * nj] +
                   indata[(ii - 1) + jj * ni + kk * ni * nj] +
                   indata[ii + (jj - 1) * ni + k * ni * nj] +
                   indata[ii + jj * ni + k * ni * nj] +
                   indata[ii + (jj + 1) * ni + k * ni * nj] +
                   indata[(ii + 1) + jj * ni + k * ni * nj] +
                   indata[ii + jj * ni + (k + 1) * ni * nj]) /
                  (double)7.0;
            }
          }
        }
      }
    }
  }
}
#else
void omp_stencil_3d7p_kernel_db(double *__restrict__ indata,
                                double *__restrict__ outdata,
                                size_t (*array_size)[3], size_t (*arr_start)[3],
                                size_t (*arr_end)[3]) {
  int ni, nj, nk;

  ni = (*array_size)[0];
  nj = (*array_size)[1];
  nk = (*array_size)[2];

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
    for (k = (*arr_start)[2] + 1; k < (*arr_end)[2]; k++) {
      for (j = (*arr_start)[1] + 1; j < (*arr_end)[1]; j++) {
        //#pragma Loop_Optimize Ivdep No_Unroll
        //#pragma unroll(8)
        i = (*arr_start)[0] + 1;
        c = i + j * ni + k * ni * nj;
        r0 = indata[c];
        rx1 = indata[c - 1];
        rx2 = indata[c + 1];
        for (; i < (*arr_end)[0]; i++) {
          outdata[i + j * ni + k * ni * nj] =
              (indata[i + j * ni + (k - 1) * ni * nj] + rx1 +
               indata[i + (j - 1) * ni + k * ni * nj] + r0 +
               indata[i + (j + 1) * ni + k * ni * nj] +
               indata[(i + 1) + j * ni + k * ni * nj] + rx2) /
              (double)7.0;
          c++;
          r0 = rx2;
          rx1 = r0;
          rx2 = indata[c + 1];
        }
      }
    }
  }
}
#endif

void omp_stencil_3d7p_kernel_fl(float *indata, float *outdata,
                                size_t (*array_size)[3], size_t (*arr_start)[3],
                                size_t (*arr_end)[3]) {
  int ni, nj, nk;

  ni = (*array_size)[0];
  nj = (*array_size)[1];
  nk = (*array_size)[2];

#pragma omp parallel shared(ni, nj, nk, indata, outdata)
  {
    int i, j, k;
#pragma omp for collapse(2) schedule(static) nowait
    //#pragma omp for
    for (k = (*arr_start)[2] + 1; k < (*arr_end)[2]; k++) {
      for (j = (*arr_start)[1] + 1; j < (*arr_end)[1]; j++) {
        //#pragma unroll (8)
        //#pragma prefetch indata:_MM_HINT_T2:8,outdata:_MM_HINT_NTA
        //#pragma loop count (256)
#pragma ivdep
#pragma vector nontemporal(outdata)
        for (i = (*arr_start)[0] + 1; i < (*arr_end)[0]; i++) {
          outdata[i + j * ni + k * ni * nj] =
              (indata[i + j * ni + (k - 1) * ni * nj] +
               indata[(i - 1) + j * ni + k * ni * nj] +
               indata[i + (j - 1) * ni + k * ni * nj] +
               indata[i + j * ni + k * ni * nj] +
               indata[i + (j + 1) * ni + k * ni * nj] +
               indata[(i + 1) + j * ni + k * ni * nj] +
               indata[i + j * ni + (k + 1) * ni * nj]) /
              (float)7.0;
        }
      }
    }
  }
}

void omp_stencil_3d7p_kernel_ul(unsigned long *indata, unsigned long *outdata,
                                size_t (*array_size)[3], size_t (*arr_start)[3],
                                size_t (*arr_end)[3]) {
  int ni, nj, nk;

  ni = (*array_size)[0];
  nj = (*array_size)[1];
  nk = (*array_size)[2];

#pragma omp parallel shared(ni, nj, nk, indata, outdata)
  {
    int i, j, k;
#pragma omp for collapse(2) schedule(static) nowait
    //#pragma omp for
    for (k = (*arr_start)[2] + 1; k < (*arr_end)[2]; k++) {
      for (j = (*arr_start)[1] + 1; j < (*arr_end)[1]; j++) {
        //#pragma unroll (8)
        //#pragma prefetch indata:_MM_HINT_T2:8,outdata:_MM_HINT_NTA
        //#pragma loop count (256)
#pragma ivdep
#pragma vector nontemporal(outdata)
        for (i = (*arr_start)[0] + 1; i < (*arr_end)[0]; i++) {
          outdata[i + j * ni + k * ni * nj] =
              (indata[i + j * ni + (k - 1) * ni * nj] +
               indata[(i - 1) + j * ni + k * ni * nj] +
               indata[i + (j - 1) * ni + k * ni * nj] +
               indata[i + j * ni + k * ni * nj] +
               indata[i + (j + 1) * ni + k * ni * nj] +
               indata[(i + 1) + j * ni + k * ni * nj] +
               indata[i + j * ni + (k + 1) * ni * nj]) /
              (unsigned long)7;
        }
      }
    }
  }
}

void omp_stencil_3d7p_kernel_in(int *indata, int *outdata,
                                size_t (*array_size)[3], size_t (*arr_start)[3],
                                size_t (*arr_end)[3]) {
  int ni, nj, nk;

  ni = (*array_size)[0];
  nj = (*array_size)[1];
  nk = (*array_size)[2];

#pragma omp parallel shared(ni, nj, nk, indata, outdata)
  {
    int i, j, k;
#pragma omp for collapse(2) schedule(static) nowait
    //#pragma omp for
    for (k = (*arr_start)[2] + 1; k < (*arr_end)[2]; k++) {
      for (j = (*arr_start)[1] + 1; j < (*arr_end)[1]; j++) {
        //#pragma unroll (8)
        //#pragma prefetch indata:_MM_HINT_T2:8,outdata:_MM_HINT_NTA
        //#pragma loop count (256)
#pragma ivdep
#pragma vector nontemporal(outdata)
        for (i = (*arr_start)[0] + 1; i < (*arr_end)[0]; i++) {
          outdata[i + j * ni + k * ni * nj] =
              (indata[i + j * ni + (k - 1) * ni * nj] +
               indata[(i - 1) + j * ni + k * ni * nj] +
               indata[i + (j - 1) * ni + k * ni * nj] +
               indata[i + j * ni + k * ni * nj] +
               indata[i + (j + 1) * ni + k * ni * nj] +
               indata[(i + 1) + j * ni + k * ni * nj] +
               indata[i + j * ni + (k + 1) * ni * nj]) /
              (int)7;
        }
      }
    }
  }
}

void omp_stencil_3d7p_kernel_ui(unsigned int *indata, unsigned int *outdata,
                                size_t (*array_size)[3], size_t (*arr_start)[3],
                                size_t (*arr_end)[3]) {
  int ni, nj, nk;

  ni = (*array_size)[0];
  nj = (*array_size)[1];
  nk = (*array_size)[2];

#pragma omp parallel shared(ni, nj, nk, indata, outdata)
  {
    int i, j, k;
#pragma omp for collapse(2) schedule(static) nowait
    //#pragma omp for
    for (k = (*arr_start)[2] + 1; k < (*arr_end)[2]; k++) {
      for (j = (*arr_start)[1] + 1; j < (*arr_end)[1]; j++) {
        //#pragma unroll (8)
        //#pragma prefetch indata:_MM_HINT_T2:8,outdata:_MM_HINT_NTA
        //#pragma loop count (256)
#pragma ivdep
#pragma vector nontemporal(outdata)
        for (i = (*arr_start)[0] + 1; i < (*arr_end)[0]; i++) {
          outdata[i + j * ni + k * ni * nj] =
              (indata[i + j * ni + (k - 1) * ni * nj] +
               indata[(i - 1) + j * ni + k * ni * nj] +
               indata[i + (j - 1) * ni + k * ni * nj] +
               indata[i + j * ni + k * ni * nj] +
               indata[i + (j + 1) * ni + k * ni * nj] +
               indata[(i + 1) + j * ni + k * ni * nj] +
               indata[i + j * ni + (k + 1) * ni * nj]) /
              (unsigned int)7;
        }
      }
    }
  }
}

// wrappers
int omp_dotProd(size_t (*grid_size)[3], size_t (*block_size)[3], void *data1,
                void *data2, size_t (*array_size)[3], size_t (*arr_start)[3],
                size_t (*arr_end)[3], void *reduction_var, meta_type_id type,
                int async) {
  int ret = 0; // Success

  // ignore grid_size, block_size, async

  switch (type) {
  case meta_db:
    omp_dotProd_kernel_db((double *)data1, (double *)data2, array_size,
                          arr_start, arr_end, (double *)reduction_var);
    break;

  case meta_fl:
    omp_dotProd_kernel_fl((float *)data1, (float *)data2, array_size, arr_start,
                          arr_end, (float *)reduction_var);
    break;

  case meta_ul:
    omp_dotProd_kernel_ul((unsigned long *)data1, (unsigned long *)data2,
                          array_size, arr_start, arr_end,
                          (unsigned long *)reduction_var);
    break;

  case meta_in:
    omp_dotProd_kernel_in((int *)data1, (int *)data2, array_size, arr_start,
                          arr_end, (int *)reduction_var);
    break;

  case meta_ui:
    omp_dotProd_kernel_ui((unsigned int *)data1, (unsigned int *)data2,
                          array_size, arr_start, arr_end,
                          (unsigned int *)reduction_var);
    break;

  default:
    fprintf(
        stderr,
        "Error: Function 'omp_dotProd' not implemented for selected type!\n");
    break;
  }

  return (ret);
}

int omp_reduce(size_t (*grid_size)[3], size_t (*block_size)[3], void *data,
               size_t (*array_size)[3], size_t (*arr_start)[3],
               size_t (*arr_end)[3], void *reduction_var, meta_type_id type,
               int async) {
  int ret = 0; // Success

  // ignore grid_size, block_size, async

  switch (type) {
  case meta_db:
    omp_reduce_kernel_db((double *)data, array_size, arr_start, arr_end,
                         (double *)reduction_var);
    break;

  case meta_fl:
    omp_reduce_kernel_fl((float *)data, array_size, arr_start, arr_end,
                         (float *)reduction_var);
    break;

  case meta_ul:
    omp_reduce_kernel_ul((unsigned long *)data, array_size, arr_start, arr_end,
                         (unsigned long *)reduction_var);
    break;

  case meta_in:
    omp_reduce_kernel_in((int *)data, array_size, arr_start, arr_end,
                         (int *)reduction_var);
    break;

  case meta_ui:
    omp_reduce_kernel_ui((unsigned int *)data, array_size, arr_start, arr_end,
                         (unsigned int *)reduction_var);
    break;

  default:
    fprintf(
        stderr,
        "Error: Function 'omp_reduce' not implemented for selected type!\n");
    break;
  }

  return (ret);
}

int omp_transpose_face(size_t (*grid_size)[3], size_t (*block_size)[3],
                       void *indata, void *outdata, size_t (*arr_dim_xy)[3],
                       size_t (*tran_dim_xy)[3], meta_type_id type, int async) {
  int ret = 0; // Success

  // ignore grid_size, block_size, async

  switch (type) {
  case meta_db:
    omp_transpose_face_kernel_db((double *)indata, (double *)outdata,
                                 arr_dim_xy, tran_dim_xy);
    break;

  case meta_fl:
    omp_transpose_face_kernel_fl((float *)indata, (float *)outdata, arr_dim_xy,
                                 tran_dim_xy);
    break;

  case meta_ul:
    omp_transpose_face_kernel_ul((unsigned long *)indata,
                                 (unsigned long *)outdata, arr_dim_xy,
                                 tran_dim_xy);
    break;

  case meta_in:
    omp_transpose_face_kernel_in((int *)indata, (int *)outdata, arr_dim_xy,
                                 tran_dim_xy);
    break;

  case meta_ui:
    omp_transpose_face_kernel_ui((unsigned int *)indata,
                                 (unsigned int *)outdata, arr_dim_xy,
                                 tran_dim_xy);
    break;

  default:
    fprintf(stderr, "Error: Function 'omp_transpose_face' not implemented for "
                    "selected type!\n");
    break;
  }

  return (ret);
}

int omp_pack_face(size_t (*grid_size)[3], size_t (*block_size)[3],
                  void *packed_buf, void *buf, meta_face *face, int *remain_dim,
                  meta_type_id type, int async) {
  int ret = 0; // Success

  // ignore grid_size, BLOCK_size, async

  switch (type) {
  case meta_db:
    omp_pack_face_kernel_db((double *)packed_buf, (double *)buf, face,
                            remain_dim);
    break;

  case meta_fl:
    omp_pack_face_kernel_fl((float *)packed_buf, (float *)buf, face,
                            remain_dim);
    break;

  case meta_ul:
    omp_pack_face_kernel_ul((unsigned long *)packed_buf, (unsigned long *)buf,
                            face, remain_dim);
    break;

  case meta_in:
    omp_pack_face_kernel_in((int *)packed_buf, (int *)buf, face, remain_dim);
    break;

  case meta_ui:
    omp_pack_face_kernel_ui((unsigned int *)packed_buf, (unsigned int *)buf,
                            face, remain_dim);
    break;

  default:
    fprintf(stderr, "Error: Function 'omp_transpose_face' not implemented for "
                    "selected type!\n");
    break;
  }

  return (ret);
}

int omp_unpack_face(size_t (*grid_size)[3], size_t (*block_size)[3],
                    void *packed_buf, void *buf, meta_face *face,
                    int *remain_dim, meta_type_id type, int async) {
  int ret = 0; // Success

  // ignore grid_size, BLOCK_size, async

  switch (type) {
  case meta_db:
    omp_unpack_face_kernel_db((double *)packed_buf, (double *)buf, face,
                              remain_dim);
    break;

  case meta_fl:
    omp_unpack_face_kernel_fl((float *)packed_buf, (float *)buf, face,
                              remain_dim);
    break;

  case meta_ul:
    omp_unpack_face_kernel_ul((unsigned long *)packed_buf, (unsigned long *)buf,
                              face, remain_dim);
    break;

  case meta_in:
    omp_unpack_face_kernel_in((int *)packed_buf, (int *)buf, face, remain_dim);
    break;

  case meta_ui:
    omp_unpack_face_kernel_ui((unsigned int *)packed_buf, (unsigned int *)buf,
                              face, remain_dim);
    break;

  default:
    fprintf(stderr, "Error: Function 'omp_transpose_face' not implemented for "
                    "selected type!\n");
    break;
  }

  return (ret);
}

int omp_stencil_3d7p(size_t (*grid_size)[3], size_t (*block_size)[3],
                     void *indata, void *outdata, size_t (*array_size)[3],
                     size_t (*arr_start)[3], size_t (*arr_end)[3],
                     meta_type_id type, int async) {
  int ret = 0; // Success

  // ignore grid_size, block_size, async

  switch (type) {
  case meta_db:
    omp_stencil_3d7p_kernel_db((double *)indata, (double *)outdata, array_size,
                               arr_start, arr_end);
    break;

  case meta_fl:
    omp_stencil_3d7p_kernel_fl((float *)indata, (float *)outdata, array_size,
                               arr_start, arr_end);
    break;

  case meta_ul:
    omp_stencil_3d7p_kernel_ul((unsigned long *)indata,
                               (unsigned long *)outdata, array_size, arr_start,
                               arr_end);
    break;

  case meta_in:
    omp_stencil_3d7p_kernel_in((int *)indata, (int *)outdata, array_size,
                               arr_start, arr_end);
    break;

  case meta_ui:
    omp_stencil_3d7p_kernel_ui((unsigned int *)indata, (unsigned int *)outdata,
                               array_size, arr_start, arr_end);
    break;

  default:
    fprintf(stderr, "Error: Function 'omp_stencil_3d7p' not implemented for "
                    "selected type!\n");
    break;
  }

  return (ret);
}

int omp_copy_d2d(void *dst, void *src, size_t size, int async) {
  int i;
  // int num_t = omp_get _num_threads();
  int num_b = size / sizeof(unsigned long);

#pragma omp parallel for
#pragma ivdep
  //#pragma vector nontemporal (dst)
  for (i = 0; i < num_b; i++) {
    // memcpy((unsigned char *) dst+i, (unsigned char *) src+i, size);
    *((unsigned long *)dst + i) = *((unsigned long *)src + i);
  }
}
