/**
 * @file
 * A simple reduction on a torus, works with or without MPI
 * Each process initializes a portion of a 3D torus, where cells are filled with
 * the sum of their global x, y, and z indices Each process has a ghost region
 * on one of their "low index" faces, and sends the opposite "high index" face
 * to it's ((rank+1)%num_ranks) neighbor, using the pack_and_send and
 * recv_and_unpack MetaMorph+MPI plugin operators After receiving values for the
 * empty ghost region, each process performs a dot product of their portion of
 * the torus against a matrix of ones, equivalent to a reduction sum.
 * Per-process values are then reduces using MPI_Allreduce
 * For the non-MPI case, the pack and unpack are still performed, but no process
 * communication or all-reduce is performed
 */
#include <metamorph.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#ifdef WITH_MPI
#include <metamorph_mpi.h>
#endif

// Should be set on the compile line, but if not, default to double
#ifdef DOUBLE
/** The global host type to run the test as */
#define G_TYPE double
/** The OpenCL type corresponding to G_TYPE */
#define CL_G_TYPE cl_double
/** The MetaMorph type corresponding to G_TYPE */
#define M_TYPE a_db
/** The MPI type corresponding to G_TYPE */
#define MPI_TYPE MPI_DOUBLE
#elif defined(FLOAT)
/** The global host type to run the test as */
#define G_TYPE float
/** The OpenCL type corresponding to G_TYPE */
#define CL_G_TYPE cl_float
/** The MetaMorph type corresponding to G_TYPE */
#define M_TYPE a_fl
/** The MPI type corresponding to G_TYPE */
#define MPI_TYPE MPI_FLOAT
#elif defined(UNSIGNED_LONG)
/** The global host type to run the test as */
#define G_TYPE unsigned long
/** The OpenCL type corresponding to G_TYPE */
#define CL_G_TYPE cl_ulong
/** The MetaMorph type corresponding to G_TYPE */
#define M_TYPE a_ul
/** The MPI type corresponding to G_TYPE */
#define MPI_TYPE MPI_UNSIGNED_LONG
#elif defined(INTEGER)
/** The global host type to run the test as */
#define G_TYPE int
/** The OpenCL type corresponding to G_TYPE */
#define CL_G_TYPE cl_int
/** The MetaMorph type corresponding to G_TYPE */
#define M_TYPE a_in
/** The MPI type corresponding to G_TYPE */
#define MPI_TYPE MPI_INT
#elif defined(UNISGNED_INTEGER)
/** The global host type to run the test as */
#define G_TYPE unsigned int
/** The OpenCL type corresponding to G_TYPE */
#define CL_G_TYPE cl_uint
/** The MetaMorph type corresponding to G_TYPE */
#define M_TYPE a_ui
/** The MPI type corresponding to G_TYPE */
#define MPI_TYPE MPI_UNSIGNED
#else
/** The global host type to run the test as */
#define G_TYPE double
/** The OpenCL type corresponding to G_TYPE */
#define CL_G_TYPE cl_double
/** The MetaMorph type corresponding to G_TYPE */
#define M_TYPE a_db
/** The MPI type corresponding to G_TYPE */
#define MPI_TYPE MPI_DOUBLE
#endif

/** Sum of a region when all cells are filled by i+j+k (i=0:ni-1, j=0:nj-1,
 * k=0:nk-1)
 * \param a The X dimension of the region
 * \param b The Y dimension of the region
 * \param c The Z dimension of the region
 */
#define SUM_REG(a, b, c) ((((a) * (b) * (c)) / 2.0) * ((a) + (b) + (c)-3))

/** Host side left buffer */
void *domain;
/** Host side right buffer */
void *domain2;

/** Device side left buffer */
void *d_domain;
/** Device side right buffer */
void *d_domain2;
/** Device side result buffer (scalar) */
void *result;
/** Device side packed staging buffer for MPI_Send */
void *d_sendbuf;
/** Device side packed staging buffer for MPI_Recv */
void *d_recvbuf;

/** Number of grid elements in the X dimension (not counting a single
 * ghost-element slab) */
int ni;
/** Number of grid elements in the Y dimension (no ghost elements) */
int nj;
/** Number of grid elements in the Z dimension (no ghost elements) */
int nk;
/** Local reduced value */
G_TYPE r_val;
/** Global reduced value */
G_TYPE global_sum;
/** Reused error value */
a_err err;
/** Reused thread grid struct */
a_dim3 grid;
/** Reused thread block struct */
a_dim3 block;
/** Size of the data arrays */
a_dim3 array;
/** Start offsets in the data arrays */
a_dim3 a_start;
/** End offsets in the data arrays */
a_dim3 a_end;

/** Create two buffers of size (ni+1)*nj*nk */
void data_allocate() {
  domain = malloc(sizeof(G_TYPE) * (ni + 1) * nj * nk);
  domain2 = malloc(sizeof(G_TYPE) * (ni + 1) * nj * nk);
#ifndef USE_UNIFIED_MEMORY
  if (err = meta_alloc(&d_domain, sizeof(G_TYPE) * (ni + 1) * nj * nk))
    fprintf(stderr, "ERROR allocating d_domain: [%d]\n", err);
  if (err = meta_alloc(&d_domain2, sizeof(G_TYPE) * (ni + 1) * nj * nk))
    fprintf(stderr, "ERROR allocating d_domain: [%d]\n", err);
#else
  d_domain = domain;
  d_domain2 = domain2;
#endif
  if (err = meta_alloc(&result, sizeof(G_TYPE)))
    fprintf(stderr, "ERROR allocating result: [%d]\n", err);
  if (err = meta_alloc(&d_sendbuf, sizeof(G_TYPE) * nj * nk))
    fprintf(stderr, "Error allocating d_sendbuf: [%d]\n", err);
  if (err = meta_alloc(&d_recvbuf, sizeof(G_TYPE) * nj * nk))
    fprintf(stderr, "Error allocating d_recvbuf: [%d]\n", err);
}

/** Fill the left data buffer with values corresponding to each cell's global
 * (i+j+k) index, and the right data buffer with ones
 * \param rank MPI rank for computing offset, assumes all ranks have the same
 * size region
 */
void data_initialize(int rank) {
  G_TYPE *l_domain = (G_TYPE *)domain;
  G_TYPE *l_domain2 = (G_TYPE *)domain2;

  int iter;
  int i, j, k;

  for (i = ni; i >= 0; i--) {
    for (j = nj - 1; j >= 0; j--) {
      for (k = nk - 1; k >= 0; k--) {
        if (i == 0) {
          l_domain[i + j * (ni + 1) + k * (ni + 1) * nj] = 0.0f;
        } else {
          l_domain[i + j * (ni + 1) + k * (ni + 1) * nj] =
              i + j + k + (ni * rank);
        }
      }
    }
  }

  for (i = ni; i >= 0; i--) {
    for (j = nj - 1; j >= 0; j--) {
      for (k = nk - 1; k >= 0; k--) {
        l_domain2[i + j * (ni + 1) + k * (ni + 1) * nj] = 1.0f;
      }
    }
  }
}

/** Release both host and device buffers */
void deallocate() {
  free(domain);
  free(domain2);
#ifndef USE_UNIFIED_MEMORY
  meta_free(d_domain);
  meta_free(d_domain2);
#endif
  meta_free(result);
  meta_free(d_sendbuf);
  meta_free(d_recvbuf);
}

#ifdef DEBUG
void print_grid(G_TYPE *grid) {
  int i, j, k;
  for (k = 0; k < nk; k++) {
    for (j = 0; j < nj; j++) {
      for (i = 0; i < ni + 1; i++) {
        printf("[%f] ", grid[i + j * (ni + 1) + k * nj * (ni + 1)]);
      }
      printf("\n");
    }
    printf("\n");
  }
}
#endif

/**
 * The main driver function
 * \param argc The number of command line arguments
 * \param argv The vector of command line argument strings
 * \return a termination status code, zero if success
 */
int main(int argc, char **argv) {
  int rank = 0;
  int comm_sz = 1;
#ifdef WITH_MPI
  MPI_Request request;

  meta_mpi_init(&argc, &argv); // MM
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

  int niters, async, autoconfig;
  if (argc < 7) {
    fprintf(stderr, "<ni> <nj> <nk> <iterations> <async> <autoconfig>\n");
    return (-1);
  }
  ni = atoi(argv[1]);
  nj = atoi(argv[2]);
  nk = atoi(argv[3]);
  niters = atoi(argv[4]);
  async = atoi(argv[5]);
  autoconfig = atoi(argv[6]);

  printf("<%d> <%d> <%d> <%d> <%d> <%d>\n", ni, nj, nk, niters, async,
         autoconfig);

  // Initialization
  // Use environment-based controls
  meta_set_acc(-1, metaModePreferGeneric);
  // metaTimersInit();
  data_allocate();
  data_initialize(rank);

#ifdef DEBUG
  print_grid((G_TYPE *)domain);
#endif

  // The zero face is actually the max face from the far process
  // so compute the sum including the max face, and subtract off the min face
  // that doesn't exist
  G_TYPE check = SUM_REG((ni * comm_sz + 1), nj, nk) - SUM_REG(1, nj, nk);
  int ct;
  struct timeval start, end;
  G_TYPE zero = 0.0;
  block[0] = 4, block[1] = 4, block[2] = 2;
  grid[0] = (ni / block[0]) + ((ni % block[0]) ? 1 : 0), grid[1] = (nj / block[1]) + ((nj % block[1]) ? 1 : 0), grid[2] = (nk / block[2]) + ((nk % block[2]) ? 1 : 0);
  array[0] = ni + 1, array[1] = nj, array[2] = nk;
  a_start[0] = a_start[1] = a_start[2] = 0;
  a_end[0] = ni - 1, a_end[1] = nj - 1, a_end[2] = nk - 1;
  if (autoconfig) printf("Using auto-configured grid and block\n");
  else printf("Using grid:[%lu][%lu][%lu] of blocks:[%lu]{%lu][%lu]\n", grid[0], grid[1], grid[2], block[0], block[1], block[2]);

  // MM: Face specs
  meta_face *send_face, *recv_face;
  send_face = make_slab_from_3d(3, ni + 1, nj, nk, 1);
  recv_face = make_slab_from_3d(2, ni + 1, nj, nk, 1);

  // MM: Data-copy
#ifndef USE_UNIFIED_MEMORY
  if (err = meta_copy_h2d(d_domain, domain, sizeof(G_TYPE) * (ni + 1) * nj * nk,
                          false, NULL, NULL))
    fprintf(stderr, "ERROR Init d_domain failed: [%d]\n", err);
  if (err = meta_copy_h2d(d_domain2, domain2,
                          sizeof(G_TYPE) * (ni + 1) * nj * nk, false, NULL, NULL))
    fprintf(stderr, "ERROR Init dev_d3 failed: [%d]\n", err);
#endif

#ifdef DEBUG
  printf("Post-H2D domain");
#ifndef USE_UNIFIED_MEMORY
  meta_copy_d2h(domain, d_domain, sizeof(G_TYPE) * (ni + 1) * nj * nk, false,
                NULL, NULL);
#endif
  print_grid(domain);
#endif

  gettimeofday(&start, NULL);
  for (ct = niters - 1; ct > -1; ct--) {
    // MM: data marshaling
#ifdef WITH_MPI
    // set up async recv and unpack
    meta_mpi_recv_and_unpack_face(
        autoconfig ? NULL : &grid, autoconfig ? NULL : &block,
        (rank + comm_sz - 1) % comm_sz, recv_face, d_domain, d_recvbuf, ct,
        &request, M_TYPE, 1);
    // pack and send
    meta_mpi_pack_and_send_face(autoconfig ? NULL : &grid,
                                      autoconfig ? NULL : &block,
                                      (rank + 1) % comm_sz, send_face, d_domain,
                                      d_sendbuf, ct, &request, M_TYPE, 0);
#else
    // Non-MPI still needs to exchange the face with itself
    err = meta_pack_face(autoconfig ? NULL : &grid, autoconfig ? NULL : &block,
                         d_sendbuf, d_domain, send_face, M_TYPE, 0, NULL, NULL,
                         NULL, NULL, NULL);
    size_t selfCopySize = sizeof(G_TYPE);
    for (int i = 0; i < send_face->count; i++)
      selfCopySize *= send_face->size[i];
    err = meta_copy_d2d(d_recvbuf, d_sendbuf, selfCopySize, 0, NULL, NULL);
    err = meta_unpack_face(autoconfig ? NULL : &grid,
                           autoconfig ? NULL : &block, d_recvbuf, d_domain,
                           recv_face, M_TYPE, 0, NULL, NULL, NULL, NULL, NULL);
#endif
    // MM flush
    meta_flush();
#ifdef DEBUG
    printf("Pre-reduce domain");
    if (err = meta_alloc(&result, sizeof(G_TYPE)))
      fprintf(stderr, "ERROR allocating result: [%d]\n", err);
#ifndef USE_UNIFIED_MEMORY
    meta_copy_d2h(domain, d_domain, sizeof(G_TYPE) * (ni + 1) * nj * nk, false,
                  NULL, NULL);
#endif
    print_grid(domain);
#endif

    // MM: global dot Product
    meta_copy_h2d(result, &zero, sizeof(G_TYPE), true, NULL, NULL);
    meta_dotProd(autoconfig ? NULL : &grid, autoconfig ? NULL : &block,
                 d_domain, d_domain2, &array, &a_start, &a_end, result, M_TYPE,
                 true, NULL, NULL);
    meta_copy_d2h(&r_val, result, sizeof(G_TYPE), false, NULL, NULL);
#ifdef WITH_MPI
    MPI_Reduce(&r_val, &global_sum, 1, MPI_TYPE, MPI_SUM, 0, MPI_COMM_WORLD);
#else
    global_sum = r_val;
#endif // WITH_MPI
  }
  gettimeofday(&end, NULL);
  double time =
      (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);
  printf("Local partial sum on rank[%d]: [%f]\n", rank, r_val);
  if (rank == 0 && abs(global_sum - check) > 0.000001)
    fprintf(
        stderr,
        "Error, computed dot-product invalid!\n\tExpect: [%f]\tReturn: [%f]\n",
        check, global_sum);
  printf("[%d] Tests completed on rank[%d] with matrices of "
         "size[%d][%d][%d]\n\t[%f] Total time (us)\n\t[%f] Average "
         "time/iteration (us)\n\t[%f] Average time/element (us)\n",
         niters, rank, ni, nj, nk, time, time / niters,
         time / niters / (ni * nj * nk));
  deallocate();

  // metaTimersFinish();

#ifdef WITH_MPI
  meta_mpi_finalize();
#endif
  return 0;
}
