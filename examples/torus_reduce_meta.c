#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <metamorph.h>

#define DEBUG2

//Should be set on the compile line, but if not, default to double
#ifdef DOUBLE 
#define G_TYPE double
#define CL_G_TYPE cl_double
#elif defined(FLOAT)
#define G_TYPE float
#define CL_G_TYPE cl_float
#else
#define G_TYPE double
#define CL_G_TYPE cl_double
#endif

//Sum of a region when all cells are filled by i+j+k (i=0:ni-1, j=0:nj-1, k=0:nk-1)
#define SUM_REG(a,b,c) ((((a)*(b)*(c))/2.0)*((a)+(b)+(c)-3))

// host buffers
void *domain, *domain2;

// dev buffers
void *d_domain, *d_domain2, *result, *d_sendbuf, *d_recvbuf;

int ni, nj, nk;
G_TYPE r_val;
G_TYPE global_sum;
a_err err;
a_dim3 grid, block, array, a_start, a_end;

void init(int rank) {
#ifdef WITH_CUDA
	if (rank == 0) meta_set_acc(0, metaModePreferCUDA);
	else fprintf(stderr, "WITH_CUDA should only be defined for rank 0, I have rank [%d]\n", rank);
#elif defined(WITH_OPENCL)
#ifdef DEBUG2
	if (rank ==0) meta_set_acc(-1, metaModePreferOpenCL);
#else
	if (rank ==2) meta_set_acc(0, metaModePreferOpenCL);
#endif
	else fprintf(stderr, "WITH_OPENCL should only be defined for rank 2, I have rank [%d]\n", rank);
#elif defined(WITH_OPENMP)
#ifdef DEBUG2
	if (rank ==0) meta_set_acc(0, metaModePreferOpenMP);
#else
	if (rank == 1 || rank == 3) meta_set_acc(0, metaModePreferOpenMP);
#endif
	else fprintf(stderr, "WITH_OPENMP should only be defined for ranks 1 and 3, I have rank [%d]\n", rank);
#else
#error MetaMorph needs either WITH_CUDA, WITH_OPENCL, or WITH_OPENMP
#endif
}

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

void data_initialize(int rank) {
	G_TYPE *l_domain = (G_TYPE *) domain;
	G_TYPE *l_domain2 = (G_TYPE *) domain2;

	int iter;
	int i, j, k;

	for (i = ni; i >= 0; i--) {
		for (j = nj - 1; j >= 0; j--) {
			for (k = nk - 1; k >= 0; k--) {
				if (i == 0) {
					l_domain[i + j * (ni + 1) + k * (ni + 1) * nj] = 0.0f;
				} else {
					l_domain[i + j * (ni + 1) + k * (ni + 1) * nj] = i + j + k
							+ (ni * rank);
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
void print_grid(G_TYPE * grid) {
	int i,j,k;
	for (k = 0; k < nk; k++) {
		for (j =0; j < nj; j++) {
			for (i=0; i < ni+1; i++) {
				printf("[%f] ", grid[i+ j*(ni+1) + k*nj*(ni+1)]);
			}
			printf("\n");
		}
		printf("\n");
	}
}
#endif

int main(int argc, char **argv) {
	int comm_sz;
	int rank;
	MPI_Request request;

	meta_mpi_init(&argc, &argv); //MM
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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

	//Initialization
	init(rank);
	metaTimersInit();
	data_allocate();
	data_initialize(rank);

#ifdef DEBUG
	print_grid((G_TYPE * ) domain);
#endif

	//The zero face is actually the max face from the far process
	// so compute the sum including the max face, and subtract off the min face that doesn't exist
	G_TYPE check = SUM_REG((ni*comm_sz+1),nj,nk) - SUM_REG(1, nj, nk);
	int ct;
	struct timeval start, end;
	G_TYPE zero = 0.0;
	grid[0] = ni / 4, grid[1] = nj / 4, grid[2] = nk / 2; //Assume powers of 2, for simplicity
	block[0] = 4, block[1] = 4, block[2] = 2;
	array[0] = ni + 1, array[1] = nj, array[2] = nk;
	a_start[0] = a_start[1] = a_start[2] = 0;
	a_end[0] = ni - 1, a_end[1] = nj - 1, a_end[2] = nk - 1;

	// MM: Face specs
	meta_face * send_face, *recv_face;
	send_face = make_slab_from_3d(3, ni + 1, nj, nk, 1);
	recv_face = make_slab_from_3d(2, ni + 1, nj, nk, 1);

	//MM: Data-copy
#ifndef USE_UNIFIED_MEMORY
	if (err = meta_copy_h2d(d_domain, domain, sizeof(G_TYPE) * (ni + 1) * nj * nk, false))
		fprintf(stderr, "ERROR Init d_domain failed: [%d]\n", err);
	if (err = meta_copy_h2d(d_domain2, domain2, sizeof(G_TYPE) * (ni + 1) * nj * nk, false))
		fprintf(stderr, "ERROR Init dev_d3 failed: [%d]\n", err);
#endif

#ifdef DEBUG
	printf("Post-H2D domain");
#ifndef USE_UNIFIED_MEMORY
	meta_copy_d2h(domain, d_domain, sizeof(G_TYPE)*(ni+1)*nj*nk, false);
#endif
	print_grid(domain);
#endif

	gettimeofday(&start, NULL);
	for (ct = niters - 1; ct > -1; ct--) {
		//MM: data marshaling
#if defined(DOUBLE)
		//set up async recv and unpack
		err = meta_mpi_recv_and_unpack_face(autoconfig ? NULL : &grid, autoconfig ? NULL : &block, (rank+comm_sz-1)%comm_sz, recv_face, d_domain, d_recvbuf, ct, &request, a_db, 1);
		//pack and send
		err = meta_mpi_pack_and_send_face(autoconfig ? NULL : &grid, autoconfig ? NULL : &block, (rank+1)%comm_sz, send_face, d_domain, d_sendbuf, ct, &request, a_db, 0);
#elif defined(FLOAT)
		//set up async recv and unpack
		err = meta_mpi_recv_and_unpack_face(autoconfig ? NULL : &grid, autoconfig ? NULL : &block, (rank+comm_sz-1)%comm_sz, recv_face, d_domain, d_recvbuf, ct, &request, a_fl, 1);
		//pack and send
		err = meta_mpi_pack_and_send_face(autoconfig ? NULL : &grid, autoconfig ? NULL : &block, (rank+1)%comm_sz, send_face, d_domain, d_sendbuf, ct, &request, a_fl, 0);
#else
#error Unsupported G_TYPE, must be double or float
#endif
		//MM flush
		meta_flush();

#ifdef DEBUG
		printf("Pre-reduce domain");
		if (err = meta_alloc(&result, sizeof(G_TYPE))) fprintf(stderr, "ERROR allocating result: [%d]\n", err);
#ifndef USE_UNIFIED_MEMORY
		meta_copy_d2h(domain, d_domain, sizeof(G_TYPE)*(ni+1)*nj*nk, false);
#endif
		print_grid(domain);
#endif	

		//MM: global dot Product
#if defined(DOUBLE)
		meta_copy_h2d(result, &zero, sizeof(G_TYPE), true);
		meta_dotProd(autoconfig ? NULL: &grid, autoconfig ? NULL : &block, d_domain, d_domain2, &array, &a_start, &a_end, result, a_db, true);
		meta_copy_d2h(&r_val, result, sizeof(G_TYPE), false);
		MPI_Reduce(&r_val, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#elif defined(FLOAT)
		meta_copy_h2d(result, &zero, sizeof(G_TYPE), true);
		meta_dotProd(autoconfig ? NULL: &grid, autoconfig ? NULL : &block, d_domain, d_domain2, &array, &a_start, &a_end, result, a_fl, true);
		meta_copy_d2h(&r_val, result, sizeof(G_TYPE), true);
		MPI_Reduce(&r_val, &global_sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
#else
#error Unsupported G_TYPE, must be double or float
#endif
	}
	gettimeofday(&end, NULL);
	double time = (end.tv_sec - start.tv_sec) * 1000000.0
			+ (end.tv_usec - start.tv_usec);
	printf("Local partial sum on rank[%d]: [%f]\n", rank, r_val);
	if (rank == 0 && global_sum != check)
		fprintf(stderr,
				"Error, computed dot-product invalid!\n\tExpect: [%f]\tReturn: [%f]\n",
				check, r_val);
	printf(
			"[%d] Tests completed on rank[%d] with matrices of size[%d][%d][%d]\n\t[%f] Total time (us)\n\t[%f] Average time/iteration (us)\n\t[%f] Average time/element (us)\n",
			niters, rank, ni, nj, nk, time, time / niters,
			time / niters / (ni * nj * nk));
	deallocate();

	metaTimersFinish();

	meta_mpi_finalize();
	return 0;
}
