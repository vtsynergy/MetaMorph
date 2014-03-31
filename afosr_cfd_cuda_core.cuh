#include <cuda_runtime.h>

#ifdef __CUDACC__
extern "C" {
#endif
	cudaError_t cuda_dotProd(size_t (* grid_size)[3], size_t (* block_size)[3], double * data1, double * data2, size_t (* array_size)[3], size_t (* arr_start)[3], size_t (* arr_end)[3], double * reduced_val, int async, cudaEvent_t ((*event)[2])); 
	cudaError_t cuda_reduce(size_t (* grid_size)[3], size_t (* block_size)[3], double * data, size_t (* array_size)[3], size_t (* arr_start)[3], size_t (* arr_end)[3], double * reduced_val, int async, cudaEvent_t (*event)[2]); 
#ifdef __CUDACC__
}
#endif
