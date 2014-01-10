#include <cuda_runtime.h>

#ifdef __CUDACC__
extern "C" {
#endif
 cudaError_t cuda_dotProd_reduce(size_t (* grid_size)[3], size_t (* block_size)[3], double * data1, double * data2, size_t (* array_size)[3], size_t (* arr_start)[3], size_t (* arr_end)[3], double * reduced_val); 
#ifdef __CUDACC__
}
#endif
