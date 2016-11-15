CC=gcc
#CC=icc

ifeq ($(CC),gcc)
CC_FLAGS= -fopenmp
SIMD_CPU_FLAGS= -ftree-vectorize -ffast-math -mavx -ftree-vectorizer-verbose=2
else
CC_FLAGS= -openmp
SIMD_CPU_FLAGS= -axavx -fp-model fast #-opt-streaming-stores auto -vec-report3
SIMD_MIC_FLAGS= -mmic -fp-model fast # -vec-report3
#SIMD_MIC_FLAGS= -mmic -no-vec -fp-model fast #-vec-report3 
endif

libmm_openmp_backend.so:
	$(CC) mm_openmp_backend.c $(OPT_LVL) $(CC_FLAGS) $(SIMD_CPU_FLAGS) $(L_FLAGS) $(INCLUDES) -o $(MM_LIB)/libmm_openmp_backend.so
	
libmm_openmp_backend_mic.so:
	icc mm_openmp_backend_mic.c $(OPT_LVL) $(CC_FLAGS) $(SIMD_MIC_FLAGS) $(L_FLAGS) $(INCLUDES) -o $(MM_LIB)/libmm_openmp_backend_mic.so