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

libmetamorph_openmp.so:
	$(CC) metamorph_openmp.c $(OPT_LVL) $(CC_FLAGS) $(OPENMP_FLAGS) $(SIMD_CPU_FLAGS) $(L_FLAGS) $(INCLUDES) -L $(MM_LIB) -lmetamorph -o $(MM_LIB)/libmetamorph_openmp.so -shared -Wl,-soname,libmetamorph_openmp.so
	
libmetamorph_openmp_mic.so:
	icc metamorph_openmp_mic.c $(OPT_LVL) $(CC_FLAGS) $(OPENMP_FLAGS) $(SIMD_MIC_FLAGS) $(L_FLAGS) $(INCLUDES) -L $(MM_LIB) -lmetamorph -o $(MM_LIB)/libmetamorph_openmp_mic.so -shared -Wl,-soname,libmetamorph_openmp_mic.so
