// Define TEXTURE_DIM as 1 to use 1D texture (more accurate, costs 1 mult per LOAD)
// Define TEXTURE_DIM as 2 to use 2D texture (less accurate, saves 1 mult per LOAD)
#ifndef TEXTURE_DIM
#define TEXTURE_DIM 1
#endif

#if TEXTURE_DIM == 1

// Read char4 from global, write char4s to shared memory avoid bank conflict.
#define LOAD(s, t)							\
  {char4 real = tex1Dfetch(tex1dchar4, array_index + (t)*NFREQUENCY*NSTATION*NPOL); \
    char4 imag = tex1Dfetch(tex1dchar4, array_index + (NTIME_PIPE/4 + t)*NFREQUENCY*NSTATION*NPOL); \
    CUBE_ADD_BYTES(sizeof(ComplexInput));				\
    *(input##s##_p) = real;						\
    *(input##s##_p + 4*TILE_WIDTH) = imag;}

#else
#error TEXTURE_DIM must be 1
#endif

// read in shared data as individual floats to avoid bank conflicts

#if COMPLEX_BLOCK_SIZE == 1
#define TWO_BY_TWO_PRELOAD(s)						\
 {char4 col1Xreal = input[s][4*tx                                   ];	\
  char4 col1Ximag = input[s][4*tx     + 4*TILE_WIDTH                ];	\
  char4 col1Yreal = input[s][4*tx + 1                               ];	\
  char4 col1Yimag = input[s][4*tx + 1 + 4*TILE_WIDTH                ];	\
  char4 col2Xreal = input[s][4*tx + 2                               ];	\
  char4 col2Ximag = input[s][4*tx + 2 + 4*TILE_WIDTH                ];	\
  char4 col2Yreal = input[s][4*tx + 3                               ];	\
  char4 col2Yimag = input[s][4*tx + 3 + 4*TILE_WIDTH                ];	\
  char4 row1Xreal = input[s][4*ty                     + 8*TILE_WIDTH];	\
  char4 row1Ximag = input[s][4*ty     + 4*TILE_HEIGHT + 8*TILE_WIDTH];	\
  char4 row1Yreal = input[s][4*ty + 1                 + 8*TILE_WIDTH];	\
  char4 row1Yimag = input[s][4*ty + 1 + 4*TILE_HEIGHT + 8*TILE_WIDTH];	\
  char4 row2Xreal = input[s][4*ty + 2                 + 8*TILE_WIDTH];	\
  char4 row2Ximag = input[s][4*ty + 2 + 4*TILE_HEIGHT + 8*TILE_WIDTH];	\
  char4 row2Yreal = input[s][4*ty + 3                 + 8*TILE_WIDTH];	\
  char4 row2Yimag = input[s][4*ty + 3 + 4*TILE_HEIGHT + 8*TILE_WIDTH];
#elif COMPLEX_BLOCK_SIZE == 32
#define TWO_BY_TWO_PRELOAD(s)						\
 {char4 col1Xreal = input[s][2*tx                                  ];	\
  char4 col1Ximag = input[s][2*tx     + 2*TILE_WIDTH               ];	\
  char4 col1Yreal = input[s][2*tx     + 4*TILE_WIDTH               ];	\
  char4 col1Yimag = input[s][2*tx     + 6*TILE_WIDTH               ];	\
  char4 col2Xreal = input[s][2*tx + 1                              ];	\
  char4 col2Yreal = input[s][2*tx + 1 + 4*TILE_WIDTH               ];	\
  char4 col2Ximag = input[s][2*tx + 1 + 2*TILE_WIDTH               ];	\
  char4 col2Yimag = input[s][2*tx + 1 + 6*TILE_WIDTH               ];	\
  char4 row1Xreal = input[s][2*ty                    + 8*TILE_WIDTH];	\
  char4 row1Ximag = input[s][2*ty     + 2*TILE_WIDTH + 8*TILE_WIDTH];	\
  char4 row1Yreal = input[s][2*ty     + 4*TILE_WIDTH + 8*TILE_WIDTH];	\
  char4 row1Yimag = input[s][2*ty     + 6*TILE_WIDTH + 8*TILE_WIDTH];	\
  char4 row2Xreal = input[s][2*ty + 1                + 8*TILE_WIDTH];	\
  char4 row2Yreal = input[s][2*ty + 1 + 4*TILE_WIDTH + 8*TILE_WIDTH];	\
  char4 row2Ximag = input[s][2*ty + 1 + 2*TILE_WIDTH + 8*TILE_WIDTH];	\
  char4 row2Yimag = input[s][2*ty + 1 + 6*TILE_WIDTH + 8*TILE_WIDTH];
#else
#error COMPLEX_BLOCK_SIZE must be 1 or 32
#endif // COMPLEX_BLOCK_SIZE

inline __device__ void dp4a(int &c, const char4 &a, const char4 &b) {
#if __CUDA_ARCH__ >= 610
  int &ai = *((int*)&a);
  int &bi = *((int*)&b);
  asm("dp4a.s32.s32 %0, %1, %2, %0;" : "+r"(c) : "r"(ai), "r"(bi));
#else
  c += a.x*b.x;
  c += a.y*b.y;
  c += a.z*b.z;
  c += a.w*b.w;
#endif
}

inline __device__ void dp4s(int &c, const char4 &a, const char4 &b) {
#if __CUDA_ARCH__ >= 610
  int &ai = *((int*)&a);
  int &bi = *((int*)&b);
  asm("dp4a.s32.s32 %0, %1, %2, %3;" : "+r"(c) : "r"(ai), "r"(bi), "r"(-c));
#else
  c -= a.x*b.x;
  c -= a.y*b.y;
  c -= a.z*b.z;
  c -= a.w*b.w;  
#endif
}

#define TWO_BY_TWO_COMPUTE(s)						\
  TWO_BY_TWO_PRELOAD(s)							\
  dp4a(sum11XXreal,row1Xreal,col1Xreal);				\
  dp4a(sum11XXreal,row1Ximag,col1Ximag);				\
  dp4a(sum11XXimag,row1Ximag,col1Xreal);				\
  dp4s(sum11XXimag,row1Xreal,col1Ximag);				\
  dp4a(sum11XYreal,row1Xreal,col1Yreal);				\
  dp4a(sum11XYreal,row1Ximag,col1Yimag);				\
  dp4a(sum11XYimag,row1Ximag,col1Yreal);				\
  dp4s(sum11XYimag,row1Xreal,col1Yimag);				\
  dp4a(sum11YXreal,row1Yreal,col1Xreal);				\
  dp4a(sum11YXreal,row1Yimag,col1Ximag);				\
  dp4a(sum11YXimag,row1Yimag,col1Xreal);				\
  dp4s(sum11YXimag,row1Yreal,col1Ximag);				\
  dp4a(sum11YYreal,row1Yreal,col1Yreal);				\
  dp4a(sum11YYreal,row1Yimag,col1Yimag);				\
  dp4a(sum11YYimag,row1Yimag,col1Yreal);				\
  dp4s(sum11YYimag,row1Yreal,col1Yimag);				\
  dp4a(sum12XXreal,row1Xreal,col2Xreal);				\
  dp4a(sum12XXreal,row1Ximag,col2Ximag);				\
  dp4a(sum12XXimag,row1Ximag,col2Xreal);				\
  dp4s(sum12XXimag,row1Xreal,col2Ximag);				\
  dp4a(sum12XYreal,row1Xreal,col2Yreal);				\
  dp4a(sum12XYreal,row1Ximag,col2Yimag);				\
  dp4a(sum12XYimag,row1Ximag,col2Yreal);				\
  dp4s(sum12XYimag,row1Xreal,col2Yimag);				\
  dp4a(sum12YXreal,row1Yreal,col2Xreal);				\
  dp4a(sum12YXreal,row1Yimag,col2Ximag);				\
  dp4a(sum12YXimag,row1Yimag,col2Xreal);				\
  dp4s(sum12YXimag,row1Yreal,col2Ximag);				\
  dp4a(sum12YYreal,row1Yreal,col2Yreal);				\
  dp4a(sum12YYreal,row1Yimag,col2Yimag);				\
  dp4a(sum12YYimag,row1Yimag,col2Yreal);				\
  dp4s(sum12YYimag,row1Yreal,col2Yimag);				\
  dp4a(sum21XXreal,row2Xreal,col1Xreal);				\
  dp4a(sum21XXreal,row2Ximag,col1Ximag);				\
  dp4a(sum21XXimag,row2Ximag,col1Xreal);				\
  dp4s(sum21XXimag,row2Xreal,col1Ximag);				\
  dp4a(sum21XYreal,row2Xreal,col1Yreal);				\
  dp4a(sum21XYreal,row2Ximag,col1Yimag);				\
  dp4a(sum21XYimag,row2Ximag,col1Yreal);				\
  dp4s(sum21XYimag,row2Xreal,col1Yimag);				\
  dp4a(sum21YXreal,row2Yreal,col1Xreal);				\
  dp4a(sum21YXreal,row2Yimag,col1Ximag);				\
  dp4a(sum21YXimag,row2Yimag,col1Xreal);				\
  dp4s(sum21YXimag,row2Yreal,col1Ximag);				\
  dp4a(sum21YYreal,row2Yreal,col1Yreal);				\
  dp4a(sum21YYreal,row2Yimag,col1Yimag);				\
  dp4a(sum21YYimag,row2Yimag,col1Yreal);				\
  dp4s(sum21YYimag,row2Yreal,col1Yimag);				\
  dp4a(sum22XXreal,row2Xreal,col2Xreal);				\
  dp4a(sum22XXreal,row2Ximag,col2Ximag);				\
  dp4a(sum22XXimag,row2Ximag,col2Xreal);				\
  dp4s(sum22XXimag,row2Xreal,col2Ximag);				\
  dp4a(sum22XYreal,row2Xreal,col2Yreal);				\
  dp4a(sum22XYreal,row2Ximag,col2Yimag);				\
  dp4a(sum22XYimag,row2Ximag,col2Yreal);				\
  dp4s(sum22XYimag,row2Xreal,col2Yimag);				\
  dp4a(sum22YXreal,row2Yreal,col2Xreal);				\
  dp4a(sum22YXreal,row2Yimag,col2Ximag);				\
  dp4a(sum22YXimag,row2Yimag,col2Xreal);				\
  dp4s(sum22YXimag,row2Yreal,col2Ximag);				\
  dp4a(sum22YYreal,row2Yreal,col2Yreal);				\
  dp4a(sum22YYreal,row2Yimag,col2Yimag);				\
  dp4a(sum22YYimag,row2Yimag,col2Yreal);				\
  dp4s(sum22YYimag,row2Yreal,col2Yimag);}
