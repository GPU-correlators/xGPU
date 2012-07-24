// Define TEXTURE_DIM as 1 to use 1D texture (more accurate, costs 1 mult per LOAD)
// Define TEXTURE_DIM as 2 to use 2D texture (less accurate, saves 1 mult per LOAD)
#ifndef TEXTURE_DIM
#define TEXTURE_DIM 1
#endif

#if TEXTURE_DIM == 1

// Read float2 from global, write individual floats
// to shared memory avoid bank conflict.


#if MULTIPLY_MODE == 0 || MULTIPLY_MODE == 1
#define LOAD(s, t)							\
  {float2 temp = tex1Dfetch(tex1dfloat2, array_index + (t)*NFREQUENCY*Nstation*NPOL);			\
    CUBE_ADD_BYTES(sizeof(ComplexInput));				\
    *(input##s##_p) = temp.x;						\
    *(input##s##_p + 4*TILE_WIDTH) = temp.y;}
#elif MULTIPLY_MODE == 2
#define LOAD(s, t)							\
  {float2 temp = tex1Dfetch(tex1dfloat2, array_index + (t)*NFREQUENCY*Nstation*NPOL);			\
    CUBE_ADD_BYTES(sizeof(ComplexInput));				\
    *(input##s##_p) = temp.x;						\
    *(input##s##_p + 4*TILE_WIDTH) = temp.y;}
#endif



#elif TEXTURE_DIM == 2


// Read float2 from global, write individual floats
// to shared memory avoid bank conflict.
#define LOAD(s, t)							\
  {float2 temp = tex2D(tex2dfloat2, array_index, t);			\
    CUBE_ADD_BYTES(sizeof(ComplexInput));				\
    *(input##s##_p) = temp.x;						\
    *(input##s##_p + 4*TILE_WIDTH) = temp.y;}

#else
#error TEXTURE_DIM must be 1 or 2
#endif



#if MULTIPLY_MODE == 0

// read in shared data as individual floats to avoid bank conflicts

#if COMPLEX_BLOCK_SIZE == 1
#define TWO_BY_TWO_PRELOAD(s)						\
 {float col1Xreal = input[s][4*tx                                   ];	\
  float col1Ximag = input[s][4*tx     + 4*TILE_WIDTH                ];	\
  float col1Yreal = input[s][4*tx + 1                               ];	\
  float col1Yimag = input[s][4*tx + 1 + 4*TILE_WIDTH                ];	\
  float col2Xreal = input[s][4*tx + 2                               ];	\
  float col2Ximag = input[s][4*tx + 2 + 4*TILE_WIDTH                ];	\
  float col2Yreal = input[s][4*tx + 3                               ];	\
  float col2Yimag = input[s][4*tx + 3 + 4*TILE_WIDTH                ];	\
  float row1Xreal = input[s][4*ty                     + 8*TILE_WIDTH];	\
  float row1Ximag = input[s][4*ty     + 4*TILE_HEIGHT + 8*TILE_WIDTH];	\
  float row1Yreal = input[s][4*ty + 1                 + 8*TILE_WIDTH];	\
  float row1Yimag = input[s][4*ty + 1 + 4*TILE_HEIGHT + 8*TILE_WIDTH];	\
  float row2Xreal = input[s][4*ty + 2                 + 8*TILE_WIDTH];	\
  float row2Ximag = input[s][4*ty + 2 + 4*TILE_HEIGHT + 8*TILE_WIDTH];	\
  float row2Yreal = input[s][4*ty + 3                 + 8*TILE_WIDTH];	\
  float row2Yimag = input[s][4*ty + 3 + 4*TILE_HEIGHT + 8*TILE_WIDTH];
#elif COMPLEX_BLOCK_SIZE == 32
#define TWO_BY_TWO_PRELOAD(s)						\
 {float col1Xreal = input[s][2*tx                                  ];	\
  float col1Ximag = input[s][2*tx     + 2*TILE_WIDTH               ];	\
  float col1Yreal = input[s][2*tx     + 4*TILE_WIDTH               ];	\
  float col1Yimag = input[s][2*tx     + 6*TILE_WIDTH               ];	\
  float col2Xreal = input[s][2*tx + 1                              ];	\
  float col2Yreal = input[s][2*tx + 1 + 4*TILE_WIDTH               ];	\
  float col2Ximag = input[s][2*tx + 1 + 2*TILE_WIDTH               ];	\
  float col2Yimag = input[s][2*tx + 1 + 6*TILE_WIDTH               ];	\
  float row1Xreal = input[s][2*ty                    + 8*TILE_WIDTH];	\
  float row1Ximag = input[s][2*ty     + 2*TILE_WIDTH + 8*TILE_WIDTH];	\
  float row1Yreal = input[s][2*ty     + 4*TILE_WIDTH + 8*TILE_WIDTH];	\
  float row1Yimag = input[s][2*ty     + 6*TILE_WIDTH + 8*TILE_WIDTH];	\
  float row2Xreal = input[s][2*ty + 1                + 8*TILE_WIDTH];	\
  float row2Yreal = input[s][2*ty + 1 + 4*TILE_WIDTH + 8*TILE_WIDTH];	\
  float row2Ximag = input[s][2*ty + 1 + 2*TILE_WIDTH + 8*TILE_WIDTH];	\
  float row2Yimag = input[s][2*ty + 1 + 6*TILE_WIDTH + 8*TILE_WIDTH];
#else
#error COMPLEX_BLOCK_SIZE must be 1 or 32
#endif // COMPLEX_BLOCK_SIZE

#define TWO_BY_TWO_COMPUTE(s)						\
  TWO_BY_TWO_PRELOAD(s)							\
  sum11XXreal += row1Xreal * col1Xreal;					\
  sum11XXreal += row1Ximag * col1Ximag;					\
  sum11XXimag += row1Ximag * col1Xreal;					\
  sum11XXimag -= row1Xreal * col1Ximag;					\
  sum11XYreal += row1Xreal * col1Yreal;					\
  sum11XYreal += row1Ximag * col1Yimag;					\
  sum11XYimag += row1Ximag * col1Yreal;					\
  sum11XYimag -= row1Xreal * col1Yimag;					\
  sum11YXreal += row1Yreal * col1Xreal;					\
  sum11YXreal += row1Yimag * col1Ximag;					\
  sum11YXimag += row1Yimag * col1Xreal;					\
  sum11YXimag -= row1Yreal * col1Ximag;					\
  sum11YYreal += row1Yreal * col1Yreal;					\
  sum11YYreal += row1Yimag * col1Yimag;					\
  sum11YYimag += row1Yimag * col1Yreal;					\
  sum11YYimag -= row1Yreal * col1Yimag;					\
  sum12XXreal += row1Xreal * col2Xreal;					\
  sum12XXreal += row1Ximag * col2Ximag;					\
  sum12XXimag += row1Ximag * col2Xreal;					\
  sum12XXimag -= row1Xreal * col2Ximag;					\
  sum12XYreal += row1Xreal * col2Yreal;					\
  sum12XYreal += row1Ximag * col2Yimag;					\
  sum12XYimag += row1Ximag * col2Yreal;					\
  sum12XYimag -= row1Xreal * col2Yimag;					\
  sum12YXreal += row1Yreal * col2Xreal;					\
  sum12YXreal += row1Yimag * col2Ximag;					\
  sum12YXimag += row1Yimag * col2Xreal;					\
  sum12YXimag -= row1Yreal * col2Ximag;					\
  sum12YYreal += row1Yreal * col2Yreal;					\
  sum12YYreal += row1Yimag * col2Yimag;					\
  sum12YYimag += row1Yimag * col2Yreal;					\
  sum12YYimag -= row1Yreal * col2Yimag;					\
  sum21XXreal += row2Xreal * col1Xreal;					\
  sum21XXreal += row2Ximag * col1Ximag;					\
  sum21XXimag += row2Ximag * col1Xreal;					\
  sum21XXimag -= row2Xreal * col1Ximag;					\
  sum21XYreal += row2Xreal * col1Yreal;					\
  sum21XYreal += row2Ximag * col1Yimag;					\
  sum21XYimag += row2Ximag * col1Yreal;					\
  sum21XYimag -= row2Xreal * col1Yimag;					\
  sum21YXreal += row2Yreal * col1Xreal;					\
  sum21YXreal += row2Yimag * col1Ximag;					\
  sum21YXimag += row2Yimag * col1Xreal;					\
  sum21YXimag -= row2Yreal * col1Ximag;					\
  sum21YYreal += row2Yreal * col1Yreal;					\
  sum21YYreal += row2Yimag * col1Yimag;					\
  sum21YYimag += row2Yimag * col1Yreal;					\
  sum21YYimag -= row2Yreal * col1Yimag;					\
  sum22XXreal += row2Xreal * col2Xreal;					\
  sum22XXreal += row2Ximag * col2Ximag;					\
  sum22XXimag += row2Ximag * col2Xreal;					\
  sum22XXimag -= row2Xreal * col2Ximag;					\
  sum22XYreal += row2Xreal * col2Yreal;					\
  sum22XYreal += row2Ximag * col2Yimag;					\
  sum22XYimag += row2Ximag * col2Yreal;					\
  sum22XYimag -= row2Xreal * col2Yimag;					\
  sum22YXreal += row2Yreal * col2Xreal;					\
  sum22YXreal += row2Yimag * col2Ximag;					\
  sum22YXimag += row2Yimag * col2Xreal;					\
  sum22YXimag -= row2Yreal * col2Ximag;					\
  sum22YYreal += row2Yreal * col2Yreal;					\
  sum22YYreal += row2Yimag * col2Yimag;					\
  sum22YYimag += row2Yimag * col2Yreal;					\
  sum22YYimag -= row2Yreal * col2Yimag;}











#elif MULTIPLY_MODE == 1
// read in shared data as individual floats to avoid bank conflicts

#if COMPLEX_BLOCK_SIZE == 1
#define TWO_BY_TWO_PRELOAD(s)						\
 {float col1Xreal = input[s][4*tx                                   ];	\
  float col1Ximag = input[s][4*tx     + 4*TILE_WIDTH                ];	\
  float col1Yreal = input[s][4*tx + 1                               ];	\
  float col1Yimag = input[s][4*tx + 1 + 4*TILE_WIDTH                ];	\
  float col2Xreal = input[s][4*tx + 2                               ];	\
  float col2Ximag = input[s][4*tx + 2 + 4*TILE_WIDTH                ];	\
  float col2Yreal = input[s][4*tx + 3                               ];	\
  float col2Yimag = input[s][4*tx + 3 + 4*TILE_WIDTH                ];	\
  float row1Xreal = input[s][4*ty                     + 8*TILE_WIDTH];	\
  float row1Ximag = input[s][4*ty     + 4*TILE_HEIGHT + 8*TILE_WIDTH];	\
  float row1Yreal = input[s][4*ty + 1                 + 8*TILE_WIDTH];	\
  float row1Yimag = input[s][4*ty + 1 + 4*TILE_HEIGHT + 8*TILE_WIDTH];	\
  float row2Xreal = input[s][4*ty + 2                 + 8*TILE_WIDTH];	\
  float row2Ximag = input[s][4*ty + 2 + 4*TILE_HEIGHT + 8*TILE_WIDTH];	\
  float row2Yreal = input[s][4*ty + 3                 + 8*TILE_WIDTH];	\
  float row2Yimag = input[s][4*ty + 3 + 4*TILE_HEIGHT + 8*TILE_WIDTH];  \
  float p1_1X = row1Xreal + row1Ximag;					\
  float p2_1X = 0 - col1Ximag - col1Xreal;				\
  float p3_1X = col1Xreal - col1Ximag;					\
  float p1_1Y = row1Yreal + row1Yimag;					\
  float p2_1Y = 0 - col1Yimag - col1Yreal;				\
  float p3_1Y = col1Yreal - col1Yimag;					\
  float p1_2X = row2Xreal + row2Ximag;					\
  float p2_2X = 0 - col2Ximag - col2Xreal;				\
  float p3_2X = col2Xreal - col2Ximag;					\
  float p1_2Y = row2Yreal + row2Yimag;					\
  float p2_2Y = 0 - col2Yimag - col2Yreal;				\
  float p3_2Y = col2Yreal - col2Yimag;					



#elif COMPLEX_BLOCK_SIZE == 32
#define TWO_BY_TWO_PRELOAD(s)						\
 {float col1Xreal = input[s][2*tx                                  ];	\
  float col1Ximag = input[s][2*tx     + 2*TILE_WIDTH               ];	\
  float col1Yreal = input[s][2*tx     + 4*TILE_WIDTH               ];	\
  float col1Yimag = input[s][2*tx     + 6*TILE_WIDTH               ];	\
  float col2Xreal = input[s][2*tx + 1                              ];	\
  float col2Yreal = input[s][2*tx + 1 + 4*TILE_WIDTH               ];	\
  float col2Ximag = input[s][2*tx + 1 + 2*TILE_WIDTH               ];	\
  float col2Yimag = input[s][2*tx + 1 + 6*TILE_WIDTH               ];	\
  float row1Xreal = input[s][2*ty                    + 8*TILE_WIDTH];	\
  float row1Ximag = input[s][2*ty     + 2*TILE_WIDTH + 8*TILE_WIDTH];	\
  float row1Yreal = input[s][2*ty     + 4*TILE_WIDTH + 8*TILE_WIDTH];	\
  float row1Yimag = input[s][2*ty     + 6*TILE_WIDTH + 8*TILE_WIDTH];	\
  float row2Xreal = input[s][2*ty + 1                + 8*TILE_WIDTH];	\
  float row2Yreal = input[s][2*ty + 1 + 4*TILE_WIDTH + 8*TILE_WIDTH];	\
  float row2Ximag = input[s][2*ty + 1 + 2*TILE_WIDTH + 8*TILE_WIDTH];	\
  float row2Yimag = input[s][2*ty + 1 + 6*TILE_WIDTH + 8*TILE_WIDTH];	\

#else

#error COMPLEX_BLOCK_SIZE must be 1 or 32
#endif // COMPLEX_BLOCK_SIZE
#define TWO_BY_TWO_COMPUTE(s)			      \
  TWO_BY_TWO_PRELOAD(s)                               \
  sum11XXk1 += p1_1X * col1Xreal;                     \
  sum11XXk2 += p2_1X * row1Xreal;                     \
  sum11XXk3 += p3_1X * row1Ximag;                     \
  sum11XYk1 += p1_1X * col1Yreal;                     \
  sum11XYk2 += p2_1Y * row1Xreal;                     \
  sum11XYk3 += p3_1Y * row1Ximag;                     \
  sum11YXk1 += p1_1Y * col1Xreal;                     \
  sum11YXk2 += p2_1X * row1Yreal;                     \
  sum11YXk3 += p3_1X * row1Yimag;                     \
  sum11YYk1 += p1_1Y * col1Yreal;                     \
  sum11YYk2 += p2_1Y * row1Yreal;                     \
  sum11YYk3 += p3_1Y * row1Yimag;                     \
  sum12XXk1 += p1_1X * col2Xreal;                     \
  sum12XXk2 += p2_2X * row1Xreal;                     \
  sum12XXk3 += p3_2X * row1Ximag;                     \
  sum12XYk1 += p1_1X * col2Yreal;                     \
  sum12XYk2 += p2_2Y * row1Xreal;                     \
  sum12XYk3 += p3_2Y * row1Ximag;                     \
  sum12YXk1 += p1_1Y * col2Xreal;                     \
  sum12YXk2 += p2_2X * row1Yreal;                     \
  sum12YXk3 += p3_2X * row1Yimag;                     \
  sum12YYk1 += p1_1Y * col2Yreal;                     \
  sum12YYk2 += p2_2Y * row1Yreal;                     \
  sum12YYk3 += p3_2Y * row1Yimag;                     \
  sum21XXk1 += p1_2X * col1Xreal;                     \
  sum21XXk2 += p2_1X * row2Xreal;                     \
  sum21XXk3 += p3_1X * row2Ximag;                     \
  sum21XYk1 += p1_2X * col1Yreal;                     \
  sum21XYk2 += p2_1Y * row2Xreal;                     \
  sum21XYk3 += p3_1Y * row2Ximag;                     \
  sum21YXk1 += p1_2Y * col1Xreal;                     \
  sum21YXk2 += p2_1X * row2Yreal;                     \
  sum21YXk3 += p3_1X * row2Yimag;                     \
  sum21YYk1 += p1_2Y * col1Yreal;                     \
  sum21YYk2 += p2_1Y * row2Yreal;                     \
  sum21YYk3 += p3_1Y * row2Yimag;                     \
  sum22XXk1 += p1_2X * col2Xreal;                     \
  sum22XXk2 += p2_2X * row2Xreal;                     \
  sum22XXk3 += p3_2X * row2Ximag;                     \
  sum22XYk1 += p1_2X * col2Yreal;                     \
  sum22XYk2 += p2_2Y * row2Xreal;                     \
  sum22XYk3 += p3_2Y * row2Ximag;                     \
  sum22YXk1 += p1_2Y * col2Xreal;                     \
  sum22YXk2 += p2_2X * row2Yreal;                     \
  sum22YXk3 += p3_2X * row2Yimag;                     \
  sum22YYk1 += p1_2Y * col2Yreal;                     \
  sum22YYk2 += p2_2Y * row2Yreal;                     \
  sum22YYk3 += p3_2Y * row2Yimag;}

















#elif MULTIPLY_MODE == 2
// read in shared data as individual floats to avoid bank conflicts

// [col/row][c][rt][p] -> [col/row][g][rt][p]

#if COMPLEX_BLOCK_SIZE == 1
#define TWO_BY_TWO_PRELOAD(s)						\
 {float col1Xreal = input[s][4*tx                                   ];	\
  float col1Ximag = input[s][4*tx     + 4*TILE_WIDTH                ];	\
  float col1Yreal = input[s][4*tx + 1                               ];	\
  float col1Yimag = input[s][4*tx + 1 + 4*TILE_WIDTH                ];	\
  float col2Xreal = input[s][4*tx + 2                               ];	\
  float col2Ximag = input[s][4*tx + 2 + 4*TILE_WIDTH                ];	\
  float col2Yreal = input[s][4*tx + 3                               ];	\
  float col2Yimag = input[s][4*tx + 3 + 4*TILE_WIDTH                ];	\
/*
	input[s][(((z*2 + c)*8 + t)*2 + r)*2 + p];

	s = shared mem buffer (0/1)
	z = row / col
	c = complexity (real / imag)
	t = thread idx (0..7) (col has tx, row has ty)
	r = register tile coord (1 / 2)
	p = polarization (X / Y)
*/
  float row1Xreal = input[s][4*ty                     + 8*TILE_WIDTH];	\
  float row1Ximag = input[s][4*ty     + 4*TILE_HEIGHT + 8*TILE_WIDTH];	\
  float row1Yreal = input[s][4*ty + 1                 + 8*TILE_WIDTH];	\
  float row1Yimag = input[s][4*ty + 1 + 4*TILE_HEIGHT + 8*TILE_WIDTH];	\
  float row2Xreal = input[s][4*ty + 2                 + 8*TILE_WIDTH];	\
  float row2Ximag = input[s][4*ty + 2 + 4*TILE_HEIGHT + 8*TILE_WIDTH];	\
  float row2Yreal = input[s][4*ty + 3                 + 8*TILE_WIDTH];	\
  float row2Yimag = input[s][4*ty + 3 + 4*TILE_HEIGHT + 8*TILE_WIDTH];

//Include p1_1X, p1_1Y, p1_2X, p1_2Y + p2... + p3...


#elif COMPLEX_BLOCK_SIZE == 32
#define TWO_BY_TWO_PRELOAD(s)						\
 {float col1Xreal = input[s][2*tx                                  ];	\
  float col1Ximag = input[s][2*tx     + 2*TILE_WIDTH               ];	\
  float col1Yreal = input[s][2*tx     + 4*TILE_WIDTH               ];	\
  float col1Yimag = input[s][2*tx     + 6*TILE_WIDTH               ];	\
  float col2Xreal = input[s][2*tx + 1                              ];	\
  float col2Yreal = input[s][2*tx + 1 + 4*TILE_WIDTH               ];	\
  float col2Ximag = input[s][2*tx + 1 + 2*TILE_WIDTH               ];	\
  float col2Yimag = input[s][2*tx + 1 + 6*TILE_WIDTH               ];	\
  float row1Xreal = input[s][2*ty                    + 8*TILE_WIDTH];	\
  float row1Ximag = input[s][2*ty     + 2*TILE_WIDTH + 8*TILE_WIDTH];	\
  float row1Yreal = input[s][2*ty     + 4*TILE_WIDTH + 8*TILE_WIDTH];	\
  float row1Yimag = input[s][2*ty     + 6*TILE_WIDTH + 8*TILE_WIDTH];	\
  float row2Xreal = input[s][2*ty + 1                + 8*TILE_WIDTH];	\
  float row2Yreal = input[s][2*ty + 1 + 4*TILE_WIDTH + 8*TILE_WIDTH];	\
  float row2Ximag = input[s][2*ty + 1 + 2*TILE_WIDTH + 8*TILE_WIDTH];	\
  float row2Yimag = input[s][2*ty + 1 + 6*TILE_WIDTH + 8*TILE_WIDTH];
#else



#error COMPLEX_BLOCK_SIZE must be 1 or 32
#endif // COMPLEX_BLOCK_SIZE



#define TWO_BY_TWO_COMPUTE(s)						\
  TWO_BY_TWO_PRELOAD(s)                                                 \
  sum11XXk1 += p1_1X * col1Xreal;                     \
  sum11XXk2 += p2_1X * row1Xreal;                     \
  sum11XXk3 += p3_1X * row1Ximag;                     \
  sum11XYk1 += p1_1X * col1Yreal;                     \
  sum11XYk2 += p2_1Y * row1Xreal;                     \
  sum11XYk3 += p3_1Y * row1Ximag;                     \
  sum11YXk1 += p1_1Y * col1Xreal;                     \
  sum11YXk2 += p2_1X * row1Yreal;                     \
  sum11YXk3 += p3_1X * row1Yimag;                     \
  sum11YYk1 += p1_1Y * col1Yreal;                     \
  sum11YYk2 += p2_1Y * row1Yreal;                     \
  sum11YYk3 += p3_1Y * row1Yimag;                     \
  sum12XXk1 += p1_1X * col2Xreal;                     \
  sum12XXk2 += p2_2X * row1Xreal;                     \
  sum12XXk3 += p3_2X * row1Ximag;                     \
  sum12XYk1 += p1_1X * col2Yreal;                     \
  sum12XYk2 += p2_2Y * row1Xreal;                     \
  sum12XYk3 += p3_2Y * row1Ximag;                     \
  sum12YXk1 += p1_1Y * col2Xreal;                     \
  sum12YXk2 += p2_2X * row1Yreal;                     \
  sum12YXk3 += p3_2X * row1Yimag;                     \
  sum12YYk1 += p1_1Y * col2Yreal;                     \
  sum12YYk2 += p2_2Y * row1Yreal;                     \
  sum12YYk3 += p3_2Y * row1Yimag;                     \
  sum21XXk1 += p1_2X * col1Xreal;                     \
  sum21XXk2 += p2_1X * row2Xreal;                     \
  sum21XXk3 += p3_1X * row2Ximag;                     \
  sum21XYk1 += p1_2X * col1Yreal;                     \
  sum21XYk2 += p2_1Y * row2Xreal;                     \
  sum21XYk3 += p3_1Y * row2Ximag;                     \
  sum21YXk1 += p1_2Y * col1Xreal;                     \
  sum21YXk2 += p2_1X * row2Yreal;                     \
  sum21YXk3 += p3_1X * row2Yimag;                     \
  sum21YYk1 += p1_2Y * col1Yreal;                     \
  sum21YYk2 += p2_1Y * row2Yreal;                     \
  sum21YYk3 += p3_1Y * row2Yimag;                     \
  sum22XXk1 += p1_2X * col2Xreal;                     \
  sum22XXk2 += p2_2X * row2Xreal;                     \
  sum22XXk3 += p3_2X * row2Ximag;                     \
  sum22XYk1 += p1_2X * col2Yreal;                     \
  sum22XYk2 += p2_2Y * row2Xreal;                     \
  sum22XYk3 += p3_2Y * row2Ximag;                     \
  sum22YXk1 += p1_2Y * col2Xreal;                     \
  sum22YXk2 += p2_2X * row2Yreal;                     \
  sum22YXk3 += p3_2X * row2Yimag;                     \
  sum22YYk1 += p1_2Y * col2Yreal;                     \
  sum22YYk2 += p2_2Y * row2Yreal;                     \
  sum22YYk3 += p3_2Y * row2Yimag;}

#endif



