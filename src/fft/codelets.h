// Written by Vasily Volkov.
// Copyright (c) 2008-2009, The Regents of the University of California. 
// All rights reserved.

#pragma once
#pragma warning(disable:4996)

#define _USE_MATH_DEFINES
#include <math.h>

//
// arrange blocks into 2D grid that fits into the GPU ( for powers of two only )
//
inline dim3 grid2D( int nblocks )
{
    int slices = 1;
    while( nblocks/slices > 65535 ) 
        slices *= 2;
    return dim3( nblocks/slices, slices );
}

//
// complex number arithmetic
//
inline __device__ float2 operator*( float2 a, float2 b ) { return make_float2( a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x ); }
inline __device__ float2 operator*( float2 a, float  b ) { return make_float2( b*a.x, b*a.y ); }
inline __device__ float2 operator+( float2 a, float2 b ) { return make_float2( a.x + b.x, a.y + b.y ); }
inline __device__ float2 operator-( float2 a, float2 b ) { return make_float2( a.x - b.x, a.y - b.y ); }

#define COS_PI_8  0.923879533f
#define SIN_PI_8  0.382683432f
#define exp_1_16  make_float2(  COS_PI_8, -SIN_PI_8 )
#define exp_3_16  make_float2(  SIN_PI_8, -COS_PI_8 )
#define exp_5_16  make_float2( -SIN_PI_8, -COS_PI_8 )
#define exp_7_16  make_float2( -COS_PI_8, -SIN_PI_8 )
#define exp_9_16  make_float2( -COS_PI_8,  SIN_PI_8 )
#define exp_1_8   make_float2(  1, -1 )//requires post-multiply by 1/sqrt(2)
#define exp_1_4   make_float2(  0, -1 )
#define exp_3_8   make_float2( -1, -1 )//requires post-multiply by 1/sqrt(2)

#define iexp_1_16  make_float2(  COS_PI_8,  SIN_PI_8 )
#define iexp_3_16  make_float2(  SIN_PI_8,  COS_PI_8 )
#define iexp_5_16  make_float2( -SIN_PI_8,  COS_PI_8 )
#define iexp_7_16  make_float2( -COS_PI_8,  SIN_PI_8 )
#define iexp_9_16  make_float2( -COS_PI_8, -SIN_PI_8 )
#define iexp_1_8   make_float2(  1, 1 )//requires post-multiply by 1/sqrt(2)
#define iexp_1_4   make_float2(  0, 1 )
#define iexp_3_8   make_float2( -1, 1 )//requires post-multiply by 1/sqrt(2)

inline __device__ float2 exp_i( float phi )
{
    return make_float2( __cosf(phi), __sinf(phi) );
}

//
//  bit reversal
//
template<int radix> inline __device__ int rev( int bits );

template<> inline __device__ int rev<2>( int bits )
{
    return bits;
}

template<> inline __device__ int rev<4>( int bits )
{
    int reversed[] = {0,2,1,3};
    return reversed[bits];
}

template<> inline __device__ int rev<8>( int bits )
{
    int reversed[] = {0,4,2,6,1,5,3,7};
    return reversed[bits];
}

template<> inline __device__ int rev<16>( int bits )
{
    int reversed[] = {0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15};
    return reversed[bits];
}

inline __device__ int rev4x4( int bits )
{
    int reversed[] = {0,2,1,3, 4,6,5,7, 8,10,9,11, 12,14,13,15};
    return reversed[bits];
}

//
//  all FFTs produce output in bit-reversed order
//
#define IFFT2 FFT2
inline __device__ void FFT2( float2 &a0, float2 &a1 )
{ 
    float2 c0 = a0;
    a0 = c0 + a1; 
    a1 = c0 - a1;
}

inline __device__ void FFT4( float2 &a0, float2 &a1, float2 &a2, float2 &a3 )
{
    FFT2( a0, a2 );
    FFT2( a1, a3 );
    a3 = a3 * exp_1_4;
    FFT2( a0, a1 );
    FFT2( a2, a3 );
}

inline __device__ void IFFT4( float2 &a0, float2 &a1, float2 &a2, float2 &a3 )
{
    IFFT2( a0, a2 );
    IFFT2( a1, a3 );
    a3 = a3 * iexp_1_4;
    IFFT2( a0, a1 );
    IFFT2( a2, a3 );
}

inline __device__ void FFT2( float2 *a ) { FFT2( a[0], a[1] ); }
inline __device__ void FFT4( float2 *a ) { FFT4( a[0], a[1], a[2], a[3] ); }
inline __device__ void IFFT4( float2 *a ) { IFFT4( a[0], a[1], a[2], a[3] ); }

inline __device__ void FFT8( float2 *a )
{
    FFT2( a[0], a[4] );
    FFT2( a[1], a[5] );
    FFT2( a[2], a[6] );
    FFT2( a[3], a[7] );
    
    a[5] = ( a[5] * exp_1_8 ) * M_SQRT1_2;
    a[6] =   a[6] * exp_1_4;
    a[7] = ( a[7] * exp_3_8 ) * M_SQRT1_2;

    FFT4( a[0], a[1], a[2], a[3] );
    FFT4( a[4], a[5], a[6], a[7] );
}

inline __device__ void IFFT8( float2 *a )
{
    IFFT2( a[0], a[4] );
    IFFT2( a[1], a[5] );
    IFFT2( a[2], a[6] );
    IFFT2( a[3], a[7] );
    
    a[5] = ( a[5] * iexp_1_8 ) * M_SQRT1_2;
    a[6] =   a[6] * iexp_1_4;
    a[7] = ( a[7] * iexp_3_8 ) * M_SQRT1_2;

    IFFT4( a[0], a[1], a[2], a[3] );
    IFFT4( a[4], a[5], a[6], a[7] );
}

inline __device__ void FFT16( float2 *a )
{
    FFT4( a[0], a[4], a[8], a[12] );
    FFT4( a[1], a[5], a[9], a[13] );
    FFT4( a[2], a[6], a[10], a[14] );
    FFT4( a[3], a[7], a[11], a[15] );

    a[5]  = (a[5]  * exp_1_8 ) * M_SQRT1_2;
    a[6]  =  a[6]  * exp_1_4;
    a[7]  = (a[7]  * exp_3_8 ) * M_SQRT1_2;
    a[9]  =  a[9]  * exp_1_16;
    a[10] = (a[10] * exp_1_8 ) * M_SQRT1_2;
    a[11] =  a[11] * exp_3_16;
    a[13] =  a[13] * exp_3_16;
    a[14] = (a[14] * exp_3_8 ) * M_SQRT1_2;
    a[15] =  a[15] * exp_9_16;

    FFT4( a[0],  a[1],  a[2],  a[3] );
    FFT4( a[4],  a[5],  a[6],  a[7] );
    FFT4( a[8],  a[9],  a[10], a[11] );
    FFT4( a[12], a[13], a[14], a[15] );
}

inline __device__ void IFFT16( float2 *a )
{
    IFFT4( a[0], a[4], a[8], a[12] );
    IFFT4( a[1], a[5], a[9], a[13] );
    IFFT4( a[2], a[6], a[10], a[14] );
    IFFT4( a[3], a[7], a[11], a[15] );

    a[5]  = (a[5]  * iexp_1_8 ) * M_SQRT1_2;
    a[6]  =  a[6]  * iexp_1_4;
    a[7]  = (a[7]  * iexp_3_8 ) * M_SQRT1_2;
    a[9]  =  a[9]  * iexp_1_16;
    a[10] = (a[10] * iexp_1_8 ) * M_SQRT1_2;
    a[11] =  a[11] * iexp_3_16;
    a[13] =  a[13] * iexp_3_16;
    a[14] = (a[14] * iexp_3_8 ) * M_SQRT1_2;
    a[15] =  a[15] * iexp_9_16;

    IFFT4( a[0],  a[1],  a[2],  a[3] );
    IFFT4( a[4],  a[5],  a[6],  a[7] );
    IFFT4( a[8],  a[9],  a[10], a[11] );
    IFFT4( a[12], a[13], a[14], a[15] );
}

inline __device__ void FFT4x4( float2 *a )
{
    FFT4( a[0],  a[1],  a[2],  a[3] );
    FFT4( a[4],  a[5],  a[6],  a[7] );
    FFT4( a[8],  a[9],  a[10], a[11] );
    FFT4( a[12], a[13], a[14], a[15] );
}

inline __device__ void IFFT4x4( float2 *a )
{
    IFFT2( a[0], a[2] );
    IFFT2( a[1], a[3] );
    IFFT2( a[4], a[6] );
    IFFT2( a[5], a[7] );
    IFFT2( a[8], a[10] );
    IFFT2( a[9], a[11] );
    IFFT2( a[12], a[14] );
    IFFT2( a[13], a[15] );

    a[3] = a[3] * iexp_1_4;
    a[7] = a[7] * iexp_1_4;
    a[11] = a[11] * iexp_1_4;
    a[15] = a[15] * iexp_1_4;

    IFFT2( a[0], a[1] );
    IFFT2( a[2], a[3] );
    IFFT2( a[4], a[5] );
    IFFT2( a[6], a[7] );
    IFFT2( a[8], a[9] );
    IFFT2( a[10], a[11] );
    IFFT2( a[12], a[13] );
    IFFT2( a[14], a[15] );
}

//
//  loads
//
template<int n> inline __device__ void load( float2 *a, float2 *x, int sx )
{
    for( int i = 0; i < n; i++ )
        a[i] = x[i*sx];
}
template<int n> inline __device__ void loadx( float2 *a, float *x, int sx )
{
    for( int i = 0; i < n; i++ )
        a[i].x = x[i*sx];
}
template<int n> inline __device__ void loady( float2 *a, float *x, int sx )
{
    for( int i = 0; i < n; i++ )
        a[i].y = x[i*sx];
}
template<int n> inline __device__ void loadx( float2 *a, float *x, int *ind )
{
    for( int i = 0; i < n; i++ )
        a[i].x = x[ind[i]];
}
template<int n> inline __device__ void loady( float2 *a, float *x, int *ind )
{
    for( int i = 0; i < n; i++ )
        a[i].y = x[ind[i]];
}

//
//  stores, input is in bit reversed order
//
template<int n> inline __device__ void store( float2 *a, float2 *x, int sx )
{
#pragma unroll
    for( int i = 0; i < n; i++ )
        x[i*sx] = a[rev<n>(i)];
}
template<int n> inline __device__ void storex( float2 *a, float *x, int sx )
{
#pragma unroll
    for( int i = 0; i < n; i++ )
        x[i*sx] = a[rev<n>(i)].x;
}
template<int n> inline __device__ void storey( float2 *a, float *x, int sx )
{
#pragma unroll
    for( int i = 0; i < n; i++ )
        x[i*sx] = a[rev<n>(i)].y;
}
inline __device__ void storex4x4( float2 *a, float *x, int sx )
{
#pragma unroll
    for( int i = 0; i < 16; i++ )
        x[i*sx] = a[rev4x4(i)].x;
}
inline __device__ void storey4x4( float2 *a, float *x, int sx )
{
#pragma unroll
    for( int i = 0; i < 16; i++ )
        x[i*sx] = a[rev4x4(i)].y;
}

//
//  multiply by twiddle factors in bit-reversed order
//
template<int radix>inline __device__ void twiddle( float2 *a, int i, int n )
{
#pragma unroll
    for( int j = 1; j < radix; j++ )
        a[j] = a[j] * exp_i((-2*M_PI*rev<radix>(j)/n)*i);
}

template<int radix>inline __device__ void itwiddle( float2 *a, int i, int n )
{
#pragma unroll
    for( int j = 1; j < radix; j++ )
        a[j] = a[j] * exp_i((2*M_PI*rev<radix>(j)/n)*i);
}

inline __device__ void twiddle4x4( float2 *a, int i )
{
    float2 w1 = exp_i((-2*M_PI/32)*i);
    a[1]  = a[1]  * w1;
    a[5]  = a[5]  * w1;
    a[9]  = a[9]  * w1;
    a[13] = a[13] * w1;
    
    float2 w2 = exp_i((-1*M_PI/32)*i);
    a[2]  = a[2]  * w2;
    a[6]  = a[6]  * w2;
    a[10] = a[10] * w2;
    a[14] = a[14] * w2;
    
    float2 w3 = exp_i((-3*M_PI/32)*i);
    a[3]  = a[3]  * w3;
    a[7]  = a[7]  * w3;
    a[11] = a[11] * w3;
    a[15] = a[15] * w3;
}

inline __device__ void itwiddle4x4( float2 *a, int i )
{
    float2 w1 = exp_i((2*M_PI/32)*i);
    a[1]  = a[1]  * w1;
    a[5]  = a[5]  * w1;
    a[9]  = a[9]  * w1;
    a[13] = a[13] * w1;
    
    float2 w2 = exp_i((1*M_PI/32)*i);
    a[2]  = a[2]  * w2;
    a[6]  = a[6]  * w2;
    a[10] = a[10] * w2;
    a[14] = a[14] * w2;
    
    float2 w3 = exp_i((3*M_PI/32)*i);
    a[3]  = a[3]  * w3;
    a[7]  = a[7]  * w3;
    a[11] = a[11] * w3;
    a[15] = a[15] * w3;
}

//
//  multiply by twiddle factors in straight order
//
template<int radix>inline __device__ void twiddle_straight( float2 *a, int i, int n )
{
#pragma unroll
    for( int j = 1; j < radix; j++ )
        a[j] = a[j] * exp_i((-2*M_PI*j/n)*i);
}

template<int radix>inline __device__ void itwiddle_straight( float2 *a, int i, int n )
{
#pragma unroll
    for( int j = 1; j < radix; j++ )
        a[j] = a[j] * exp_i((2*M_PI*j/n)*i);
}

//
//  transpose via shared memory, input is in bit-reversed layout
//
template<int n> inline __device__ void transpose( float2 *a, float *s, int ds, float *l, int dl, int sync = 0xf )
{
    storex<n>( a, s, ds );	if( sync&8 ) __syncthreads();
    loadx<n> ( a, l, dl );	if( sync&4 ) __syncthreads();
    storey<n>( a, s, ds );	if( sync&2 ) __syncthreads();
    loady<n> ( a, l, dl );  if( sync&1 ) __syncthreads();
}

template<int n> inline __device__ void transpose( float2 *a, float *s, int ds, float *l, int *il, int sync = 0xf )
{
    storex<n>( a, s, ds );  if( sync&8 ) __syncthreads();
    loadx<n> ( a, l, il );  if( sync&4 ) __syncthreads();
    storey<n>( a, s, ds );  if( sync&2 ) __syncthreads();
    loady<n> ( a, l, il );  if( sync&1 ) __syncthreads();
}

inline __device__ void transpose4x4( float2 *a, float *s, int ds, float *l, int dl, int sync = 0xf )
{
    storex4x4( a, s, ds );  if( sync&8 ) __syncthreads();
    loadx<16>( a, l, dl );  if( sync&4 ) __syncthreads();
    storey4x4( a, s, ds );  if( sync&2 ) __syncthreads();
    loady<16>( a, l, dl );  if( sync&1 ) __syncthreads();
}
