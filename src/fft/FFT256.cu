// Written by Vasily Volkov.
// Copyright (c) 2008-2009, The Regents of the University of California. 
// All rights reserved.

#include "codelets.h"

__global__ void FFT256_device( float2 *dst, float2 *src )
{	
    int tid = threadIdx.x;
    int hi = tid>>4;
    int lo = tid&15;
    
    int index = (blockIdx.y * gridDim.x + blockIdx.x) * 1024 + lo + hi*256;
    src += index;
    dst += index;
	
    //
    //  no sync in transpose is needed here if warpSize >= 32
    //  since the permutations are within-warp
    //
    
    float2 a[16];
    __shared__ float smem[64*17];
    
    load<16>( a, src, 16 );

    FFT16( a );
    
    twiddle<16>( a, lo, 256 );
    transpose<16>( a, &smem[hi*17*16 + 17*lo], 1, &smem[hi*17*16+lo], 17, 0 );
    
    FFT16( a );

    store<16>( a, dst, 16 );
}	
    
extern "C" void FFT256( float2 *work, int batch )
{	
    FFT256_device<<< grid2D(batch/4), 64 >>>( work, work );
}	
