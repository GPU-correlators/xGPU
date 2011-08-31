// Written by Vasily Volkov.
// Copyright (c) 2008-2009, The Regents of the University of California. 
// All rights reserved.

#include "codelets.h"

__global__ void FFT512_device( float2 *work )
{	
    int tid = threadIdx.x;
    int hi = tid>>3;
    int lo = tid&7;
    
    work += (blockIdx.y * gridDim.x + blockIdx.x) * 512 + tid;
	
    float2 a[8];
    __shared__ float smem[8*8*9];
    
    load<8>( a, work, 64 );

    FFT8( a );
	
    twiddle<8>( a, tid, 512 );
    transpose_br<8>( a, &smem[hi*8+lo], 66, &smem[lo*66+hi], 8 );
	
    FFT8( a );
	
    twiddle<8>( a, hi, 64);
    transpose_br<8>( a, &smem[hi*8+lo], 8*9, &smem[hi*8*9+lo], 8, 0xE );
    
    FFT8( a );

    store<8>( a, work, 64 );
}	
    
extern "C" void FFT512( float2 *work, int batch )
{	
    FFT512_device<<< grid2D(batch), 64 >>>( work );
}	
