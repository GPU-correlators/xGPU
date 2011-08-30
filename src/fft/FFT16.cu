// Written by Vasily Volkov.
// Copyright (c) 2008-2009, The Regents of the University of California. 
// All rights reserved.

#include "codelets.h"

__global__ void FFT16_device( float2 *dst, float2 *src )
{	
    int tid = threadIdx.x;
    
    int iblock = blockIdx.y * gridDim.x + blockIdx.x;
    int index = iblock * 1024 + tid;
    src += index;
    dst += index;
    
    float2 a[16];
    
    load<16>( a, src, 64 );

    FFT16( a );

    store<16>( a, dst, 64 );
}	
    
extern "C" void FFT16( float2 *work, int batch )
{	
    FFT16_device<<< grid2D(batch/64), 64 >>>( work, work );
}	
