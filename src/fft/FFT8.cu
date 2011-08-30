// Written by Vasily Volkov.
// Copyright (c) 2008-2009, The Regents of the University of California. 
// All rights reserved.

#include "codelets.h"

__global__ void FFT8_device( float2 *work )
{	
    int tid = threadIdx.x;

    work += (blockIdx.y * gridDim.x + blockIdx.x) * 512 + tid;
    
    float2 a[8];
  
    load<8>( a, work, 64 );

    FFT8( a );

    store<8>( a, work, 64 );
}	
    
extern "C" void FFT8( float2 *work, int batch )
{	
    FFT8_device<<< grid2D(batch/64), 64 >>>( work );
}	
