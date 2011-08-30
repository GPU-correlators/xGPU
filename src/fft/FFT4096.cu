// Written by Vasily Volkov.
// Copyright (c) 2008-2009, The Regents of the University of California. 
// All rights reserved.

#include "codelets.h"

__global__ void FFT512_device( float2 *work );

#define rank 8
__global__ void FFT8_device_( float2 *work )
{	
    int tid = threadIdx.x;

    int bid = blockIdx.y * gridDim.x + blockIdx.x;
    int lo = bid & (4096/rank/64-1);
    int hi = bid &~(4096/rank/64-1);

    int i = lo*64 + tid;
    
    work += hi * (rank*64) + i;
    
    float2 a[rank];
    load<rank>( a, work, 512 );
    FFT8( a );
    twiddle<rank>( a, i, 4096 );
    store<rank>( a, work, 512 );
}	

extern "C" void FFT4096( float2 *work, int batch )
{	
    FFT8_device_<<< grid2D(batch*(4096/rank)/64), 64 >>>( work );
    FFT512_device<<< grid2D(batch*rank), 64 >>>( work );
}	
