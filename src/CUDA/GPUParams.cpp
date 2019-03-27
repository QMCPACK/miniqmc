#include "CUDA/GPUParams.h"


void
Gpu::initCUDAEvents()
{
    //cudaEventCreateWithFlags(&syncEvent, cudaEventDisableTiming);
    //cudaEventCreateWithFlags(&gradientSyncDiracEvent, cudaEventDisableTiming);
    //cudaEventCreateWithFlags(&gradientSyncOneBodyEvent, cudaEventDisableTiming);
    //cudaEventCreateWithFlags(&gradientSyncTwoBodyEvent, cudaEventDisableTiming);
    //cudaEventCreateWithFlags(&ratioSyncDiracEvent, cudaEventDisableTiming);
    //cudaEventCreateWithFlags(&ratioSyncOneBodyEvent, cudaEventDisableTiming);
    //cudaEventCreateWithFlags(&ratioSyncTwoBodyEvent, cudaEventDisableTiming);
}

void
Gpu::initCublas()
{
    //cublasCreate(&cublasHandle);
}

void
Gpu::finalizeCUDAStreams()
{
    //cudaStreamDestroy(kernelStream);
    //cudaStreamDestroy(memoryStream);
}

void
Gpu::finalizeCUDAEvents()
{
    //cudaEventDestroy(syncEvent);
    //cudaEventDestroy(gradientSyncDiracEvent);
    //cudaEventDestroy(gradientSyncOneBodyEvent);
    //  cudaEventDestroy(gradientSyncTwoBodyEvent);
    //cudaEventDestroy(ratioSyncDiracEvent);
    //cudaEventDestroy(ratioSyncOneBodyEvent);
    //cudaEventDestroy(ratioSyncTwoBodyEvent);
}

void
Gpu::finalizeCublas()
{
    //cublasDestroy(cublasHandle);
}

void
Gpu::synchronize()
{
    //cudaDeviceSynchronize();
}

void
Gpu::streamsSynchronize()
{
    //cudaEventRecord(syncEvent, 0);
}


