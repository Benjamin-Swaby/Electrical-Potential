#include <kernel.cuh>
#include <chrono>
#include <iostream>

using namespace bodies;

const double k = 8.99e9;


cudaDeviceProp getDetails(int deviceId)
{
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);
    return props;
}


void PrintProps(cudaDeviceProp props) {
    std::cout << "Device: " << props.name << std::endl;
    std::cout << "\tSMs: " << props.multiProcessorCount << std::endl;
    std::cout << "\tMax Blocks Per SM: " << props.maxBlocksPerMultiProcessor << std::endl;
    std::cout << "\tMax Threads Per SM: " << props.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "\tMax Threads Per Block: " << props.maxThreadsPerBlock << std::endl;
    std::cout << "\tWarpSize: " << props.warpSize << std::endl;
    

}


__device__ double getMagnitudeDistance(bodies::pVec p, bodies::pVec p2) {

    double dx = p.x - p2.x;
    double dy = p.y - p2.y;
    double dz = p.z - p2.z;

    double magnitude = sqrt((dx * dx) + (dy * dy) + (dz * dz));  // |x| = sqrt(x^2 + y^2 + z^2)
    return magnitude;
}

__global__ void calculatePotentials(bodies::pVec *point, bodies::body *particles, double *Ve, int N, int Np) {
    
    int index =  blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    double total;

    for (int i = index; i < Np; i+= stride) {
        total = 0.0;
        
        for (int j = 0; j < N; j++) {
            double r = getMagnitudeDistance(point[index], particles[i].position);
            total += k * (particles[i].charge / r);
        }
        
        Ve[i] = total;
    }
    
}


double hostGetMagnitudeDistance(bodies::pVec p, bodies::pVec p2) {
    double dx = p.x - p2.x;
    double dy = p.y - p2.y;
    double dz = p.z - p2.z;

    double magnitude = sqrt((dx * dx) + (dy * dy) + (dz * dz));  // |x| = sqrt(x^2 + y^2 + z^2)
    return magnitude;
}

double* kernel::CPUlaunch(bodies::pVec *points, bodies::body *particles, int N, int Np) {

    double r;
    double total = 0.0f;

    auto Ve = new double[Np];

    for (int j = 0; j < Np; j++) {
        total = 0.0;
        for (int i = 0; i < N; i++) {
            r = hostGetMagnitudeDistance(points[j], particles->position);
            total += k * (particles[i].charge / r);
            
        }
        Ve[j] = total;
    }


    /**
    printf("CPU Result = \n");
    for (int i = 0; i < Np; i++) {
        printf("%ld,", Ve[i]);
    }
    printf("\n");
    **/ 
    return Ve;
}

double *kernel::launch(bodies::pVec *points, bodies::body *particles, int N, int Np) {   

    int deviceID;
    cudaGetDevice(&deviceID);
    cudaDeviceProp props = getDetails(deviceID);

    size_t size = sizeof(bodies::body) * N;

    cudaError_t stepError;
    cudaError_t asyncError;

    bodies::body *d_particles;
    double *d_Ve;
    double *Ve;
    bodies::pVec *d_points;
    
    cudaMalloc((void **)&d_points, sizeof(bodies::pVec) * Np);
    cudaMalloc((void **)&d_Ve, sizeof(double) * Np);
    cudaMalloc((void **)&d_particles, size);
    Ve = new double[Np];

    cudaStream_t particleStream, pointStream;
    cudaStreamCreate(&particleStream);
    cudaStreamCreate(&pointStream);

    stepError = cudaGetLastError();
    if (stepError != cudaSuccess){std::cout << "! (Alloc) STEP ERROR: " << cudaGetErrorString(stepError) << std::endl;}
    asyncError = cudaDeviceSynchronize();
    if(asyncError != cudaSuccess){std::cout << "! (Alloc) ASYNC ERROR" << std::endl;}

    cudaMemcpyAsync(d_particles, particles, size, cudaMemcpyHostToDevice, particleStream);
    cudaMemcpyAsync(d_points, points, size, cudaMemcpyHostToDevice, pointStream);

    int threadsPerBlock = 512; // (16 warps worth)
    int blocks = props.multiProcessorCount * props.maxBlocksPerMultiProcessor;
    //std::cout << "Kernel configuration = <" << threadsPerBlock << "," << blocks << ">\n";
    
    calculatePotentials<<<threadsPerBlock, blocks>>>(d_points, d_particles, d_Ve, N, Np);
    
    // copy back array of N potentials
    cudaMemcpy(Ve, d_Ve, sizeof(double) * Np, cudaMemcpyDeviceToHost);
    // sum potentials

    stepError = cudaGetLastError();
    //if (stepError != cudaSuccess){std::cout << "! STEP ERROR: " << cudaGetErrorString(stepError) << std::endl;}
    asyncError = cudaDeviceSynchronize();
    if(asyncError != cudaSuccess){std::cout << "! ASYNC ERROR" << std::endl;}

    cudaFree(d_particles); cudaFree(d_Ve); cudaFree(d_points);

    /**
    printf("GPU Result = \n");
    for (int i = 0; i < Np; i++) {
        printf("%ld,", Ve[i]);
    }
    printf("\n");
    **/

    return Ve;
}