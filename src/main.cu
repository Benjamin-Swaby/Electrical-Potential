#include <iostream>
#include <chrono>
#include <cstdlib>

#include "bodies.cuh"
#include "kernel.cuh"

using namespace bodies;


body* generateBodies(int N) {

    auto arr = new bodies::body[N];
    

    for (int i = 0; i < N; i++) {
        arr[i] = {1e-9, i/10, 1, 2};
    }

    return arr;

}

void printBodies(body *arr, int N) {

    for (int i = 0; i < N; i++) {
        std::cout << arr[i].charge << std::endl;
    }
    

}

bodies::pVec* generatePoints(int N) {
    auto p = new bodies::pVec[N];

    for (int i = 0; i < N; i++) {
        p[i] = {0.01, 0.01, 0};
    }

    return p;
}

int main(int argc, char **argv) {    

    int n = 30;
    int np = 10;

    if (argc > 2) {
        n = std::stoi(argv[1]);
        np = std::stoi(argv[2]);
    }

    auto VeC = new double(np);
    auto Ve = new double(np);

    auto start = std::chrono::steady_clock::now();    
    auto arr = generateBodies(n);
    auto p = generatePoints(np);
    auto end = std::chrono::steady_clock::now();
    std::cout << "* Generated: " << n << " Bodies and " << np << " points in: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n"; 

    start = std::chrono::steady_clock::now();
    Ve = kernel::launch(p, arr, n, np);
    end = std::chrono::steady_clock::now();
    std::cout << "* kernel::launch() in : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n"; 


    start = std::chrono::steady_clock::now();
    VeC = kernel::CPUlaunch(p, arr, n, np);
    end = std::chrono::steady_clock::now();
    std::cout << "* kernel::CPUlaunch() in : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n"; 
}