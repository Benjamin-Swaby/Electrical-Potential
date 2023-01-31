#pragma once

#include "bodies.cuh"

namespace kernel {

    double *launch(bodies::pVec *points, bodies::body *particles, int N, int Np);
    double *CPUlaunch(bodies::pVec *points, bodies::body *particles, int N, int Np);  
}