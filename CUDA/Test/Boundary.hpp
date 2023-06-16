#pragma once

#include <vector>

#include "Problems.hpp"

using namespace std;

//? ====================================================================================== ?//

#ifdef __CUDACC__

    __global__ void PBC(double* vector, size_t N)
    {
        vector[N-3] = vector[3];
        vector[N-2] = vector[4];
        vector[N-1] = vector[5];

        vector[0] = vector[N-6];
        vector[1] = vector[N-5];
        vector[2] = vector[N-4];
    }

#else

    void PBC(double* vector, size_t N)
    {
        vector[N-3] = vector[3];
        vector[N-2] = vector[4];
        vector[N-1] = vector[5];

        vector[0] = vector[N-6];
        vector[1] = vector[N-5];
        vector[2] = vector[N-4];
    }

#endif

void PBC(vector<double> vector, size_t N)
{
    vector[N-3] = vector[3];
    vector[N-2] = vector[4];
    vector[N-1] = vector[5];

    vector[0] = vector[N-6];
    vector[1] = vector[N-5];
    vector[2] = vector[N-4];
}

//? ====================================================================================== ?//