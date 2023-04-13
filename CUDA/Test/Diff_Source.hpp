#pragma once

#include "Problems.hpp"

using namespace std;

//? ====================================================================================== ?//

__global__ void Dif_Source(int N, double* X, double dx, double velocity, double* z, double* v)
{
    int ii = blockDim.x * blockIdx.x + threadIdx.x;

    if(ii < N)
    {
        //? Diffusion
        v[ii] =   1.0/(dx*dx) * z[(ii + 1) % N] 
                - 2.0/(dx*dx) * z[ii] 
                + 1.0/(dx*dx) * z[(ii + N - 1) % N]
                + 40*exp(-((X[ii] - 0.5)**2)/0.03);
    }
}

struct RHS_Dif_Source:public Problems
{
    //! Constructor
    RHS_Dif_Source(int _N, double _dx, double _velocity) : Problems(_N, _dx, _velocity) {}

    void operator()(double* input, double* output)
    {
        Dif_Source<<<(N/128) + 1, 128>>>(N, dx, velocity, input, output);
    }

    //! Destructor
    ~RHS_Dif_Source() {}
};

//? ====================================================================================== ?//