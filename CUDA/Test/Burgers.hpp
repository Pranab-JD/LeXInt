#pragma once

#include "Problems.hpp"

using namespace std;

//? ====================================================================================== ?//

__global__ void Burgers(int N, double dx, double velocity, double* z, double* v)
{
    int ii = blockDim.x * blockIdx.x + threadIdx.x;

    if(ii < N)
    {
        //? Diffusion
        v[ii] =   1.0/(dx*dx) * z[(ii + 1) % N]
                - 2.0/(dx*dx) * z[ii]
                + 1.0/(dx*dx) * z[(ii + N - 1) % N];
        
        //? Advection
        v[ii] = v[ii] - 2.0/6.0*velocity/dx * z[(ii + N - 1)%N] * z[(ii + N - 1)%N]/2.0
                        - 3.0/6.0*velocity/dx * z[ii] * z[ii]/2.0
                        + 6.0/6.0*velocity/dx * z[(ii + 1)%N] * z[(ii + 1)%N]/2.0
                        - 1.0/6.0*velocity/dx * z[(ii + 2)%N] * z[(ii + 2)%N]/2.0;
    }
}

struct RHS_Burgers:public Problems
{

    //! Constructor
    RHS_Burgers(int _N, double _dx, double _velocity) : Problems(_N, _dx, _velocity) {}

    void operator()(double* input, double* output)
    {
        Burgers<<<(N/128) + 1, 128>>>(N, dx, velocity, input, output);
    }

    //! Destructor
    ~RHS_Burgers() {}
};

//? ====================================================================================== ?//