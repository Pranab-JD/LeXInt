#pragma once

#include "Problems.hpp"

using namespace std;

//? ====================================================================================== ?//

#ifdef __CUDACC__

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

#endif

struct RHS_Burgers:public Problems
{
    //? RHS = A_adv.u^2/2.0 + A_dif.u

    //! Constructor
    RHS_Burgers(int _N, double _dx, double _velocity) : Problems(_N, _dx, _velocity) {}

    void operator()(double* input, double* output)
    {
        #ifdef __CUDACC__

            Burgers<<<(N/128) + 1, 128>>>(N, dx, velocity, input, output);
        
        #else

        #pragma omp parallel for
        for (int ii = 0; ii < N; ii++)
        {
            //? Diffusion
            output[ii] =  1.0/(dx*dx) * input[(ii + 1) % N]
                        - 2.0/(dx*dx) * input[ii]
                        + 1.0/(dx*dx) * input[(ii + N - 1) % N];
            
            //? Advection
            output[ii] = output[ii] - 2.0/6.0*velocity/dx * input[(ii + N - 1)%N] * input[(ii + N - 1)%N]/2.0
                                    - 3.0/6.0*velocity/dx * input[ii] * input[ii]/2.0
                                    + 6.0/6.0*velocity/dx * input[(ii + 1)%N] * input[(ii + 1)%N]/2.0
                                    - 1.0/6.0*velocity/dx * input[(ii + 2)%N] * input[(ii + 2)%N]/2.0;
        }
        #endif
    }

    //! Destructor
    ~RHS_Burgers() {}
};

//? ====================================================================================== ?//