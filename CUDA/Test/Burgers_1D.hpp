#pragma once

#include "Problems.hpp"

using namespace std;

//? ====================================================================================== ?//

#ifdef __CUDACC__

__global__ void Burgers_1D(int N, double dx, double velocity, double* input, double* output)
{
    int ii = blockDim.x * blockIdx.x + threadIdx.x;

    if(ii < N)
    {
        //? Diffusion
        output[ii] =  1.0/(dx*dx) * input[(ii + 1) % N]
                    - 2.0/(dx*dx) * input[ii]
                    + 1.0/(dx*dx) * input[(ii + N - 1) % N];
        
        //? Advection
        output[ii] = output[ii] + velocity/dx
                    * (- 2.0/6.0 * input[(ii + N - 1)%N] * input[(ii + N - 1)%N]/2
                       - 3.0/6.0 * input[ii] * input[ii]/2
                       + 6.0/6.0 * input[(ii + 1)%N] * input[(ii + 1)%N]/2
                       - 1.0/6.0 * input[(ii + 2)%N] * input[(ii + 2)%N]/2);
    }
}

#endif

struct RHS_Burgers_1D:public Problems_1D
{
    //? RHS = A_adv.u^2/2.0 + A_dif.u

    //! Constructor
    RHS_Burgers_1D(int _N, double _dx, double _velocity) : Problems_1D(_N, _dx, _velocity) {}

    void operator()(double* input, double* output)
    {
        #ifdef __CUDACC__

            Burgers_1D<<<(N/128) + 1, 128>>>(N, dx, velocity, input, output);
        
        #else

            #pragma omp parallel for
            for (int ii = 0; ii < N; ii++)
            {
                //? Diffusion
                output[ii] =  1.0/(dx*dx) * input[(ii + 1) % N]
                            - 2.0/(dx*dx) * input[ii]
                            + 1.0/(dx*dx) * input[(ii + N - 1) % N];
                
                //? Advection
                output[ii] = output[ii] + velocity/dx
                            * (- 2.0/6.0 * input[(ii + N - 1)%N] * input[(ii + N - 1)%N]/2
                               - 3.0/6.0 * input[ii] * input[ii]/2
                               + 6.0/6.0 * input[(ii + 1)%N] * input[(ii + 1)%N]/2
                               - 1.0/6.0 * input[(ii + 2)%N] * input[(ii + 2)%N]/2);
            }
            
        #endif
    }

    //! Destructor
    ~RHS_Burgers_1D() {}
};

//? ====================================================================================== ?//