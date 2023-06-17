#pragma once

#include "Problems.hpp"

using namespace std;

//? ====================================================================================== ?//

#ifdef __CUDACC__

__global__ void Dif_Adv_1D(int N, double dx, double velocity, double* input, double* output)
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
                    * (- 2.0/6.0 * input[(ii + N - 1)%N]
                       - 3.0/6.0 * input[ii]
                       + 6.0/6.0 * input[(ii + 1)%N]
                       - 1.0/6.0 * input[(ii + 2)%N]);


    }
}

#endif

struct RHS_Dif_Adv_1D:public Problems_1D
{
    //? RHS = (A_adv + A_dif).u

    //! Constructor
    RHS_Dif_Adv_1D(int _N, double _dx, double _velocity) : Problems_1D(_N, _dx, _velocity) {}

    void operator()(double* input, double* output)
    {
        #ifdef __CUDACC__

            Dif_Adv_1D<<<(N/128) + 1, 128>>>(N, dx, velocity, input, output);

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
                           * (- 2.0/6.0 * input[(ii + N - 1)%N]
                              - 3.0/6.0 * input[ii]
                              + 6.0/6.0 * input[(ii + 1)%N]
                              - 1.0/6.0 * input[(ii + 2)%N]);
            }
        
        #endif
    }

    //! Destructor
    ~RHS_Dif_Adv_1D() {}
};

//? ====================================================================================== ?//