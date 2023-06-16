#pragma once

#include "Problems.hpp"

using namespace std;

//? ====================================================================================== ?//

#ifdef __CUDACC__

__global__ void Burgers_2D(int N, double dx, double velocity, double* input, double* output)
{
    int ii = blockDim.x * blockIdx.x + threadIdx.x;
    int jj = blockDim.y * blockIdx.y + threadIdx.y;

    if(ii < N)
    {
        if(jj < N)
        {
            //? Diffusion
            output[N*ii + jj] = (input[N*ii + (jj + 1) % N] - (4.0 * input[N*ii + jj]) + input[N*ii + (jj + N - 1) % N])/(dx*dx)
                            + (input[N*((ii + 1) % N) + jj] + input[N*((ii + N - 1) % N) + jj])/(dy*dy);
            
            //? Advection
            output[N*ii + jj] = output[N*ii + jj] + velocity/dx 
                                * (- 2.0/6.0 * input[(jj + N - 1) % N + N*ii] * input[(jj + N - 1) % N + N*ii]/2
                                - 3.0/6.0 * input[jj + N*ii] * input[jj + N*ii]/2
                                + 6.0/6.0 * input[(jj + 1) % N + N*ii] * input[(jj + 1) % N + N*ii]/2
                                - 1.0/6.0 * input[(jj + 2) % N + N*ii] * input[(jj + 2) % N + N*ii]/2)
                                + velocity/dy
                                * (- 2.0/6.0 * input[(ii + N - 1) % N + N*jj] * input[(ii + N - 1) % N + N*jj]/2
                                - 3.0/6.0 * input[ii + N*jj] * input[ii + N*jj]/2
                                + 6.0/6.0 * input[(ii + 1) % N + N*jj] * input[(ii + 1) % N + N*jj]/2
                                - 1.0/6.0 * input[(ii + 2) % N + N*jj] * input[(ii + 2) % N + N*jj]/2);
        }
    }
}

#endif

struct RHS_Burgers_2D:public Problems_2D
{
    //? RHS = A_adv.u^2/2.0 + A_dif.u

    //! Constructor
    RHS_Burgers_2D(int _N, double _dx, double _dy, double _velocity) : Problems_2D(_N, _dx, _dy, _velocity) {}

    void operator()(double* input, double* output)
    {
        #ifdef __CUDACC__

            Burgers_2D<<<(N/128) + 1, 128>>>(N, dx, dy, velocity, input, output);
        
        #else

            #pragma omp parallel for
            for (int ii = 0; ii < N; ii++)
            {
                for (int jj = 0; jj < N; jj++)
                {
                    //? Diffusion
                    output[N*ii + jj] = (input[N*ii + (jj + 1) % N] - (4.0 * input[N*ii + jj]) + input[N*ii + (jj + N - 1) % N])/(dx*dx)
                                    + (input[N*((ii + 1) % N) + jj] + input[N*((ii + N - 1) % N) + jj])/(dy*dy);
                    
                    //? Advection
                    output[N*ii + jj] = output[N*ii + jj] + velocity/dx 
                                        * (- 2.0/6.0 * input[(jj + N - 1) % N + N*ii] * input[(jj + N - 1) % N + N*ii]/2
                                        - 3.0/6.0 * input[jj + N*ii] * input[jj + N*ii]/2
                                        + 6.0/6.0 * input[(jj + 1) % N + N*ii] * input[(jj + 1) % N + N*ii]/2
                                        - 1.0/6.0 * input[(jj + 2) % N + N*ii] * input[(jj + 2) % N + N*ii]/2)
                                        + velocity/dy
                                        * (- 2.0/6.0 * input[(ii + N - 1) % N + N*jj] * input[(ii + N - 1) % N + N*jj]/2
                                        - 3.0/6.0 * input[ii + N*jj] * input[ii + N*jj]/2
                                        + 6.0/6.0 * input[(ii + 1) % N + N*jj] * input[(ii + 1) % N + N*jj]/2
                                        - 1.0/6.0 * input[(ii + 2) % N + N*jj] * input[(ii + 2) % N + N*jj]/2);
                }
            }
            
        #endif
    }

    //! Destructor
    ~RHS_Burgers_2D() {}
};

//? ====================================================================================== ?//