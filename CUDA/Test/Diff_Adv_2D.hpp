#pragma once

#include "Problems.hpp"

using namespace std;

//? ====================================================================================== ?//

#ifdef __CUDACC__

__global__ void Dif_Adv_2D(int N, double dx, double dy, double velocity, double* input, double* output)
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
                                * (- 2.0/6.0 * input[(jj + N - 1) % N + N*ii]
                                - 3.0/6.0 * input[jj + N*ii]
                                + 6.0/6.0 * input[(jj + 1) % N + N*ii]
                                - 1.0/6.0 * input[(jj + 2) % N + N*ii])
                                + velocity/dy 
                                * (- 2.0/6.0 * input[(ii + N - 1) % N + N*jj]
                                - 3.0/6.0 * input[ii + N*jj]
                                + 6.0/6.0 * input[(ii + 1) % N + N*jj]
                                - 1.0/6.0 * input[(ii + 2) % N + N*jj]);
        }
    }
}

#endif

struct RHS_Dif_Adv_2D:public Problems_2D
{
    //? RHS = (A_adv + A_dif).u

    //! Constructor
    RHS_Dif_Adv_2D(int _N, double _dx, double _dy, double _velocity) : Problems_2D(_N, _dx, _dy, _velocity) {}

    void operator()(double* input, double* output)
    {
        #ifdef __CUDACC__

            Dif_Adv_2D<<<(N/128) + 1, 128>>>(N, dx, dy, velocity, input, output);

        #else

            #pragma omp parallel for
            for (int ii = 0; ii < N; ii++)
            {
                for (int jj = 0; jj < N; jj++)
                {
                    //? Diffusion
                    output[N*ii + jj] =   (input[N*ii + (jj + 1) % N] - (4.0 * input[N*ii + jj]) + input[N*ii + (jj + N - 1) % N])/(dx*dx)
                                        + (input[N*((ii + 1) % N) + jj] + input[N*((ii + N - 1) % N) + jj])/(dy*dy);
                    
                    //? Advection
                    output[N*ii + jj] = output[N*ii + jj] + velocity/dx 
                                        * (- 2.0/6.0 * input[(jj + N - 1) % N + N*ii]
                                           - 3.0/6.0 * input[jj + N*ii]
                                           + 6.0/6.0 * input[(jj + 1) % N + N*ii]
                                           - 1.0/6.0 * input[(jj + 2) % N + N*ii])
                                           + velocity/dy 
                                        * (- 2.0/6.0 * input[(ii + N - 1) % N + N*jj]
                                           - 3.0/6.0 * input[ii + N*jj]
                                           + 6.0/6.0 * input[(ii + 1) % N + N*jj]
                                           - 1.0/6.0 * input[(ii + 2) % N + N*jj]);
                }
            }
        
        #endif
    }

    //! Destructor
    ~RHS_Dif_Adv_2D() {}
};

//? ====================================================================================== ?//