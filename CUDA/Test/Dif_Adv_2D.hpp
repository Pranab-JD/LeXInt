#pragma once

#include "Problems.hpp"
#include "../error_check.hpp"

using namespace std;

//? ====================================================================================== ?//

#ifdef __CUDACC__

__global__ void Dif_Adv_2D(int N, double dx, double dy, double velocity, double* input, double* output)
{
    int ii = threadIdx.y + blockIdx.y * blockDim.y;
    int jj = threadIdx.x + blockIdx.x * blockDim.x;

    if ((ii >= N) || (jj >= N))
        return;
    
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

#endif

struct RHS_Dif_Adv_2D:public Problems_2D
{
    //? RHS = A_adv.u^2/2.0 + A_dif.u

    //! Constructor
    RHS_Dif_Adv_2D(int _N, double _dx, double _dy, double _velocity) : Problems_2D(_N, _dx, _dy, _velocity) {}

    void operator()(double* input, double* output)
    {
        #ifdef __CUDACC__

            dim3 threads(32, 32);
            dim3 blocks((N+31)/32, (N+31)/32);
            Dif_Adv_2D<<<blocks, threads>>>(N, dx, dy, velocity, input, output);
        
        #else

            int num_threads = 32;

            #pragma omp parallel for collapse(2)
            for (int blockIdxx = 0; blockIdxx < (N + num_threads - 1)/num_threads; blockIdxx++)
            {
                for (int blockIdxy = 0; blockIdxy < (N + num_threads - 1)/num_threads; blockIdxy++)
                {
                    for (int threadIdxx = 0; threadIdxx < num_threads; threadIdxx++)
                    {
                        for (int threadIdxy = 0; threadIdxy < num_threads; threadIdxy++)
                        {
                            int ii = (blockIdxx * num_threads) + threadIdxx;
                            int jj = (blockIdxy * num_threads) + threadIdxy;

                            if ((ii < N) && (jj < N))
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
                }
            }
            
        #endif
    }

    //! Destructor
    ~RHS_Dif_Adv_2D() {}
};

//? ====================================================================================== ?//