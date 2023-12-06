#pragma once

#include "Problems.hpp"
#include "../error_check.hpp"

using namespace std;

//! This function has 2 vector reads and writes.

//? ====================================================================================== ?//

#ifdef __CUDACC__

__global__ void Dif_Adv_2D(int N, double dx, double dy, double velocity, double* input, double* output)
{
    int ii = threadIdx.y + blockIdx.y * blockDim.y;
    int jj = threadIdx.x + blockIdx.x * blockDim.x;

    if ((ii >= N) || (jj >= N))
        return;
                        //? Diffusion
    output[N*ii + jj] =   (input[PBC(ii, jj + 1, N)] - (4.0 * input[PBC(ii, jj, N)]) + input[PBC(ii, jj - 1, N)])/(dx*dx)
                        + (input[PBC(ii + 1, jj, N)] + input[PBC(ii - 1, jj, N)])/(dy*dy)
                        
                        //? Advection
                        + velocity/dx 
                        * (- 2.0/6.0 * input[PBC(ii, jj - 1, N)]
                        - 3.0/6.0 * input[PBC(ii, jj, N)]
                        + 6.0/6.0 * input[PBC(ii, jj + 1, N)]
                        - 1.0/6.0 * input[PBC(ii, jj + 2, N)])
                        + velocity/dy
                        * (- 2.0/6.0 * input[PBC(ii - 1, jj, N)]
                        - 3.0/6.0 * input[PBC(ii, jj, N)]
                        + 6.0/6.0 * input[PBC(ii + 1, jj, N)]
                        - 1.0/6.0 * input[PBC(ii + 2, jj, N)]);
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

            int num_threads = 16;
            dim3 threads(num_threads, num_threads);
            dim3 blocks((N + num_threads - 1)/num_threads, (N + num_threads - 1)/num_threads);
            
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
                                output[N*ii + jj] =   (input[PBC(ii, jj + 1, N)] - (4.0 * input[PBC(ii, jj, N)]) + input[PBC(ii, jj - 1, N)])/(dx*dx)
                                                    + (input[PBC(ii + 1, jj, N)] + input[PBC(ii - 1, jj, N)])/(dy*dy)
                                                    
                                                    //? Advection
                                                    + velocity/dx 
                                                    * (- 2.0/6.0 * input[PBC(ii, jj - 1, N)]
                                                    - 3.0/6.0 * input[PBC(ii, jj, N)]
                                                    + 6.0/6.0 * input[PBC(ii, jj + 1, N)]
                                                    - 1.0/6.0 * input[PBC(ii, jj + 2, N)])
                                                    + velocity/dy
                                                    * (- 2.0/6.0 * input[PBC(ii - 1, jj, N)]
                                                    - 3.0/6.0 * input[PBC(ii, jj, N)]
                                                    + 6.0/6.0 * input[PBC(ii + 1, jj, N)]
                                                    - 1.0/6.0 * input[PBC(ii + 2, jj, N)]);
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