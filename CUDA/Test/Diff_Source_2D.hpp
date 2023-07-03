#pragma once

#include <cmath>
#include "Problems.hpp"

using namespace std;

//? ====================================================================================== ?//

#ifdef __CUDACC__

__global__ void Dif_Source_2D(int N, double dx, double dy, double velocity, double* input, double* output)
{
    int ii = threadIdx.y + blockIdx.y * blockDim.y;
    int jj = threadIdx.x + blockIdx.x * blockDim.x;

    if ((ii >= N) || (jj >= N))
        return;

    double X = -1 + ii*dx;  double Y = -1 + jj*dy;
    double Source = exp(-((X + 0.5)*(X + 0.5) + (Y + 0.5)*(Y + 0.5))/0.01);
    
    //? Diffusion
    output[N*ii + jj] =   (input[N*ii + (jj + 1) % N] - (4.0 * input[N*ii + jj]) + input[N*ii + (jj + N - 1) % N])/(dx*dx)
                        + (input[N*((ii + 1) % N) + jj] + input[N*((ii + N - 1) % N) + jj])/(dy*dy) + Source;


}

#endif

struct RHS_Dif_Source_2D:public Problems_2D
{
    //? RHS = A_dif.u + S(x, y)

    //! Constructor
    RHS_Dif_Source_2D(int _N, double _dx, double _dy, double _velocity) : Problems_2D(_N, _dx, _dy, _velocity) {}

    void operator()(double* input, double* output)
    {
        #ifdef __CUDACC__

            dim3 threads(32, 32);
            dim3 blocks((N+31)/32, (N+31)/32);
            Dif_Source_2D<<<blocks, threads>>>(N, dx, dy, velocity, input, output);

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
                                double X = -1 + ii*dx;  double Y = -1 + jj*dy;
                                double Source = exp(-((X + 0.5)*(X + 0.5) + (Y + 0.5)*(Y + 0.5))/0.01);
                                
                                //? Diffusion
                                output[N*ii + jj] =   (input[N*ii + (jj + 1) % N] - (4.0 * input[N*ii + jj]) + input[N*ii + (jj + N - 1) % N])/(dx*dx)
                                                    + (input[N*((ii + 1) % N) + jj] + input[N*((ii + N - 1) % N) + jj])/(dy*dy) + Source;
                            }
                        }
                    }
                }
            }
        
        #endif
    }

    //! Destructor
    ~RHS_Dif_Source_2D() {}
};

//? ====================================================================================== ?//