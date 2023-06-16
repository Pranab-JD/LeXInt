#pragma once

#include "Problems.hpp"

using namespace std;

//? ====================================================================================== ?//

#ifdef __CUDACC__

__global__ void Dif_Source_2D(int N, double* X, double* Y, double dx, double dy, double velocity, double* input, double* output)
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
        }
    }
}

#endif

struct RHS_Dif_Source_2D:public Problems_2D
{
    //? RHS = A_dif.u + S(x, y)

    //! Constructor
    RHS_Dif_Source_2D(int _N, double* X, double* Y, double _dx, double _dy, double _velocity) : Problems_2D(_N, _dx, _dy, _velocity) {}

    void operator()(double* input, double* output)
    {
        #ifdef __CUDACC__

            Dif_Source_2D<<<(N/128) + 1, 128>>>(N, X, Y, dx, dy, velocity, input, output);

        #else

            #pragma omp parallel for
            for (int ii = 0; ii < N; ii++)
            {
                for (int jj = 0; jj < N; jj++)
                {
                    //? Diffusion
                    output[N*ii + jj] =   (input[N*ii + (jj + 1) % N] - (4.0 * input[N*ii + jj]) + input[N*ii + (jj + N - 1) % N])/(dx*dx)
                                        + (input[N*((ii + 1) % N) + jj] + input[N*((ii + N - 1) % N) + jj])/(dy*dy)
                                        + 40*exp(-((X[ii] - 0.5)**2)/0.03);;
                }
            }
        
        #endif
    }

    //! Destructor
    ~RHS_Dif_Source_2D() {}
};

//? ====================================================================================== ?//










        v[ii] =   1.0/(dx*dx) * z[(ii + 1) % N] 
                - 2.0/(dx*dx) * z[ii] 
                + 1.0/(dx*dx) * z[(ii + N - 1) % N]
                + 40*exp(-((X[ii] - 0.5)**2)/0.03);