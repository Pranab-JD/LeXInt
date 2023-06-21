#pragma once

#include <cmath>
#include "Problems.hpp"

using namespace std;

//? ====================================================================================== ?//

#ifdef __CUDACC__

__global__ void Dif_Source_2D(int N, double dx, double dy, double velocity, double* input, double* output)
{
    int ii = blockIdx.x;
    int jj = threadIdx.x;

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

            Dif_Source_2D<<<(N, N)>>>(N, dx, dy, velocity, input, output);

        #else

            #pragma omp parallel for
            for (int ii = 0; ii < N; ii++)
            {
                for (int jj = 0; jj < N; jj++)
                {
                    double X = -1 + ii*dx;  double Y = -1 + jj*dy;
                    double Source = exp(-((X + 0.5)*(X + 0.5) + (Y + 0.5)*(Y + 0.5))/0.01);
                    
                    //? Diffusion
                    output[N*ii + jj] =   (input[N*ii + (jj + 1) % N] - (4.0 * input[N*ii + jj]) + input[N*ii + (jj + N - 1) % N])/(dx*dx)
                                        + (input[N*((ii + 1) % N) + jj] + input[N*((ii + N - 1) % N) + jj])/(dy*dy) + Source;
                }
            }
        
        #endif
    }

    //! Destructor
    ~RHS_Dif_Source_2D() {}
};

//? ====================================================================================== ?//