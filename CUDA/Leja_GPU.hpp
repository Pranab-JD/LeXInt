#pragma once

using namespace std;

#include "Divided_Differences.hpp"
#include "Jacobian_vector.hpp"
#include "Leja.hpp"

//? CUDA 
#include "Kernels_CUDA_Cpp.hpp"
#include "error_check.hpp"

template <typename rhs>
struct Leja_GPU
{
    int N;                          //? Number of grid points
    int num_vectors;                //? Number of vectors for an exponential integrator
    string integrator_name;         //? Name of the exponential integrator
    GPU_handle cublas_handle;       //? Modified handle for cublas

    //? Allocate memory
    //! These are device vectors if GPU support is activated
    double* auxillary_expint;       //? Internal vectors for an exponential integrator
    double* auxillary_Leja;         //? Internal vectors for Leja interpolation
    double* auxillary_NL;           //? Internal vectors for computation of NL remainder

    //! Constructor
    Leja_GPU(int _N, string _integrator_name) :  N(_N), integrator_name(_integrator_name)
    {
        if (integrator_name == "Rosenbrock_Euler")
        {
            num_vectors = 2;
        }
        else if (integrator_name == "EXPRB32")
        {
            num_vectors = 6;
        }
        else if (integrator_name == "EXPRB42")
        {
            num_vectors = 8;
        }
        else if (integrator_name == "EXPRB43")
        {
            num_vectors = 15;
        }
        else if (integrator_name == "EXPRB53s3")
        {
            num_vectors = 19;
        }
        else if (integrator_name == "EXPRB54s4")
        {
            num_vectors = 25;
        }
        else if (integrator_name == "EPIRK4s3")
        {
            num_vectors = 15;
        }
        else if (integrator_name == "EPIRK4s3A")
        {
            num_vectors = 15;
        }
        else if (integrator_name == "EPIRK4s3B")
        {
            num_vectors = 15;
        }
        else if (integrator_name == "EPIRK5P1")
        {
            num_vectors = 17;
        }
        else if (integrator_name == "Hom_Linear")
        {
            //? Homogeneous Linear Differential Equations 
        }
        else if (integrator_name == "NonHom_Linear")
        {
            //? Nonhomogeneous Linear Differential Equations 
        }
        else
        {
            cout << "Incorrect integrator!! (Leja_GPU.hpp)";
        }

        #ifdef __CUDACC__

            //? Allocate memory on device
            cudaMalloc(&auxillary_expint, num_vectors * N * sizeof(double));
            cudaMalloc(&auxillary_Leja, 6 * N * sizeof(double));
            cudaMalloc(&auxillary_NL, 7 * N * sizeof(double));

        #else

            auxillary_expint = (double*)malloc(num_vectors * N * sizeof(double));
            auxillary_Leja = (double*)malloc(6 * N * sizeof(double));
            auxillary_NL = (double*)malloc(7 * N * sizeof(double));

        #endif
    }

    //! Destructor
    ~Leja_GPU()
    {
        //? Deallocate memory

        #ifdef __CUDACC__
            
            cudaFree(auxillary_expint);
            cudaFree(auxillary_Leja);
            cudaFree(auxillary_NL);
        
        #else

            free(auxillary_expint);
            free(auxillary_Leja);
            free(auxillary_NL);

        #endif
    }

    //! ============ Functions for Exponential Integrators ============ !//

    //? Integrators without embedded error estimate
    void operator()(rhs& RHS, 
                    double* u_input, 
                    double* u_output, 
                    int N, 
                    vector<double>& Leja_X, 
                    double c,
                    double Gamma,
                    double tol,
                    double dt,
                    bool GPU
                    )
    {
        //! Call the required integrator
        if (integrator_name == "Rosenbrock_Euler")
        {
            LeXInt::Ros_Eu(RHS, u_input, u_output,
                           auxillary_expint, auxillary_Leja,
                           N, Leja_X, c, Gamma, tol, dt, GPU, cublas_handle);
        }
        else if (integrator_name == "EPIRK4s3B")
        {
            LeXInt::EPIRK4s3B(RHS, u_input, u_output, 
                              auxillary_expint, auxillary_Leja, auxillary_NL, 
                              N, Leja_X, c, Gamma, tol, dt, GPU, cublas_handle);
        }
        else
        {
            cout << "ERROR: Only 1 output vector for Rosenbrock_Euler and EPIRK4s3B." << endl;
            return;
        }
    }

    //? Embedded integrators
    void operator()(rhs& RHS, 
                    double* u_input, 
                    double* u_output_low, 
                    double* u_output_high, 
                    int N, 
                    vector<double>& Leja_X, 
                    double c,
                    double Gamma,
                    double tol,
                    double dt,
                    bool GPU
                    )
    {
        //! Call the required integrator
        if (integrator_name == "EXPRB32")
        {
            LeXInt::EXPRB32(RHS, u_input, u_output_low, u_output_high, 
                            auxillary_expint, auxillary_Leja, auxillary_NL, 
                            N, Leja_X, c, Gamma, tol, dt, GPU, cublas_handle);
        }
        else if (integrator_name == "EXPRB42")
        {
            LeXInt::EXPRB42(RHS, u_input, u_output_low, u_output_high, 
                            auxillary_expint, auxillary_Leja, auxillary_NL, 
                            N, Leja_X, c, Gamma, tol, dt, GPU, cublas_handle);        
        }
        else if (integrator_name == "EXPRB43")
        {
            LeXInt::EXPRB43(RHS, u_input, u_output_low, u_output_high, 
                            auxillary_expint, auxillary_Leja, auxillary_NL, 
                            N, Leja_X, c, Gamma, tol, dt, GPU, cublas_handle);        
        }
        else if (integrator_name == "EXPRB53s3")
        {
            LeXInt::EXPRB53s3(RHS, u_input, u_output_low, u_output_high, 
                              auxillary_expint, auxillary_Leja, auxillary_NL, 
                              N, Leja_X, c, Gamma, tol, dt, GPU, cublas_handle);        
        }
        else if (integrator_name == "EXPRB54s4")
        {
            LeXInt::EXPRB54s4(RHS, u_input, u_output_low, u_output_high, 
                              auxillary_expint, auxillary_Leja, auxillary_NL, 
                              N, Leja_X, c, Gamma, tol, dt, GPU, cublas_handle);        
        }
        else if (integrator_name == "EPIRK4s3")
        {
            LeXInt::EPIRK4s3(RHS, u_input, u_output_low, u_output_high, 
                             auxillary_expint, auxillary_Leja, auxillary_NL, 
                             N, Leja_X, c, Gamma, tol, dt, GPU, cublas_handle);         
        }
        else if (integrator_name == "EPIRK4s3A")
        {
            LeXInt::EPIRK4s3A(RHS, u_input, u_output_low, u_output_high, 
                             auxillary_expint, auxillary_Leja, auxillary_NL, 
                             N, Leja_X, c, Gamma, tol, dt, GPU, cublas_handle);         
        }
        else if (integrator_name == "EPIRK5P1")
        {
            LeXInt::EPIRK5P1(RHS, u_input, u_output_low, u_output_high, 
                             auxillary_expint, auxillary_Leja, auxillary_NL, 
                             N, Leja_X, c, Gamma, tol, dt, GPU, cublas_handle);         
        }
        else
        {
            cout << "ERROR: 2 output vectors for EXPRB32, EXPRB43, EXPRB53s3, EXPRB54s4, EPIRK4s3, EPIRK4s3A, and EPIRK5P1." << endl;
            return;
        }
    }
};