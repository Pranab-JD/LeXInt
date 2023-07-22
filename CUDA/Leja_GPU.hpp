#pragma once

#include "Eigenvalues.hpp"
#include "Divided_Differences.hpp"
#include "Leja.hpp"
#include "error_check.hpp"

template <typename rhs>
struct Leja_GPU
{
    int N;                          //? Number of grid points
    int num_vectors;                //? Number of vectors in an exponential integrator
    string integrator_name;         //? Name of the exponential integrator
    GPU_handle cublas_handle;       //? Modified handle for cublas

    //? Allocate memory
    //! These are device vectors if GPU support is activated
    double* auxiliary_Leja;         //? Internal vectors for Leja interpolation
    double* auxiliary_expint;       //? Internal vectors for an exponential integrator

    //! Constructor
    Leja_GPU(int _N, string _integrator_name) :  N(_N), integrator_name(_integrator_name)
    {
        if (integrator_name == "Rosenbrock_Euler")
        {
            num_vectors = 1;
        }
        else if (integrator_name == "EXPRB32")
        {
            num_vectors = 1;
        }
        else if (integrator_name == "EXPRB42")
        {
            num_vectors = 2;
        }
        else if (integrator_name == "EXPRB43")
        {
            num_vectors = 3;
        }
        else if (integrator_name == "EXPRB53s3")
        {
            num_vectors = 6;
        }
        else if (integrator_name == "EXPRB54s4")
        {
            num_vectors = 25;
        }
        else if (integrator_name == "EPIRK4s3")
        {
            num_vectors = 4;
        }
        else if (integrator_name == "EPIRK4s3A")
        {
            num_vectors = 4;
        }
        else if (integrator_name == "EPIRK4s3B")
        {
            num_vectors = 5;
        }
        else if (integrator_name == "EPIRK5P1")
        {
            num_vectors = 8;
        }
        else if (integrator_name == "Hom_Linear")
        {
            //? Homogeneous Linear Differential Equations
            num_vectors = 0;
        }
        else if (integrator_name == "NonHom_Linear")
        {
            //? Nonhomogeneous Linear Differential Equations
            num_vectors = 0;
        }
        else
        {
            std::cout << "Incorrect integrator!! (Leja_GPU.hpp)" << std::endl;
        }

        #ifdef __CUDACC__

            //? Allocate memory on device
            cudaMalloc(&auxiliary_Leja, 4 * N * sizeof(double));
            cudaMalloc(&auxiliary_expint, num_vectors * N * sizeof(double));

        #else

            auxiliary_Leja = (double*)malloc(4 * N * sizeof(double));
            auxiliary_expint = (double*)malloc(num_vectors * N * sizeof(double));

        #endif
    }

    //! Destructor
    ~Leja_GPU()
    {
        //? Deallocate memory
        #ifdef __CUDACC__
            
            cudaFree(auxiliary_Leja);
            cudaFree(auxiliary_expint);
        
        #else

            free(auxiliary_Leja);
            free(auxiliary_expint);

        #endif
    }

    //! ============ Operator Functions ============ !//

    //? Power Iterations
    void operator()(rhs& RHS, 
                    double* u_input,
                    int N, 
                    double& eigenvalue,
                    bool GPU
                    )
    {
        LeXInt::Power_iterations(RHS, u_input, N, eigenvalue, auxiliary_Leja, GPU, cublas_handle);
    }

    //? Real Leja Phi
    void operator()(rhs& RHS, 
                    double* u_input, 
                    double* u_output, 
                    int N,
                    double (* phi_function) (double),
                    vector<double>& Leja_X, 
                    double c,
                    double Gamma,
                    double tol,
                    double dt,
                    int& iters,
                    bool GPU
                    )
    {
        LeXInt::real_Leja_phi_nl(RHS, u_input, u_output, 
                                 auxiliary_Leja, 
                                 N, (* phi_function), Leja_X, c, Gamma, tol, dt, iters, GPU, cublas_handle);
    }

    //? Solvers without an error estimate
    void operator()(rhs& RHS, 
                    double* u_input, 
                    double* u_output, 
                    int N, 
                    vector<double>& Leja_X, 
                    double c,
                    double Gamma,
                    double tol,
                    double dt,
                    int& iters,
                    bool GPU
                    )
    {
        //! Call the required integrator
        if (integrator_name == "Hom_Linear")
        {
            LeXInt::real_Leja_exp(RHS, u_input, u_output,
                                  auxiliary_Leja, 
                                  N, Leja_X, c, Gamma, tol, dt, iters, GPU, cublas_handle);
        }

        else if (integrator_name == "Rosenbrock_Euler")
        {
            LeXInt::Ros_Eu(RHS, u_input, u_output,
                           auxiliary_expint, auxiliary_Leja,
                           N, Leja_X, c, Gamma, tol, dt, iters, GPU, cublas_handle);
        }
        else if (integrator_name == "EPIRK4s3B")
        {
            LeXInt::EPIRK4s3B(RHS, u_input, u_output, 
                              auxiliary_expint, auxiliary_Leja,
                              N, Leja_X, c, Gamma, tol, dt, iters, GPU, cublas_handle);
        }
        else
        {
            std::cout << "ERROR: Only 1 output vector for Rosenbrock_Euler and EPIRK4s3B." << std::endl;
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
                    int& iters,
                    bool GPU
                    )
    {
        //! Call the required integrator
        if (integrator_name == "EXPRB32")
        {
            LeXInt::EXPRB32(RHS, u_input, u_output_low, u_output_high, 
                            auxiliary_expint, auxiliary_Leja, 
                            N, Leja_X, c, Gamma, tol, dt, iters, GPU, cublas_handle);
        }
        else if (integrator_name == "EXPRB42")
        {
            LeXInt::EXPRB42(RHS, u_input, u_output_low, u_output_high, 
                            auxiliary_expint, auxiliary_Leja,
                            N, Leja_X, c, Gamma, tol, dt, iters, GPU, cublas_handle);        
        }
        else if (integrator_name == "EXPRB43")
        {
            LeXInt::EXPRB43(RHS, u_input, u_output_low, u_output_high, 
                            auxiliary_expint, auxiliary_Leja,
                            N, Leja_X, c, Gamma, tol, dt, iters, GPU, cublas_handle);        
        }
        else if (integrator_name == "EXPRB53s3")
        {
            LeXInt::EXPRB53s3(RHS, u_input, u_output_low, u_output_high, 
                              auxiliary_expint, auxiliary_Leja,
                              N, Leja_X, c, Gamma, tol, dt, iters, GPU, cublas_handle);        
        }
        // else if (integrator_name == "EXPRB54s4")
        // {
        //     LeXInt::EXPRB54s4(RHS, u_input, u_output_low, u_output_high, 
        //                       auxiliary_expint, auxiliary_Leja, auxiliary_NL, 
        //                       N, Leja_X, c, Gamma, tol, dt, iters, GPU, cublas_handle);        
        // }
        else if (integrator_name == "EPIRK4s3")
        {
            LeXInt::EPIRK4s3(RHS, u_input, u_output_low, u_output_high, 
                             auxiliary_expint, auxiliary_Leja,
                             N, Leja_X, c, Gamma, tol, dt, iters, GPU, cublas_handle);         
        }
        else if (integrator_name == "EPIRK4s3A")
        {
            LeXInt::EPIRK4s3A(RHS, u_input, u_output_low, u_output_high, 
                             auxiliary_expint, auxiliary_Leja,
                             N, Leja_X, c, Gamma, tol, dt, iters, GPU, cublas_handle);         
        }
        else if (integrator_name == "EPIRK5P1")
        {
            LeXInt::EPIRK5P1(RHS, u_input, u_output_low, u_output_high, 
                             auxiliary_expint, auxiliary_Leja,
                             N, Leja_X, c, Gamma, tol, dt, iters, GPU, cublas_handle);         
        }
        else
        {
            std::cout << "ERROR: 2 output vectors for EXPRB32, EXPRB43, EXPRB53s3, EXPRB54s4, EPIRK4s3, EPIRK4s3A, and EPIRK5P1." << std::endl;
            return;
        }
    }
};