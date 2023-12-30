#pragma once

#include <fstream>

#include "Timer.hpp"
#include "Eigenvalues.hpp"
#include "error_check.hpp"
#include "Kernels_CUDA_Cpp.hpp"

//! Include Leja interpolation functions
#include "real_Leja_exp.hpp"
#include "real_Leja_phi.hpp"
#include "real_Leja_phi_nl.hpp"

//! Include all integrators
#include "./Integrators/Rosenbrock_Euler.hpp"       //! 2nd order; no embedded error estimate
#include "./Integrators/EXPRB32.hpp"                //! 2nd and 3rd order
#include "./Integrators/EXPRB42.hpp"                //! 2nd and 4th order
#include "./Integrators/EXPRB43.hpp"                //! 3rd and 4th order
#include "./Integrators/EXPRB53s3.hpp"              //! 3rd and 5th order
#include "./Integrators/EXPRB54s4.hpp"              //! 4th and 5th order

#include "./Integrators/EPIRK4s3.hpp"               //! 3rd and 4th order
#include "./Integrators/EPIRK4s3A.hpp"              //! 3rd and 4th order
#include "./Integrators/EPIRK4s3B.hpp"              //! 4th order; no embedded error estimate
#include "./Integrators/EPIRK5P1.hpp"               //! 4th and 5th order

template <typename rhs>
struct Leja
{
    int N;                          //? Number of grid points
    int num_vectors;                //? Number of vectors in an exponential integrator
    std::string integrator_name;    //? Name of the exponential integrator
    GPU_handle cublas_handle;       //? Modified handle for cublas

    //! Allocate memory - these are device vectors if GPU support is activated
    double* auxiliary_Leja;         //? Internal vectors for Leja interpolation and power iterations
    double* auxiliary_expint;       //? Internal vectors for an exponential integrator

    //! Set of Leja points
    std::vector<double> Leja_X;

    //! Constructor
    Leja(int _N, std::string _integrator_name) :  N(_N), integrator_name(_integrator_name)
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
            num_vectors = 5;
        }
        else if (integrator_name == "EXPRB54s4")
        {
            num_vectors = 5;
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
        else if (integrator_name == "Hom_Linear" or integrator_name == "NonHom_Linear")
        {
            //? Linear Differential Equations
            num_vectors = 0;
        }
        else
        {
            std::cout << "Incorrect integrator!! See list of integrators in Leja.hpp" << std::endl;
        }

        #ifdef __CUDACC__

            //? Allocate memory on device
            cudaMalloc(&auxiliary_Leja, 4 * N * sizeof(double));
            cudaMalloc(&auxiliary_expint, num_vectors * N * sizeof(double));

        #else

            auxiliary_Leja = (double*)malloc(4 * N * sizeof(double));
            auxiliary_expint = (double*)malloc(num_vectors * N * sizeof(double));

        #endif

        Leja_X = Leja_Points();

    }

    //! Destructor
    ~Leja()
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

    //! ========================== LeXInt Functions========================== !//

    //! Read Leja points from file
    std::vector<double> Leja_Points()
    {
        //* Load Leja points
        std::ifstream inputFile("/home/pranab/PJD/LeXInt/CUDA/Leja_10000.txt");

        if (!inputFile.is_open()) 
        {
            std::cout << "Unable to open Leja_10000.txt file." << std::endl;
        }

        int max_Leja_pts = 150;                        // Max. number of Leja points
        std::vector<double> Leja_X(max_Leja_pts);      // Initialize array

        //* Read Leja_points from file into the vector Leja_X
        int count = 0;
        while(count < max_Leja_pts && inputFile >> Leja_X[count])
        {
            count = count + 1;
        }

        inputFile.close();

        return Leja_X;
    }

    //! ---------------- Generic functions  ---------------- !//

    double l2norm(double *x, size_t N, bool GPU)
    {
        double norm = LeXInt::l2norm(x, N, GPU, cublas_handle);
        return norm;
    }

    void copy(double *x, double *y, size_t N, bool GPU)
    {
        //? Set x = y
        LeXInt::copy(x, y, N, GPU);
    }

    void ones(double *x, size_t N, bool GPU)
    {
        //? ones(x) = (x[0:N] =) 1.0
        LeXInt::ones(x, N, GPU);
    }
    
    void axpby(double a, double *x, double *y, size_t N, bool GPU)
    {
        //? y = ax
        LeXInt::axpby(a, x, y, N, GPU);
    }

    void axpby(double a, double *x, double b, double *y, double *z, size_t N, bool GPU)
    {
        //? z = ax + by
        LeXInt::axpby(a, x, b, y, z, N, GPU);
    }

    void axpby(double a, double *x, double b, double *y, double c, double *z, double *w, size_t N, bool GPU)
    {
        //? w = ax + by + cz
        LeXInt::axpby(a, x, b, y, c, z, w, N, GPU);
    }

    void axpby(double a, double *x, double b, double *y, double c, double *z, double d, double *w, double *v, size_t N, bool GPU)
    {
        //? v = ax + by + cz + dw
        LeXInt::axpby(a, x, b, y, c, z, w, N, GPU);
    }

    void Power_iterations(rhs& RHS, double* u_input, double& eigenvalue, bool GPU)
    {
        LeXInt::Power_iterations(RHS, u_input, N, eigenvalue, auxiliary_Leja, GPU, cublas_handle);
    }

    //! ---------------- Leja & Exp Int ---------------- !//

    //? Real Leja Phi NL (Nonhomogenous linear equations)
    void real_Leja_phi_nl(rhs& RHS, 
                          double* u_input, 
                          double* u_output, 
                          double (* phi_function) (double),
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

    //? Real Leja Exp (Homogenous linear equations - matrix exponential)
    void real_Leja_exp(rhs& RHS, 
                       double* u_input, 
                       double* u_output, 
                       double c,
                       double Gamma,
                       double tol,
                       double dt,
                       int& iters,
                       bool GPU
                       )
    {
        LeXInt::real_Leja_exp(RHS, u_input, u_output,
                              auxiliary_Leja, 
                              N, Leja_X, c, Gamma, tol, dt, iters, GPU, cublas_handle);
    }

    //? Solvers without an error estimate
    void exp_int(rhs& RHS, 
                 double* u_input, 
                 double* u_output, 
                 double c,
                 double Gamma,
                 double tol,
                 double dt,
                 int& iters,
                 bool GPU
                 )
    {
        //! Call the required integrator
        if (integrator_name == "Rosenbrock_Euler")
        {
            LeXInt::Ros_Eu(RHS, u_input, u_output,
                           auxiliary_expint, auxiliary_Leja,
                           N, Leja_X, c, Gamma, tol, dt, iters, GPU, cublas_handle);
        }
        // else if (integrator_name == "EPIRK4s3B")
        // {
        //     LeXInt::EPIRK4s3B(RHS, u_input, u_output, 
        //                       auxiliary_expint, auxiliary_Leja,
        //                       N, Leja_X, c, Gamma, tol, dt, iters, GPU, cublas_handle);
        // }
        else
        {
            std::cout << "ERROR: Only 1 output vector for RosEu and EPIRK4s3B." << std::endl;
        }
    }

    //? Embedded integrators
    void embed_exp_int(rhs& RHS, 
                       double* u_input, 
                       double* u_output_low, 
                       double* u_output_high,
                       double& error,
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
            LeXInt::EXPRB32(RHS, u_input, u_output_low, u_output_high, error,
                            auxiliary_expint, auxiliary_Leja,
                            N, Leja_X, c, Gamma, tol, dt, iters, GPU, cublas_handle);
        }
        // else if (integrator_name == "EXPRB42")
        // {
        //     LeXInt::EXPRB42(RHS, u_input, u_output_low, u_output_high, error, 
        //                     auxiliary_expint, auxiliary_Leja,
        //                     N, Leja_X, c, Gamma, tol, dt, iters, GPU, cublas_handle);        
        // }
        // else if (integrator_name == "EXPRB43")
        // {
        //     LeXInt::EXPRB43(RHS, u_input, u_output_low, u_output_high, error,
        //                     auxiliary_expint, auxiliary_Leja,
        //                     N, Leja_X, c, Gamma, tol, dt, iters, GPU, cublas_handle);        
        // }
        // else if (integrator_name == "EXPRB53s3")
        // {
        //     LeXInt::EXPRB53s3(RHS, u_input, u_output_low, u_output_high, error,
        //                       auxiliary_expint, auxiliary_Leja,
        //                       N, Leja_X, c, Gamma, tol, dt, iters, GPU, cublas_handle);        
        // }
        // else if (integrator_name == "EXPRB54s4")
        // {
        //     LeXInt::EXPRB54s4(RHS, u_input, u_output_low, u_output_high, error,
        //                       auxiliary_expint, auxiliary_Leja, 
        //                       N, Leja_X, c, Gamma, tol, dt, iters, GPU, cublas_handle);        
        // }
        // else if (integrator_name == "EPIRK4s3")
        // {
        //     LeXInt::EPIRK4s3(RHS, u_input, u_output_low, u_output_high, error,
        //                      auxiliary_expint, auxiliary_Leja,
        //                      N, Leja_X, c, Gamma, tol, dt, iters, GPU, cublas_handle);         
        // }
        // else if (integrator_name == "EPIRK4s3A")
        // {
        //     LeXInt::EPIRK4s3A(RHS, u_input, u_output_low, u_output_high, error,
        //                      auxiliary_expint, auxiliary_Leja,
        //                      N, Leja_X, c, Gamma, tol, dt, iters, GPU, cublas_handle);         
        // }
        // else if (integrator_name == "EPIRK5P1")
        // {
        //     LeXInt::EPIRK5P1(RHS, u_input, u_output_low, u_output_high, error,
        //                      auxiliary_expint, auxiliary_Leja,
        //                      N, Leja_X, c, Gamma, tol, dt, iters, GPU, cublas_handle);         
        // }
        else
        {
            std::cout << "ERROR: 2 output vectors for EXPRB32, EXPRB52, EXPRB43,\
                          EXPRB53s3, EXPRB54s4, EPIRK4s3, EPIRK4s3A, and EPIRK5P1." << std::endl;
            return;
        }
    }
};