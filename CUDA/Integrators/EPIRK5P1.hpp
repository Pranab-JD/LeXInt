#pragma once

#include "../Phi_functions.hpp"
#include "../real_Leja_phi.hpp"
#include "../Timer.hpp"

//? CUDA 
#include "../error_check.hpp"
#include "../Leja_GPU.hpp"
#include "../Kernels_CUDA_Cpp.hpp"

namespace LeXInt
{
    //? Phi functions interpolated on real Leja points
    template <typename rhs>
    void EPIRK5P1(rhs& RHS,                   //? RHS function
                  double* u,                  //? Input state variable(s)
                  double* u_epirk4,           //? Output state variable(s) (lower order)
                  double* u_epirk5,           //? Output state variable(s) (higher order)
                  double& error,              //? Embedded error estimate
                  double* auxiliary_expint,   //? Internal auxiliary variables (EPIRK5P1)
                  double* auxiliary_Leja,     //? Internal auxiliary variables (Leja)
                  size_t N,                   //? Number of grid points
                  vector<double>& Leja_X,     //? Array of Leja points
                  double c,                   //? Shifting factor
                  double Gamma,               //? Scaling factor
                  double tol,                 //? Tolerance (normalised desired accuracy)
                  double dt,                  //? Step size
                  int& iters,                 //? # of iterations needed to converge (iteration variable)
                  bool GPU,                   //? false (0) --> CPU; true (1) --> GPU
                  GPU_handle& cublas_handle   //? CuBLAS handle
                  )
    {
        //* -------------------------------------------------------------------------

        //! u, u_epirk4, u_epirk5, auxiliary_expint, and auxiliary_Leja
        //! are device vectors if GPU support is activated.

        //*    Returns
        //*    ----------
        //*     u_epirk4                : double*
        //*                                 4th order solution after time dt
        //*     
        //*     u_epirk5                : double* 
        //*                                 5th order solution after time dt
        //*
        //*
        //*    Reference:
        //*         M. Tokman, J. Loffeld, P. Tranquilli, New Adaptive Exponential Propagation Iterative Methods of Runge-Kutta Type, SIAM J. Sci. Comput. 34 (5) (2012) A2650-A2669.
        //*         doi:10.1137/110849961

        //* -------------------------------------------------------------------------

        //? Parameters of EPIRK5P1 (5th order)
        double a11 = 0.35129592695058193092;
        double a21 = 0.84405472011657126298;
        double a22 = 1.6905891609568963624;

        double b1  = 1.0;
        double b2  = 1.2727127317356892397;
        double b3  = 2.2714599265422622275;

        double g11 = 0.35129592695058193092;
        double g21 = 0.84405472011657126298;
        double g22 = 1.0;
        double g31 = 1.0;
        double g32 = 0.71111095364366870359;
        double g33 = 0.62378111953371494809;
        
        //? 4th order
        double g32_4 = 0.5;
        double g33_4 = 1.0;

        //? Counters for Leja iterations
        int iters_1 = 0, iters_2 = 0, iters_3 = 0;

        //? Assign names and variables
        double* f_u = &u_epirk4[0]; double* u_flux = &auxiliary_expint[0]; double* NL_u = &u_epirk4[0];
        double* u_nl_1 = &auxiliary_expint[3*N]; double* u_nl_2 = &auxiliary_expint[6*N];
        double* a = &u_nl_2[0]; double* NL_a = &u_epirk5[0]; double* R_a = &u_flux[0];
        double* b = &u_nl_2[0]; double* NL_b = &u_epirk5[0]; double* R_b = &u_flux[N];
        double* R_3 = &u_flux[N]; double* error_vector = &u_flux[2*N];

        //? RHS evaluated at 'u' multiplied by 'dt'; f_u = RHS(u)*dt
        RHS(u, f_u);
        axpby(dt, f_u, f_u, N, GPU);

        //? Vertical interpolation of RHS(u) at g11, g21, and g31; u_flux = phi_1({g11, g21, g31} J(u) dt) f(u) dt
        vector<double> coeffs_1 = {g11, g21, g31};
        real_Leja_phi(RHS, u, f_u, u_flux, auxiliary_Leja, N, coeffs_1, 
                        phi_1, Leja_X, c, Gamma, tol, dt, iters_1, GPU, cublas_handle);

        //? Internal stage 1; a = u + a11 phi_1(g11 J(u) dt) f(u) dt 
        axpby(1.0, u, a11, &u_flux[0], a, N, GPU);

        //? R_a = (NL_a - NL_u) * dt
        Nonlinear_remainder(RHS, u, u, NL_u, auxiliary_Leja, N, GPU, cublas_handle);
        Nonlinear_remainder(RHS, u, a, NL_a, auxiliary_Leja, N, GPU, cublas_handle);
        axpby(dt, NL_a, -dt, NL_u, R_a, N, GPU);

        //? Vertical interpolation of R_a at g32_4, g32, and g22; phi_1({g32_4, g32, g22} J(u) dt) R(a) dt
        vector<double> coeffs_2 = {g32_4, g32, g22};
        real_Leja_phi(RHS, u, R_a, u_nl_1, auxiliary_Leja, N, coeffs_2, 
                        phi_1, Leja_X, c, Gamma, tol, dt, iters_2, GPU, cublas_handle);

        //? Internal stage 2; b = u + a21 phi_1(g21 J(u) dt) f(u) dt + a22 phi_1(g22 J(u) dt) R(a) dt
        axpby(1.0, u, a21, &u_flux[N], a22, &u_nl_1[2*N], b, N, GPU);

        //? R_b = (NL_b - NL_u) * dt
        Nonlinear_remainder(RHS, u, b, NL_b, auxiliary_Leja, N, GPU, cublas_handle);
        axpby(dt, NL_b, -dt, NL_u, R_b, N, GPU);

        //? (-2*R(a) + R(b))
        axpby(-2.0, R_a, 1.0, R_b, R_3, N, GPU);

        //? Vertical interpolation of (-2*R(a) + R(b)) at g33 and g33_4; phi_3({g33, g33_4} J(u) dt) (-2*R(a) + R(b)) dt
        vector<double> coeffs_3 = {g33, g33_4};
        real_Leja_phi(RHS, u, R_3, u_nl_2, auxiliary_Leja, N, coeffs_3, 
                        phi_3, Leja_X, c, Gamma, tol, dt, iters_3, GPU, cublas_handle);

        //! 4th order solution; u_4 = u + b1 phi_1(g31 J(u) dt) f(u) dt + b2 phi_1(g32_4 J(u) dt) R(a) dt + b3 phi_3(g33_4 J(u) dt) (-2*R(a) + R(b)) dt
        axpby(1.0, u, b1, &u_flux[2*N], b2, &u_nl_1[0], b3, &u_nl_2[N], u_epirk4, N, GPU);

        //! 5th order solution; u_5 = u + b1 phi_1(g31 J(u) dt) f(u) dt + b2 phi_1(g32 J(u) dt) R(a) dt + b3 phi_3(g33 J(u) dt) (-2*R(a) + R(b)) dt
        axpby(1.0, u, b1, &u_flux[2*N], b2, &u_nl_1[N], b3, &u_nl_2[0], u_epirk5, N, GPU);

        //? Error estimate
        axpby(1.0, u_epirk5, -1.0, u_epirk4, error_vector, N, GPU);
        error = l2norm(error_vector, N, GPU, cublas_handle);

        //? Total number of Leja iterations
        iters = iters_1 + iters_2 + iters_3;
    }
}