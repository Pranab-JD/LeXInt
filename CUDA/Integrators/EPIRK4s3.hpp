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
    void EPIRK4s3(rhs& RHS,                   //? RHS function
                  double* u,                  //? Input state variable(s)
                  double* u_epirk3,           //? Output state variable(s) (lower order)
                  double* u_epirk4,           //? Output state variable(s) (higher order)
                  double* auxiliary_expint,   //? Internal auxiliary variables (EPIRK4s3)
                  double* auxiliary_Leja,     //? Internal auxiliary variables (Leja)
                  double* auxiliary_NL,       //? Internal auxiliary variables (EPIRK4s3 NL)
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

        //! u, u_epirk3, u_epirk4, auxiliary_expint, auxiliary_Leja, and auxiliary_NL
        //! are device vectors if GPU support is activated.

        //*
        //*    Returns
        //*    ----------
        //*     u_epirk3                : double*
        //*                                 3rd order solution after time dt
        //*     
        //*     u_epirk4                : double* 
        //*                                 4th order solution after time dt
        //*
        //*
        //*    Reference:
        //*         1. D. L. Michels, V. T. Luan, M. Tokman, A stiffly accurate integrator for elastodynamic problems, ACM Trans. Graph. 36 (4) (2017). 
        //*         doi:10.1145/3072959.3073706
        //*
        //*         2. G. Rainwater, M. Tokman, Designing efficient exponential integrators with EPIRK framework, in: International Conference of Numerical
        //*         Analysis and Applied Mathematics (ICNAAM 2016), Vol. 1863 of American Institute of Physics Conference Series, 2017, p. 020007.
        //*         doi:10.1063/1.4992153

        //* -------------------------------------------------------------------------

        //? Counters for Leja iterations
        int iters_1 = 0, iters_2 = 0, iters_3 = 0;

        //? RHS evaluated at 'u' multiplied by 'dt'
        double* rhs_u = &auxiliary_expint[0];
        RHS(u, rhs_u);
        axpby(dt, rhs_u, rhs_u, N, GPU);

        //? Vertical interpolation of RHS(u) at 1/8, 1/9, and 1; phi_1({1./8., 1./9., 1.0} J(u) dt) f(u) dt
        double* u_flux = &auxiliary_expint[N];
        real_Leja_phi(RHS, u, rhs_u, u_flux, auxiliary_Leja, N, {1./8., 1./9., 1.0}, 
                        phi_1, Leja_X, c, Gamma, tol, dt, iters_1, GPU, cublas_handle);

        //? Internal stage 1; a = u + 1/8 phi_1(1/8 J(u) dt) f(u) dt
        double* a = &auxiliary_expint[4*N];
        axpby(1.0, u, 1./8., &u_flux[0], a, N, GPU);

        //? Internal stage 2; b = u + 1/9 phi_1(1/9 J(u) dt) f(u) dt
        double* b = &auxiliary_expint[5*N];
        axpby(1.0, u, 1./9, &u_flux[N], b, N, GPU);

        //? Nonlinear remainder at u, a, and b
        double* NL_u = &auxiliary_expint[6*N];
        double* NL_a = &auxiliary_expint[7*N];
        double* NL_b = &auxiliary_expint[8*N];

        double* R_a = &auxiliary_expint[9*N];
        double* R_b = &auxiliary_expint[10*N];

        //? Difference of nonlinear remainders at a and b
        Nonlinear_remainder(RHS, u, u, NL_u, auxiliary_NL, N, GPU, cublas_handle);
        Nonlinear_remainder(RHS, u, a, NL_a, auxiliary_NL, N, GPU, cublas_handle);
        Nonlinear_remainder(RHS, u, b, NL_b, auxiliary_NL, N, GPU, cublas_handle);
        axpby(dt, NL_a, -dt, NL_u, R_a, N, GPU);
        axpby(dt, NL_b, -dt, NL_u, R_b, N, GPU);

        //? Final nonlinear stages
        double* R_3 = &auxiliary_expint[11*N];
        double* R_4 = &auxiliary_expint[12*N];
        axpby(-1024.0, R_a,   1458.0, R_b, R_3, N, GPU);
        axpby(27648.0, R_a, -34992.0, R_b, R_4, N, GPU);

        //? phi_3(J(u) dt) (-1024R(a) + 1458R(b)) dt
        double* u_nl_3 = &auxiliary_expint[13*N];
        real_Leja_phi(RHS, u, R_3, u_nl_3, auxiliary_Leja, N, {1.0}, 
                        phi_3, Leja_X, c, Gamma, tol, dt, iters_2, GPU, cublas_handle);

        //? phi_4(J(u) dt) (27648R(a) - 34992R(b)) dt
        double* u_nl_4 = &auxiliary_expint[14*N];
        real_Leja_phi(RHS, u, R_4, u_nl_4, auxiliary_Leja, N, {1.0}, 
                        phi_4, Leja_X, c, Gamma, tol, dt, iters_3, GPU, cublas_handle);

        //? 3rd order solution; u_3 = u + phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) (-1024R(a) + 1458R(b)) dt
        axpby(1.0, u, 1.0, &u_flux[2*N], 1.0, u_nl_3, u_epirk3, N, GPU);

        //? 4th order solution; u_4 = u_3 + phi_4(J(u) dt) (27648R(a) - 34992R(b)) dt
        axpby(1.0, u_epirk3, 1.0, u_nl_4, u_epirk4, N, GPU);

        //? Total number of Leja iterations
        iters = iters_1 + iters_2 + iters_3;
    }
}