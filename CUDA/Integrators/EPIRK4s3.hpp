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

        //? Assign required variables
        double* u_flux = &auxiliary_expint[0];
        double* a = &auxiliary_expint[3*N];

        //? RHS evaluated at 'u' multiplied by 'dt'; u_epirk3 = f(u)
        RHS(u, u_epirk3);
        axpby(dt, u_epirk3, u_epirk3, N, GPU);

        //? Vertical interpolation of RHS(u) at 1/8, 1/9, and 1; phi_1({1./8., 1./9., 1.0} J(u) dt) f(u) dt
        real_Leja_phi(RHS, u, u_epirk3, u_flux, auxiliary_Leja, N, {1./8., 1./9., 1.0}, 
                        phi_1, Leja_X, c, Gamma, tol, dt, iters_1, GPU, cublas_handle);

        //? Internal stage 1; a = u + 1/8 phi_1(1/8 J(u) dt) f(u) dt
        axpby(1.0, u, 1./8., &u_flux[0], a, N, GPU);

        //? Internal stage 2; b = u + 1/9 phi_1(1/9 J(u) dt) f(u) dt
        double* b = &u_flux[0];
        axpby(1.0, u, 1./9, &u_flux[N], b, N, GPU);

        //? Difference of nonlinear remainders at a and b
        double* temp_vec = &u_flux[N];
        Nonlinear_remainder(RHS, u, u, temp_vec, auxiliary_Leja, N, GPU, cublas_handle);
        Nonlinear_remainder(RHS, u, a, u_epirk3, auxiliary_Leja, N, GPU, cublas_handle);
        Nonlinear_remainder(RHS, u, b, u_epirk4, auxiliary_Leja, N, GPU, cublas_handle);
        axpby(dt, u_epirk3, -dt, temp_vec, a, N, GPU);
        axpby(dt, u_epirk4, -dt, temp_vec, b, N, GPU);

        //? Final nonlinear stages
        axpby(-1024.0, a,   1458.0, b, u_epirk3, N, GPU);
        axpby(27648.0, a, -34992.0, b, u_epirk4, N, GPU);

        //? a = phi_3(J(u) dt) (-1024R(a) + 1458R(b)) dt
        real_Leja_phi(RHS, u, u_epirk3, a, auxiliary_Leja, N, {1.0}, 
                        phi_3, Leja_X, c, Gamma, tol, dt, iters_2, GPU, cublas_handle);

        //? b = phi_4(J(u) dt) (27648R(a) - 34992R(b)) dt
        real_Leja_phi(RHS, u, u_epirk4, b, auxiliary_Leja, N, {1.0}, 
                        phi_4, Leja_X, c, Gamma, tol, dt, iters_3, GPU, cublas_handle);

        //! 3rd order solution; u_3 = u + phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) (-1024R(a) + 1458R(b)) dt
        axpby(1.0, u, 1.0, &u_flux[2*N], 1.0, a, u_epirk3, N, GPU);

        //! 4th order solution; u_4 = u_3 + phi_4(J(u) dt) (27648R(a) - 34992R(b)) dt
        axpby(1.0, u_epirk3, 1.0, b, u_epirk4, N, GPU);

        //? Total number of Leja iterations
        iters = iters_1 + iters_2 + iters_3;
    }
}