#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <functional>

#include "../Phi_functions.hpp"
#include "../real_Leja_phi.hpp"

//? CUDA 
#include <cublas_v2.h>
#include "../error_check.hpp"
#include "../Leja_GPU.hpp"

using namespace std;

//? Phi functions interpolated on real Leja points
template <typename rhs>
void Ros_Eu(rhs& RHS, 
            double* device_u, 
            double* device_u_exprb2, 
            int N, 
            vector<double>& Leja_X, 
            double c,
            double Gamma,
            double tol,
            double dt,
            struct Leja_GPU<rhs> leja_gpu,
            double* device_auxillary
            )
{
    //* -------------------------------------------------------------------------

    //*
    //*    Returns
    //*    ----------
    //*     u_exprb2                : double*
    //*                                     2nd order solution after time dt
    //*
    //*
    //*    Reference:
    //*         D. A. Pope, An exponential method of numerical integration of ordinary differential equations, Commun. ACM 6 (8) (1963) 491-493.
    //*         doi:10.1145/366707.367592

    //* -------------------------------------------------------------------------

    //? RHS evaluated at 'u' multiplied by 'dt'
    double* device_rhs_u = &device_auxillary[0];
    RHS(device_u, device_rhs_u);
    axpby<<<(N/128) + 1, 128>>>(dt, device_rhs_u, device_rhs_u, N);

    //? Internal stage 1; interpolation of RHS(u) at 1
    double* device_u_flux = &device_auxillary[N];
    leja_gpu.real_Leja_phi(RHS, device_u, device_rhs_u, device_u_flux, {1.0}, phi_1, Leja_X, c, Gamma, tol, dt);

    //? 2nd order solution; u_2 = u + phi_1(J(u) dt) f(u) dt
    axpby<<<(N/128) + 1, 128>>>(1.0, device_u, 1.0, device_u_flux, device_u_exprb2, N);
}
