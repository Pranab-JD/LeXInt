#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <functional>

#include "../Phi_functions.hpp"
#include "../real_Leja_phi.hpp"

using namespace std;

//? Phi functions interpolated on real Leja points
template <typename state, typename rhs>
embedded_solutions<state> EPIRK4s3(rhs& RHS, state& u, int N, vector<double>& Leja_X, double c, double Gamma, double tol, double dt, int Real_Imag)
{
    //* -------------------------------------------------------------------------

    //* Computes the polynomial interpolation of matrix exponential applied to 'u' at real Leja points.
    //*
    //*    Parameters
    //*    ----------
    //*
    //*    Leja_X                  : vector <double>
    //*                                Set of Leja points
    //*
    //*    c                       : double
    //*                                Shifting factor
    //*
    //*    Gamma                   : double
    //*                                Scaling factor
    //*
    //*    tol                     : double
    //*                                Accuracy of the polynomial so formed
    //*
    //*    dt                      : double
    //*                                Step size
    //*
    //*    Returns
    //*    ----------
    //*     u_epirk3                : state
    //*                                 3rd order solution after time dt
    //*     
    //*     u_epirk4                : state 
    //*                                 4th order solution after time dt
    //*
    //*
    //*    Reference:
    //*         1. D. L. Michels, V. T. Luan, M. Tokman, A stiffly accurate integrator for elastodynamic problems, ACM Trans. Graph. 36 (4) (2017). 
    //*         doi:10.1145/3072959.3073706.
    //*
    //*         2. G. Rainwater, M. Tokman, Designing efficient exponential integrators with EPIRK framework, in: International Conference of Numerical
    //*         Analysis and Applied Mathematics (ICNAAM 2016), Vol. 1863 of American Institute of Physics Conference Series, 2017, p. 020007.
    //*         doi:10.1063/1.4992153.

    //* -------------------------------------------------------------------------

    //? RHS evaluated at 'u' multiplied by 'dt'
    state rhs_u = RHS(u);                               
    rhs_u = axpby(dt, rhs_u, N);

    //? Vertical interpolation of RHS(u) at 1/2, 2/3, and 1
    state u_flux_1 = real_Leja_phi(RHS, u, rhs_u, {1./8.}, N, phi_1, Leja_X, c, Gamma, tol, dt);
    state u_flux_2 = real_Leja_phi(RHS, u, rhs_u, {1./9.}, N, phi_1, Leja_X, c, Gamma, tol, dt);
    state u_flux_3 = real_Leja_phi(RHS, u, rhs_u, {1.0},   N, phi_1, Leja_X, c, Gamma, tol, dt);

    //? Internal stage 1; a = u + 1/8 phi_1(1/8 J(u) dt) f(u) dt
    state a = axpby(1.0, u, 1./8., u_flux_1, N);

    //? Internal stage 2; b = u + 1/9 phi_1(1/9 J(u) dt) f(u) dt
    state b = axpby(1.0, u, 1./9., u_flux_2, N);

    //? Nonlinear remainder at u, a, and b
    state NL_u = Nonlinear_remainder(RHS, u, u, N);
    state NL_a = Nonlinear_remainder(RHS, u, a, N);
    state NL_b = Nonlinear_remainder(RHS, u, b, N);
    
    state R_a = axpby(dt, NL_a, -dt, NL_u, N);
    state R_b = axpby(dt, NL_b, -dt, NL_u, N);

    //? Final nonlinear stages
    state R_3 = axpby(-1024.0, R_a,   1458.0, R_b, N);
    state R_4 = axpby(27648.0, R_a, -34992.0, R_b, N);

    //* phi_3(J(u) dt) (-1024R(a) + 1458R(b)) dt
    state u_nl_3 = real_Leja_phi(RHS, u, R_3, {1.0}, N, phi_3, Leja_X, c, Gamma, tol, dt);

    //* phi_4(J(u) dt) (27648R(a) - 34992R(b)) dt
    state u_nl_4 = real_Leja_phi(RHS, u, R_4, {1.0}, N, phi_4, Leja_X, c, Gamma, tol, dt);

    //? 3rd order solution; u_3 = u + phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) (-1024R(a) + 1458R(b)) dt
    state u_epirk3 = axpby(1.0, u, 1.0, u_flux_3, 1.0, u_nl_3, N);

    //? 4th order solution; u_4 = u_3 + phi_4(J(u) dt) (27648R(a) - 34992R(b)) dt
    state u_epirk4 = axpby(1.0, u_epirk3, 1.0, u_nl_4, N);

    return {u_epirk3, u_epirk4};
}
