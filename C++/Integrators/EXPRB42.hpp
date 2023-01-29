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
embedded_solutions<state> EXPRB42(rhs& RHS, state& u, int N, vector<double>& Leja_X, double c, double Gamma, double tol, double dt, int Real_Imag)
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
    //*     u_exprb2                : state
    //*                                 2nd order solution after time dt
    //*     
    //*     u_exprb4                : state 
    //*                                 4th order solution after time dt
    //*
    //*
    //*    Reference:
    //*         V. T. Luan, Fourth-order two-stage explicit exponential integrators for time-dependent PDEs, Appl. Numer. Math. 112 (2017) 91-103. 
    //*             doi:10.1016/j.apnum.2016.10.008.

    //* -------------------------------------------------------------------------

    //? RHS evaluated at 'u' multiplied by 'dt'
    state rhs_u = RHS(u);                               
    rhs_u = axpby(dt, rhs_u, N);

    //? Internal stage 1; interpolation of RHS(u) at 3/4 and 1
    state u_flux_1 = real_Leja_phi(RHS, u, rhs_u, {3./4.}, N, phi_1, Leja_X, c, Gamma, tol, dt);
    state u_flux_2 = real_Leja_phi(RHS, u, rhs_u, {1.0},   N, phi_1, Leja_X, c, Gamma, tol, dt);

    //? Internal stage 1; a = u + 3/4 phi_1(3/4 J(u) dt) f(u) dt
    state a = axpby(1.0, u, 3./4., u_flux_1, N);

    //? Difference of nonlinear remainders at a
    state NL_u = Nonlinear_remainder(RHS, u, u, N);
    state NL_a = Nonlinear_remainder(RHS, u, a, N);
    state R_a = axpby(dt, NL_a, -dt, NL_u, N);

    //? Final nonlinear stage
    state u_nl_3 = real_Leja_phi(RHS, u, R_a, {1.0}, N, phi_3, Leja_X, c, Gamma, tol, dt);

    //? 2nd order solution; u_2 = u + phi_1(J(u) dt) f(u) dt
    state u_exprb2 = axpby(1.0, u, 1.0, u_flux_2, N);

    //? 4th order solution; u_4 = u + phi_1(J(u) dt) f(u) dt + 32/9 phi_3(J(u) dt) R(a) dt
    state u_exprb4 = axpby(1.0, u_exprb2, 32./9., u_nl_3, N);

    return {u_exprb2, u_exprb4};
}
