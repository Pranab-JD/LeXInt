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
embedded_solutions<state> EXPRB43(rhs& RHS, state& u, int N, vector<double>& Leja_X, double c, double Gamma, double tol, double dt, int Real_Imag)
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
    //*     u_exprb3                : state
    //*                                 3rd order solution after time dt
    //*     
    //*     u_exprb4                : state 
    //*                                 4th order solution after time dt
    //*
    //*
    //*    Reference:
    //*         M. Hochbruck, A. Ostermann, Exponential Integrators, Acta Numer. 19 (2010) 209-286. 
    //*         doi:10.1017/S0962492910000048.

    //* -------------------------------------------------------------------------

    //? RHS evaluated at 'u' multiplied by 'dt'
    state rhs_u = RHS(u);                               
    rhs_u = axpby(dt, rhs_u, N);

    //? Internal stage 1; interpolation of RHS(u) at 1
    state u_flux_1 = real_Leja_phi(RHS, u, rhs_u, {0.5}, N, phi_1, Leja_X, c, Gamma, tol, dt);
    state u_flux_2 = real_Leja_phi(RHS, u, rhs_u, {1.0}, N, phi_1, Leja_X, c, Gamma, tol, dt);

    //? 2nd order solution; u_2 = u + phi_1(J(u) dt) f(u) dt
    state a = axpby(1.0, u, 0.5, u_flux_1, N);

    //? Difference of nonlinear remainders at u_exprb2
    state NL_u = Nonlinear_remainder(RHS, u, u, N);
    state NL_a = Nonlinear_remainder(RHS, u, a, N);
    state R_a = axpby(dt, NL_a, -dt, NL_u, N);

    //? Final nonlinear stage
    state b_nl = real_Leja_phi(RHS, u, R_a, {1.0}, N, phi_1, Leja_X, c, Gamma, tol, dt);

    state b = axpbypcz(1.0, u, 1.0, u_flux_2, 1.0, b_nl, N);

    state NL_b = Nonlinear_remainder(RHS, u, b, N);
    state R_b = axpby(dt, NL_b, -dt, NL_u, N);

    state R_3 = axpby(16.0, R_a, -2.0, R_b, N);
    state R_4 = axpby(-48.0, R_a, 12.0, R_b, N);

    state u_nl_3 = real_Leja_phi(RHS, u, R_3, {1.0}, N, phi_3, Leja_X, c, Gamma, tol, dt);
    state u_nl_4 = real_Leja_phi(RHS, u, R_4, {1.0}, N, phi_4, Leja_X, c, Gamma, tol, dt);

    //? 3rd order solution; u_3 = u_2 + 2 phi_3(J(u) dt) R(a) dt
    state u_exprb3 = axpby(1.0, u, 1.0, u_flux_2, 1.0, u_nl_3, N);
    state u_exprb4 = axpby(1.0, u_exprb3, 1.0, u_nl_4, N);

    return {u_exprb3, u_exprb4};
}
