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
embedded_solutions<state> EXPRB32(rhs& RHS, state& u, int N, vector<double>& Leja_X, double c, double Gamma, double tol, double dt, int Real_Imag)
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
    //*     u_exprb3                : state 
    //*                                 3rd order solution after time dt
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
    state u_flux = real_Leja_phi(RHS, u, rhs_u, {1.0}, N, phi_1, Leja_X, c, Gamma, tol, dt);

    //? 2nd order solution; u_2 = u + phi_1(J(u) dt) f(u) dt
    state u_exprb2 = axpby(1.0, u, 1.0, u_flux, N);

    //? Difference of nonlinear remainders at u_exprb2
    state NL_u = Nonlinear_remainder(RHS, u, u, N);
    state NL_u_exprb2 = Nonlinear_remainder(RHS, u, u_exprb2, N);
    state R_a = axpby(dt, NL_u_exprb2, -dt, NL_u, N);

    //? Final nonlinear stage
    state u_nl_3 = real_Leja_phi(RHS, u, R_a, {1.0}, N, phi_3, Leja_X, c, Gamma, tol, dt);

    //? 3rd order solution; u_3 = u_2 + 2 phi_3(J(u) dt) R(a) dt
    state u_exprb3 = axpby(1.0, u_exprb2, 2.0, u_nl_3, N);

    return {u_exprb2, u_exprb3};
}
