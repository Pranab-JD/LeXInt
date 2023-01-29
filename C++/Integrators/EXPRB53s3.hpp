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
embedded_solutions<state> EXPRB53s3(rhs& RHS, state& u, int N, vector<double>& Leja_X, double c, double Gamma, double tol, double dt, int Real_Imag)
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
    //*     u_exprb5                : state 
    //*                                 5th order solution after time dt
    //*
    //*
    //*    Reference:
    //*         V. T. Luan, A. Ostermann, Exponential Rosenbrock methods of order five — construction, analysis and numerical comparisons, J. Comput. Appl. Math. 255 (2014) 417-431. 
    //*         doi:10.1016/j.cam.2013.04.041.

    //* -------------------------------------------------------------------------

    //? RHS evaluated at 'u' multiplied by 'dt'
    state rhs_u = RHS(u);
    rhs_u = axpby(dt, rhs_u, N);

    //? Vertical interpolation of RHS(u) at 1/2, 2/3, and 1
    state u_flux_1 = real_Leja_phi(RHS, u, rhs_u, {0.5}, N, phi_1, Leja_X, c, Gamma, tol, dt);
    state u_flux_2 = real_Leja_phi(RHS, u, rhs_u, {0.9}, N, phi_1, Leja_X, c, Gamma, tol, dt);
    state u_flux_3 = real_Leja_phi(RHS, u, rhs_u, {1.0}, N, phi_1, Leja_X, c, Gamma, tol, dt);

    //? Internal stage 1; a = u + 1/2 phi_1(1/2 J(u) dt) f(u) dt
    state a = axpby(1.0, u, 0.5, u_flux_1, N);

    //? Nonlinear remainder at u and a
    state NL_u = Nonlinear_remainder(RHS, u, u, N);
    state NL_a = Nonlinear_remainder(RHS, u, a, N);
    state R_a = axpby(dt, NL_a, -dt, NL_u, N);

    //? Vertical interpolation of R(a) at 1/2 and 9/10
    state b_nl_1 = real_Leja_phi(RHS, u, R_a, {0.5}, N, phi_3, Leja_X, c, Gamma, tol, dt);
    state b_nl_2 = real_Leja_phi(RHS, u, R_a, {0.9}, N, phi_3, Leja_X, c, Gamma, tol, dt);

    //? Internal stage 2; b = u + 9/10 phi_1(9/10 J(u) dt) f(u) dt + (27/25 phi_3(1/2 J(u) dt) + 729/125 phi_3(9/10 J(u) dt)) R(a) dt
    state b = axpby(1.0, u, 0.9, u_flux_2, 27.0/25.0, b_nl_1, 729.0/125.0, b_nl_2, N);

    //? Nonlinear remainder at b
    state NL_b = Nonlinear_remainder(RHS, u, b, N);
    state R_b = axpby(dt, NL_b, -dt, NL_u, N);

    //? Final nonlinear stages
    state R_3a = axpby(2.0,   R_a,  150.0/81.0, R_b, N);
    state R_3b = axpby(18.0,  R_a, -250.0/81.0, R_b, N);
    state R_4  = axpby(-60.0, R_a,  500.0/27.0, R_b, N);

    //* phi_3(J(u) dt) (2R(a) + 150/81R(b)) dt
    state u_nl_4_3 = real_Leja_phi(RHS, u, R_3a, {1.0}, N, phi_3, Leja_X, c, Gamma, tol, dt);

    //* phi_3(J(u) dt) (18R(a) - 250/81R(b)) dt
    state u_nl_5_3 = real_Leja_phi(RHS, u, R_3b, {1.0}, N, phi_3, Leja_X, c, Gamma, tol, dt);

    //* phi_4(J(u) dt) (-60R(a) + (500/27)R(b)) dt
    state u_nl_5_4 = real_Leja_phi(RHS, u, R_4,  {1.0}, N, phi_4, Leja_X, c, Gamma, tol, dt);

    //? 3rd order solution; u_3 = u + phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) (2R(a) + (150/81)R(b)) dt
    state u_exprb3 = axpby(1.0, u, 1.0, u_flux_3, 1.0, u_nl_4_3, N);

    //? 5th order solution; u_5 = u + phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) (18R(a) - (250/81)R(b)) dt + phi_4(J(u) dt) (-60R(a) + (500/27)R(b)) dt
    state u_exprb5 = axpby(1.0, u, 1.0, u_flux_3, 1.0, u_nl_5_3, 1.0, u_nl_5_4, N);

    return {u_exprb3, u_exprb5};
}
