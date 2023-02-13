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
embedded_solutions<state> EXPRB54s4(rhs& RHS,                 //? RHS function
                                    state& u,                 //? State variable(s)
                                    int N,                    //? Number of grid points
                                    vector<double>& Leja_X,   //? Array of Leja points
                                    double c,                 //? Shifting factor
                                    double Gamma,             //? Scaling factor
                                    double tol,               //? Tolerance (normalised desired accuracy)
                                    double dt,                //? Step size
                                    int Real_Imag             //? 0 --> Real Leja, 1 --> Imaginary Leja
                                    )
{
    //* -------------------------------------------------------------------------

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
    //*         doi:10.1016/j.cam.2013.04.041

    //* -------------------------------------------------------------------------

    //? RHS evaluated at 'u' multiplied by 'dt'
    state rhs_u = RHS(u);
    rhs_u = axpby(dt, rhs_u, N);

    //? Vertical interpolation of RHS(u) at 1/4, 1/2, 9/10, and 1
    vector<state> u_flux = real_Leja_phi(RHS, u, rhs_u, {0.25, 0.5, 0.9, 1.0}, N, phi_1, Leja_X, c, Gamma, tol, dt);

    //? Internal stage 1; a = u + 1/4 phi_1(1/4 J(u) dt) f(u) dt
    state a_n = axpby(1.0, u, 0.25, u_flux[0], N);

    //? Nonlinear remainder at u and a
    state NL_u = Nonlinear_remainder(RHS, u, u, N);
    state NL_a = Nonlinear_remainder(RHS, u, a_n, N);
    state R_a = axpby(dt, NL_a, -dt, NL_u, N);

    //? Interpolation of R(a) at 1/2
    vector<state> b_nl = real_Leja_phi(RHS, u, R_a, {0.5}, N, phi_3, Leja_X, c, Gamma, tol, dt);

    //? Internal stage 2; b = u + (1/2) phi_1(1/2 J(u) dt) f(u) dt + 4 phi_3(1/2 J(u) dt) R(a) dt
    state b_n = axpby(1.0, u, 0.5, u_flux[1], 4.0, b_nl[0], N);

    //? Nonlinear remainder at b
    state NL_b = Nonlinear_remainder(RHS, u, b_n, N);
    state R_b = axpby(dt, NL_b, -dt, NL_u, N);

    //? Interpolation of R(b) at 9/10
    vector<state> c_nl = real_Leja_phi(RHS, u, R_b, {0.9}, N, phi_3, Leja_X, c, Gamma, tol, dt);

    //? Internal stage 2; b = u + (9/10) phi_1(9/10 J(u) dt) f(u) dt + (729/125) phi_3(9/10 J(u) dt) R(b) dt
    state c_n = axpby(1.0, u, 0.9, u_flux[2], 729.0/125.0, c_nl[0], N);

    //? Nonlinear remainder at c
    state NL_c = Nonlinear_remainder(RHS, u, c_n, N);
    state R_c = axpby(dt, NL_c, -dt, NL_u, N);

    //? Final nonlinear stages
    state R_4a = axpby(64.0,  R_a, -8.0, R_b, N);
    state R_4b = axpby(-60.0, R_a, -285.0/8.0,  R_b, 125.0/8.0, R_c, N);
    state R_5a = axpby(18.0,  R_b, -250.0/81.0, R_c, N);
    state R_5b = axpby(-60.0, R_b,  500.0/27.0, R_c, N);

    //? phi_3(J(u) dt) (64R(a) - 8R(b)) dt
    vector<state> u_nl_4_3 = real_Leja_phi(RHS, u, R_4a, {1.0}, N, phi_3, Leja_X, c, Gamma, tol, dt);

    //? phi_4(J(u) dt) (-60R(a) - (285/8)R(b) + (125/8)R(c)) dt
    vector<state> u_nl_4_4 = real_Leja_phi(RHS, u, R_4b, {1.0}, N, phi_4, Leja_X, c, Gamma, tol, dt);

    //? phi_3(J(u) dt) (18R(a) - (250/81)R(b)) dt
    vector<state> u_nl_5_3 = real_Leja_phi(RHS, u, R_5a, {1.0}, N, phi_3, Leja_X, c, Gamma, tol, dt);

    //? phi_3(J(u) dt) (-60R(a) + (500/27)R(b)) dt
    vector<state> u_nl_5_4 = real_Leja_phi(RHS, u, R_5b, {1.0}, N, phi_4, Leja_X, c, Gamma, tol, dt);

    //? 4th order solution; u_4 = u + phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) (64R(a) - 8R(b)) dt + phi_4(J(u) dt) (-60R(a) - (285/8)R(b) + (125/8)R(c)) dt
    state u_exprb4 = axpby(1.0, u, 1.0, u_flux[3], 1.0, u_nl_4_3[0], 1.0, u_nl_4_4[0], N);

    //? 5th order solution; u_5 = u + phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) (18R(a) - (250/81)R(b)) dt + phi_4(J(u) dt) (-60R(a) + (500/27)R(b)) dt
    state u_exprb5 = axpby(1.0, u, 1.0, u_flux[3], 1.0, u_nl_5_3[0], 1.0, u_nl_5_4[0], N);

    return {u_exprb4, u_exprb5};
}