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
embedded_solutions<state> EXPRB43(rhs& RHS,                 //? RHS function
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
    //*         doi:10.1017/S0962492910000048

    //* -------------------------------------------------------------------------

    //? RHS evaluated at 'u' multiplied by 'dt'
    state rhs_u = RHS(u);                               
    rhs_u = axpby(dt, rhs_u, N);

    //? Vertical interpolation of RHS(u) at 0.5 and 1; phi_1({0.5, 1.0} J(u) dt) f(u) dt
    vector<state> u_flux = real_Leja_phi(RHS, u, rhs_u, {0.5, 1.0}, N, phi_1, Leja_X, c, Gamma, tol, dt);

    //? Internal stage 1; a = u + 1/2 phi_1(1/2 J(u) dt) f(u) dt
    state a = axpby(1.0, u, 0.5, u_flux[0], N);

    //? Difference of nonlinear remainders at a
    state NL_u = Nonlinear_remainder(RHS, u, u, N);
    state NL_a = Nonlinear_remainder(RHS, u, a, N);
    state R_a = axpby(dt, NL_a, -dt, NL_u, N);

    //? Internal stage 2; phi_1(J(u) dt) R(a) dt
    vector<state> b_nl = real_Leja_phi(RHS, u, R_a, {1.0}, N, phi_1, Leja_X, c, Gamma, tol, dt);

    //? b = u + phi_1(J(u) dt) f(u) dt + phi_1(J(u) dt) R(a) dt
    state b = axpby(1.0, u, 1.0, u_flux[1], 1.0, b_nl[0], N);

    //? Difference of nonlinear remainders at a
    state NL_b = Nonlinear_remainder(RHS, u, b, N);
    state R_b = axpby(dt, NL_b, -dt, NL_u, N);

    //? Final nonlinear stages
    state R_3 = axpby(16.0, R_a, -2.0, R_b, N);
    state R_4 = axpby(-48.0, R_a, 12.0, R_b, N);

    //? phi_3(J(u) dt) (16R(a) - 2R(b)) dt
    vector<state> u_nl_3 = real_Leja_phi(RHS, u, R_3, {1.0}, N, phi_3, Leja_X, c, Gamma, tol, dt);
    
    //? phi_4(J(u) dt) (-48R(a) + 12R(b)) dt
    vector<state> u_nl_4 = real_Leja_phi(RHS, u, R_4, {1.0}, N, phi_4, Leja_X, c, Gamma, tol, dt);

    //? 3rd order solution; u_3 = u + phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) (16R(a) - 2R(b)) dt
    state u_exprb3 = axpby(1.0, u, 1.0, u_flux[1], 1.0, u_nl_3[0], N);

    //? 4th order solution; u_4 = u_3 + phi_4(J(u) dt) (-48R(a) + 12R(b)) dt
    state u_exprb4 = axpby(1.0, u_exprb3, 1.0, u_nl_4[0], N);

    return {u_exprb3, u_exprb4};
}