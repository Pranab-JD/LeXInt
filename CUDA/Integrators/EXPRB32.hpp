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
embedded_solutions<state> EXPRB32(rhs& RHS,                 //? RHS function
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
    //*     u_exprb2                : state
    //*                                 2nd order solution after time dt
    //*     
    //*     u_exprb3                : state 
    //*                                 3rd order solution after time dt
    //*
    //*
    //*    Reference:
    //*         M. Hochbruck, A. Ostermann, Exponential Integrators, Acta Numer. 19 (2010) 209-286. 
    //*         doi:10.1017/S0962492910000048

    //* -------------------------------------------------------------------------

    //? RHS evaluated at 'u' multiplied by 'dt'
    state rhs_u = RHS(u);                               
    rhs_u = axpby(dt, rhs_u, N);

    //? Interpolation of RHS(u) at 1; phi_1(J(u) dt) f(u) dt
    vector<state> u_flux = real_Leja_phi(RHS, u, rhs_u, {1.0}, N, phi_1, Leja_X, c, Gamma, tol, dt);

    //? Internal stage 1; 2nd order solution; u_2 = u + phi_1(J(u) dt) f(u) dt
    state u_exprb2 = axpby(1.0, u, 1.0, u_flux[0], N);

    //? Difference of nonlinear remainders at u_exprb2
    state NL_u = Nonlinear_remainder(RHS, u, u, N);
    state NL_u_exprb2 = Nonlinear_remainder(RHS, u, u_exprb2, N);
    state R_a = axpby(dt, NL_u_exprb2, -dt, NL_u, N);

    //? Final nonlinear stage; phi_3(J(u) dt) R(a) dt
    vector<state> u_nl_3 = real_Leja_phi(RHS, u, R_a, {1.0}, N, phi_3, Leja_X, c, Gamma, tol, dt);

    //? 3rd order solution; u_3 = u_2 + 2 phi_3(J(u) dt) R(a) dt
    state u_exprb3 = axpby(1.0, u_exprb2, 2.0, u_nl_3[0], N);

    return {u_exprb2, u_exprb3};
}
