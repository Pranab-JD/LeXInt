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
embedded_solutions<state> EPIRK5P1(rhs& RHS,                 //? RHS function
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
    //*     u_epirk4                : state
    //*                                 4th order solution after time dt
    //*     
    //*     u_epirk5                : state 
    //*                                 5th order solution after time dt
    //*
    //*
    //*    Reference:
    //*         M. Tokman, J. Loffeld, P. Tranquilli, New Adaptive Exponential Propagation Iterative Methods of Runge-Kutta Type, SIAM J. Sci. Comput. 34 (5) (2012) A2650-A2669.
    //*         doi:10.1137/110849961

    //* -------------------------------------------------------------------------

    //? Parameters of EPIRK5P1 (5th order)
    double a11 = 0.35129592695058193092;
    double a21 = 0.84405472011657126298;
    double a22 = 1.6905891609568963624;

    double b1  = 1.0;
    double b2  = 1.2727127317356892397;
    double b3  = 2.2714599265422622275;

    double g11 = 0.35129592695058193092;
    double g21 = 0.84405472011657126298;
    double g22 = 1.0;
    double g31 = 1.0;
    double g32 = 0.71111095364366870359;
    double g33 = 0.62378111953371494809;
    
    //? 4th order
    double g32_4 = 0.5;
    double g33_4 = 1.0;

    //? RHS evaluated at 'u' multiplied by 'dt'
    state rhs_u = RHS(u);
    rhs_u = axpby(dt, rhs_u, N);

    //? Vertical interpolation of RHS(u) at g11, g21, and g31; phi_1({g11, g21, g31} J(u) dt) f(u) dt
    vector<state> u_flux = real_Leja_phi(RHS, u, rhs_u, {g11, g21, g31}, N, phi_1, Leja_X, c, Gamma, tol, dt);

    //? Internal stage 1; a = u + a11 phi_1(g11 J(u) dt) f(u) dt
    state a = axpby(1.0, u, a11, u_flux[0], N);

    //? Nonlinear remainder at u and a
    state NL_u = Nonlinear_remainder(RHS, u, u, N);
    state NL_a = Nonlinear_remainder(RHS, u, a, N);
    state R_a = axpby(dt, NL_a, -dt, NL_u, N);

    //? Vertical interpolation of R_a at g32_4, g32, and g22; phi_1({g32_4, g32, g22} J(u) dt) R(a) dt
    vector<state> u_nl_1 = real_Leja_phi(RHS, u, R_a, {g32_4, g32, g22}, N, phi_1, Leja_X, c, Gamma, tol, dt);

    //? Internal stage 2; b = u + a21 phi_1(g21 J(u) dt) f(u) dt + a22 phi_1(g22 J(u) dt) R(a) dt
    state b = axpby(1.0, u, a21, u_flux[1], a22, u_nl_1[2], N);

    //? Nonlinear remainder at b
    state NL_b = Nonlinear_remainder(RHS, u, b, N);
    state R_b = axpby(dt, NL_b, -dt, NL_u, N);

    //? (-2*R(a) + R(b))
    state nl_2 = axpby(-2.0, R_a, 1.0, R_b, N);

    //? Vertical interpolation of (-2*R(a) + R(b)) at g33 and g33_4; phi_3({g33, g33_4} J(u) dt) (-2*R(a) + R(b)) dt
    vector<state> u_nl_2 = real_Leja_phi(RHS, u, nl_2, {g33, g33_4}, N, phi_3, Leja_X, c, Gamma, tol, dt);

    //? 4th order solution; u_4 = u + b1 phi_1(g31 J(u) dt) f(u) dt + b2 phi_1(g32 J(u) dt) R(a) dt + b3 phi_3(g33 J(u) dt) (-2*R(a) + R(b)) dt
    state u_epirk4 = axpby(1.0, u, 1.0, u_flux[2], b2, u_nl_1[0], b3, u_nl_2[1], N);

    //? 5th order solution; u_5 = u + b1 phi_1(g31 J(u) dt) f(u) dt + b2 phi_1(g32_4 J(u) dt) R(a) dt + b3 phi_3(g33_4 J(u) dt) (-2*R(a) + R(b)) dt
    state u_epirk5 = axpby(1.0, u, 1.0, u_flux[2], b2, u_nl_1[1], b3, u_nl_2[0], N);

    return {u_epirk4, u_epirk5};
}
