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
state Ros_Eu(rhs& RHS, state& u, int N, vector<double>& Leja_X, double c, double Gamma, double tol, double dt, int Real_Imag)
{
    //* -------------------------------------------------------------------------

    //*
    //*    Returns
    //*    ----------
    //*     u_exprb2                : state
    //*                                 2nd order solution after time dt
    //*
    //*
    //*    Reference:
    //*         D. A. Pope, An exponential method of numerical integration of ordinary differential equations, Commun. ACM 6 (8) (1963) 491-493.
    //*         doi:10.1145/366707.367592

    //* -------------------------------------------------------------------------

    //? RHS evaluated at 'u' multiplied by 'dt'
    state rhs_u = RHS(u);                               
    rhs_u = axpby(dt, rhs_u, N);

    //? Internal stage 1; interpolation of RHS(u) at 1
    vector<state> u_flux = real_Leja_phi(RHS, u, rhs_u, {1.0}, N, phi_1, Leja_X, c, Gamma, tol, dt);

    //? 2nd order solution; u_2 = u + phi_1(J(u) dt) f(u) dt
    state u_exprb2 = axpby(1.0, u, 1.0, u_flux[0], N);

    return u_exprb2;
}
