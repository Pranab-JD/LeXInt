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
state EPIRK4s3B(rhs& RHS, state& u, int N, vector<double>& Leja_X, double c, double Gamma, double tol, double dt, int Real_Imag)
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
    //*         G. Rainwater, M. Tokman, A new approach to constructing efficient stiffly accurate EPIRK methods, J. Comput. Phys. 323 (2016) 283-309.
    //*         doi:10.1016/j.jcp.2016.07.026.

    //* -------------------------------------------------------------------------

    //? RHS evaluated at 'u' multiplied by 'dt'
    state rhs_u = RHS(u);
    rhs_u = axpby(dt, rhs_u, N);

    //? Vertical interpolation of RHS(u) at 1/2, 3/4, and 1
    state u_flux_1 = real_Leja_phi(RHS, u, rhs_u, {1./2.}, N, phi_2, Leja_X, c, Gamma, tol, dt);
    state u_flux_2 = real_Leja_phi(RHS, u, rhs_u, {3./4.}, N, phi_2, Leja_X, c, Gamma, tol, dt);
    state u_flux_3 = real_Leja_phi(RHS, u, rhs_u, {1.0},   N, phi_1, Leja_X, c, Gamma, tol, dt);

    //? Internal stage 1; a = u + 2/3 phi_2(1/2 J(u) dt) f(u) dt
    state a = axpby(1.0, u, 2./3., u_flux_1, N);

    //? Internal stage 2; b = u + phi_2(3/4 J(u) dt) f(u) dt
    state b = axpby(1.0, u, 1.0, u_flux_2, N);

    //? Nonlinear remainder at u, a, and b
    state NL_u = Nonlinear_remainder(RHS, u, u, N);
    state NL_a = Nonlinear_remainder(RHS, u, a, N);
    state NL_b = Nonlinear_remainder(RHS, u, b, N);
    
    state R_a = axpby(dt, NL_a, -dt, NL_u, N);
    state R_b = axpby(dt, NL_b, -dt, NL_u, N);

    //? Final nonlinear stages
    state R_3 = axpby(  54.0, R_a, -16.0, R_b, N);
    state R_4 = axpby(-324.0, R_a, 144.0, R_b, N);

    //* phi_3(J(u) dt) (54R(a) - 16R(b)) dt
    state u_nl_3 = real_Leja_phi(RHS, u, R_3, {1.0}, N, phi_3, Leja_X, c, Gamma, tol, dt);

    //* phi_4(J(u) dt) (-324R(a) + 144R(b)) dt
    state u_nl_4 = real_Leja_phi(RHS, u, R_4, {1.0}, N, phi_4, Leja_X, c, Gamma, tol, dt);

    //? 4th order solution; u_4 = u + phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) (54R(a) - 16R(b)) dt + phi_4(J(u) dt) (-324R(a) + 144R(b)) dt
    state u_epirk4 = axpby(1.0, u, 1.0, u_flux_3, 1.0, u_nl_3, 1.0, u_nl_4, N);

    return u_epirk4;
}