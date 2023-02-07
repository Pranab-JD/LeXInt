#include <vector>
#include <cmath>
#include <functional>

#include "functions.hpp"

using namespace std;

//? RK2
template <typename state, typename rhs>
state RK2(rhs& RHS, state& u, int N, double dt)
{
    //? Internal stage 1; k1 = dt * RHS(u)
    state k1 = RHS(u);
    k1 = axpby(dt, k1, N);

    //? Internal stage 2; k2 = RHS(u + k1)
    state u_rk2 = axpby(1.0, u, 1.0, k1, N);
    state k2 = RHS(u_rk2);

    //? 2nd order solution; u^{n+1} = u^n + 0.5*(k1 + k2*dt)
    u_rk2 = axpbypcz(1.0, u, 0.5, k1, 0.5*dt, k2, N);

    return u_rk2;
}

//? RK4
template <typename U, typename rhs>
U RK4(rhs& RHS, U& u, int N, double dt)
{
    U k1(N), k2(N), k3(N), k4(N), u_rk4(N);

    k1 = RHS(u);

    for (int ii = 0; ii < N; ii++)
    {
        k1[ii] = dt * k1[ii];
        u_rk4[ii] = u[ii] + k1[ii]/2;
    }

    k2 = RHS(u_rk4);

    for (int ii = 0; ii < N; ii++)
    {
        k2[ii] = dt * k2[ii];
        u_rk4[ii] = u[ii] + k2[ii]/2;
    }

    k3 = RHS(u_rk4);

    for (int ii = 0; ii < N; ii++)
    {
        k3[ii] = dt * k3[ii];
        u_rk4[ii] = u[ii] + k3[ii];
    }

    k4 = RHS(u_rk4);

    for (int ii = 0; ii < N; ii++)
    {
        u_rk4[ii] = u[ii] + 1.0/6.0*(k1[ii] + 2*k2[ii] + 2*k3[ii] + (dt*k4[ii]));
    }

    return u_rk4;
}