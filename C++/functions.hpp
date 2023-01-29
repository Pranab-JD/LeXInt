#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

using namespace std;

//? ----------------------------------------------------------
//?
//? Description:
//?     A pleothera of functions are defined here that
//?     are used throughout the code.
//?
//? ----------------------------------------------------------

//! ======================================================================================== !//

//! Return void !//

//* Function to compute precise time
enum { NS_PER_SECOND = 1000000000 };

void sub_timespec(struct timespec t1, struct timespec t2, struct timespec *td)
{
    td->tv_nsec = t2.tv_nsec - t1.tv_nsec;
    td->tv_sec  = t2.tv_sec - t1.tv_sec;
    if (td->tv_sec > 0 && td->tv_nsec < 0)
    {
        td->tv_nsec += NS_PER_SECOND;
        td->tv_sec--;
    }
    else if (td->tv_sec < 0 && td->tv_nsec > 0)
    {
        td->tv_nsec -= NS_PER_SECOND;
        td->tv_sec++;
    }
}

//! ======================================================================================== !//

//! Return double !//

template <typename T>
double l2norm(T vector_1, size_t N)
{
    double norm = 0.0;

    for (int ii = 0; ii < N; ii++)
    {
        norm = norm + (vector_1[ii] * vector_1[ii]);
    }

    return sqrt(norm);
}

double factorial(int number)
{
    double fact = 1.0;

    if (number == 0)
    {
        fact = 1.0;
    }

    else
    {
        for(int ii = 1; ii <= abs(number); ii++)
        {    
            fact = fact*ii;    
        }
    }

    return fact;
}

//! ======================================================================================== !//

//! Return typename T !//

template <typename T>
T axpby(double a, const T& vector_x, size_t N)
{
    T vector_z(N);

    for (int ii = 0; ii < N; ii++)
    {
        vector_z[ii] = a * vector_x[ii];
    }

    return vector_z;
}

template <typename T>
T axpby(double a, const T& vector_x, double b, const T& vector_y, size_t N)
{
    T vector_z(N);

    for (int ii = 0; ii < N; ii++)
    {
        vector_z[ii] = (a * vector_x[ii]) + (b * vector_y[ii]);
    }

    return vector_z;
}

template <typename T>
T axpby(double a, const T& vector_x, double b, const T& vector_y, double c, const T& vector_z, size_t N)
{
    T vector_w(N);

    for (int ii = 0; ii < N; ii++)
    {
        vector_w[ii] = (a * vector_x[ii]) + (b * vector_y[ii]) + (c * vector_z[ii]);
    }

    return vector_w;
}

template <typename T>
T axpby(double a, const T& vector_x, double b, const T& vector_y, double c, const T& vector_z, double d, const T& vector_w, size_t N)
{
    T vector_v(N);

    for (int ii = 0; ii < N; ii++)
    {
        vector_v[ii] = (a * vector_x[ii]) + (b * vector_y[ii]) + (c * vector_z[ii]) + (d * vector_w[ii]);
    }

    return vector_v;
}

template <typename rhs, typename T>
T Jacobian_vector(rhs& RHS, const T& vector_x, const T& vector_y, size_t N)
{
    //* epsilon has to be normalised to RHS(u)
    T rhs_u = RHS(vector_x); 
    double epsilon = 1e-7*l2norm(rhs_u, N);
    
    //? u_eps1 = u + epsilon*y
    T vector_x_eps_1 = axpby(1.0, vector_x, epsilon, vector_y, N); 

    //? u_eps2 = u - epsilon*y
    T vector_x_eps_2 = axpby(1.0, vector_x, -epsilon, vector_y, N); 

    //? RHS(u + epsilon*y)
    T rhs_u_eps_1 = RHS(vector_x_eps_1);

    //? RHS(u - epsilon*y)
    T rhs_u_eps_2 = RHS(vector_x_eps_2);

    //? J(u) * y = (RHS(u + epsilon*y) - RHS(u))/epsilon
    T Jac_vec = axpby(1.0/(2.0*epsilon), rhs_u_eps_1, -1.0/(2.0*epsilon), rhs_u_eps_2, N);

    return Jac_vec;
}

template <typename rhs, typename T>
T Nonlinear_remainder(rhs& RHS, const T& vector_x, const T& vector_y, size_t N)
{
    //? J(u) * y = (RHS(u + epsilon*y) - RHS(u))/epsilon
    T Linear_y = Jacobian_vector(RHS, vector_x, vector_y, N);

    //? F(y) = f(y) - (J(u) * y)
    T rhs_y = RHS(vector_y);
    T Nonlinear_y = axpby(1.0, rhs_y, -1.0, Linear_y, N);

    return Nonlinear_y;
}

//! ======================================================================================== !//

//! Structs !//

template<typename T>
struct embedded_solutions
{

    T lower_order_solution;
    T higher_order_solution;
};

//! ======================================================================================== !//
