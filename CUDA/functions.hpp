#pragma once

//? ----------------------------------------------------------
//?
//? Description:
//?     A pleothera of functions are defined here that
//?     are used throughout the code.
//?
//? ----------------------------------------------------------

//! ======================================================================================== !//

//! Return double !//

double l1norm_Cpp(double* vector, size_t N)
{

    double norm = 0.0;

    #pragma omp parallel for reduction(+:norm)
    for (int ii = 0; ii < N; ii++)
    {
        norm = norm + abs(vector[ii]);
    }

    return norm;
}

double l2norm_Cpp(double* vector, size_t N)
{

    double norm = 0.0;

    #pragma omp parallel for reduction(+:norm)
    for (int ii = 0; ii < N; ii++)
    {
        norm = norm + (vector[ii] * vector[ii]);
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

//! Return double* !//

//? ones(y) = (y[0:N] =) 1.0
void ones_Cpp(double *x, size_t N)                    
{
    #pragma omp parallel for
    for (int ii = 0; ii < N; ii++)
    {
        x[ii] = 1.0;
    }
}

//? y = ax
void axpby_Cpp(double a, double *x, 
                         double *y, size_t N)                    
{
    #pragma omp parallel for
    for (int ii = 0; ii < N; ii++)
    {
        y[ii] = (a * x[ii]);
    }
}

//? z = ax + by
void axpby_Cpp(double a, double *x, 
               double b, double *y, 
                         double *z, size_t N)
{
    #pragma omp parallel for
    for (int ii = 0; ii < N; ii++)
    {
        z[ii] = (a * x[ii]) + (b * y[ii]);
    }

}

//? w = ax + by + cz
void axpby_Cpp(double a, double *x,
               double b, double *y,
               double c, double *z, 
                         double *w, size_t N)
{
    #pragma omp parallel for
    for (int ii = 0; ii < N; ii++)
    {
        w[ii] = (a * x[ii]) + (b * y[ii]) + (c * z[ii]);
    }
}

//? v = ax + by + cz + dw
void axpby_Cpp(double a, double *x,
               double b, double *y,
               double c, double *z,
               double d, double *w,
                         double *v, size_t N)
{
    #pragma omp parallel for
    for (int ii = 0; ii < N; ii++)
    {
        v[ii] = (a * x[ii]) + (b * y[ii]) + (c * z[ii]) + (d * w[ii]);
    }
}

//! ======================================================================================== !//