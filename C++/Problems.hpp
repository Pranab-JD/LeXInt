#pragma once
#include <vector>

using namespace std;
using matrix = vector<vector<double>>;
using vec = vector<double>;

//? ====================================================================================== ?//

struct Problems
{
    int N;
    double dx;
    double velocity;
    matrix A_dif;
    matrix A_adv;

    //! Constructor
    Problems(int _N, double _dx, double _velocity) : N{_N}, dx{_dx}, velocity{_velocity}
    {
        A_dif = Diffusion();
        A_adv = Advection();
    }

    //? Diffusion matrix (periodic BC)
    matrix Diffusion()
    {
        matrix A_dif(N, vec(N, 0));

        for (int ii = 0; ii < N; ii++)
        {
            for (int jj = 0; jj < N; jj++)
            {
                A_dif[ii][(ii + N - 1)%N] =  1.0/(dx*dx);
                A_dif[ii][ii]             = -2.0/(dx*dx);
                A_dif[ii][(ii + 1)%N]     =  1.0/(dx*dx);
            }
        }

        return A_dif;
    }

    //? Advection matrix (periodic BC)
    matrix Advection()
    {
        matrix A_adv(N, vec(N, 0));

        for (int ii = 0; ii < N; ii++)
        {
            for (int jj = 0; jj < N; jj++)
            {
                A_adv[ii][(ii + N - 1)%N] = -2.0/6.0*velocity/dx;
                A_adv[ii][ii]             = -3.0/6.0*velocity/dx;
                A_adv[ii][(ii + 1)%N]     =  6.0/6.0*velocity/dx;
                A_adv[ii][(ii + 2)%N]     = -1.0/6.0*velocity/dx;
            }
        }

        return A_adv;
    }

    //! Destructor
    ~Problems() {}
};
