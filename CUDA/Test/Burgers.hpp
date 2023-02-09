#pragma once
#include "Problems.hpp"

using namespace std;

//? ====================================================================================== ?//

struct RHS_Burgers:public Problems
{

    //! Constructor
    RHS_Burgers(int _N, double _dx, double _velocity) : Problems(_N, _dx, _velocity) {}

    vec operator()(const vec& z)
    {
        //* Return vector, v
        vec v(N, 0.0);

        //! (A_adv/2.0 + A_dif).u^2
        for (int ii = 0; ii < N; ii++)
        {
            //? Diffusion
            v[ii] =   1.0/(dx*dx) * z[(ii + 1) % N]
                    - 2.0/(dx*dx) * z[ii]
                    + 1.0/(dx*dx) * z[(ii + N - 1) % N];
            
            //? Advection
            v[ii] = v[ii] - 2.0/6.0*velocity/dx * z[(ii + N - 1)%N] * z[(ii + N - 1)%N]/2.0
                          - 3.0/6.0*velocity/dx * z[ii] * z[ii]/2.0
                          + 6.0/6.0*velocity/dx * z[(ii + 1)%N] * z[(ii + 1)%N]/2.0
                          - 1.0/6.0*velocity/dx * z[(ii + 2)%N] * z[(ii + 2)%N]/2.0;
        }

        return v;
    }

    //! Destructor
    ~RHS_Burgers() {}
};

//? ====================================================================================== ?//