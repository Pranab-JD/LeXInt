#pragma once
#include "Problems.hpp"

using namespace std;

//? ====================================================================================== ?//

struct RHS_Dif_Adv:public Problems
{

    //! Constructor
    RHS_Dif_Adv(int _N, double _dx, double _velocity) : Problems(_N, _dx, _velocity) {}

    vec operator()(const vec& z)
    {
        //* Return vector, v
        vec v(N, 0.0);

        //! (A_adv + A_dif).u
        for (int ii = 0; ii < N; ii++)
        {
            //? Diffusion
            v[ii] =   1.0/(dx*dx) * z[(ii + 1) % N] 
                    - 2.0/(dx*dx) * z[ii] 
                    + 1.0/(dx*dx) * z[(ii + N - 1) % N];
            
            //? Advection
            v[ii] = v[ii] - 2.0/6.0*velocity/dx*z[(ii + N - 1)%N]
                          - 3.0/6.0*velocity/dx*z[ii]
                          + 6.0/6.0*velocity/dx*z[(ii + 1)%N]
                          - 1.0/6.0*velocity/dx*z[(ii + 2)%N];
        }

        return v;
    }

    //! Destructor
    ~RHS_Dif_Adv() {}
};

//? ====================================================================================== ?//