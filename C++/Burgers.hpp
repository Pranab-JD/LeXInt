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
        vec v(N);

        //! (A_adv + A_dif).u
        for (int jj = 0; jj < N; jj++)
        {
            for (int ii = 0; ii < N; ii++)
            {
                v[ii] = v[ii] + (A_dif[ii][jj]*z[jj]) + (A_adv[ii][jj]*z[jj]*z[jj])/2.0;
            }
        }

        return v;
    }

    //! Destructor
    ~RHS_Burgers() {}
};

//? ====================================================================================== ?//