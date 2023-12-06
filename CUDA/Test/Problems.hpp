#pragma once

using namespace std;

//? ====================================================================================== ?//

struct Problems_2D
{
    int N;
    double dx;
    double dy;
    double velocity;

    //! Constructor
    Problems_2D(int _N, double _dx, double _dy, double _velocity) : N{_N}, dx{_dx}, dy{_dy}, velocity{_velocity} {}

    //! Destructor
    ~Problems_2D() {}
};

//? Periodic BC
#ifdef __CUDACC__
    __host__ __device__
#endif
int PBC(int ii, int jj, int N)
{
    if(ii < 0)
        ii = ii + N;
    if(ii >= N)
        ii = ii - N;
    if(jj < 0)
        jj = jj + N;
    if(jj >= N)
        jj = jj - N;
    return N*ii + jj;
}

//? ====================================================================================== ?//