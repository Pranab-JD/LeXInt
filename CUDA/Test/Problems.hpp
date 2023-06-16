#pragma once

using namespace std;

//? ====================================================================================== ?//

struct Problems_1D
{
    int N;
    double dx;
    double velocity;

    //! Constructor
    Problems_1D(int _N, double _dx, double _velocity) : N{_N}, dx{_dx}, velocity{_velocity} {}

    //! Destructor
    ~Problems_1D() {}
};

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

//? ====================================================================================== ?//