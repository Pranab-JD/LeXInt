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

    //! Constructor
    Problems(int _N, double _dx, double _velocity) : N{_N}, dx{_dx}, velocity{_velocity} {}

    //! Destructor
    ~Problems() {}
};
