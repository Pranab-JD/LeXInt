#pragma once

//? Include Leja interpolation functions
#include "real_Leja_exp.hpp"
#include "real_Leja_phi.hpp"
#include "real_Leja_phi_nl.hpp"

//? Include all solvers
#include "./Integrators/Rosenbrock_Euler.hpp"       //! 2nd order; no embedded error estimate
// #include "./Integrators/EXPRB32.hpp"                //! 2nd and 3rd order
// #include "./Integrators/EXPRB42.hpp"                //! 2nd and 4th order
// #include "./Integrators/EXPRB43.hpp"                //! 3rd and 4th order
// #include "./Integrators/EXPRB53s3.hpp"              //! 3rd and 5th order
// #include "./Integrators/EXPRB54s4.hpp"              //! 4th and 5th order

// #include "./Integrators/EPIRK4s3.hpp"               //! 3rd and 4th order
// #include "./Integrators/EPIRK4s3A.hpp"              //! 3rd and 4th order
// #include "./Integrators/EPIRK4s3B.hpp"              //! 4th order; no embedded error estimate
// #include "./Integrators/EPIRK5P1.hpp"               //! 4th and 5th order
