#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <functional>
#include <fstream>
#include <sstream>
#include <string>
#include <typeinfo>
#include <algorithm>

#include <sys/types.h>
#include <sys/stat.h>

//? Problems
#include "Diff_Adv.hpp"
#include "Burgers.hpp"

//? Solvers
#include "RK.hpp"
#include "real_Leja_exp.hpp"
#include "real_Leja_phi.hpp"

//! Include all solvers simultaneously
#include "./Integrators/Rosenbrock_Euler.hpp"
#include "./Integrators/EXPRB32.hpp"
#include "./Integrators/EXPRB42.hpp"
#include "./Integrators/EXPRB43.hpp"
#include "./Integrators/EXPRB53s3.hpp"
#include "./Integrators/EXPRB54s4.hpp"

#include "./Integrators/EPIRK4s3.hpp"
#include "./Integrators/EPIRK4s3A.hpp"
#include "./Integrators/EPIRK4s3B.hpp"

using namespace std;

using matrix = vector<vector<double>>;

//? ====================================================================================== ?//

template <typename T1, typename T2>
double Gershgorin(T1 A, T2 N)
{
    vector<double> eigen_list(N);
        
    for (int ii = 0; ii < N; ii++)
    {
        double eigenvalue = 0;

        for (int jj = 0; jj < N; jj++)
        {
            eigenvalue = eigenvalue + abs(A[ii][jj]);
        }

        eigen_list[ii] = eigenvalue;
    }
    return *max_element(begin(eigen_list), end(eigen_list));
}

//! Read Leja points from file
vector<double> Leja_Points()
{
    int max_Leja_pts = 100;                         // Max. number of Leja points
    vector<double> Leja_X(max_Leja_pts);            // Initialize static array
    int count = 0;                                  // Loop counter variable

    //* Load Leja points
    ifstream inputFile;
    inputFile.open("./Leja_10000.txt");

    //* Read Leja_points from file into the vector Leja_X
    while(count < max_Leja_pts && inputFile >> Leja_X[count])
    {
        count = count + 1;
    }

    inputFile.close();

    return Leja_X;
}

//? ====================================================================================== ?//

int main()
{
    //* Initialise parameters
    int N = 100;                                    // # grid points
    double xmin = -1.0;                             // Left boundary (limit)
    double xmax =  1.0;                             // Right boundary (limit)
    vec X(N);                                       // Array of grid points
    vec u(N);                                       // Initial condition
    vec u_sol(N);                                   // Solution vector
    embedded_solutions<vec> u_sol_embed;

    //* Set up X array and initial condition
    for (int ii = 0; ii < N; ii++)
    {
        X[ii] = xmin + ii*(xmax - xmin)/N;
        u[ii] = sin(X[ii]);
    }

    //* Initialise additional parameters
    double dx = X[2] - X[1];                        // Grid spacing
    double velocity = 50;                           // Advection speed
    double dif_cfl = dx*dx;                         // Diffusion CFL
    double adv_cfl = dx/velocity;                   // Advection CFL
    double dt = 0.075*min(dif_cfl, adv_cfl);         // Step size
    stringstream step_size;
    step_size << fixed << scientific << setprecision(1) << dt;
    cout << "Step size: " << dt << endl;

    //? Define problems
    // RHS_Dif_Adv RHS = RHS_Dif_Adv(N, dx, velocity);
    RHS_Burgers RHS = RHS_Burgers(N, dx, velocity);
    
    //* Temporal parameters
    double time = 0;                                // Simulation time
    double t_final = 0.05;                          // Final simulation time
    int time_steps = 0;                             // # time steps

    //? Shifting and scaling parameters
    double eigen_dif = -1.0*Gershgorin(RHS.A_dif, N);   // Real eigenvalue has to be negative
    double c = eigen_dif/2.0;
    double Gamma = -eigen_dif/4.0;

    //* Set of Leja points
    vec Leja_X = Leja_Points();

    cout << "Eigenvalue Diffusion: " << eigen_dif << endl;
    vec error_vec;
    double error;

    //! Time Loop
    while (time < t_final)
    {
        //* Final time step
        if (time + dt >= t_final)
        {
            dt = t_final - time;
        }

        //* Solve (choose required integrator)
        // u_sol = RK2(RHS, u, N, dt);

        // u_sol = real_Leja_exp(RHS, u, N, Leja_X, c, Gamma, 1e-8, dt);

        // u_sol = EPIRK4s3B(RHS, u, N, Leja_X, c, Gamma, 1e-10, dt, 0);

        u_sol_embed = EXPRB42(RHS, u, N, Leja_X, c, Gamma, 1e-10, dt, 0);

        // error_vec = axpby(1.0, u_sol_embed.higher_order_solution, -1.0, u_sol_embed.lower_order_solution, N);
        
        // cout << "Embedded error: " << *max_element(begin(error_vec), end(error_vec)) << endl << endl;

        //* Update variables
        time = time + dt;
        u = u_sol_embed.higher_order_solution;
        // u = u_sol;
        time_steps = time_steps + 1;
    }

    cout << endl << "==================================================" << endl;
    cout << "Simulation time: " << time << endl;
    cout << "Total # of time steps: " << time_steps << endl;
    cout << "==================================================" << endl << endl;

    //! Create nested directories
    system(("mkdir -p ../LeXInt_Test/Cpp/Burgers/EXPRB42_4/dt_" + step_size.str()).c_str());

    //* Write data to files
    ofstream outfile("../LeXInt_Test/Cpp/Burgers/EXPRB42_4/dt_" + step_size.str() + "/Final_data.txt");
    for(int ii = 0; ii < u.size(); ii++)
    {
        outfile << setprecision(16) << u[ii] << endl;
    }
    outfile.close();

    return 0;
}
