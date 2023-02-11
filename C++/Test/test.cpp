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
#include <numeric>

#include <sys/types.h>
#include <sys/stat.h>

//? Problems
#include "Diff_Adv.hpp"
#include "Burgers.hpp"

//! ---------------------------------------------------------------------------

//! Include Exponential Integrators and Leja functions 
//! (This has to be included to use Leja and/or exponential integrators)
#include "../Leja.hpp"

//! Functions to compute the largest eigenvalue (in magnitude)
#include "../Eigenvalues.hpp"

//! ---------------------------------------------------------------------------

using namespace std;

//? Simplfied definition of vector<vector<double>>
using matrix = vector<vector<double>>;

//? ====================================================================================== ?//

//! Read Leja points from file
vector<double> Leja_Points()
{
    int max_Leja_pts = 100;                         // Max. number of Leja points
    vector<double> Leja_X(max_Leja_pts);            // Initialize static array
    int count = 0;                                  // Loop counter variable

    //* Load Leja points
    ifstream inputFile;
    inputFile.open("../Leja_10000.txt");

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
    double dt = 0.4*min(dif_cfl, adv_cfl);         // Step size
    stringstream step_size;
    step_size << fixed << scientific << setprecision(1) << dt;
    cout << endl << "Step size: " << dt << endl;

    //? Define problems
    // RHS_Dif_Adv RHS = RHS_Dif_Adv(N, dx, velocity);
    RHS_Burgers RHS = RHS_Burgers(N, dx, velocity);
    
    //* Temporal parameters
    double time = 0;                                // Simulation time
    double t_final = 0.05;                          // Final simulation time
    int time_steps = 0;                             // # time steps

    //? Shifting and scaling parameters
    double eigen_power;                                         // Eigenvalue using power iterations
    double eigen_dif = -1.2*Power_iterations(RHS, u, N);        // Real eigenvalue has to be negative
    double c = eigen_dif/2.0;
    double Gamma = -eigen_dif/4.0;

    cout << "Eigenvalue Diffusion: " << eigen_dif << endl;

    //* Set of Leja points
    vec Leja_X = Leja_Points();

    //? Timer
    struct timespec total_start, total_finish, total_elapsed;
    struct timespec leja_start, leja_finish, leja_elapsed;
    vec time_leja;

    //! Time Loop
    clock_gettime(CLOCK_REALTIME, &total_start);

    while (time < t_final)
    {
        //* Final time step
        if (time + dt >= t_final)
        {
            dt = t_final - time;
        }

        //* Solve (choose required integrator)

        //? ---------------------------------------------------------------- ?//

        //? RK2, RK4

        // u_sol = RK2(RHS, u, N, dt);

        //? ---------------------------------------------------------------- ?//

        //? Linear equations

        // u_sol = real_Leja_exp(RHS, u, N, Leja_X, c, Gamma, 1e-8, dt);

        //? ---------------------------------------------------------------- ?//

        //? Embedded integrators

        clock_gettime(CLOCK_REALTIME, &leja_start);

        //? Largest eigenvalue of the Jacobian, for nonlinear equations, changes at every time step
        eigen_power = -1.2*Power_iterations(RHS, u, N); 
        // cout << "Eigenvalue PowerIters : " << eigen_power << endl;

        // u_sol = EPIRK4s3B(RHS, u, N, Leja_X, c, Gamma, 1e-10, dt, 0);

        u_sol_embed = EPIRK5P1(RHS, u, N, Leja_X, c, Gamma, 1e-10, dt, 0);
        // vec error_vec = axpby(1.0, u_sol_embed.higher_order_solution, -1.0, u_sol_embed.lower_order_solution, N);
        // cout << "Embedded error: " << *max_element(begin(error_vec), end(error_vec)) << endl << endl;


        clock_gettime(CLOCK_REALTIME, &leja_finish);
        sub_timespec(leja_start, leja_finish, &leja_elapsed);
        double leja_time = (int)leja_elapsed.tv_sec + (pow(10, -9) * leja_elapsed.tv_nsec);
        time_leja.push_back(leja_time);

        //? ---------------------------------------------------------------- ?//

        //* Update variables
        time = time + dt;
        u = u_sol_embed.higher_order_solution;
        // u = u_sol;
        time_steps = time_steps + 1;
    }

    //? Timers
    clock_gettime(CLOCK_REALTIME, &total_finish);
    sub_timespec(total_start, total_finish, &total_elapsed);
    double total_time = (int)total_elapsed.tv_sec + (pow(10, -9) * total_elapsed.tv_nsec);

    double leja_cost = accumulate(time_leja.begin(), time_leja.end(), 0.0f);


    cout << endl << "==================================================" << endl;
    cout << "Simulation time: " << time << endl;
    cout << "Total number of time steps: " << time_steps << endl;
    cout << "Total time elapsed (s): " << total_time << endl;
    cout << "Time elapsed for Leja method (s): " << leja_cost << endl;
    cout << "==================================================" << endl << endl;

    //! Create nested directories
    system(("mkdir -p ../../LeXInt_Test/Cpp/Burgers/EPIRK5P1_5/dt_" + step_size.str()).c_str());

    // Write data to files
    ofstream outfile("../../LeXInt_Test/Cpp/Burgers/EPIRK5P1_5/dt_" + step_size.str() + "/Final_data.txt");
    for(int ii = 0; ii < u.size(); ii++)
    {
        outfile << setprecision(16) << u[ii] << endl;
    }
    outfile.close();

    return 0;
}
