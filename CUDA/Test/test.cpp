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

#include "../Timer.hpp"

#include <sys/types.h>
#include <sys/stat.h>

//? Problems
#include "Diff_Adv.hpp"
#include "Burgers.hpp"

//! ---------------------------------------------------------------------------

//! Include Exponential Integrators and Leja functions 
//! (This has to be included to use Leja and/or exponential integrators)
#include "../Leja.hpp"
#include "../Leja_GPU.hpp"

//! Functions to compute the largest eigenvalue (in magnitude)
#include "../Eigenvalues.hpp"

//! ---------------------------------------------------------------------------

using namespace std;

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
    int N = 1e7;                                    // # grid points
    double xmin = -100.0;                           // Left boundary (limit)
    double xmax =  100.0;                           // Right boundary (limit)
    vec X(N);                                       // Array of grid points
    vec u_init(N);                                       // Initial condition
    vec u_sol(N);                                   // Solution vector

    //* Set up X array and initial condition
    for (int ii = 0; ii < N; ii++)
    {
        X[ii] = xmin + ii*(xmax - xmin)/N;
        u_init[ii] = sin(X[ii]);
    }

    //* Initialise additional parameters
    double dx = X[2] - X[1];                        // Grid spacing
    double velocity = 20;                           // Advection speed
    double dif_cfl = dx*dx;                         // Diffusion CFL
    double adv_cfl = dx/velocity;                   // Advection CFL
    double dt = 1*min(dif_cfl, adv_cfl);         // Step size
    stringstream step_size;
    step_size << fixed << scientific << setprecision(1) << dt;
    cout << "Step size: " << dt << endl;

    //! Set GPU spport to false
    bool GPU_access = false;

    //! Allocate memory on CPU
    size_t N_size = N * sizeof(double);
    double *u = (double*)malloc(N_size);
    double *u_low = (double*)malloc(N_size);
    double *u_high = (double*)malloc(N_size);
    double *error = (double*)malloc(N_size);
    double *auxillary_Jv = (double*)malloc(7*N_size);
    copy(u_init.begin(), u_init.end(), u);


    //? Define problems
    // RHS_Dif_Adv RHS = RHS_Dif_Adv(N, dx, velocity);
    //TODO: pass X as function argument
    // RHS_Dif_Source RHS = RHS_Dif_Source(N, X, dx, velocity);
    RHS_Burgers RHS = RHS_Burgers(N, dx, velocity);
    
    //* Temporal parameters
    double time = 0;                                // Simulation time
    double t_final = 0.05;                          // Final simulation time
    int time_steps = 0;                             // # time steps

    //? Shifting and scaling parameters
    // double eigen_power = -1.2*Power_iterations(RHS, u, N);        // Real eigenvalue has to be negative
    double eigen_power = -1.2*1e10;        // Real eigenvalue has to be negative
    double c = eigen_power/2.0;
    double Gamma = -eigen_power/4.0;

    // cout << "Largest eigenvalue: " << eigenvalue << endl;

    //* Set of Leja points
    vec Leja_X = Leja_Points();

    //! Construct Leja_GPU
    string integrator = "EXPRB53s3";
    Leja_GPU<RHS_Burgers> leja_gpu{N, integrator};

    //! Time Loop
    timer time_loop;
    time_loop.start();

    GPU_handle cublas_h;

    while (time < t_final)
    {
        //* Final time step
        if (time + dt >= t_final)
        {
            dt = t_final - time;
        }

        //* Solve (choose required integrator)

        //? ---------------------------------------------------------------- ?//

        //? Linear equations

        // u_sol = real_Leja_exp(RHS, u, N, Leja_X, c, Gamma, 1e-8, dt);

        //? ---------------------------------------------------------------- ?//

        //? Embedded integrators

        //? Largest eigenvalue of the Jacobian, for nonlinear equations, changes at every time step
        if (integrator == "EXPRB32" or integrator == "EXPRB42" or integrator == "EXPRB43" or integrator == "EXPRB53s3" 
        or integrator == "EPIRK4s3" or integrator == "EPIRK4s3A")
        {
            //? Embedded integrators
            leja_gpu(RHS, u, u_low, u_high, N, Leja_X, c, Gamma, 1e-8, dt, GPU_access);
            
            axpby(1.0, u_low, -1.0, u_high, error, N, GPU_access);
            double error_norm = l2norm(error, N, GPU_access, cublas_h);
            cout << "Embedded error: " << error_norm << endl;
        }
        else
        {
            cout << "ERROR: Please choose an available integator. See 'Leja.hpp'." << endl;
        }


        //? ---------------------------------------------------------------- ?//

        //* Update variables
        time = time + dt;
        swap(u, u_high);
        time_steps = time_steps + 1;
        cout << endl << endl;

        if (time_steps == 10)
            break;
    }

    time_loop.stop();

    cout << endl << "==================================================" << endl;
    cout << "Simulation time: " << time << endl;
    cout << "Total number of time steps: " << time_steps << endl;
    cout << "Total time elapsed (s): " << time_loop.total() << endl;
    cout << "==================================================" << endl << endl;

    // //! Create nested directories
    // system(("mkdir -p ../../LeXInt_Test/Cpp/Burgers/EPIRK5P1_5/dt_" + step_size.str()).c_str());

    // // Write data to files
    // ofstream outfile("../../LeXInt_Test/Cpp/Burgers/EPIRK5P1_5/dt_" + step_size.str() + "/Final_data.txt");
    // for(int ii = 0; ii < u.size(); ii++)
    // {
    //     outfile << setprecision(16) << u[ii] << endl;
    // }
    // outfile.close();

    return 0;
}
