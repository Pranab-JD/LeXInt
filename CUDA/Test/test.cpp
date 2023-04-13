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
#include "../functions.hpp"

//? CUDA
#include "../Leja.hpp"
#include "../Leja_GPU.hpp"

//! ---------------------------------------------------------------------------

//! Include Exponential Integrators and Leja functions 
//! (This has to be included to use Leja and/or exponential integrators)
#include "../Leja.hpp"

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
    double velocity = 20;                           // Advection speed
    double dif_cfl = dx*dx;                         // Diffusion CFL
    double adv_cfl = dx/velocity;                   // Advection CFL
    double dt = 0.1*min(dif_cfl, adv_cfl);         // Step size
    stringstream step_size;
    step_size << fixed << scientific << setprecision(1) << dt;
    cout << "Step size: " << dt << endl;

    //! Transfer to GPU
    size_t N_size = N * sizeof(double);
    double *device_u; cudaMalloc(&device_u, N_size);
    double *device_u_sol; cudaMalloc(&device_u_sol, N_size);
    cudaMemcpy(device_u, &u[0], N_size, cudaMemcpyHostToDevice);
        
    //? Cublas Handle
    cublasHandle_t cublas_handle;
    cublasCreate_v2(&cublas_handle);

    // int threads_per_block = 128;
    // int blocks_per_grid = (N/threads_per_block) + 1;

    //? Define problems
    // RHS_Dif_Adv RHS = RHS_Dif_Adv(N, dx, velocity);
    //TODO: pass X as function argument
    // RHS_Dif_Source RHS = RHS_Dif_Source(N, X, dx, velocity);
    RHS_Burgers RHS = RHS_Burgers(N, dx, velocity);
    
    //* Temporal parameters
    double time = 0;                                // Simulation time
    double t_final = 0.01;                          // Final simulation time
    int time_steps = 0;                             // # time steps

    //? Shifting and scaling parameters
    double eigenvalue = 0.0;
    Power_iterations(RHS, device_u, N, eigenvalue, cublas_handle);         // Real eigenvalue has to be negative
    eigenvalue = -1.2*eigenvalue;
    double c = eigenvalue/2.0; double Gamma = -eigenvalue/4.0;

    cout << "Largest eigenvalue: " << eigenvalue << endl;

    //* Set of Leja points
    vec Leja_X = Leja_Points();

    //? Timer
    struct timespec total_start, total_finish, total_elapsed;
    struct timespec leja_start, leja_finish, leja_elapsed;
    vec time_leja;

    //! Copy state variable to device
    cudaMemcpy(device_u, &u[0], N_size, cudaMemcpyHostToDevice);

    //! Construct Leja_GPU
    Leja_GPU<RHS_Burgers> leja_gpu{N, cublas_handle};

    //! Time Loop
    clock_gettime(CLOCK_REALTIME, &total_start);

    // Ros_Eu(RHS, device_u, device_u_sol, N, Leja_X, c, Gamma, 1e-8, dt, leja_gpu);

    while (time < t_final)
    {
        //* Final time step
        if (time + dt >= t_final)
        {
            dt = t_final - time;
        }

        //* Timer
        clock_gettime(CLOCK_REALTIME, &leja_start);

        //? ---------------------------------------------------------------- ?//

        //? Linear equations

        // leja_gpu.real_Leja_exp(RHS, device_u, device_u_sol, Leja_X, c, Gamma, 1e-8, dt);

        //? ---------------------------------------------------------------- ?//

        //? Embedded integrators

        //* -------------------------------- *//

        //? Largest eigenvalue of the Jacobian, for nonlinear equations, changes at every time step
        Power_iterations(RHS, device_u, N, eigenvalue, cublas_handle);         // Real eigenvalue has to be negative
        eigenvalue = -1.2*eigenvalue;
        c = eigenvalue/2.0; Gamma = -eigenvalue/4.0;

        cout << "Largest eigenvalue: " << eigenvalue << endl;

        //* -------------------------------- *//

        Ros_Eu(RHS, device_u, device_u_sol, N, Leja_X, c, Gamma, 1e-8, dt, leja_gpu);

        // u_sol_embed = EPIRK5P1(RHS, u, N, Leja_X, c, Gamma, 1e-10, dt, 0);
        // vec error_vec = axpby(1.0, u_sol_embed.higher_order_solution, -1.0, u_sol_embed.lower_order_solution, N);
        // cout << "Embedded error: " << *max_element(begin(error_vec), end(error_vec)) << endl << endl;


        clock_gettime(CLOCK_REALTIME, &leja_finish);
        sub_timespec(leja_start, leja_finish, &leja_elapsed);
        double leja_time = (int)leja_elapsed.tv_sec + (pow(10, -9) * leja_elapsed.tv_nsec);
        time_leja.push_back(leja_time);

        //? ---------------------------------------------------------------- ?//

        //* Update variables
        time = time + dt;
        // u = u_sol_embed.higher_order_solution;
        device_u = device_u_sol;
        time_steps = time_steps + 1;

        // break;
    }

    //? Timers
    clock_gettime(CLOCK_REALTIME, &total_finish);
    sub_timespec(total_start, total_finish, &total_elapsed);
    double total_time = (int)total_elapsed.tv_sec + (pow(10, -9) * total_elapsed.tv_nsec);

    double leja_cost = accumulate(time_leja.begin(), time_leja.end(), 0.0f);

    // cout << setprecision(16) << "Norm of u (test.cpp): " << l2norm(u, N) << endl;

    cout << endl << "==================================================" << endl;
    cout << "Simulation time: " << time << endl;
    cout << "Total number of time steps: " << time_steps << endl;
    cout << "Total time elapsed (s): " << total_time << endl;
    cout << "Time elapsed for Leja method (s): " << leja_cost << endl;
    cout << "==================================================" << endl << endl;

    //! Create nested directories
    // system(("mkdir -p ../../LeXInt_Test/Cpp/Burgers/EPIRK5P1_5/dt_" + step_size.str()).c_str());

    // // Write data to files
    // ofstream outfile("../../LeXInt_Test/Cpp/Burgers/EPIRK5P1_5/dt_" + step_size.str() + "/Final_data.txt");
    // for(int ii = 0; ii < u.size(); ii++)
    // {
    //     outfile << setprecision(16) << u[ii] << endl;
    // }
    // outfile.close();
    
    cublasDestroy_v2(cublas_handle);

    return 0;
}
