#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>

#include "../Timer.hpp"

//? Problems
#include "Diff_Adv_1D.hpp"
#include "Burgers_1D.hpp"

//! ---------------------------------------------------------------------------

//! Include Exponential Integrators and Leja functions 
//! (This has to be included to use Leja and/or exponential integrators)
#include "../Leja_GPU.hpp"

//! Functions to compute the largest eigenvalue (in magnitude)
#include "../Eigenvalues.hpp"

//! ---------------------------------------------------------------------------

using namespace std;

//? ====================================================================================== ?//

//! Read Leja points from file
vector<double> Leja_Points()
{
    int max_Leja_pts = 1000;                        // Max. number of Leja points
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
    int N = 1e3;                                    // # grid points
    double xmin = -1;                       // Left boundary (limit)
    double xmax =  1;                       // Right boundary (limit)
    vector<double> X(N);                            // Array of grid points
    vector<double> u_init(N);                       // Initial condition

    //* Set up X array and initial condition
    for (int ii = 0; ii < N; ii++)
    {
        X[ii] = xmin + ii*(xmax - xmin)/N;
    }

    //* Initialise additional parameters
    double dx = X[12] - X[11];                      // Grid spacing
    double velocity = 150;                          // Advection speed
    double dif_cfl = dx*dx;                         // Diffusion CFL
    double adv_cfl = dx/velocity;                   // Advection CFL
    double dt = 0.8*min(dif_cfl, adv_cfl);          // Step size
    stringstream step_size;
    step_size << fixed << scientific << std::setprecision(1) << dt;
    cout << "Step size: " << dt << endl;

    //* Temporal parameters
    double time = 0;                                // Simulation time elapsed
    double t_final = 0.1;                           // Final simulation time
    int time_steps = 0;                             // # time steps

    //* Set of Leja points
    vector<double>Leja_X = Leja_Points();

    //? Choose problem and integrator
    double tol = 1e-7;
    string problem = "Burgers_1D";
    string integrator = "EXPRB43";

    RHS_Dif_Adv_1D RHS(N, dx, velocity);                    //* Default problem
    Leja_GPU<RHS_Dif_Adv_1D> leja_gpu{N, integrator};       //* Default problem

    if (problem == "Diff_Adv_1D")
    {
        //? Initial condition
        for (int ii = 0; ii < N; ii++)
        {
            u_init[ii] = 1 + exp(-(X[ii] - 10.0)*(X[ii] - 10.0)/0.4);
        }
    }
    else if (problem == "Burgers_1D")
    {
        RHS_Burgers_1D RHS = RHS_Burgers_1D(N, dx, velocity);
        Leja_GPU<RHS_Burgers_1D> leja_gpu{N, integrator};

        //? Initial condition
        for (int ii = 0; ii < N; ii++)
        {
            u_init[ii] = 2 + 0.01*sin(2*M_PI*X[ii]) + 0.01*sin(8*M_PI*X[ii] + 0.3);
        }
    }
    else
    {
        cout << "Undefined problem!" << endl;
    } 

    //! Allocate memory on CPU
    size_t N_size = N * sizeof(double);
    double *u = (double*)malloc(N_size);
    double *u_low = (double*)malloc(N_size);
    double *u_sol = (double*)malloc(N_size);
    double *u_error = (double*)malloc(N_size);
    double *auxillary_Leja = (double*)malloc(N_size);
    double *auxillary_Jv = (double*)malloc(7*N_size);
    copy(u_init.begin(), u_init.end(), u);

    //! Set GPU spport to false
    bool GPU_access = false;
    GPU_handle cublas_h;            //! To be ignored if compiled with g++ (for C++ implementation) 

    //? Shifting and scaling parameters
    double eigenvalue = 0.0;
    LeXInt::Power_iterations(RHS, u, N, eigenvalue, auxillary_Jv, GPU_access, cublas_h);         // Real eigenvalue has to be negative
    eigenvalue = -1.2*eigenvalue;
    double c = eigenvalue/2.0; double Gamma = -eigenvalue/4.0;
    cout << "Largest eigenvalue: " << eigenvalue << endl;

    //! Create nested directories
    int sys_value = system(("mkdir -p ../../LeXInt_Test/B1/"));
    string directory = "../../LeXInt_Test/B1/";

    //! Time Loop
    LeXInt::timer time_loop;
    time_loop.start();

    while (time < t_final)
    {
        //* Final time step
        if (time + dt >= t_final)
        {
            dt = t_final - time;
        }

        //? ---------------------------------------------------------------- ?//

        //? Homogenous Linear Equations

        if (integrator == "Hom_Linear")
        {
            LeXInt::real_Leja_exp(RHS, u, u_sol, auxillary_Leja, N, Leja_X, c, Gamma, tol, dt, GPU_access, cublas_h);
        }
        
        //? ---------------------------------------------------------------- ?//

        //? Nonlinear Equations

        //* Non-embedded Intergators
        else if (integrator == "Rosenbrock_Euler" or integrator == "EPIRK4s3B")
        {
            // * ----------- Eigenvalue (Spectrum) ----------- *//

            if (time_steps % 100 == 0)
            {
                //? Largest eigenvalue of the Jacobian; changes at every time step for nonlinear equations
                LeXInt::Power_iterations(RHS, u, N, eigenvalue, auxillary_Jv, GPU_access, cublas_h);         // Real eigenvalue has to be negative
                eigenvalue = -1.2*eigenvalue;
                double c = eigenvalue/2.0; double Gamma = -eigenvalue/4.0;
                cout << "Largest eigenvalue: " << eigenvalue << endl;
            }

            //* ---------------------------------------------- *//

            //? Non-embedded integrators
            leja_gpu(RHS, u, u_sol, N, Leja_X, c, Gamma, tol, dt, GPU_access);
        }

        //* Embedded Integrators 
        else if (integrator == "EXPRB32" or integrator == "EXPRB42" or integrator == "EXPRB43" or integrator == "EXPRB53s3" 
        or integrator == "EXPRB54s4" or integrator == "EPIRK4s3" or integrator == "EPIRK4s3A" or integrator == "EPIRK5P1")
        {
            // * ----------- Eigenvalue (Spectrum) ----------- *//
            
            if (time_steps % 100 == 0)
            {
                //? Largest eigenvalue of the Jacobian; changes at every time step for nonlinear equations
                LeXInt::Power_iterations(RHS, u, N, eigenvalue, auxillary_Jv, GPU_access, cublas_h);         // Real eigenvalue has to be negative
                eigenvalue = -1.2*eigenvalue;
                double c = eigenvalue/2.0; double Gamma = -eigenvalue/4.0;
                cout << "Largest eigenvalue: " << eigenvalue << endl;
            }

            //* ---------------------------------------------- *//

            //? Embedded integrators
            leja_gpu(RHS, u, u_low, u_sol, N, Leja_X, c, Gamma, tol, dt, GPU_access);
            
            LeXInt::axpby(1.0, u_low, -1.0, u_sol, u_error, N, GPU_access);
            double error = LeXInt::l2norm(u_error, N, GPU_access, cublas_h);
            // cout << "Embedded error: " << error << endl;
        }
        else
        {
            cout << "ERROR: Please choose an available integator. See 'Leja.hpp'." << endl;
        }

        //? ---------------------------------------------------------------- ?//

        //* Update variables
        time = time + dt;
        swap(u, u_sol);
        time_steps = time_steps + 1;
        
        if (time_steps % 100 == 0)
        {
            cout << "Time steps: " << time_steps << endl;
            cout << "Time elapsed: " << time << endl;
            cout << endl;
        }

        //? Write data to files
        string output_data = directory + "/" +  to_string(time_steps) + ".txt";
        ofstream data;
        data.open(output_data); 
        for(int ii = 0; ii < N; ii++)
        {
            data << setprecision(16) << u[ii] << endl;
        }
        data.close();
    }

    time_loop.stop();

    cout << endl << "==================================================" << endl;
    cout << "Simulation time: " << time << endl;
    cout << "Total number of time steps: " << time_steps << endl;
    cout << "Total time elapsed (s): " << time_loop.total() << endl;
    cout << "==================================================" << endl << endl;

    //! Create nested directories
    // int sys_value = system(("mkdir -p ../../LeXInt_Test/" + to_string(GPU_access) + "/Constant/" + problem + "/dt_" + step_size.str()).c_str());
    // string directory = "../../LeXInt_Test/" + to_string(GPU_access) + "/Constant/" + problem + "/dt_" + step_size.str();

    // //? Write data to files
    // string final_data = directory + "/1.txt";
    // ofstream data;
    // data.open(final_data);
    // for(int ii = 0; ii < N; ii++)
    // {
    //     data << setprecision(16) << u[ii] << endl;
    // }
    // data.close();

    // string results = directory + "/Results.txt";
    // ofstream params;
    // params.open(results);
    // params << "Simulation time: " << time << endl;
    // params << "Total number of time steps: " << time_steps << endl;
    // params << setprecision(16) << "Total time elapsed (s): " << time_loop.total() << endl;
    // params.close();

    // cout << "Writing data to files complete!" << endl << endl;

    return 0;
}