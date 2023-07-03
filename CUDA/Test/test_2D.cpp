#include <cmath>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>

#include "../Timer.hpp"

//? Problems
#include "Dif_Adv_2D.hpp"
#include "Burgers_2D.hpp"

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
    int max_Leja_pts = 1000;                         // Max. number of Leja points
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
    int n = 8192;                                   // # grid points
    int N = n*n;                                    // # grid points
    double xmin = -1;                               // Left boundary (limit)
    double xmax =  1;                               // Right boundary (limit)
    double ymin = -1;                               // Left boundary (limit)
    double ymax =  1;                               // Right boundary (limit)
    vector<double> X(n);                            // Array of grid points
    vector<double> Y(n);                            // Array of grid points
    vector<double> u_init(N);                       // Initial condition

    //* Set up X, Y arrays and initial condition
    for (int ii = 0; ii < n; ii++)
    {
        X[ii] = xmin + ii*(xmax - xmin)/n;
        Y[ii] = ymin + ii*(ymax - ymin)/n;
    }

    //* Initialise additional parameters
    double dx = X[12] - X[11];                              // Grid spacing
    double dy = Y[12] - Y[11];                              // Grid spacing
    double velocity = 50;                                   // Advection speed

    //* Temporal parameters
    double time = 0;                                        // Simulation time elapsed
    double t_final = 0.00005;                                // Final simulation time
    int time_steps = 0;                                     // # time steps

    double dif_cfl = (dx*dx * dy*dy)/(2*dx*dx + 2*dy*dy);   // Diffusion CFL
    double adv_cfl = dx*dy/(velocity * (dx + dy));          // Advection CFL
    double dt = 4.0*min(dif_cfl, adv_cfl);                  // Step size
    cout << endl << "Step size: " << dt << endl;

    //? Strings for directory names
    stringstream step_size, tf, grid;
    step_size << fixed << scientific << setprecision(1) << dt;
    tf << fixed << scientific << setprecision(1) << t_final;
    grid << fixed << scientific << setprecision(0) << n;

    //* Set of Leja points
    vector<double> Leja_X = Leja_Points();

    //? Choose problem and integrator
    double tol = 1e-12;
    string problem = "Diff_Adv_2D";
    string integrator = "Hom_Linear";

    RHS_Dif_Adv_2D RHS(n, dx, dy, velocity);                //* Default problem
    Leja_GPU<RHS_Dif_Adv_2D> leja_gpu{N, integrator};       //* Default problem

    if (problem == "Diff_Adv_2D")
    {
        //? Initial condition
        for (int ii = 0; ii < n; ii++)
        {
            for (int jj = 0; jj< n; jj++)
            {
                u_init[n*ii + jj] = 1 + exp(-((X[ii] + 0.5)*(X[ii] + 0.5) + (Y[jj] + 0.5)*(Y[jj] + 0.5))/0.01);
            }
        }
    }
    else if (problem == "Burgers_2D")
    {
        RHS_Burgers_2D RHS = RHS_Burgers_2D(n, dx, dy, velocity);
        Leja_GPU<RHS_Burgers_2D> leja_gpu{N, integrator};

        //? Initial condition
        for (int ii = 0; ii < n; ii++)
        {
            for (int jj = 0; jj< n; jj++)
            {
                u_init[n*ii + jj] = 2 + 0.01*sin(2*M_PI*X[ii]) + 0.01*sin(8*M_PI*X[ii] + 0.3)
                                      + 0.01*sin(2*M_PI*Y[jj]) + 0.01*sin(8*M_PI*Y[jj] + 0.3);
            }
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

    //! Create nested directories (for movies)
    // int sys_value = system(("mkdir -p ../../LeXInt_Test/DA/"));
    // string directory = "../../LeXInt_Test/DA/";

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

            //? Embedded integrators
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

        if (time_steps % 500 == 0)
        {
            cout << "Time steps: " << time_steps << endl;
            cout << "Time elapsed: " << time << endl;
            cout << endl;
        }

        //! Write data to files (for movies)
        // string output_data = directory + "/" +  to_string(time_steps) + ".txt";
        // ofstream data;
        // data.open(output_data); 
        // for(int ii = 0; ii < N; ii++)
        // {
        //     data << setprecision(16) << u[ii] << endl;
        // }
        // data.close();
    }

    time_loop.stop();

    cout << endl << "==================================================" << endl;
    cout << "Simulation time: " << time << endl;
    cout << "Total number of time steps: " << time_steps << endl;
    cout << "Total time elapsed (s): " << time_loop.total() << endl;
    cout << "==================================================" << endl << endl;

    //! Create nested directories
    int sys_value_f = system(("mkdir -p ../../LeXInt_Test/" + to_string(GPU_access) + "/Constant/" + problem 
                                + "/N_" + grid.str().c_str() + "/t_" + tf.str().c_str() + "/dt_" + step_size.str()).c_str());
    string directory_f = "../../LeXInt_Test/" + to_string(GPU_access) + "/Constant/" + problem 
                                + "/N_" + grid.str().c_str() + "/t_" + tf.str().c_str() + "/dt_" + step_size.str().c_str();

    //? Write data to files
    string final_data = directory_f + "/Final_data.txt";
    ofstream data;
    data.open(final_data);
    for(int ii = 0; ii < N; ii++)
    {
        data << setprecision(16) << u[ii] << endl;
    }
    data.close();

    string results = directory_f + "/Results.txt";
    ofstream params;
    params.open(results);
    params << "Simulation time: " << time << endl;
    params << "Total number of time steps: " << time_steps << endl;
    params << setprecision(16) << "Total time elapsed (s): " << time_loop.total() << endl;
    params.close();

    cout << "Writing data to files complete!" << endl;

    return 0;
}