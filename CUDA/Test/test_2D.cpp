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
    //! Set GPU spport to false
    bool GPU_access = false;
    GPU_handle cublas_h;            //! To be ignored if compiled with g++ (for C++ implementation) 

    //* Initialise parameters
    int n = pow(2, 11);                             // # grid points (1D)
    int N = n*n;                                    // # grid points (2D)
    double xmin = -1;                               // Left boundary (limit)
    double xmax =  1;                               // Right boundary (limit)
    double ymin = -1;                               // Left boundary (limit)
    double ymax =  1;                               // Right boundary (limit)
    vector<double> X(n);                            // Array of grid points
    vector<double> Y(n);                            // Array of grid points
    vector<double> u_init(N);                       // Initial condition
    vector<double> Source(N);                       // Source term

    //* Set up X, Y arrays and initial condition
    for (int ii = 0; ii < n; ii++)
    {
        X[ii] = xmin + ii*(xmax - xmin)/n;
        Y[ii] = ymin + ii*(ymax - ymin)/n;
    }

    //* Initialise additional parameters
    double dx = X[12] - X[11];                              // Grid spacing
    double dy = Y[12] - Y[11];                              // Grid spacing
    double velocity = 500;                                   // Advection speed

    //* Temporal parameters
    double time = 0;                                        // Simulation time elapsed
    double t_final = 5e-4;                                  // Final simulation time
    int time_steps = 0;                                     // # time steps

    double dif_cfl = (dx*dx * dy*dy)/(2*dx*dx + 2*dy*dy);   // Diffusion CFL
    double adv_cfl = dx*dy/(velocity * (dx + dy));          // Advection CFL
    double dt = 10.0*min(dif_cfl, adv_cfl);                 // Step size
    cout << endl << "Step size: " << dt << endl;

    //* Set of Leja points
    vector<double> Leja_X = Leja_Points();
    int iters = 0;                                          //* # of Leja points used per iteration (iteration variable for Leja interpolation)
    int iters_total = 0;                                    //* Total # of Leja iterations during the simulation

    //? Choose problem and integrator
    double tol = 1e-5;
    string problem = "Burgers_2D";
    string integrator = "EPIRK4s3A";

    RHS_Dif_Adv_2D RHS(n, dx, dy, velocity);                //* Default problem
    Leja_GPU<RHS_Dif_Adv_2D> leja_gpu{N, integrator};       //* Default problem

    //? Strings for directory names
    stringstream step_size, tf, grid, acc;
    step_size << fixed << scientific << setprecision(1) << dt;
    tf << fixed << scientific << setprecision(1) << t_final;
    grid << fixed << scientific << setprecision(0) << n;
    acc << fixed << scientific << setprecision(0) << tol;

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
    else if (problem == "Diff_Adv_Source_2D")
    {
        //? Initial condition and Source
        for (int ii = 0; ii < n; ii++)
        {
            for (int jj = 0; jj< n; jj++)
            {
                Source[n*ii + jj] = 10*exp(-((X[ii] - 0.4)*(X[ii] - 0.4) + (Y[jj] - 0.4)*(Y[jj] - 0.4))/0.01);
                u_init[n*ii + jj] = 1 + 1e-2*exp(-((X[ii] + 0.75)*(X[ii] + 0.75) + (Y[jj] + 0.3)*(Y[jj] + 0.3))/0.01);
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
    copy(u_init.begin(), u_init.end(), u);
    double *u_sol = (double*)malloc(N_size);            //* Solution vector
    double *auxiliary_Jv = (double*)malloc(4*N_size);   //* To compute spectrum using power iterations
    
    double *auxiliary_Leja;                             //? Only for linear problems
    double *u_low;                                      //? Only for nonlinear problems
    double *u_error;                                    //? Only for nonlinear problems
    double error;
    double *interp_vector;                              //? Only for nonhomogenous linear problems
    double *source;                                     //? Only for nonhomogenous linear problems
    
    if (integrator == "Hom_Linear" or integrator == "Rosenbrock_Euler" or integrator == "EPIRK4s3B" )
    {
        auxiliary_Leja = (double*)malloc(N_size);
    }
    else if (integrator == "NonHom_Linear")
    {
        //? Vector to be interpolated
        interp_vector = (double*)malloc(N_size);
        
        //? Source
        source = (double*)malloc(N_size);
        copy(Source.begin(), Source.end(), source);         //? Only for nonhomogenous linear problems
    }
    else
    {
        u_low = (double*)malloc(N_size);
        u_error = (double*)malloc(N_size);
    }

    //? Shifting and scaling parameters
    double eigenvalue = 0.0;
    LeXInt::Power_iterations(RHS, u, N, eigenvalue, auxiliary_Jv, GPU_access, cublas_h);         // Real eigenvalue has to be negative
    eigenvalue = -1.2*eigenvalue;
    double c = eigenvalue/2.0; double Gamma = -eigenvalue/4.0;
    cout << "Largest eigenvalue: " << eigenvalue << endl;

    //! Create nested directories (for movies)
    // int sys_value = system(("mkdir -p ../../LeXInt_Test/DAS/"));
    // string directory = "../../LeXInt_Test/DAS/";

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

        //! Homogenous Linear Equations

        if (integrator == "Hom_Linear")
        {
            LeXInt::real_Leja_exp(RHS, u, u_sol, auxiliary_Leja, N, Leja_X, c, Gamma, tol, dt, iters, GPU_access, cublas_h);
        }

        //? ---------------------------------------------------------------- ?//

        //! Nonhomogenous Linear Equations

        else if (integrator == "NonHom_Linear")
        {
            LeXInt::axpby(1.0, source, 1.0, u, interp_vector, N, GPU_access);
            LeXInt::axpby(dt, interp_vector, interp_vector, N, GPU_access);

            LeXInt::real_Leja_phi_nl(RHS, interp_vector, u_sol, auxiliary_Leja, N, LeXInt::phi_1, Leja_X, c, Gamma, tol, dt, iters, GPU_access, cublas_h);
        }
        
        //? ---------------------------------------------------------------- ?//

        //! Nonlinear Equations

        //* Non-embedded Intergators
        else if (integrator == "Rosenbrock_Euler" or integrator == "EPIRK4s3B")
        {
			//* Largest eigenvalue (spectrum) every 500 time steps
            if (time_steps != 0 && time_steps % 500 == 0)
            {
                //? Largest eigenvalue of the Jacobian; changes at every time step for nonlinear equations
                LeXInt::Power_iterations(RHS, u, N, eigenvalue, auxiliary_Jv, GPU_access, cublas_h);         // Real eigenvalue has to be negative
                eigenvalue = -1.2*eigenvalue;
                double c = eigenvalue/2.0; double Gamma = -eigenvalue/4.0;
            }

           //? Non-embedded integrators
            leja_gpu(RHS, u, u_sol, N, Leja_X, c, Gamma, tol, dt, iters, GPU_access);
        }

        //* Embedded Integrators 
        else if (integrator == "EXPRB32" or integrator == "EXPRB42" or integrator == "EXPRB43" or integrator == "EXPRB53s3" 
        or integrator == "EXPRB54s4" or integrator == "EPIRK4s3" or integrator == "EPIRK4s3A" or integrator == "EPIRK5P1")
        {
			//* Largest eigenvalue (spectrum) every 500 time steps
            if (time_steps != 0 && time_steps % 500 == 0)
            {
                //? Largest eigenvalue of the Jacobian; changes at every time step for nonlinear equations
                double eigenvalue = 0.0;
                LeXInt::Power_iterations(RHS, u, N, eigenvalue, auxiliary_Jv, GPU_access, cublas_h);         // Real eigenvalue has to be negative
                eigenvalue = -1.2*eigenvalue;
                double c = eigenvalue/2.0; double Gamma = -eigenvalue/4.0;
            }

            //? Embedded integrators
            leja_gpu(RHS, u, u_low, u_sol, N, Leja_X, c, Gamma, tol, dt, iters, GPU_access);
            
            //? Error estimate
            LeXInt::axpby(1.0, u_low, -1.0, u_sol, u_error, N, GPU_access);
            error = LeXInt::l2norm(u_error, N, GPU_access, cublas_h);

            if (time_steps % 1 == 0)
            {
                cout << "Embedded error: " << error << endl;
            }
        }
        else
        {
            cout << "ERROR: Please choose an available integator. See 'Leja.hpp'." << endl;
        }

        //? ---------------------------------------------------------------- ?//

        //* Update variables
        time = time + dt;
        time_steps = time_steps + 1;
        iters_total = iters_total + iters;

        if (time_steps == 10)
            break;

        if (integrator == "NonHom_Linear")
        {
            LeXInt::axpby(1.0, u, 1.0, u_sol, u, N, GPU_access);
        }
        else
        {
            swap(u, u_sol);
        }

        if (time_steps % 500 == 0)
        {
            cout << "Largest eigenvalue: " << eigenvalue << endl;
            cout << "Time steps: " << time_steps << endl;
            cout << "Time elapsed: " << time << endl;
            cout << endl;
        }

        //! Write data to files (for movies)
        // if (time_steps % 50 == 0)
        // {
        
        //     string output_data = directory + "/" +  to_string(time_steps) + ".txt";
        //     ofstream data;
        //     data.open(output_data); 
        //     for(int ii = 0; ii < N; ii++)
        //     {
        //         data << setprecision(16) << u[ii] << endl;
        //     }
        //     data.close();
        // }
    }

    time_loop.stop();

    //? ---------------------- Bandwidth computation ---------------------- ?//

    //! Number of vector reads and writes (to compute achieved bandwidth):
    //! Note: Vector reads and writes per RHS computation = 2.
    //! This number can vary depending on how the RHS function is defined.
    //? real_Leja_phi = 21 + (num_interpolations * 3)

    double reads_writes;
    if (integrator == "Hom_Linear")
    {
        reads_writes = 11.0 * iters_total;
    }
    else if (integrator == "NonHom_Linear")
    {
        reads_writes = (8.0 * time_steps) + (10.0 * iters_total);
    }
    else if (integrator == "Rosenbrock_Euler")
    {
        reads_writes = (8.0 * time_steps) + (24 * iters_total);
    }
    else if (integrator == "EXPRB32")
    {
        reads_writes = (56.0 * time_steps) + ((24 + 24)/2.0 * iters_total);
    }
    else if (integrator == "EXPRB42")
    {
        reads_writes = (59.0 * time_steps) + ((27 + 24)/2.0 * iters_total);
    }
    else if (integrator == "EXPRB43")
    {
        reads_writes = (93.0 * time_steps) + ((27 + 24 + 24 + 24)/4.0 * iters_total);
    }
    else if (integrator == "EPIRK4s3" or integrator == "EPIRK4s3A")
    {
        reads_writes = (93.0 * time_steps) + ((30 + 24 + 24)/3.0 * iters_total);
    }
        else if (integrator == "EPIRK5P1")
    {
        reads_writes = (94.0 * time_steps) + ((30 + 30 + 27)/3.0 * iters_total);
    }
    else if (integrator == "EXPRB53s3")
    {
        reads_writes = (100.0 * time_steps) + ((30 + 27 + 24 + 24 + 24)/5.0 * iters_total);
    }
    else if (integrator == "EXPRB54s4")
    {
        reads_writes = (132.0 * time_steps) + ((33 + (24 * 6))/7.0 * iters_total);
    }
    else
    {
        
    }

    double bandwidth = (1.0 * N * 8 *  reads_writes * 1e-9)/time_loop.total();

    cout << endl << "==================================================" << endl;
    cout << "Simulation time: " << time << endl;
    cout << "Total number of time steps: " << time_steps << endl;
    cout << "Total number of Leja iterations: " << iters_total << endl;
    cout << "Total time elapsed (s): " << time_loop.total() << endl;
    cout << "Average Bandwidth (GB/s): " << bandwidth << endl;
    cout << "==================================================" << endl << endl;

    // //! Create nested directories
    // int sys_value_f = system(("mkdir -p ../../LeXInt_Test/" + to_string(GPU_access) + "/Constant/" + problem + "/" + integrator
    //                             + "/N_" + grid.str().c_str() + "/t_" + tf.str().c_str() + "/dt_" + step_size.str().c_str() + "/tol_" + acc.str()).c_str());
    // string directory_f = "../../LeXInt_Test/" + to_string(GPU_access) + "/Constant/" + problem + "/" + integrator
    //                             + "/N_" + grid.str().c_str() + "/t_" + tf.str().c_str() + "/dt_" + step_size.str().c_str() + "/tol_" + acc.str().c_str();

    // //? Write data to files
    // string final_data = directory_f + "/Final_data.txt";
    // ofstream data;
    // data.open(final_data);
    // for(int ii = 0; ii < N; ii++)
    // {
    //     data << setprecision(16) << u[ii] << endl;
    // }
    // data.close();

    // string results = directory_f + "/Results.txt";
    // ofstream params;
    // params.open(results);
    // params << "Simulation time: " << time << endl;
    // params << "Total number of time steps: " << time_steps << endl;
    // params << "Total number of Leja iterations: " << iters_total << endl;
    // params << "Average Bandwidth (GB/s): " << bandwidth << endl;
    // params << setprecision(16) << "Total time elapsed (s): " << time_loop.total() << endl;
    // params.close();

    // cout << "Writing data to files complete!" << endl;

    return 0;
}