#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <sys/types.h>
#include <sys/stat.h>

#include <cuda.h>
#include "../Timer.hpp"

//? Problems
#include "Dif_Adv_2D.hpp"
#include "Burgers_2D.hpp"

//! ---------------------------------------------------------------------------

//! Include Exponential Integrators and Leja functions 
//! (This has to be included to use Leja and/or exponential integrators)
#include "../Leja.hpp"

//! Functions to compute the largest eigenvalue (in magnitude)
#include "../Eigenvalues.hpp"

//! ---------------------------------------------------------------------------

using namespace std;

//? ====================================================================================== ?//

int main(int argc, char** argv)
{
    int index = atoi(argv[1]);          // N = 2^index * 2^index
    double n_cfl = atoi(argv[2]);       // dt = n_cfl * dt_cfl
    double tol = atof(argv[3]);         // User-specified tolerance
    double t_final = atof(argv[4]);     // Final simulation time

    cout << "N = " << index << ", N_cfl = " << n_cfl <<
    ", tol = " << tol << ", T_f = " << t_final << endl;

    //! Set GPU support to true
    bool GPU_access = true;

    //* Initialise parameters
    int n = pow(2, index);                          // # grid points (1D)
    int N = n*n;                                    // # grid points (2D)
    double xmin = -1;                               // Left boundary (limit)
    double xmax =  1;                               // Right boundary (limit)
    double ymin = -1;                               // Left boundary (limit)
    double ymax =  1;                               // Right boundary (limit)
    vector<double> X(n);                            // Array of grid points
    vector<double> Y(n);                            // Array of grid points
    vector<double> u(N);                            // Initial condition
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
    double velocity = 10.0;                                 // Advection speed

    //* Temporal parameters
    double time = 0;                                        // Simulation time elapsed
    int time_steps = 0;                                     // # time steps

    double dif_cfl = (dx*dx * dy*dy)/(2*dx*dx + 2*dy*dy);   // Diffusion CFL
    double adv_cfl = dx*dy/(velocity * (dx + dy));          // Advection CFL
    double dt = n_cfl*min(dif_cfl, adv_cfl);                // Step size
    cout << endl << "Step size: " << dt << endl;

    int iters = 0;                                          //* # of Leja points used per iteration (iteration variable for Leja interpolation)
    int iters_total = 0;                                    //* Total # of Leja iterations during the simulation

    //? Choose problem and integrator
    string problem = "Burgers_2D";
    string integrator = "EXPRB43";

    //! Diffusion-Advection or Diffusion + Sources
    // RHS_Dif_Adv_2D RHS(n, dx, dy, velocity); 
    // Leja<RHS_Dif_Adv_2D> leja_gpu{N, integrator};

    //! Burgers' Equation
    RHS_Burgers_2D RHS(n, dx, dy, velocity);
    Leja<RHS_Burgers_2D> leja_gpu{N, integrator};

    //? Strings for directory names
    stringstream step_size, tf, grid, acc;
    step_size << fixed << scientific << setprecision(1) << dt;
    tf << fixed << scientific << setprecision(1) << t_final;
    grid << fixed << scientific << setprecision(0) << n;
    acc << fixed << scientific << setprecision(0) << tol;

    //! Error Check
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    if (problem == "Diff_Adv_2D")
    {
        //? Initial condition
        for (int ii = 0; ii < n; ii++)
        {
            for (int jj = 0; jj< n; jj++)
            {
                u[n*ii + jj] = 1 + exp(-((X[ii] + 0.5)*(X[ii] + 0.5) + (Y[jj] + 0.5)*(Y[jj] + 0.5))/0.01);
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
                u[n*ii + jj] = 1 + 1e-2*exp(-((X[ii] + 0.75)*(X[ii] + 0.75) + (Y[jj] + 0.3)*(Y[jj] + 0.3))/0.01);
            }
        }
    }
    else if (problem == "Burgers_2D")
    {
        //? Initial condition
        for (int ii = 0; ii < n; ii++)
        {
            for (int jj = 0; jj< n; jj++)
            {
                u[n*ii + jj] = 2 + 0.1*sin(2*M_PI*X[ii]) + 0.1*sin(8*M_PI*X[ii] + 0.3)
                                 + 0.1*sin(2*M_PI*Y[jj]) + 0.1*sin(8*M_PI*Y[jj] + 0.3);
            }
        }
    }
    else
    {
        cout << "Undefined problem!" << endl;
    } 

    //! Allocate memory on GPU
    size_t N_size = N * sizeof(double);
    double *device_u; cudaMalloc(&device_u, N_size);
    cudaMemcpy(device_u, &u[0], N_size, cudaMemcpyHostToDevice);                    //* Copy state variable to device
    double *device_u_sol; cudaMalloc(&device_u_sol, N_size);                        //* Solution vector

    double *device_u_low;                                                           //? Only for nonlinear problems
    double *device_interp_vector;                                                   //? Only for nonhomogenous linear problems
    double *device_source;                                                          //? Only for nonhomogenous linear problems
    double error;

    if (integrator == "NonHom_Linear")
    {
        //? Vector to be interpolated
        cudaMalloc(&device_interp_vector, N_size);

        //? Source
        cudaMalloc(&device_source, N_size);
        cudaMemcpy(device_source, &Source[0], N_size, cudaMemcpyHostToDevice);
    }
    else
    {
        cudaMalloc(&device_u_low, N_size);
    }

    //? Shifting and scaling parameters 
    double eigenvalue = 0.0;
    leja_gpu.Power_iterations(RHS, device_u, eigenvalue, GPU_access);
    eigenvalue = -1.05*eigenvalue;                                   //! Real eigenvalue has to be negative
    double c = eigenvalue/2.0; double Gamma = -eigenvalue/4.0;
    cout << "Largest eigenvalue: " << eigenvalue << endl << endl;

    //! Error Check
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

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

        cudaDeviceSynchronize();

        //? ---------------------------------------------------------------- ?//

        //! Homogenous Linear Equations

        if (integrator == "Hom_Linear")
        {
            leja_gpu.real_Leja_exp(RHS, device_u, device_u_sol, c, Gamma, tol, dt, iters, GPU_access);
        }

        //? ---------------------------------------------------------------- ?//

        //! Nonhomogenous Linear Equations

        else if (integrator == "NonHom_Linear")
        {
            //? interp_vector * dt = (u + source) * dt
            LeXInt::axpby(1.0, device_source, 1.0, device_u, device_interp_vector, N, GPU_access);
            LeXInt::axpby(dt, device_interp_vector, device_interp_vector, N, GPU_access);

            leja_gpu.real_Leja_phi_nl(RHS, device_interp_vector, device_u_sol, LeXInt::phi_1, c, Gamma, tol, dt, iters, GPU_access);
        }
        
        //? ---------------------------------------------------------------- ?//

        //! Nonlinear Equations

        //* Non-embedded Intergators
        else if (integrator == "Rosenbrock_Euler" or integrator == "EPIRK4s3B")
        {
            //* Largest eigenvalue (spectrum) every 250 time steps
            if (time_steps != 0 && time_steps % 250 == 0)
            {
                //? Largest eigenvalue of the Jacobian; changes at every time step for nonlinear equations
                leja_gpu.Power_iterations(RHS, device_u, eigenvalue, GPU_access);
                eigenvalue = -1.05*eigenvalue;       //! Real eigenvalue has to be negative
                c = eigenvalue/2.0; Gamma = -eigenvalue/4.0;
            }

            //? Non-embedded integrators
            leja_gpu.exp_int(RHS, device_u, device_u_sol, c, Gamma, tol, dt, iters, GPU_access);
        }

        //* Embedded Integrators 
        else if (integrator == "EXPRB32" or integrator == "EXPRB42" or integrator == "EXPRB43" or integrator == "EXPRB53s3" 
        or integrator == "EXPRB54s4" or integrator == "EPIRK4s3" or integrator == "EPIRK4s3A" or integrator == "EPIRK5P1")
        {
            //* Largest eigenvalue (spectrum) every 250 time steps
            if (time_steps != 0 && time_steps % 250 == 0)
            {
                //? Largest eigenvalue of the Jacobian; changes at every time step for nonlinear equations
                leja_gpu.Power_iterations(RHS, device_u, eigenvalue, GPU_access);
                eigenvalue = -1.05*eigenvalue;       //! Real eigenvalue has to be negative
                c = eigenvalue/2.0; Gamma = -eigenvalue/4.0;

                cout << "Embedded error: " << error << endl;
            }

            //? Embedded integrators
            leja_gpu.embed_exp_int(RHS, device_u, device_u_low, device_u_sol, error, c, Gamma, tol, dt, iters, GPU_access);
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

        if (integrator == "NonHom_Linear")
        {
            LeXInt::axpby(1.0, device_u, 1.0, device_interp_vector, device_u, N, GPU_access);
        }
        else
        {
            LeXInt::copy(device_u_sol, device_u, N, GPU_access);
        }

        // if (time_steps % 250 == 0)
        // {
        //     cout << "Largest eigenvalue: " << eigenvalue << endl;
        //     cout << "Time steps: " << time_steps << endl;
        //     cout << "Time elapsed: " << time << endl;
        //     cout << endl;
        // }

        if (time_steps == 20)
        {
            break;
        }
    }

    cudaDeviceSynchronize(); 
    time_loop.stop();

    //? ---------------------- Bandwidth computation ---------------------- ?//

    //! Number of vector reads and writes (to compute achieved bandwidth):
    //! Note: Vector reads and writes per RHS computation = 2.
    //! This number can vary depending on how the RHS function is defined.
    //? real_Leja_phi = 21 + num_interpolations

    double reads_writes;
    if (integrator == "Hom_Linear")
    {
        reads_writes = (1.0 * time_steps) + (10.0 * iters_total);
    }
    else if (integrator == "NonHom_Linear")
    {
        reads_writes = (8.0 * time_steps) + (10.0 * iters_total);
    }
    else if (integrator == "Rosenbrock_Euler")
    {
        reads_writes = (9.0 * time_steps) + (24.0 * iters_total);
    }
    else if (integrator == "EXPRB32")
    {
        reads_writes = (60.0 * time_steps) + ((24 + 24)/2.0 * iters_total);
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
        reads_writes = (95.0 * time_steps) + ((30 + 30 + 27)/3.0 * iters_total);
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

    //! Create nested directories
    // int sys_value_f = system(("mkdir -p ../../LeXInt_Test/" + to_string(GPU_access) + "/Constant/" + problem + "/" + integrator
    //                             + "/N_" + grid.str().c_str() + "/t_" + tf.str().c_str() + "/dt_" + step_size.str().c_str() + "/tol_" + acc.str()).c_str());
    // string directory_f = "../../LeXInt_Test/" + to_string(GPU_access) + "/Constant/" + problem + "/" + integrator
    //                             + "/N_" + grid.str().c_str() + "/t_" + tf.str().c_str() + "/dt_" + step_size.str().c_str() + "/tol_" + acc.str().c_str();

    // //* Copy state variable from device to host
    // cudaMemcpy(&u[0], device_u, N_size, cudaMemcpyDeviceToHost);   

    // //? Write data to files
    // // string final_data = directory_f + "/Final_data.txt";
    // // ofstream data;
    // // data.open(final_data);
    // // for(int ii = 0; ii < N; ii++)
    // // {
    // //     data << setprecision(16) << u[ii] << endl;
    // // }
    // // data.close();

    // string results = directory_f + "/Results.txt";
    // ofstream params;
    // params.open(results);
    // params << "Simulation time: " << time << endl;
    // params << "Total number of time steps: " << time_steps << endl;
    // params << "Total number of Leja iterations: " << iters_total << endl;
    // params << "Average Bandwidth (GB/s): " << bandwidth << endl;
    // params << setprecision(16) << "Total time elapsed (s): " << time_loop.total() << endl;
    // params.close();

    cout << "Writing data to files complete!" << endl;

    return 0;
}