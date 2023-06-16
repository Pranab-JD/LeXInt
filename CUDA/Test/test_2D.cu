#include <fstream>
#include <string>
#include <sstream>
#include <sys/types.h>
#include <sys/stat.h>

#include "../Timer.hpp"

//? Problems
#include "Diff_Adv_2D.hpp"
// #include "Burgers.hpp"

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
    int N = 100;                                            // # grid points
    double xmin = -1;                       // Left boundary (limit)
    double xmax =  1;                       // Right boundary (limit)
    double ymin = -1;                       // Left boundary (limit)
    double ymax =  1;                       // Right boundary (limit)
    vector<double> X(N);                            // Array of grid points
    vector<double> Y(N);                            // Array of grid points
    vector<double> u(N*N);                          // Initial condition

    //* Set up X, Y arrays and initial condition
    for (int ii = 0; ii < N; ii++)
    {
        X[ii] = xmin + ii*(xmax - xmin)/N;
        Y[ii] = ymin + ii*(ymax - ymin)/N;
    }

    //* Initialise additional parameters
    double dx = X[12] - X[11];                              // Grid spacing
    double dy = Y[12] - Y[11];                              // Grid spacing
    double velocity = 50;                                   // Advection speed
    double dif_cfl = (dx*dx * dy*dy)/(2*dx*dx + 2*dy*dy);   // Diffusion CFL
    double adv_cfl = dx*dy/(velocity * (dx + dy));          // Advection CFL
    double dt = 0.4*min(dif_cfl, adv_cfl);                  // Step size
    stringstream step_size;
    step_size << fixed << scientific << setprecision(1) << dt;
    cout << "Step size: " << dt << endl;

    //* Temporal parameters
    double time = 0;                                  // Simulation time elapsed
    double t_final = 0.05;                            // Final simulation time
    int time_steps = 0;                               // # time steps

    //* Set of Leja points
    vector<double> Leja_X = Leja_Points();

    //? Choose problem and integrator
    double tol = 1e-10;
    string problem = "Diff_Adv_2D";
    string integrator = "Hom_Linear";

    RHS_Dif_Adv_2D RHS(N, dx, dy, velocity);                //* Default problem
    Leja_GPU<RHS_Dif_Adv_2D> leja_gpu{N, integrator};       //* Default problem

    if (problem == "Diff_Adv")
    {
        //? Initial condition
        for (int ii = 0; ii < N; ii++)
        {
            for (int jj = 0; jj< N; jj++)
            {
                u[N*ii + jj] = 1 + exp(-((X[ii] + 0.5)*(X[ii] + 0.5) + (Y[jj] + 0.5)*(Y[jj] + 0.5))/0.01);
            }
        }
    }
    // else if (problem == "Burgers")
    // {
    //     RHS_Burgers RHS = RHS_Burgers(N, dx, velocity);
    //     Leja_GPU<RHS_Burgers> leja_gpu{N, integrator};

    //     //? Initial condition
    //     for (int ii = 3; ii < N-3; ii++)
    //     {
    //         u[ii] = 4 + sin(X[ii]/10) + 2*sin(X[ii]/5 + 20) ;
    //     }
    // }
    else
    {
        cout << "Undefined problem!" << endl;
    } 

    // //! Allocate memory on GPU
    // size_t N_size = N * N * sizeof(double);
    // double *device_u; cudaMalloc(&device_u, N_size);
    // double *device_u_low; cudaMalloc(&device_u_low, N_size);
    // double *device_u_sol; cudaMalloc(&device_u_sol, N_size);
    // double *device_error; cudaMalloc(&device_error, N_size);
    // double *device_auxillary_Leja; cudaMalloc(&device_auxillary_Leja, N_size);
    // double *device_auxillary_Jv; cudaMalloc(&device_auxillary_Jv, 7*N_size);        //* To compute spectrum using power iterations
    // cudaMemcpy(device_u, &u[0], N_size, cudaMemcpyHostToDevice);                    //* Copy state variable to device

    // //! Set GPU spport to true
    // bool GPU_access = true;
    // GPU_handle cublas_h;

    //? Shifting and scaling parameters
    // double eigenvalue = 0.0;
    // LeXInt::Power_iterations(RHS, device_u, N, eigenvalue, device_auxillary_Jv, GPU_access, cublas_h);         // Real eigenvalue has to be negative
    // eigenvalue = -1.2*eigenvalue;
    // double c = eigenvalue/2.0; double Gamma = -eigenvalue/4.0;
    // cout << "Largest eigenvalue: " << eigenvalue << endl;

    //! Time Loop
    // LeXInt::timer time_loop;
    // time_loop.start();

    // while (time < t_final)
    // {
    //     //* Final time step
    //     if (time + dt >= t_final)
    //     {
    //         dt = t_final - time;
    //     }

    //     //? ---------------------------------------------------------------- ?//

    //     //? Homogenous Linear Equations

    //     if (integrator == "Hom_Linear")
    //     {
    //         real_Leja_exp(RHS, device_u, device_u_sol, device_auxillary_Leja, N, Leja_X, c, Gamma, tol, dt, GPU_access, cublas_h);
    //     }
        
    //     //? ---------------------------------------------------------------- ?//

    //     //? Nonlinear Equations

    //     //* Non-embedded Intergators
    //     else if (integrator == "Rosenbrock_Euler" or integrator == "EPIRK4s3B")
    //     {
    //         // * ----------- Eigenvalue (Spectrum) ----------- *//

    //         //? Largest eigenvalue of the Jacobian; changes at every time step for nonlinear equations
    //         Power_iterations(RHS, device_u, N, eigenvalue, device_auxillary_Jv, GPU_access, cublas_h);         // Real eigenvalue has to be negative
    //         eigenvalue = -1.2*eigenvalue;
    //         c = eigenvalue/2.0; Gamma = -eigenvalue/4.0;
    //         cout << "Largest eigenvalue: " << eigenvalue << endl;

    //         //* ---------------------------------------------- *//

    //         //? Embedded integrators
    //         leja_gpu(RHS, device_u, device_u_sol, N, Leja_X, c, Gamma, tol, dt, GPU_access);
    //     }

    //     //* Embedded Integrators 
    //     else if (integrator == "EXPRB32" or integrator == "EXPRB42" or integrator == "EXPRB43" or integrator == "EXPRB53s3" 
    //     or integrator == "EXPRB54s4" or integrator == "EPIRK4s3" or integrator == "EPIRK4s3A" or integrator == "EPIRK5P1")
    //     {
            // * ----------- Eigenvalue (Spectrum) ----------- *//

    //         //? Largest eigenvalue of the Jacobian; changes at every time step for nonlinear equations
    //         Power_iterations(RHS, device_u, N, eigenvalue, device_auxillary_Jv, GPU_access, cublas_h);         // Real eigenvalue has to be negative
    //         eigenvalue = -1.2*eigenvalue;
    //         c = eigenvalue/2.0; Gamma = -eigenvalue/4.0;
    //         cout << "Largest eigenvalue: " << eigenvalue << endl;

    //         //* ---------------------------------------------- *//

    //         //? Embedded integrators
    //         leja_gpu(RHS, device_u, device_u_low, device_u_sol, N, Leja_X, c, Gamma, tol, dt, GPU_access);
            
    //         axpby(1.0, device_u_low, -1.0, device_u_sol, device_error, N, GPU_access);
    //         double error = l2norm(device_error, N, GPU_access, cublas_h);
    //         cout << "Embedded error: " << error << endl;
    //     }
    //     else
    //     {
    //         cout << "ERROR: Please choose an available integator. See 'Leja.hpp'." << endl;
    //     }

    //     //? ---------------------------------------------------------------- ?//

    //     //* Update variables
    //     time = time + dt;
    //     swap(device_u, device_u_sol);
        // time_steps = time_steps + 1;

        // cout << "Time steps: " << time_steps << endl;
        // cout << "Time elapsed: " << time << endl;
        // cout << endl;

        // //! Create nested directories
        // int sys_value = system(("mkdir -p ../../LeXInt_Test/DA_GPU/"));
        // string directory = "../../LeXInt_Test/DA_GPU/";

        // //? Write data to files
        // string output_data = directory + "/" +  to_string(time_steps) + ".txt";
        // ofstream data;
        // data.open(output_data); 
        // for(int ii = 0; ii < N*N; ii++)
        // {
        //     data << setprecision(16) << u[ii] << endl;
        // }
        // data.close();

    // }

    // time_loop.stop();

    // //* Copy state variable from device to host
    // cudaMemcpy(&u[0], device_u, N_size, cudaMemcpyDeviceToHost);                

    // cout << endl << "==================================================" << endl;
    // cout << "Simulation time: " << time << endl;
    // cout << "Total number of time steps: " << time_steps << endl;
    // cout << "Total time elapsed (s): " << time_loop.total() << endl;
    // cout << "==================================================" << endl << endl;

    //! Create nested directories
    // system(("mkdir -p ../../LeXInt_Test/" + to_string(GPU_access) + "/Constant/" + problem + "/EXPRB32_3/dt_" + step_size.str()).c_str());
    // string directory = "../../LeXInt_Test/" + to_string(GPU_access) + "/Constant/" + problem + "/EXPRB32_3/dt_" + step_size.str();

    // //? Write data to files
    // string final_data = directory + "/Final_data.txt";
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