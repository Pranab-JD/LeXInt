#include <fstream>
#include <string>
#include <sstream>
#include <sys/types.h>
#include <sys/stat.h>

#include "../Timer.hpp"

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
    int N = 1e3+6;                                  // # grid points
    double xmin = -10.0*M_PI;                       // Left boundary (limit)
    double xmax =  10.0*M_PI;                       // Right boundary (limit)
    vec X(N);                                       // Array of grid points
    vec u(N);                                       // Initial condition

    //* Set up X array and initial condition
    for (int ii = 2; ii < N-2; ii++)
    {
        X[ii] = xmin + ii*(xmax - xmin)/N;
    }
    //* Periodic BC
    X[N-3] = X[3]; X[N-2] = X[4]; X[N-1] = X[5]; 
    X[0] = X[N-6]; X[1] = X[N-5]; X[2] = X[N-4];

    //* Initialise additional parameters
    double dx = X[11] - X[10];                       // Grid spacing
    double velocity = 80;                            // Advection speed
    double dif_cfl = dx*dx;                          // Diffusion CFL
    double adv_cfl = dx/velocity;                    // Advection CFL
    double dt = 0.8*min(dif_cfl, adv_cfl);           // Step size
    stringstream step_size;
    step_size << fixed << scientific << setprecision(1) << dt;
    cout << "Step size: " << dt << endl;

    //* Temporal parameters
    double time = 0;                                 // Simulation time elapsed
    double t_final = 0.5;                            // Final simulation time
    int time_steps = 0;                              // # time steps

    //* Set of Leja points
    vec Leja_X = Leja_Points();

    //? Choose problem and integrator
    double tol = 1e-10;
    string problem = "Burgers";
    string integrator = "EXPRB32";

    RHS_Dif_Adv RHS = RHS_Dif_Adv(N, dx, velocity);     //* Default problem
    Leja_GPU<RHS_Dif_Adv> leja_gpu{N, integrator};      //* Default problem

    if (problem == "Diff_Adv")
    {
        RHS_Dif_Adv RHS = RHS_Dif_Adv(N, dx, velocity);
        Leja_GPU<RHS_Dif_Adv> leja_gpu{N, integrator};

        //? Initial condition
        for (int ii = 0; ii < N; ii++)
        {
            u[ii] = 1 + exp((X[ii] - 50.0)/0.4);
        }
    }
    else if (problem == "Burgers")
    {
        RHS_Burgers RHS = RHS_Burgers(N, dx, velocity);
        Leja_GPU<RHS_Burgers> leja_gpu{N, integrator};

        //? Initial condition
        for (int ii = 2; ii < N-2; ii++)
        {
            u[ii] = 2 + sin(X[ii]/10);
        }
        
        //* Periodic BC
        u[N-3] = u[3]; u[N-2] = u[4]; u[N-1] = u[5]; 
        u[0] = u[N-6]; u[1] = u[N-5]; u[2] = u[N-4];
    
    }
    else
    {
        cout << "Undefined problem!" << endl;
    } 

    //! Allocate memory on GPU
    size_t N_size = N * sizeof(double);
    double *device_u; cudaMalloc(&device_u, N_size);
    double *device_u_low; cudaMalloc(&device_u_low, N_size);
    double *device_u_sol; cudaMalloc(&device_u_sol, N_size);
    double *device_error; cudaMalloc(&device_error, N_size);
    double *device_auxillary_Leja; cudaMalloc(&device_auxillary_Leja, N_size);
    double *device_auxillary_Jv; cudaMalloc(&device_auxillary_Jv, 7*N_size);    //* To compute spectrum using power iterations
    cudaMemcpy(device_u, &u[0], N_size, cudaMemcpyHostToDevice);                //* Copy state variable to device

    //! Set GPU spport to true
    bool GPU_access = true;
    GPU_handle cublas_h;

    //? Shifting and scaling parameters
    double eigenvalue = 0.0;
    Power_iterations(RHS, device_u, N, eigenvalue, device_auxillary_Jv, GPU_access, cublas_h);         // Real eigenvalue has to be negative
    eigenvalue = -1.2*eigenvalue;
    double c = eigenvalue/2.0; double Gamma = -eigenvalue/4.0;
    cout << "Largest eigenvalue: " << eigenvalue << endl;

    //! Time Loop
    timer time_loop;
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
            real_Leja_exp(RHS, device_u, device_u_sol, device_auxillary_Leja, N, Leja_X, c, Gamma, tol, dt, GPU_access, cublas_h);
        }
        
        //? ---------------------------------------------------------------- ?//

        //? Nonlinear Equations

        //* Non-embedded Intergators
        else if (integrator == "Rosenbrock_Euler" or integrator == "EPIRK4s3B")
        {
            // * ----------- Eigenvalue (Spectrum) ----------- *//

            //? Largest eigenvalue of the Jacobian; changes at every time step for nonlinear equations
            Power_iterations(RHS, device_u, N, eigenvalue, device_auxillary_Jv, GPU_access, cublas_h);         // Real eigenvalue has to be negative
            eigenvalue = -1.2*eigenvalue;
            c = eigenvalue/2.0; Gamma = -eigenvalue/4.0;
            cout << "Largest eigenvalue: " << eigenvalue << endl;

            //* ---------------------------------------------- *//

            //? Embedded integrators
            leja_gpu(RHS, device_u, device_u_sol, N, Leja_X, c, Gamma, tol, dt, GPU_access);
        }

        //* Embedded Integrators 
        else if (integrator == "EXPRB32" or integrator == "EXPRB42" or integrator == "EXPRB43" or integrator == "EXPRB53s3" 
        or integrator == "EXPRB54s4" or integrator == "EPIRK4s3" or integrator == "EPIRK4s3A" or integrator == "EPIRK5P1")
        {
            // * ----------- Eigenvalue (Spectrum) ----------- *//

            //? Largest eigenvalue of the Jacobian; changes at every time step for nonlinear equations
            Power_iterations(RHS, device_u, N, eigenvalue, device_auxillary_Jv, GPU_access, cublas_h);         // Real eigenvalue has to be negative
            eigenvalue = -1.2*eigenvalue;
            c = eigenvalue/2.0; Gamma = -eigenvalue/4.0;
            cout << "Largest eigenvalue: " << eigenvalue << endl;

            //* ---------------------------------------------- *//

            //? Embedded integrators
            leja_gpu(RHS, device_u, device_u_low, device_u_sol, N, Leja_X, c, Gamma, tol, dt, GPU_access);
            
            axpby(1.0, device_u_low, -1.0, device_u_sol, device_error, N, GPU_access);
            double error = l2norm(device_error, N, GPU_access, cublas_h);
            cout << "Embedded error: " << error << endl;
        }
        else
        {
            cout << "ERROR: Please choose an available integator. See 'Leja.hpp'." << endl;
        }

        //? ---------------------------------------------------------------- ?//

        //* Update variables
        time = time + dt;
        swap(device_u, device_u_sol);
        time_steps = time_steps + 1;
        cout << endl;

        if (time_steps == 10)
            break;
    }

    time_loop.stop();

    //* Copy state variable from device to host
    cudaMemcpy(&u[0], device_u, N_size, cudaMemcpyDeviceToHost);                

    cout << endl << "==================================================" << endl;
    cout << "Simulation time: " << time << endl;
    cout << "Total number of time steps: " << time_steps << endl;
    cout << "Total time elapsed (s): " << time_loop.total() << endl;
    cout << "==================================================" << endl << endl;

    //! Create nested directories
    system(("mkdir -p ../../LeXInt_Test/" + to_string(GPU_access) + "/Constant/" + problem + "/EXPRB32_3/dt_" + step_size.str()).c_str());
    string directory = "../../LeXInt_Test/" + to_string(GPU_access) + "/Constant/" + problem + "/EXPRB32_3/dt_" + step_size.str();

    //? Write data to files
    string final_data = directory + "/Final_data.txt";
    ofstream data;
    data.open(final_data);
    for(int ii = 0; ii < N; ii++)
    {
        data << setprecision(16) << u[ii] << endl;
    }
    data.close();

    string results = directory + "/Results.txt";
    ofstream params;
    params.open(results);
    params << "Simulation time: " << time << endl;
    params << "Total number of time steps: " << time_steps << endl;
    params << setprecision(16) << "Total time elapsed (s): " << time_loop.total() << endl;
    params.close();

    cout << "Writing data to files complete!" << endl << endl;

    return 0;
}