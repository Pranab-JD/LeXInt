#include <cmath>
#include <fstream>
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
    vec u_init(N);                                  // Initial condition

    //* Set up X array and initial condition
    for (int ii = 2; ii < N-2; ii++)
    {
        X[ii] = xmin + ii*(xmax - xmin)/N;
    }
    //* Periodic BC
    X[N-3] = X[3]; X[N-2] = X[4]; X[N-1] = X[5]; 
    X[0] = X[N-6]; X[1] = X[N-5]; X[2] = X[N-4];

    //* Initialise additional parameters
    double dx = X[12] - X[11];                      // Grid spacing
    double velocity = 80;                           // Advection speed
    double dif_cfl = dx*dx;                         // Diffusion CFL
    double adv_cfl = dx/velocity;                   // Advection CFL
    double dt = 0.8*min(dif_cfl, adv_cfl);            // Step size
    stringstream step_size;
    step_size << fixed << scientific << setprecision(1) << dt;
    cout << "Step size: " << dt << endl;

    //* Temporal parameters
    double time = 0;                                // Simulation time elapsed
    double t_final = 0.5;                        // Final simulation time
    int time_steps = 0;                             // # time steps

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
            u_init[ii] = 1 + exp((X[ii] - 50.0)/0.4);
        }
    }
    else if (problem == "Burgers")
    {
        RHS_Burgers RHS = RHS_Burgers(N, dx, velocity);
        Leja_GPU<RHS_Burgers> leja_gpu{N, integrator};

        //? Initial condition
        for (int ii = 2; ii < N-2; ii++)
        {
            u_init[ii] = 2 + sin(X[ii]/10); //+ 2*sin(25*X[ii] + 20);
        }
        
        //* Periodic BC
        u_init[N-3] = u_init[3]; u_init[N-2] = u_init[4]; u_init[N-1] = u_init[5]; 
        u_init[0] = u_init[N-6]; u_init[1] = u_init[N-5]; u_init[2] = u_init[N-4];
    
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
    Power_iterations(RHS, u, N, eigenvalue, auxillary_Jv, GPU_access, cublas_h);         // Real eigenvalue has to be negative
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
            real_Leja_exp(RHS, u, u_sol, auxillary_Leja, N, Leja_X, c, Gamma, tol, dt, GPU_access, cublas_h);
        }
        
        //? ---------------------------------------------------------------- ?//

        //? Nonlinear Equations

        //* Non-embedded Intergators
        else if (integrator == "Rosenbrock_Euler" or integrator == "EPIRK4s3B")
        {
            // * ----------- Eigenvalue (Spectrum) ----------- *//

            //? Largest eigenvalue of the Jacobian; changes at every time step for nonlinear equations
            Power_iterations(RHS, u, N, eigenvalue, auxillary_Jv, GPU_access, cublas_h);         // Real eigenvalue has to be negative
            eigenvalue = -1.2*eigenvalue;
            double c = eigenvalue/2.0; double Gamma = -eigenvalue/4.0;
            cout << "Largest eigenvalue: " << eigenvalue << endl;

            //* ---------------------------------------------- *//

            //? Embedded integrators
            leja_gpu(RHS, u, u_sol, N, Leja_X, c, Gamma, tol, dt, GPU_access);
        }

        //* Embedded Integrators 
        else if (integrator == "EXPRB32" or integrator == "EXPRB42" or integrator == "EXPRB43" or integrator == "EXPRB53s3" 
        or integrator == "EXPRB54s4" or integrator == "EPIRK4s3" or integrator == "EPIRK4s3A" or integrator == "EPIRK5P1")
        {
            // * ----------- Eigenvalue (Spectrum) ----------- *//

            //? Largest eigenvalue of the Jacobian; changes at every time step for nonlinear equations
            Power_iterations(RHS, u, N, eigenvalue, auxillary_Jv, GPU_access, cublas_h);         // Real eigenvalue has to be negative
            eigenvalue = -1.2*eigenvalue;
            double c = eigenvalue/2.0; double Gamma = -eigenvalue/4.0;
            cout << "Largest eigenvalue: " << eigenvalue << endl;

            //* ---------------------------------------------- *//

            //? Embedded integrators
            leja_gpu(RHS, u, u_low, u_sol, N, Leja_X, c, Gamma, tol, dt, GPU_access);
            
            axpby(1.0, u_low, -1.0, u_sol, u_error, N, GPU_access);
            double error = l2norm(u_error, N, GPU_access, cublas_h);
            cout << "Embedded error: " << error << endl;
        }
        else
        {
            cout << "ERROR: Please choose an available integator. See 'Leja.hpp'." << endl;
        }

        //? ---------------------------------------------------------------- ?//

        //* Update variables
        time = time + dt;
        swap(u, u_sol);
        //* Periodic BC
        u[N-3] = u[3]; u[N-2] = u[4]; u[N-1] = u[5]; 
        u[0] = u[N-6]; u[1] = u[N-5]; u[2] = u[N-4];
        time_steps = time_steps + 1;
        cout << endl;
    }

    time_loop.stop();

    cout << endl << "==================================================" << endl;
    cout << "Simulation time: " << time << endl;
    cout << "Total number of time steps: " << time_steps << endl;
    cout << "Total time elapsed (s): " << time_loop.total() << endl;
    cout << "==================================================" << endl << endl;

    //! Create nested directories
    system(("mkdir -p ../../LeXInt_Test/" + to_string(GPU_access) + "/Constant/" + problem + "/EXPRB32_3/dt_" + step_size.str()).c_str());
    string directory = "../../LeXInt_Test/" + to_string(GPU_access) + "/Constant/" + problem + "/EXPRB32_3/dt_" + step_size.str();

    //? Write data to files
    string final_data = directory + "/1.txt";
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