/*
 * 3D_CPP_PUSHER_FETCHER_1.cpp
 *
 *  Created on: 10 Jan 2019
 *      Author: Wendi Liu
 */

#define IMUI_MULTIDOMAIN 1

#include "mui.h"
#include <iostream>
#include <fstream>
#include "pusher_fetcher_config.h"

int main(int argc, char ** argv) {
    using namespace mui;

#if IMUI_MULTIDOMAIN

    // Declare MPI common world with the scope of MUI
    MPI_Comm  world = mui::mpi_split_by_app();
    // Define the name of MUI interfaces
    std::vector<std::string> interfaces;
    std::string domainName="PUSHER_FETCHER_1";
    std::string appName="threeDInterface0";

    interfaces.emplace_back(appName);

    // Declare MUI objects using MUI configure file
    auto ifs = mui::create_uniface<mui::pusher_fetcher_config>( domainName, interfaces );

#else

    uniface<mui::pusher_fetcher_config> ifs( "mpi://PUSHER_FETCHER_1/threeDInterface0"  );

    MPI_Comm  world = mui::mpi_split_by_app();

#endif

    std::ofstream heatFluxOutFile("heatFluxCpp.txt");

    int rank, size;
    MPI_Comm_rank( world, &rank );
    MPI_Comm_size( world, &size );

    // setup parameters
    constexpr static int    Nx        = 2; // number of grid points in x axis
    constexpr static int    Ny        = 10; // number of grid points in y axis
    constexpr static int    Nz        = 10; // number of grid points in z axis
    const char* name_fetchX = "heatFluxx";
    const char* name_fetchY = "heatFluxy";
    const char* name_fetchZ = "heatFluxz";
    const char* name_push = "temperature";
    double r    = 1.0;                      // search radius
    int Nt = Nx * Ny * Nz; // total time steps
    int steps = 200; // total time steps
    int nSubIter = 1; // total time steps
    double timeStepSize    = 5e-3;
    double local_x0 = -1.5; // local origin
    double local_y0 = -2.;
    double local_z0 = -1.;
    double local_x1 = -1.;
    double local_y1 = 2.;
    double local_z1 = 1.;
    double local_x2 = -1.5; // local origin
    double local_y2 = -2.;
    double local_z2 = -1.;
    double local_x3 = -1.;
    double local_y3 = 2.;
    double local_z3 = 1.;
    double monitorX = -1.;
    double monitorY = 0.;
    double monitorZ = 0.;
    double pp[Nx][Ny][Nz][3], pf[Nx][Ny][Nz][3];
    double temperature_push[Nx][Ny][Nz];
    double heatFlux_fetchX[Nx][Ny][Nz], heatFlux_fetchY[Nx][Ny][Nz], heatFlux_fetchZ[Nx][Ny][Nz];
    double heatFlux_fetchX_Store[Nx][Ny][Nz], heatFlux_fetchY_Store[Nx][Ny][Nz], heatFlux_fetchZ_Store[Nx][Ny][Nz];

    // Push points generation and evaluation
    for ( int i = 0; i < Nx; ++i ) {
        for ( int j = 0; j < Ny; ++j ) {
            for ( int k = 0; k < Nz; ++k ) {
                double x = local_x0+(i*(local_x1-local_x0)/(Nx-1));
                double y = local_y0+(j*(local_y1-local_y0)/(Ny-1));
                double z = local_z0+(k*(local_z1-local_z0)/(Nz-1));
                pp[i][j][k][0] = x;
                pp[i][j][k][1] = y;
                pp[i][j][k][2] = z;
                temperature_push[i][j][k] = 0.;
            }
        }
    }

    // Fetch points generation and evaluation
    for ( int i = 0; i < Nx; ++i ) {
        for ( int j = 0; j < Ny; ++j ) {
            for ( int k = 0; k < Nz; ++k ) {
                double x = local_x2+(i*(local_x3-local_x2)/(Nx-1));
                double y = local_y2+(j*(local_y3-local_y2)/(Ny-1));
                double z = local_z2+(k*(local_z3-local_z2)/(Nz-1));
                pf[i][j][k][0] = x;
                pf[i][j][k][1] = y;
                pf[i][j][k][2] = z;
                heatFlux_fetchX[i][j][k] = 0.0;
                heatFlux_fetchY[i][j][k] = 0.0;
                heatFlux_fetchZ[i][j][k] = 0.0;
                heatFlux_fetchX_Store[i][j][k] = 0.0;
                heatFlux_fetchY_Store[i][j][k] = 0.0;
                heatFlux_fetchZ_Store[i][j][k] = 0.0;
            }
        }
    }

   // annouce send span
    geometry::box<mui::pusher_fetcher_config> send_region( {local_x0, local_y0, local_z0}, {local_x1, local_y1, local_z1} );
    geometry::box<mui::pusher_fetcher_config> recv_region( {local_x2, local_y2, local_z2}, {local_x3, local_y3, local_z3} );
    printf( "{PUSHER_FETCHER_1} send region for rank %d: %lf %lf %lf - %lf %lf %lf\n", rank, local_x0, local_y0, local_z0, local_x1, local_y1, local_z1 );

#if IMUI_MULTIDOMAIN
    ifs[0]->announce_send_span( 0, steps*10, send_region );
    ifs[0]->announce_recv_span( 0, steps*10, recv_region );
#else
    ifs.announce_send_span( 0, steps*10, send_region );
    ifs.announce_recv_span( 0, steps*10, recv_region );
#endif

    // define spatial and temporal samplers
    sampler_pseudo_nearest_neighbor<mui::pusher_fetcher_config> s1(r);
    temporal_sampler_exact<mui::pusher_fetcher_config> s2;

#if IMUI_MULTIDOMAIN
    // commit ZERO step
    ifs[0]->commit(0);
#else
    // commit ZERO step
    ifs.commit(0);
#endif

    // Begin time loops
    for ( int n = 1; n <= steps; ++n ) {

        printf("\n");
        printf("{PUSHER_FETCHER_1} %d Step \n", n);

        // Begin iteration loops
        for ( int iter = 1; iter <= nSubIter; ++iter ) {

            printf("{PUSHER_FETCHER_1} %d iteration \n", iter);

            int totalIter = ( (n - 1) * nSubIter ) + iter;
            double total_flux=0.0;
            // push data to the other solver
            for ( int i = 0; i < Nx; ++i ) {
                for ( int j = 0; j < Ny; ++j ) {
                    for ( int k = 0; k < Nz; ++k ) {
                        temperature_push[i][j][k] = 500.;

                        if (std::abs(pp[i][j][k][0] + 1.0) <= 0.00001 ){
                            point3d locp( pp[i][j][k][0], pp[i][j][k][1], pp[i][j][k][2] );
#if IMUI_MULTIDOMAIN
                            ifs[0]->push( name_push, locp, temperature_push[i][j][k] );
#else
                            ifs.push( name_push, locp, temperature_push[i][j][k] );
#endif
                            // std::cout << "!!{PUSHER_FETCHER_1} push point: " <<  locp[0] << ", " <<  locp[1] << ", "<<  locp[2] << std::endl;
                        }
                    }
                }
            }

#if IMUI_MULTIDOMAIN
            int sent = ifs[0]->commit( totalIter );
#else
            int sent = ifs.commit( totalIter );
#endif
            if ((totalIter-1)>=1){
                // push data to the other solver
                for ( int i = 0; i < Nx; ++i ) {
                    for ( int j = 0; j < Ny; ++j ) {
                        for ( int k = 0; k < Nz; ++k ) {
                            if (std::abs(pf[i][j][k][0] + 1.0) <= 0.00001 ){
                                point3d locf( pf[i][j][k][0], pf[i][j][k][1], pf[i][j][k][2] );
#if IMUI_MULTIDOMAIN
                                heatFlux_fetchX[i][j][k] = ifs[0]->fetch( name_fetchX, locf,
                                    (totalIter-1),
                                    s1,
                                    s2 );
                                heatFlux_fetchY[i][j][k] = ifs[0]->fetch( name_fetchY, locf,
                                    (totalIter-1),
                                    s1,
                                    s2 );
                                heatFlux_fetchZ[i][j][k] = ifs[0]->fetch( name_fetchZ, locf,
                                    (totalIter-1),
                                    s1,
                                    s2 );
#else
                                heatFlux_fetchX[i][j][k] = ifs.fetch( name_fetchX, locf,
                                    (totalIter-1),
                                    s1,
                                    s2 );
                                heatFlux_fetchY[i][j][k] = ifs.fetch( name_fetchY, locf,
                                    (totalIter-1),
                                    s1,
                                    s2 );
                                heatFlux_fetchZ[i][j][k] = ifs.fetch( name_fetchZ, locf,
                                    (totalIter-1),
                                    s1,
                                    s2 );
#endif
                                total_flux += heatFlux_fetchX[i][j][k];
                            }
                        }
                    }
                }
                printf( "{PUSHER_FETCHER_1} total_flux: %lf at time: %d [s]\n", total_flux, (totalIter-1));
            }

            heatFluxOutFile.open("heatFluxCpp.txt", std::ios_base::app);
            heatFluxOutFile << n*timeStepSize << " " << total_flux << std::endl;
            heatFluxOutFile.close();
        }
    }

    return 0;
}