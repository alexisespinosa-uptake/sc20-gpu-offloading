#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include "globals.h" //contains array sizes and needed externs
#ifndef _JUSTOMP_
   #include "functions_acc.h"
#endif
#ifndef _JUSTACC_
   #include "functions_omp.h"
#endif


#if defined (_UPDATE_INTERNAL_) || defined (_ALL_INTERNAL_)
   #if !defined (_PGI_) && !defined (_NVCPP_)
      #define MAX(X,Y) ((X) > (Y) ? (X) : (Y))
   #endif
#endif

// smallest permitted change in temperature
#define MAX_TEMP_ERROR 0.02

// Global arrays
//double *restrict T_new; // temperature grid
//double *restrict T; // temperature grid from last iteration

//   initialisation routine
void init(double *restrict T, double *restrict T_new);

int main(int argc, char *argv[]) {

    int i, j;                                            // grid indexes
    int max_iterations;                                  // maximal number of iterations
    int iteration=1;                                     // iteration
    double dt=100;                                       // largest change in temperature
    struct timeval start_time, stop_time, elapsed_time;  // timers

    double *restrict T_new=(double*)malloc(sizeof(double)*(GRIDX+2)*(GRIDY+2)); // temperature grid
    double *restrict T=(double*)malloc(sizeof(double)*(GRIDX+2)*(GRIDY+2)); // temperature grid from last iteration


    if(argc!=2) {
      printf("Usage: %s number_of_iterations\n",argv[0]);
      exit(1);
    } else {
      max_iterations=atoi(argv[1]);
    }

    gettimeofday(&start_time,NULL); 

    init(T,T_new);                  
   
    #ifndef _NOPRELOAD_
       #if defined(_JUSTOMP_) || defined(_PRELOADOMP_)
          //#pragma omp target data map(tofrom:T) map(alloc:T_new)
                  //:gcc11:fails in runtime: illegal memory access
          #pragma omp target data map(tofrom:T[:(GRIDX+2)*(GRIDY+2)]) map(alloc:T_new[:(GRIDX+2)*(GRIDY+2)])
                  //:gcc11:works
       #else
          //#pragma acc data copy(T) create(T_new)
                  //:pgi:fails in compilation: error says "cannot determine bounds"
                  //:gcc11:fails in runtime: illegal memory access
          #pragma acc data copy(T[:(GRIDX+2)*(GRIDY+2)]) create(T_new[:(GRIDX+2)*(GRIDY+2)])
                  //:pgi:works
                  //:gcc11:works
       #endif
    #endif

    // simulation iterations
    while ( dt > MAX_TEMP_ERROR && iteration <= max_iterations ) {
    /*for ( iteration=1; iteration <=max_iterations; iteration++){
    if (dt > MAX_TEMP_ERROR) {*/

        // main computational kernel, average over neighbours in the grid
        #if defined (_AVERAGE_INTERNAL_) || defined (_ALL_INTERNAL_)
           #ifndef _JUSTOMP_
              //#pragma acc kernels
              //   #pragma acc loop independent //together with kernels above
                   //:pgi:justacc:(internal):works (fast:only copies data outside the while)
              //#pragma acc parallel loop collapse(2)
                   //:pgi:justacc:(internal):works (fast:only copies data outside the while)
                   //:gcc11:justacc:(internal):works (fast:only copies data outside the while)
              //#pragma acc parallel loop copyin(T) copyout(T_new) collapse(2)
                   //:pgi:justacc:(internal):works (fast:only copies data outside the while)
                   //:gcc11:justacc:(internal):fails at runtime:illegal memory access
              #pragma acc parallel loop copyin(T[:(GRIDX+2)*(GRIDY+2)]) copyout(T_new[:(GRIDX+2)*(GRIDY+2)]) collapse(2)
                   //:pgi:justacc:(internal):works (fast:only copies data outside the while)
                   //:gcc11:justacc:(internal):works (fast:only copies data outside the while)
              //#pragma acc parallel loop pcopyin(T[:(GRIDX+2)*(GRIDY+2)]) pcopyout(T_new[:(GRIDX+2)*(GRIDY+2)]) collapse(2)
                   //:pgi:justacc:(internal):works (fast:only copies data outside the while)
                   //:gcc11:justacc:(internal):works (fast:only copies data outside the while)
              //#pragma acc parallel loop present(T) present(T_new) collapse(2)
                   //:pgi:justacc:(internal):works (fast:only copies data outside the while)
                   //:gcc11:justacc:(internal):fails at runtime:present clause error
              //#pragma acc parallel loop present(T[:(GRIDX+2)*(GRIDY+2)]) present(T_new[:(GRIDX+2)*(GRIDY+2)]) collapse(2)
                   //:pgi:justacc:(internal):works (fast:only copies data outside the while)
                   //:gcc11:justacc:(internal):works (fast:only copies data outside the while)
           #else
              //#pragma omp target
                   //:gcc11:justomp:(internal):works (fast:only copies data outside the while)
              //#pragma omp target map(to:T) map(from:T_new)
                   //:gcc11:justomp:(internal):fails at execution time: illegal memory access
              #pragma omp target map(to:T[:(GRIDX+2)*(GRIDY+2)]) map(from:T_new[:(GRIDX+2)*(GRIDY+2)])
                   //:gcc11:justomp:(internal):works (fast:only copies data outside the while)
              #pragma omp teams distribute parallel for collapse(2) private(i,j)
           #endif
           for(i = 1; i <= GRIDX; i++)
              #ifndef _JUSTOMP_
              //   #pragma acc loop independent //together with kernels above
              #endif
              for(j = 1; j <= GRIDY; j++)
                 T_new[OFFSET(i,j)] = 0.25 * (T[OFFSET(i+1,j)] + T[OFFSET(i-1,j)] +
                                       T[OFFSET(i,j+1)] + T[OFFSET(i,j-1)]);
        #else
           #ifndef _JUSTOMP_
           getAverage_acc(T,T_new);
           #else
           getAverage_omp(T,T_new);
           #endif
        #endif
        // reset dt
        dt = 0.0;

        // compute the largest change and copy T_new to T
        #if defined (_UPDATE_INTERNAL_) || (_ALL_INTERNAL_)
           #ifndef _JUSTACC_
              //#pragma omp target map(dt)
              //#pragma omp target map(tofrom:T,dt) map(to:T_new)
              #pragma omp target map(tofrom:T[:(GRIDX+2)*(GRIDY+2)],dt) map(to:T_new[:(GRIDX+2)*(GRIDY+2)])
              #pragma omp teams distribute parallel for collapse(2) reduction(max:dt) private(i,j)
           #else
              //#pragma acc kernels
              //   #pragma acc loop independent //together with kernels above
              //#pragma acc parallel loop reduction(max:dt) collapse(2)
              //#pragma acc parallel loop copy(T) copyin(T_new) reduction(max:dt) collapse(2)
              #pragma acc parallel loop copy(T[:(GRIDX+2)*(GRIDY+2)]) copyin(T_new[:(GRIDX+2)*(GRIDY+2)]) reduction(max:dt) collapse(2)
              //#pragma acc parallel loop pcopy(T[:(GRIDX+2)*(GRIDY+2)]) pcopyin(T_new[:(GRIDX+2)*(GRIDY+2)]) reduction(max:dt) collapse(2)
              //#pragma acc parallel loop present(T) present(T_new) reduction(max:dt) collapse(2)
              //#pragma acc parallel loop present(T[:(GRIDX+2)*(GRIDY+2)]) present(T_new[:(GRIDX+2)*(GRIDY+2)]) reduction(max:dt) collapse(2)
           #endif
           for(i = 1; i <= GRIDX; i++){
              #ifndef _JUSTACC_
                 #define papa 0 
              #else
              //   #pragma acc loop independent //together with kernels above
              #endif
              for(j = 1; j <= GRIDY; j++){
                 #if defined (_PGI_) || defined (_NVCPP_)
                 dt = fmax( fabs(T_new[OFFSET(i,j)]-T[OFFSET(i,j)]), dt);
                 #else
                 dt = MAX( fabs(T_new[OFFSET(i,j)]-T[OFFSET(i,j)]), dt);
                 #endif
                 T[OFFSET(i,j)] = T_new[OFFSET(i,j)];
              }
           }
        #else
           #ifndef _JUSTACC_
           dt = updateT_omp(T,T_new,dt);
           #else
           //dt = updateT_acc(GRIDX,GRIDY,T,T_new,dt);
           dt = updateT_acc(T,T_new,dt);
           #endif
        #endif
        // periodically print largest change
        if((iteration % 100) == 0) 
            printf("Iteration %4.0d, dt %f\n",iteration,dt);
        
	     iteration++;
    /*}else
    {
       break;
    }*/
    }

    gettimeofday(&stop_time,NULL);
    timersub(&stop_time, &start_time, &elapsed_time); // measure time

    printf("Total time was %f seconds.\n", elapsed_time.tv_sec+elapsed_time.tv_usec/1000000.0);

    return 0;
}


// initialize grid and boundary conditions
void init(double *restrict T, double *restrict T_new){

    int i,j;

    for(i = 0; i <= GRIDX+1; i++){
        for (j = 0; j <= GRIDY+1; j++){
            T[OFFSET(i,j)] = 0.0;
        }
    }

    // these boundary conditions never change throughout run

    // set left side to 0 and right to a linear increase
    for(i = 0; i <= GRIDX+1; i++) {
        T[OFFSET(i,0)] = 0.0;
        T[OFFSET(i,GRIDY+1)] = (128.0/GRIDX)*i;
    }
    
    // set top to 0 and bottom to linear increase
    for(j = 0; j <= GRIDY+1; j++) {
        T[OFFSET(0,j)] = 0.0;
        T[OFFSET(GRIDX+1,j)] = (128.0/GRIDY)*j;
    }
}
