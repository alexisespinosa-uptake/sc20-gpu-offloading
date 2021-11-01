#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "functions_omp.h"
#include "globals.h"

#ifndef _PGI_
   #define MAX(X,Y) ((X) > (Y) ? (X) : (Y))
#endif

// --------------------------------------------------------------------
void getAverage_omp(double *restrict T, double *restrict T_new)
{
   int i,j;
   //#pragma omp target
       //:gcc11:justomp:static(non-managedMemory):fails at execution: wrong results
       //:gcc11:justomp:dynamic(non-managedMemory):fails at execution: wrong results
   //#pragma omp target map(to:T) map(from:T_new)
       //:gcc11:justomp:static(non-managedMemory):fails at runtime:illegal memory access encountered
       //:gcc11:justomp:dynamic(non-managedMemory):fails at runtime:illegal memory access encountered
   #pragma omp target map(to:T[:(GRIDX+2)*(GRIDY+2)]) map(from:T_new[:(GRIDX+2)*(GRIDY+2)])
       //:gcc11:justomp:static(non-managedMemory):works (fast: preloaded data works fine)
       //:gcc11:justomp:dynamic(non-managedMemory):works (fast: preloaded data works fine)
   #pragma omp teams distribute parallel for collapse(2)
   for(i = 1; i <= GRIDX; i++)
      for(j = 1; j <= GRIDY; j++)
         T_new[OFFSET(i,j)] = 0.25 * (T[OFFSET(i+1,j)] + T[OFFSET(i-1,j)] +
                                      T[OFFSET(i,j+1)] + T[OFFSET(i,j-1)]);
}

// --------------------------------------------------------------------
double updateT_omp(double *restrict T, double *restrict T_new,double dt_old)
{
   double dt=dt_old;
   int i,j;
   
   // compute the largest change and copy T_new to T
   //#pragma omp target
   //#pragma omp target map(tofrom:dt,T) map(to:T_new)
   #pragma omp target map(tofrom:dt,T[:(GRIDX+2)*(GRIDY+2)]) map(to:T_new[:(GRIDX+2)*(GRIDY+2)])
   #pragma omp teams distribute parallel for collapse(2) reduction(max:dt)
   for(i = 1; i <= GRIDX; i++){
      for(j = 1; j <= GRIDY; j++){
         #ifdef _PGI_
         dt = fmax( fabs(T_new[OFFSET(i,j)]-T[OFFSET(i,j)]), dt);
         #else
         dt = MAX( fabs(T_new[OFFSET(i,j)]-T[OFFSET(i,j)]), dt);
#endif
         T[OFFSET(i,j)] = T_new[OFFSET(i,j)];
      }
   }
   return dt; 
}
