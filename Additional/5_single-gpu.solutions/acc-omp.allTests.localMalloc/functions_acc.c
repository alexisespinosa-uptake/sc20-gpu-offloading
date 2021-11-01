#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "functions_acc.h"
#include "globals.h"

#if !defined (_PGI_) && !defined (_NVCPP_)
   #define MAX(X,Y) ((X) > (Y) ? (X) : (Y))
#endif

// --------------------------------------------------------------------
void getAverage_acc(double *restrict T,double *restrict T_new)
{
   int i,j;
   //#pragma acc kernels
   //   #pragma acc loop independent //to activate together with kernels above
        //:pgi:toBeTested
   //#pragma acc parallel loop collapse(2)
        //:pgi:toBeTested
   #pragma acc parallel loop copyin(T[:(GRIDX+2)*(GRIDY+2)]) copyout(T_new[:(GRIDX+2)*(GRIDY+2)]) collapse(2)
        //:pgi:justacc:static(non-managedMemory):works (fast:preloaded data works fine:avoids the indicated copy)
        //:pgi:justacc:dynamic(non-managedMemory):works (fast:preloaded data works fine:avoids the indicated copy)
        //:gcc11:justacc:static(non-managedMemory):works (fast:preloaded data works fine:avoids the indicated copy)
        //:gcc11:justacc:dynamic(non-managedMemory):works (fast:preloaded data works fine:avoids the indicated copy)
   //#pragma acc parallel loop pcopyin(T[:(GRIDX+2)*(GRIDY+2)]) pcopyout(T_new[:(GRIDX+2)*(GRIDY+2)]) collapse(2)
        //:pgi:justacc:static(non-managedMemory):works (fast:preloaded data works fine
        //                                              simply works as above)
        //:pgi:justacc:dynamic(non-managedMemory):works (fast:preloaded data works fine
        //                                              simply works as above)
        //:gcc11:justacc:static(non-managedMemory):works (fast:preloaded data works fine
        //                                              simply works as above)
        //:gcc11:justacc:dynamic(non-managedMemory):works (fast:preloaded data works fine
        //                                              simply works as above)
   //#pragma acc parallel loop present(T) present(T_new) collapse(2)
        //:pgi:justacc:static(non-managedMemory):works (fast:preloaded data works fine)
        //:pgi:justacc:dynamic(non-managedMemory):works (fast:preloaded data works fine)
        //:gcc11:justacc:static(non-managedMemory):fails at runtime: present clause error
        //:gcc11:justacc:dynamic(non-managedMemory):fails at runtime: present clause error
   //#pragma acc parallel loop present(T[:(GRIDX+2)*(GRIDY+2)]) present(T_new[:(GRIDX+2)*(GRIDY+2)]) collapse(2)
        //:pgi:justacc:static(non-managedMemory):works (fast:preloaded data works fine)
        //:pgi:justacc:dynamic(non-managedMemory):works (fast:preloaded data works fine)
        //:gcc11:justacc:static(non-managedMemory):works (fast:preloaded data works fine)
        //:gcc11:justacc:dynamic(non-managedMemory):works (fast:preloaded data works fine)
   for(i = 1; i <= GRIDX; i++) 
      //#pragma acc loop independent //to activate together with kernels above
      for(j = 1; j <= GRIDY; j++) 
         T_new[OFFSET(i,j)] = 0.25 * (T[OFFSET(i+1,j)] + T[OFFSET(i-1,j)] +
                                      T[OFFSET(i,j+1)] + T[OFFSET(i,j-1)]);
}

// --------------------------------------------------------------------
double updateT_acc(double *restrict T, double *restrict T_new, double dt_old)
{
   double dt=dt_old;
   int i,j;
   // compute the largest change and copy T_new to T
   //#pragma acc kernels
   //   #pragma acc loop independent //to activate together with kernels above
   //#pragma acc parallel loop reduction(max:dt) collapse(2)
   #pragma acc parallel loop copy(T[:(GRIDX+2)*(GRIDY+2)]) copyin(T_new[:(GRIDX+2)*(GRIDY+2)]) reduction(max:dt) collapse(2)
   //#pragma acc parallel loop pcopy(T[:(GRIDX+2)*(GRIDY+2)]) pcopyin(T_new[:(GRIDX+2)*(GRIDY+2)]) reduction(max:dt) collapse(2)
   //#pragma acc parallel loop present(T) present(T_new) reduction(max:dt) collapse(2)
   //#pragma acc parallel loop present(T[:(GRIDX+2)*(GRIDY+2)]) present(T_new[:(GRIDX+2)*(GRIDY+2)]) reduction(max:dt) collapse(2)
   for(i = 1; i <= GRIDX; i++){
      //#pragma acc loop independent //to activate together with kernels above
      for(j = 1; j <= GRIDY; j++){
         #if defined (_PGI_) || defined (_NVCPP_)
         dt = fmax( fabs(T_new[OFFSET(i,j)]-T[OFFSET(i,j)]), dt);
         #else
         dt = MAX( fabs(T_new[OFFSET(i,j)]-T[OFFSET(i,j)]), dt);
         #endif
         T[OFFSET(i,j)] = T_new[OFFSET(i,j)];
      }
   }
   return dt;
}
