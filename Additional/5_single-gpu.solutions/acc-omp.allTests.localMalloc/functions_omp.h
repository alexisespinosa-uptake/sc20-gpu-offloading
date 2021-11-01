// Headers of functions that will make use of OpenMP
void getAverage_omp(double *restrict T, double *restrict T_new);

double updateT_omp(double *restrict T, double *restrict T_new,double dt_old);
