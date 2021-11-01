// Headers of functions that will make use of openacc
void getAverage_acc(double * restrict T, double *restrict T_new);

double updateT_acc(double * restrict T, double *restrict T_new,double dt_old);
