#include "track.h"
#include "cuda_runtime.h"

static void HandleError(cudaError_t err, const char *file, int line)
{
	if (err != cudaSuccess){
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}

#define HANDLE_ERROR(err) (HandleError(err,__FILE__,__LINE__))


//int init(Particle *particle,int myrank,const int particle_num);
__global__ void track(Particle *particle, const int c, const int particle_num);
__device__ void rk4(double T, double *x, double *y, double *z, double *vx, double *vy, double *vz, double n0, double v0);
__device__ void Force(double T, double x, double y, double z, double *Fx, double *Fy, double *Fz, double n0, double v0);
__device__ void Srand(unsigned int s);
__device__ unsigned int Rand();
__device__ void func(Elements *elements, double yyyy, double mm, double dd, double hh, double *xc, double *yc, double *zc);

__device__ static unsigned int randx = 1;

Particle *particle_pn;
Particle *particle_cn;

int compute_p(int myrank)
{
	int i, p;
	FILE *fp;
	char filename[256];

	Particle *dev_particle;

	cudaEvent_t t_start, t_stop;
	float elapsedTime;

	// printf("device %d is used\n",myrank); 

	HANDLE_ERROR(cudaSetDevice(0));//Tesla C2075

	HANDLE_ERROR(cudaEventCreate(&t_start));
	HANDLE_ERROR(cudaEventCreate(&t_stop));

	HANDLE_ERROR(cudaMalloc((void**)&dev_particle, PN*sizeof(Particle)));

	HANDLE_ERROR(cudaMemcpy(dev_particle, particle_pn, PN*sizeof(Particle), cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaEventRecord(t_start, 0));

	for (i = 0; i < Step; i++){
		track << <128, 128 >> >(dev_particle, i, PN);
	}

	HANDLE_ERROR(cudaEventRecord(t_stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(t_stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, t_start, t_stop));
	printf("Time %d:  %3.1f ms\n", myrank, elapsedTime);

	HANDLE_ERROR(cudaMemcpy(particle_pn, dev_particle, PN*sizeof(Particle), cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaFree(dev_particle));

	return(0);
}

int compute_c(int myrank)
{
	int i, p;
	FILE *fp;
	char filename[256];

	Particle *dev_particle;

	cudaEvent_t t_start, t_stop;
	float elapsedTime;

	//printf("device %d is used\n",myrank); 

	HANDLE_ERROR(cudaSetDevice(0));//titan

	HANDLE_ERROR(cudaEventCreate(&t_start));
	HANDLE_ERROR(cudaEventCreate(&t_stop));

	HANDLE_ERROR(cudaMalloc((void**)&dev_particle, CN*sizeof(Particle)));

	HANDLE_ERROR(cudaMemcpy(dev_particle, particle_cn, CN*sizeof(Particle), cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaEventRecord(t_start, 0));

	for (i = 0; i < Step; i++){
		track << <128, 128 >> >(dev_particle, i, CN);
	}

	HANDLE_ERROR(cudaEventRecord(t_stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(t_stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, t_start, t_stop));
	printf("Time %d:  %3.1f ms\n", myrank, elapsedTime);

	HANDLE_ERROR(cudaMemcpy(particle_cn, dev_particle, CN*sizeof(Particle), cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaFree(dev_particle));

	return(0);
}

__global__ void track(Particle *particle, const int c, const int particle_num)
{
	int i;
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	double x, y, z, vx, vy, vz, n0, v0;
	double x_p, y_p, z_p;
	double temp0, temp1, angle;

	while (tid < particle_num){
		x = particle[tid].x;
		y = particle[tid].y;
		z = particle[tid].z;
		vx = particle[tid].vx;
		vy = particle[tid].vy;
		vz = particle[tid].vz;
		n0 = N0;//*(0.5+0.015*(int)(tid/100));
		v0 = V0;//*(0.5+0.015*(int)(tid%100));

		for (i = 0; i < local_step; i++){
			rk4((c*local_step + i)*dT, &x, &y, &z, &vx, &vy, &vz, n0, v0);
		}

		particle[tid].x = x;
		particle[tid].y = y;
		particle[tid].z = z;
		particle[tid].vx = vx;
		particle[tid].vy = vy;
		particle[tid].vz = vz;

		//func(Ven,YYYY,MM,DD+(c*local_step+i+1)*dT/(24.*3600.),0.,&x_p,&y_p,&z_p);
		x_p = Re*cos(40.*Pi / 180.)*cos(2 * Pi / 24. / 3600. * (c + 1)*local_step*dT);
		y_p = Re*cos(40.*Pi / 180.)*sin(2 * Pi / 24. / 3600. * (c + 1)*local_step*dT);
		z_p = Re*sin(40.*Pi / 180.);
		temp0 = pow((particle[tid].x) / Re, 2) + pow((particle[tid].y) / Re, 2) + pow((particle[tid].z) / Re, 2);
		temp1 = pow((x_p - particle[tid].x) / Re, 2) + pow((y_p - particle[tid].y) / Re, 2) + pow((z_p - particle[tid].z) / Re, 2);

		angle = (x_p*(particle[tid].x - x_p) + y_p*(particle[tid].y - y_p) + z_p*(particle[tid].z - z_p)) / Re / Re / sqrt(temp0 * temp1);

		if (temp0 > 1 && temp1 < (1000000. / Re)*(1000000. / Re) && angle>cos(Pi / 180. * 80)){
			particle[tid].p_func += local_step*dT;
		}
		else if (temp0 <= 1){
			particle[tid].p_func = -1;
			particle[tid].x = 0.;
			particle[tid].y = 0.;
			particle[tid].z = 0.;
			particle[tid].vx = 0.;
			particle[tid].vy = 0.;
			particle[tid].vz = 0.;
		}

		tid += blockDim.x*gridDim.x;
	}
}

__device__ void rk4(double T, double *x, double *y, double *z, double *vx, double *vy, double *vz, double n0, double v0)
{
	double Fx, Fy, Fz;
	double kx1, kx2, kx3, kx4;
	double ky1, ky2, ky3, ky4;
	double kz1, kz2, kz3, kz4;
	double kvx1, kvx2, kvx3, kvx4;
	double kvy1, kvy2, kvy3, kvy4;
	double kvz1, kvz2, kvz3, kvz4;

	kx1 = *vx*dT;
	ky1 = *vy*dT;
	kz1 = *vz*dT;

	Force(T, *x, *y, *z, &Fx, &Fy, &Fz, n0, v0);
	kvx1 = Fx*dT;
	kvy1 = Fy*dT;
	kvz1 = Fz*dT;

	kx2 = (*vx + 0.5*kvx1)*dT;
	ky2 = (*vy + 0.5*kvy1)*dT;
	kz2 = (*vz + 0.5*kvz1)*dT;

	Force(T, *x + 0.5*kx1, *y + 0.5*ky1, *z + 0.5*kz1, &Fx, &Fy, &Fz, n0, v0);
	kvx2 = Fx*dT;
	kvy2 = Fy*dT;
	kvz2 = Fz*dT;

	kx3 = (*vx + 0.5*kvx2)*dT;
	ky3 = (*vy + 0.5*kvy2)*dT;
	kz3 = (*vz + 0.5*kvz2)*dT;

	Force(T, *x + 0.5*kx2, *y + 0.5*ky2, *z + 0.5*kz2, &Fx, &Fy, &Fz, n0, v0);
	kvx3 = Fx*dT;
	kvy3 = Fy*dT;
	kvz3 = Fz*dT;

	kx4 = (*vx + kvx3)*dT;
	ky4 = (*vy + kvy3)*dT;
	kz4 = (*vz + kvz3)*dT;

	Force(T, *x + kx3, *y + ky3, *z + kz3, &Fx, &Fy, &Fz, n0, v0);
	kvx4 = Fx*dT;
	kvy4 = Fy*dT;
	kvz4 = Fz*dT;

	*x += (kx1 + 2.*kx2 + 2.*kx3 + kx4) / 6.;
	*y += (ky1 + 2.*ky2 + 2.*ky3 + ky4) / 6.;
	*z += (kz1 + 2.*kz2 + 2.*kz3 + kz4) / 6.;

	*vx += (kvx1 + 2.*kvx2 + 2.*kvx3 + kvx4) / 6.;
	*vy += (kvy1 + 2.*kvy2 + 2.*kvy3 + kvy4) / 6.;
	*vz += (kvz1 + 2.*kvz2 + 2.*kvz3 + kvz4) / 6.;

	if ((*x)*(*x) + (*y)*(*y) + (*z)*(*z) < 0.09*Re*Re){
		*x = 0.;
		*y = 0.;
		*z = 0.;
		*vx = 0.;
		*vy = 0.;
		*vz = 0.;
	}
}

__device__ void Force(double T, double x, double y, double z, double *Fx, double *Fy, double *Fz, double n0, double v0)
{
	double r, N, V;
	double L0, L, Cd, F, F_ear, F_mps, al, be, ga;
	double cal, cbe, cga;
	double sal, sbe, sga;
	double x_p, y_p, z_p;

	r = sqrt(x*x + y*y + z*z);

	F_ear = -Grav*M_ear / (r*r);
	F_mps = Tmr*pow(r / Re, -2.3);
	al = 0.;
	be = 0.;
	ga = 0.;

	cal = cos(al);
	sal = sin(al);
	cbe = cos(be);
	sbe = sin(be);
	cga = cos(ga);
	sga = sin(ga);

	*Fx = F_ear*x / r + F_mps / r*((x*cal - y*sal)*cga - ((x*sal + y*cal)*cbe + z*sbe)*sga);
	*Fy = F_ear*y / r + F_mps / r*((x*cal - y*sal)*sga + ((x*sal + y*cal)*cbe + z*sbe)*cga);
	*Fz = F_ear*z / r + F_mps / r*((x*sal + y*cal)*sbe + z*cbe);

	/*func(Mer,YYYY,MM,DD+T,0.,&x_p,&y_p,&z_p);
	r=sqrt((x-x_p)*(x-x_p)+(y-y_p)*(y-y_p)+(z-z_p)*(z-z_p));
	F=-Grav*M_mer/(r*r);
	*Fx+=F*(x-x_p)/r;
	*Fy+=F*(y-y_p)/r;
	*Fz+=F*(z-z_p)/r;*/

}

__device__ void Srand(unsigned int s)
{
	randx = s;
}

__device__ unsigned int Rand()
{
	randx = randx * 1103515245 + 12345;
	return(randx & 2147483647);
}

__device__ void func(Elements *elements, double yyyy, double mm, double dd, double hh, double *xc, double *yc, double *zc)
{
	double y, m, A, B, JDT;

	if (mm > 2){
		y = yyyy;
		m = mm;
	}
	else{
		y = yyyy - 1;
		m = mm + 12;
	}

	A = (int)(y / 100);
	B = 2 - A + (int)(A / 4);

	JDT = (int)(365.25*y) + (int)(30.6001*(m + 1)) + dd + hh / 24 + 1720994.5 + B;
	//printf("%10f\n",JDT);

	double A0, A1, E;
	double T = (JDT - 2451545.0) / 36525.0;
	double Omega = Pi / 180.*(elements->Omega + elements->Omega_dot*T);//long.node
	double omega_bar = Pi / 180.*(elements->omega_bar + elements->omega_bar_dot*T);
	double omega = omega_bar - Omega;//long.peri - long.node
	double a = Re*(elements->a + elements->a_dot*T);
	double e = elements->e + elements->e_dot*T;
	double I = Pi / 180.*(elements->I + elements->I_dot*T);
	double L = Pi / 180.*(elements->L + elements->L_dot*T);
	double M = L - omega_bar;

	double co, so, cO, sO, cE, sE, ci, si, x0, y0, xo, yo;

	M = M - (int)((M + Pi) / (2 * Pi))*2.*Pi;

	A0 = 0.;
	A1 = 1.;

	while (fabs(A1 - A0) > 1e-7){
		A1 = e*sin(A0 + M);

		A0 = e*sin(A1 + M);
	}

	E = A0 + M;

	co = cos(omega);
	so = sin(omega);

	cO = cos(Omega);
	sO = sin(Omega);

	cE = cos(E);
	sE = sin(E);

	ci = cos(I);
	si = sin(I);

	x0 = a*(cE - e);
	y0 = a*sqrt(1 - e*e)*sE;

	xo = co*x0 - so*y0;
	yo = so*x0 + co*y0;

	*xc = cO*xo - sO*ci*yo;
	*yc = sO*xo + cO*ci*yo;
	*zc = si*yo;

}
