#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "track.h"
#include <mpi.h>

#pragma comment(lib,"msmpi.lib")

int init(Particle *particle, int myrank, const int particle_num);
int gen_children();
int roulette(Particle a[]);
void quicksort(Particle a[], int first, int last);
int output(const int myrank, const int gen);
int output_gene(Particle a[], const int myrank, const int gen);
int sr_particle(const int myrank);
void func_init(Elements *elements, double yyyy, double mm, double dd, double hh, double *xc, double *yc, double *zc, double *vxc, double *vyc, double *vzc);

Particle send_buf[PN / 2];
Particle recv_buf[PN / 2];

int main(int argc, char *argv[])
{
	int myid, myrank, p;
	int i;
	Particle *particle_an;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	myrank = myid;

	if (p > 2){
		printf("too many processes\n");

		MPI_Finalize();
		exit(0);
	}

	particle_pn = (Particle*)malloc(PN*sizeof(Particle));
	particle_cn = (Particle*)malloc(CN*sizeof(Particle));
	particle_an = (Particle*)malloc((PN + CN)*sizeof(Particle));

	init(particle_pn, myrank, PN);

	//p_func of parents PN
	compute_p(myrank);

	for (i = 0; i < Generation; i++){

		//generate children PN+CN
		gen_children();

		//p_func of children
		compute_c(myrank);

		//selection PN+CN -> PN
		memcpy(particle_an, particle_pn, PN*sizeof(Particle));
		memcpy(particle_an + PN, particle_cn, CN*sizeof(Particle));

		quicksort(particle_an, 0, PN + CN - 1);

		printf("Gen %d ---- max ---- %E\n", i, particle_an[PN + CN - 1].p_func);
		output_gene(particle_an, myrank, i);

		memcpy(particle_pn, particle_an + PN + CN - EN, EN*sizeof(Particle));

		roulette(particle_an);

		sr_particle(myrank);

		if (i % 10 == 0){
			output(myrank, i);
		}

		MPI_Barrier(MPI_COMM_WORLD);
	}

	MPI_Finalize();

	return(0);
}

int init_elements(Elements *elements)
{
	elements->e = 0.5*(double)rand() / (double)(RAND_MAX + 1.);
	elements->a = (1 + 1000000./Re*(double)rand() / (double)(RAND_MAX + 1.))/(1-elements->e);
	elements->I = 180.*(double)rand() / (double)(RAND_MAX + 1.);
	elements->L = 360.*(double)rand() / (double)(RAND_MAX + 1.);
	elements->omega_bar = 360.*(double)rand() / (double)(RAND_MAX + 1.);
	elements->Omega = 360.*(double)rand() / (double)(RAND_MAX + 1.);
	elements->a_dot = 0.;
	elements->e_dot = 0.;
	elements->I_dot = 0.;
	elements->L_dot = 0.;
	elements->omega_bar_dot = 0.;
	elements->Omega_dot = 0.;

	return(0);
}

int init(Particle *particle, int myrank, const int particle_num)
{
	int i, j;
	double x, y, z, vx, vy, vz;

	srand(myrank*100.);

	for (i = 0; i < particle_num; i++){
		init_elements(&particle[i].elements);
		func_init(&particle[i].elements, YYYY, MM, DD, 0., &x, &y, &z, &vx, &vy, &vz);

		particle[i].x = x;
		particle[i].y = y;
		particle[i].z = z;
		particle[i].vx = vx;
		particle[i].vy = vy;
		particle[i].vz = vz;
		particle[i].p_func = 0.;
	}

	return(0);
}

int gen_children()
{
	int p0, p1, s;
	int i, j;
	double x, y, z, vx, vy, vz;
	Elements elements;

	for (i = 0; i < CN / 2; i++){
		p0 = floor(PN*(double)rand() / (double)(RAND_MAX + 1.));
		p1 = floor(PN*(double)rand() / (double)(RAND_MAX + 1.));
		s = floor((N_gene + 1)*(double)rand() / (double)(RAND_MAX + 1.));

		switch (s){
		case 0:
			particle_cn[2 * i].elements.a = particle_pn[p0].elements.a;
			particle_cn[2 * i + 1].elements.a = particle_pn[p1].elements.a;
			particle_cn[2 * i].elements.e = particle_pn[p0].elements.e;
			particle_cn[2 * i + 1].elements.e = particle_pn[p1].elements.e;
			particle_cn[2 * i].elements.I = particle_pn[p0].elements.I;
			particle_cn[2 * i + 1].elements.I = particle_pn[p1].elements.I;
			particle_cn[2 * i].elements.L = particle_pn[p0].elements.L;
			particle_cn[2 * i + 1].elements.L = particle_pn[p1].elements.L;
			particle_cn[2 * i].elements.omega_bar = particle_pn[p0].elements.omega_bar;
			particle_cn[2 * i + 1].elements.omega_bar = particle_pn[p1].elements.omega_bar;
			particle_cn[2 * i].elements.Omega = particle_pn[p0].elements.Omega;
			particle_cn[2 * i + 1].elements.Omega = particle_pn[p1].elements.Omega;

			break;
		case 1:
			particle_cn[2 * i].elements.a = particle_pn[p1].elements.a;
			particle_cn[2 * i + 1].elements.a = particle_pn[p0].elements.a;
			particle_cn[2 * i].elements.e = particle_pn[p0].elements.e;
			particle_cn[2 * i + 1].elements.e = particle_pn[p1].elements.e;
			particle_cn[2 * i].elements.I = particle_pn[p0].elements.I;
			particle_cn[2 * i + 1].elements.I = particle_pn[p1].elements.I;
			particle_cn[2 * i].elements.L = particle_pn[p0].elements.L;
			particle_cn[2 * i + 1].elements.L = particle_pn[p1].elements.L;
			particle_cn[2 * i].elements.omega_bar = particle_pn[p0].elements.omega_bar;
			particle_cn[2 * i + 1].elements.omega_bar = particle_pn[p1].elements.omega_bar;
			particle_cn[2 * i].elements.Omega = particle_pn[p0].elements.Omega;
			particle_cn[2 * i + 1].elements.Omega = particle_pn[p1].elements.Omega;

			break;
		case 2:
			particle_cn[2 * i].elements.a = particle_pn[p1].elements.a;
			particle_cn[2 * i + 1].elements.a = particle_pn[p0].elements.a;
			particle_cn[2 * i].elements.e = particle_pn[p1].elements.e;
			particle_cn[2 * i + 1].elements.e = particle_pn[p0].elements.e;
			particle_cn[2 * i].elements.I = particle_pn[p0].elements.I;
			particle_cn[2 * i + 1].elements.I = particle_pn[p1].elements.I;
			particle_cn[2 * i].elements.L = particle_pn[p0].elements.L;
			particle_cn[2 * i + 1].elements.L = particle_pn[p1].elements.L;
			particle_cn[2 * i].elements.omega_bar = particle_pn[p0].elements.omega_bar;
			particle_cn[2 * i + 1].elements.omega_bar = particle_pn[p1].elements.omega_bar;
			particle_cn[2 * i].elements.Omega = particle_pn[p0].elements.Omega;
			particle_cn[2 * i + 1].elements.Omega = particle_pn[p1].elements.Omega;

			break;
		case 3:
			particle_cn[2 * i].elements.a = particle_pn[p1].elements.a;
			particle_cn[2 * i + 1].elements.a = particle_pn[p0].elements.a;
			particle_cn[2 * i].elements.e = particle_pn[p1].elements.e;
			particle_cn[2 * i + 1].elements.e = particle_pn[p0].elements.e;
			particle_cn[2 * i].elements.I = particle_pn[p1].elements.I;
			particle_cn[2 * i + 1].elements.I = particle_pn[p0].elements.I;
			particle_cn[2 * i].elements.L = particle_pn[p0].elements.L;
			particle_cn[2 * i + 1].elements.L = particle_pn[p1].elements.L;
			particle_cn[2 * i].elements.omega_bar = particle_pn[p0].elements.omega_bar;
			particle_cn[2 * i + 1].elements.omega_bar = particle_pn[p1].elements.omega_bar;
			particle_cn[2 * i].elements.Omega = particle_pn[p0].elements.Omega;
			particle_cn[2 * i + 1].elements.Omega = particle_pn[p1].elements.Omega;

			break;
		case 4:
			particle_cn[2 * i].elements.a = particle_pn[p1].elements.a;
			particle_cn[2 * i + 1].elements.a = particle_pn[p0].elements.a;
			particle_cn[2 * i].elements.e = particle_pn[p1].elements.e;
			particle_cn[2 * i + 1].elements.e = particle_pn[p0].elements.e;
			particle_cn[2 * i].elements.I = particle_pn[p1].elements.I;
			particle_cn[2 * i + 1].elements.I = particle_pn[p0].elements.I;
			particle_cn[2 * i].elements.L = particle_pn[p1].elements.L;
			particle_cn[2 * i + 1].elements.L = particle_pn[p0].elements.L;
			particle_cn[2 * i].elements.omega_bar = particle_pn[p0].elements.omega_bar;
			particle_cn[2 * i + 1].elements.omega_bar = particle_pn[p1].elements.omega_bar;
			particle_cn[2 * i].elements.Omega = particle_pn[p0].elements.Omega;
			particle_cn[2 * i + 1].elements.Omega = particle_pn[p1].elements.Omega;

			break;
		case 5:
			particle_cn[2 * i].elements.a = particle_pn[p1].elements.a;
			particle_cn[2 * i + 1].elements.a = particle_pn[p0].elements.a;
			particle_cn[2 * i].elements.e = particle_pn[p1].elements.e;
			particle_cn[2 * i + 1].elements.e = particle_pn[p0].elements.e;
			particle_cn[2 * i].elements.I = particle_pn[p1].elements.I;
			particle_cn[2 * i + 1].elements.I = particle_pn[p0].elements.I;
			particle_cn[2 * i].elements.L = particle_pn[p1].elements.L;
			particle_cn[2 * i + 1].elements.L = particle_pn[p0].elements.L;
			particle_cn[2 * i].elements.omega_bar = particle_pn[p1].elements.omega_bar;
			particle_cn[2 * i + 1].elements.omega_bar = particle_pn[p0].elements.omega_bar;
			particle_cn[2 * i].elements.Omega = particle_pn[p0].elements.Omega;
			particle_cn[2 * i + 1].elements.Omega = particle_pn[p1].elements.Omega;

			break;
		case 6:
			particle_cn[2 * i].elements.a = particle_pn[p1].elements.a;
			particle_cn[2 * i + 1].elements.a = particle_pn[p0].elements.a;
			particle_cn[2 * i].elements.e = particle_pn[p1].elements.e;
			particle_cn[2 * i + 1].elements.e = particle_pn[p0].elements.e;
			particle_cn[2 * i].elements.I = particle_pn[p1].elements.I;
			particle_cn[2 * i + 1].elements.I = particle_pn[p0].elements.I;
			particle_cn[2 * i].elements.L = particle_pn[p1].elements.L;
			particle_cn[2 * i + 1].elements.L = particle_pn[p0].elements.L;
			particle_cn[2 * i].elements.omega_bar = particle_pn[p1].elements.omega_bar;
			particle_cn[2 * i + 1].elements.omega_bar = particle_pn[p0].elements.omega_bar;
			particle_cn[2 * i].elements.Omega = particle_pn[p1].elements.Omega;
			particle_cn[2 * i + 1].elements.Omega = particle_pn[p0].elements.Omega;

			break;
		default:
			break;
		}
	}

	//mutation
	for (i = 0; i < MN; i++){
		p0 = floor(CN*(double)rand() / (double)(RAND_MAX + 1.));
		init_elements(&particle_cn[p0].elements);
	}

	for (i = 0; i < CN; i++){
		func_init(&particle_cn[i].elements, YYYY, MM, DD, 0., &x, &y, &z, &vx, &vy, &vz);

		particle_cn[i].x = x;
		particle_cn[i].y = y;
		particle_cn[i].z = z;
		particle_cn[i].vx = vx;
		particle_cn[i].vy = vy;
		particle_cn[i].vz = vz;
		particle_cn[i].p_func = 0.;
	}

	return(0);
}

void quicksort(Particle a[], int first, int last)
{
	int i, j;
	double x;
	Particle t;

	x = a[(first + last) / 2].p_func;
	i = first;
	j = last;

	for (;;){
		while (a[i].p_func < x) i++;
		while (x < a[j].p_func) j--;

		if (i >= j) break;
		memcpy(&t, &a[i], sizeof(Particle));
		memcpy(&a[i], &a[j], sizeof(Particle));
		memcpy(&a[j], &t, sizeof(Particle));
		i++;
		j--;
	}
	if (first < i - 1) quicksort(a, first, i - 1);
	if (j + 1 < last) quicksort(a, j + 1, last);
}

int roulette(Particle a[])
{
	int i;
	int p;

	for (i = EN; i < PN; i++){
		p = floor((PN + CN - EN)*sqrt((double)rand() / (double)(RAND_MAX + 1.)));
		memcpy(&particle_pn[i], &a[p], sizeof(Particle));
	}

	return(0);
}

int output(const int myrank, const int gen)
{
	int p;
	FILE *fp;
	char filename[256];

	sprintf(filename, "output%d-%d.txt", myrank, gen);
	fp = fopen(filename, "w");
	for (p = 0; p < PN; p++){
		fprintf(fp, "%d %E\n", p, particle_pn[p].p_func);
	}
	fclose(fp);

	return(0);
}

int output_gene(Particle a[], const int myrank, const int gen)
{
	int i;
	FILE *fp;
	char filename[256];

	sprintf(filename, "gene%d-%d.txt", myrank, gen);
	fp = fopen(filename, "w");
	for (i = 1; i < 6; i++){
		fprintf(fp, "%05d | %e %e %e %e %e %e | %e\n", i,
			a[PN + CN - i].elements.a,
			a[PN + CN - i].elements.e,
			a[PN + CN - i].elements.I,
			a[PN + CN - i].elements.L,
			a[PN + CN - i].elements.omega_bar,
			a[PN + CN - i].elements.Omega,
			a[PN + CN - i].p_func
			);
	}

	for (i = PN+CN-4; i < PN+CN+1; i++){
		fprintf(fp, "%05d | %e %e %e %e %e %e | %e\n", i,
			a[PN + CN - i].elements.a,
			a[PN + CN - i].elements.e,
			a[PN + CN - i].elements.I,
			a[PN + CN - i].elements.L,
			a[PN + CN - i].elements.omega_bar,
			a[PN + CN - i].elements.Omega,
			a[PN + CN - i].p_func
			);
	}
	fclose(fp);

	return(0);
}

int sr_particle(const int myrank)
{
	int src, dest, tag, count;
	MPI_Status stat;
	MPI_Request request;
	int i;

	tag = 1000;

	count = PN*sizeof(Particle) / sizeof(int) / 2;

	for (i = 0; i < PN / 2; i++){
		memcpy(&send_buf[i], &particle_pn[2 * i], sizeof(Particle));
	}

	if (myrank == 0){
		src = 1;
	}
	else{
		src = 0;
	}

	MPI_Irecv(&recv_buf, count, MPI_INT, src, tag, MPI_COMM_WORLD, &request);

	if (myrank == 0){
		dest = 1;
	}
	else{
		dest = 0;
	}

	MPI_Send(&send_buf, count, MPI_INT, dest, tag, MPI_COMM_WORLD);

	MPI_Wait(&request, &stat);

	for (i = 0; i < PN / 2; i++){
		memcpy(&particle_pn[2 * i], &recv_buf[i], sizeof(Particle));
	}

	return(0);
}

void func_init(Elements *elements, double yyyy, double mm, double dd, double hh, double *xc, double *yc, double *zc, double *vxc, double *vyc, double *vzc)
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

	double co, so, cO, sO, cE, sE, ci, si, x0, y0, xo, yo, vx0, vy0, vxo, vyo, r;

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

	r = sqrt((*xc)*(*xc) + (*yc)*(*yc) + (*zc)*(*zc));

	vx0 = sqrt(Grav*a*M_ear) / r*(-sE);
	vy0 = sqrt(Grav*a*M_ear) / r*sqrt(1 - e*e)*cE;

	vxo = co*vx0 - so*vy0;
	vyo = so*vx0 + co*vy0;

	*vxc = cO*vxo - sO*ci*vyo;
	*vyc = sO*vxo + cO*ci*vyo;
	*vzc = si*vyo;
}
