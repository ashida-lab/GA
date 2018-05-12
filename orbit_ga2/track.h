/*******************************************************************
track mps trajectroy by GPGPU

nvcc test1.cu -o test1 -arch sm_20
******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define _USE_MATH_DEFINES

#include <math.h>

#define dT 1.
#define Pi M_PI
#define mu0 1.2566370614e-6
#define Grav 6.67259E-11
#define Au 1.49597870700e11
#define Re 6378137.

#define PN 400
#define CN 400
#define EN 240
#define MN 160 //mutation

#define mi 1.672621777e-27
#define M_sun 1.9891e30
#define M_mer 3.301e23
#define M_ven 4.869e24
#define M_ear 5.9736e24
#define M_mar 6.4191e24
#define M_jup 1.8986e27
#define m_spacecraft 500

#define Vinit (-3e3)
#define Thrust 5e-3
#define Tmr 0 //4e-5

#define N_gene (6)

#define Step (4*24*6)
#define local_step (6)

#define N0 5e6
#define V0 5e5
#define R 2.
//#define I 1e20

#define Generation 4000

#define YYYY 2013.
#define MM 11.
#define DD 18.

/*******************************************************************
 type holding particle infomation
 ******************************************************************/
typedef struct{
	double a, e, I, L, omega_bar, Omega;
	double a_dot, e_dot, I_dot, L_dot, omega_bar_dot, Omega_dot;
}Elements;

typedef struct particle_info{
  double x,y,z;
  double vx,vy,vz;
  Elements elements;
  double p_func;
}Particle;

typedef struct particle_trace{
  double x,y,z;
  double vx,vy,vz;
  Elements elements;
  double p_func;
  double delta_v;
  double delta_v_para;
  double delta_v_perp;
}Trace;

extern Particle *particle_pn;
extern Particle *particle_cn;

extern int compute_p(int myrank);
extern int compute_c(int myrank);



