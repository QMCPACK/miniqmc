//#include "hwi/include/bqc/A2_inlines.h"
#include <stdio.h>
#include <cstdlib>
#include "bspline_base.h"
#include "multi_bspline_structs.h"
#include <vector>
#include <omp.h>


#define max(a,b) ((a<b)?b:a)
#define min(a,b) ((a<b)?a:b)


#pragma omp declare target
void
eval_multi_UBspline_3d_s_vgh (const multi_UBspline_3d_s *spline,
			      float x, float y, float z,
			      float * restrict vals,
			      float* restrict grads,
			      float* restrict hess,
                              float *Af,float *dAf, float *d2Af);
#pragma omp end declare target

/*
inline unsigned long long timebase()
{
         return (unsigned long long)_rdtsc();
}
*/

const int WSIZE = 120;           // Walker
const int NSIZE = 2003;           // Values
const int MSIZE = NSIZE*3+3;      // Gradient vectors
const int OSIZE = NSIZE*9+9;      // Hessian Matrices 



const int NSIZE_round = NSIZE%16 ? NSIZE+16-NSIZE%16: NSIZE;
const size_t SSIZE = (size_t)NSIZE_round*48*48*48;  //Coefs size 
// const int L1SIZE = 16384;   // L1 cache size in bytes



#pragma omp declare target
#pragma omp declare simd simdlen(32)
static inline void eval_UBspline_3d_s_vgh_simd ( const float * restrict coefs_init,
                              const intptr_t xs,
                              const intptr_t ys,
                              const intptr_t zs,
                              float * restrict vals,
                              float * restrict grads,
                              float * restrict hess,
                              float *   a, float *   b, float *   c,
                              float *  da, float *  db, float *  dc,
                              float * d2a, float * d2b, float * d2c,
                              float dxInv, float dyInv, float dzInv);
#pragma omp end declare target

#pragma omp declare target
void  eval_abc(const float * restrict Af, float tx, float * restrict a);
#pragma omp end declare target




#pragma omp  declare target
class walker {
  public:
    float x,y,z;
    int   NSIZE, MSIZE, OSIZE;
    float resVal;
    float resGrad;
    float resHess;
    float * restrict vals;
    float * restrict grads;
    float * restrict hess;

    walker(const int NSIZE_inp, const int MSIZE_inp, const int OSIZE_inp){
      NSIZE = NSIZE_inp;
      MSIZE = MSIZE_inp;
      OSIZE = OSIZE_inp;
      vals  = new float [NSIZE];
      grads = new float [MSIZE];
      hess  = new float [OSIZE];
    }

    ~walker(){ delete[] vals; delete[] grads; delete[] hess;}

    void collect(){
      resVal = 0.0;
      resGrad = 0.0;
      resHess = 0.0;
      for( int i = 0; i < NSIZE; i++ ) resVal = resVal + vals[i];
      for( int i = 0; i < MSIZE; i++ ) resGrad = resGrad + grads[i];
      for( int i = 0; i < OSIZE; i++ ) resHess = resHess + hess[i];
    }
};
#pragma omp end declare target

 
//using namespace std;
int main(int argc, char ** argv){

  // char FillCache[ L1SIZE], FillCache2[ L1SIZE ];
  bool USE_GPU = true;


  float *Af,*dAf,*d2Af;
  Af=new float [16];
  dAf=new float [16];
  d2Af=new float [16];
  float x,y,z;
  multi_UBspline_3d_s *spline;
  spline = new multi_UBspline_3d_s;
  
  //posix_memalign ((void**)&hess, 16, ((size_t)sizeof(float)*OSIZE));
  spline->coefs= new float [SSIZE];
  //posix_memalign ((void**)&spline->coefs, 16, ((size_t)sizeof(float)*SSIZE));


  #pragma omp parallel for
  for(size_t i=0;i<SSIZE;i++)
      spline->coefs[i]=sqrt(0.22+i*1.0)*sin(i*1.0);


  spline->num_splines=NSIZE;
  spline->x_grid.start=0;
  spline->y_grid.start=0;
  spline->z_grid.start=0;
  spline->x_grid.num=45;
  spline->y_grid.num=45;
  spline->z_grid.num=45;
  spline->x_stride=NSIZE_round*48*48;
  spline->y_stride=NSIZE_round*48;
  spline->z_stride=NSIZE_round;
  spline->x_grid.delta_inv=45;
  spline->y_grid.delta_inv=45;
  spline->z_grid.delta_inv=45;

  x=0.822387;  y=0.989919;  z=0.104573;
  walker ** walkers = new walker *[WSIZE];
  for (int i=0; i<WSIZE; i++) {
    walkers[i]=new walker(NSIZE, MSIZE, OSIZE);
    for(int j=0; j<NSIZE; j++)
      walkers[i]->vals[j]  = 0.0;
    for(int j=0; j<MSIZE; j++)
      walkers[i]->grads[j] = 0.0;
    for(int j=0; j<OSIZE; j++)
      walkers[i]->hess[j]  = 0.0;
    walkers[i]->x = x + i*1.0/WSIZE;
    walkers[i]->y = y + i*1.0/WSIZE;
    walkers[i]->z = z + i*1.0/WSIZE;
  }


  Af[0]=-0.166667;
  Af[1]=0.500000;
  Af[2]=-0.500000;
  Af[3]=0.166667;
  Af[4]=0.500000;
  Af[5]=-1.000000;
  Af[6]=0.000000;
  Af[7]=0.666667;
  Af[8]=-0.500000;
  Af[9]=0.500000;
  Af[10]=0.500000;
  Af[11]=0.166667;
  Af[12]=0.166667;
  Af[13]=0.000000;
  Af[14]=0.000000;
  Af[15]=0.000000;
  dAf[0]=0.000000; d2Af[0]=0.000000;
  dAf[1]=-0.500000; d2Af[1]=0.000000;
  dAf[2]=1.000000; d2Af[2]=-1.000000;
  dAf[3]=-0.500000; d2Af[3]=1.000000;
  dAf[4]=0.000000; d2Af[4]=0.000000;
  dAf[5]=1.500000; d2Af[5]=0.000000;
  dAf[6]=-2.000000; d2Af[6]=3.000000;
  dAf[7]=0.000000; d2Af[7]=-2.000000;
  dAf[8]=0.000000; d2Af[8]=0.000000;
  dAf[9]=-1.500000; d2Af[9]=0.000000;
  dAf[10]=1.000000; d2Af[10]=-3.00000;
  dAf[11]=0.500000; d2Af[11]=1.000000;
  dAf[12]=0.000000; d2Af[12]=0.000000;
  dAf[13]=0.500000; d2Af[13]=0.000000;
  dAf[14]=0.000000; d2Af[14]=1.000000;
  dAf[15]=0.000000; d2Af[15]=0.000000;


  // for( i = 0; i < L1SIZE; i++ ) FillCache[i] = 'a';
  // for( i = 0; i < L1SIZE; i++ ) FillCache2[i] = FillCache[i] + 1;
  // for( i = 0; i < L1SIZE; i++ ) FillCache2[i] = FillCache2[i] - 1;

  int numthreads = omp_get_max_threads();
  fprintf(stdout, "Total number of CPU threads %d\n", numthreads);
  if (USE_GPU) fprintf(stdout,"using GPU\n");
  else         fprintf(stdout,"using CPU\n");

  if (USE_GPU == true){

   /* move data to GPU */
    #pragma omp target enter data map(alloc:walkers[0:WSIZE]) 

    for (int i=0; i<WSIZE; i++) {

      walker *&wlk_ptr = walkers[i];
      float * restrict  &mval = walkers[i]->vals;
      float * restrict  &mgrad = walkers[i]->grads;
      float * restrict  &mhess = walkers[i]->hess;

      #pragma omp target enter data map(to:wlk_ptr[0:1],mval[0:NSIZE],mgrad[0:MSIZE],mhess[0:OSIZE])

      #pragma omp target  map(alloc:wlk_ptr[:1],mval[0:NSIZE],mgrad[0:MSIZE],mhess[0:OSIZE])
      {
        walkers[i] = wlk_ptr;
        walkers[i]->vals  = mval;
        walkers[i]->grads = mgrad;
        walkers[i]->hess = mhess;
      }
    }

    multi_UBspline_3d_s *&mspl = spline;
    float * restrict &mcfs =  spline->coefs;

    #pragma omp target enter data map(to:mspl[0:1],mcfs[0:SSIZE]) 

    #pragma omp target map (alloc:mspl[0:1],mcfs[0:SSIZE])
    {
      spline->coefs = mcfs;
    } 
  }
/*  end moving data to GPU */

//  unsigned long long t0,t1,dt;
//  t0=timebase();
  int nteams, nthreads;
  if (USE_GPU){
     nteams = 128;
     nthreads = 4;
  }
  else{
     nteams = omp_get_max_threads();
     nthreads = 1;
  }

  printf("num_teams = %d, thread_limit = %d\n",nteams,nthreads);


  #pragma omp target enter data map(to:Af[0:16],dAf[0:16],d2Af[0:16]) if (USE_GPU)

  double t0 = omp_get_wtime();

  #pragma omp target data map(alloc:walkers[0:WSIZE],spline[0:1],Af[0:16],dAf[0:16],d2Af[0:16]) if (USE_GPU)
  {
     #pragma omp  target if(USE_GPU)
     {
        #pragma omp teams distribute  num_teams(nteams)  thread_limit(nthreads)  
        for(int i=0; i<WSIZE; i++){
          float x = walkers[i]->x, y = walkers[i]->y, z = walkers[i]->z;

          float * restrict vals  = walkers[i]->vals;
          float * restrict grads = walkers[i]->grads;
          float * restrict hess  = walkers[i]->hess;
          float ux = x*spline->x_grid.delta_inv;
          float uy = y*spline->y_grid.delta_inv;
          float uz = z*spline->z_grid.delta_inv;
          float ipartx, iparty, ipartz, tx, ty, tz;
          float a[4], b[4], c[4], da[4], db[4], dc[4], d2a[4], d2b[4], d2c[4]; //in shared memory 
          intptr_t xs = spline->x_stride;
          intptr_t ys = spline->y_stride;
          intptr_t zs = spline->z_stride;

          x -= spline->x_grid.start;
          y -= spline->y_grid.start;
          z -= spline->z_grid.start;
          ipartx = (int) ux; tx = ux-ipartx;    int ix = min(max(0,(int) ipartx),spline->x_grid.num-1);
          iparty = (int) uy; ty = uy-iparty;    int iy = min(max(0,(int) iparty),spline->y_grid.num-1);
          ipartz = (int) uz; tz = uz-ipartz;    int iz = min(max(0,(int) ipartz),spline->z_grid.num-1);

          eval_abc(Af,tx,&a[0]);
          eval_abc(Af,ty,&b[0]);
          eval_abc(Af,tz,&c[0]);
          eval_abc(dAf,tx,&da[0]);
          eval_abc(dAf,ty,&db[0]);
          eval_abc(dAf,tz,&dc[0]);
          eval_abc(d2Af,tx,&d2a[0]);
          eval_abc(d2Af,ty,&d2b[0]);
          eval_abc(d2Af,tz,&d2c[0]);              

          const float * restrict coefs_init = spline->coefs + ix*xs + iy*ys + iz*zs;

          #pragma omp parallel if(USE_GPU) // need to check if this "if" is required for CPUs 
          {
          int tid = omp_get_thread_num();
          int ntrds = omp_get_num_threads();
          int chunk = (spline->num_splines + ntrds - 1)/ntrds;
          int n_beg = tid*chunk;
          int n_end = min( (tid+1)*chunk, spline->num_splines);

        //for (int n=0; n<spline->num_splines; n++)
          #pragma omp  simd safelen(32) //for schedule(static,1)
          for (int n=n_beg; n<n_end; n++)
          eval_UBspline_3d_s_vgh_simd ( &coefs_init[n],
                                       xs, ys, zs, &vals[n], &grads[n*3], &hess[n*9],
                                       a, b, c, da, db, dc, d2a, d2b, d2c,
                                       spline->x_grid.delta_inv,
                                       spline->y_grid.delta_inv,
                                       spline->z_grid.delta_inv );


          }


        }
     } 
  }
  
  double t1 = omp_get_wtime();



  //test results
  if (USE_GPU){
    #pragma omp target teams distribute parallel for map(alloc:walkers[0:WSIZE]) // if (USE_GPU)
    for(int i=0; i<WSIZE; i++){
      walkers[i]->collect();
      if (i==0)
        printf("GPU walkers[%d]->collect([resVal resGrad resHess]) = [%e %e %e]\n",i,walkers[i]->resVal,walkers[i]->resGrad, walkers[i]->resHess);  
    }
  }
  else{
    #pragma omp parallel for 
    for(int i=0; i<WSIZE; i++){
      walkers[i]->collect();
      if (i==0)
        printf("CPU walkers[%d]->collect([resVal resGrad resHess]) = [%e %e %e]\n",i,walkers[i]->resVal,walkers[i]->resGrad, walkers[i]->resHess);
    }
  }




  if (USE_GPU){ 
    //copy vals, grads, hess data back 
    for(int i=0; i<WSIZE; i++){
      float * restrict  &mval = walkers[i]->vals;
      float * restrict  &mgrad = walkers[i]->grads;
      float * restrict  &mhess = walkers[i]->hess;
      #pragma omp target exit data map(from:mval[0:NSIZE],mgrad[0:MSIZE],mhess[0:OSIZE])
    }
  }

  
//  dt = t1 - t0;
  printf("msec =%.3lf\n\n",1.0e3*(t1-t0));

  delete []Af;
  delete []dAf;
  delete []d2Af;
  delete spline;
  delete []walkers;

  return 0;
}



#pragma omp declare target

void  eval_abc(const float * restrict Af, float tx, float * restrict a){
   
    a[0] = ( ( Af[0]  * tx + Af[1] ) * tx + Af[2] ) * tx + Af[3];
    a[1] = ( ( Af[4]  * tx + Af[5] ) * tx + Af[6] ) * tx + Af[7];
    a[2] = ( ( Af[8]  * tx + Af[9] ) * tx + Af[10] ) * tx + Af[11];
    a[3] = ( ( Af[12] * tx + Af[13] ) * tx + Af[14] ) * tx + Af[15];
}

#pragma omp end declare target


#pragma omp declare target

#pragma omp declare simd simdlen(32)
static inline void eval_UBspline_3d_s_vgh_simd ( const float * restrict coefs_init,
                              const intptr_t xs,
                              const intptr_t ys,
                              const intptr_t zs,
                              float * restrict vals,
                              float * restrict grads,
                              float * restrict hess,
                              float *   a, float *   b, float *   c,
                              float *  da, float *  db, float *  dc,
                              float * d2a, float * d2b, float * d2c,
                              float dxInv, float dyInv, float dzInv)
{


  float h[9];
  float v0 = 0.0f;
  for (int i = 0; i < 9; ++i) h[i] = 0.0f;

  for (int i=0; i<4; i++)
    for (int j=0; j<4; j++){
        float pre20 = d2a[i]*  b[j];
        float pre10 =  da[i]*  b[j];
        float pre00 =   a[i]*  b[j];
        float pre11 =  da[i]* db[j];
        float pre01 =   a[i]* db[j];
        float pre02 =   a[i]*d2b[j];

        const float * restrict coefs = coefs_init + i*xs + j*ys;

        float sum0 =   c[0] * coefs[0] +   c[1] * coefs[zs] +   c[2] * coefs[zs*2] +   c[3] * coefs[zs*3];
        float sum1 =  dc[0] * coefs[0] +  dc[1] * coefs[zs] +  dc[2] * coefs[zs*2] +  dc[3] * coefs[zs*3];
        float sum2 = d2c[0] * coefs[0] + d2c[1] * coefs[zs] + d2c[2] * coefs[zs*2] + d2c[3] * coefs[zs*3];

        h[0]  += pre20 * sum0;
        h[1]  += pre11 * sum0;
        h[2]  += pre10 * sum1;
        h[4]  += pre02 * sum0;
        h[5]  += pre01 * sum1;
        h[8]  += pre00 * sum2;
        h[3]  += pre10 * sum0;
        h[6]  += pre01 * sum0;
        h[7]  += pre00 * sum1;
        v0    += pre00 * sum0;
    }
  vals[0] = v0;
  grads[0]  = h[3] * dxInv;
  grads[1]  = h[6] * dyInv;
  grads[2]  = h[7] * dzInv;

  hess [0] = h[0]*dxInv*dxInv;
  hess [1] = h[1]*dxInv*dyInv;
  hess [2] = h[2]*dxInv*dzInv;
  hess [3] = h[1]*dxInv*dyInv; // Copy hessian elements into lower half of 3x3 matrix
  hess [4] = h[4]*dyInv*dyInv;
  hess [5] = h[5]*dyInv*dzInv;
  hess [6] = h[2]*dxInv*dzInv; // Copy hessian elements into lower half of 3x3 matrix
  hess [7] = h[5]*dyInv*dzInv; //Copy hessian elements into lower half of 3x3 matrix
  hess [8] = h[8]*dzInv*dzInv;
}
#pragma omp end declare target

