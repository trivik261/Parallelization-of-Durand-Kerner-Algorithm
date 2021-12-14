#include <complex>
#include <stdio.h>
#include <math.h>
#include<complex.h>
#include <cuComplex.h>
#define M_PI 3.14159265358979323846
#define coff_size 500
#define threads 8
using namespace std;


//----------------------------------------------------Complex Variables---------------------------------
__managed__ int nsize;


double R=0; 
complex<double> z[coff_size]; 
complex<double> deltaZ[coff_size]; 

__managed__  double deltaZMax;
complex<double> cList2[coff_size];

double epsilon = 1e-6;
//complex<double> QsubJ,fz; 
int max_iter = 800;



//----------------------------------------------------Complex Variables---------------------------------


//--------------------------------------------------Function Prototypes-------------------------------
void durand_kerner(complex<double> cList[],int n); //Prototypes
void calc_theta(int n);
double max_cof(complex<double> cList[],int n);
void printz(complex<double> cList[],int n);
void update_fz(complex<double> cList[],int n,int o);
void printfile(complex<double> cList[],int n,int k,float st);


//--------------------------------------------------Function Prototypes-------------------------------

//--------------------------------------------------GPU Function---------------------------------------

__global__ void calc_delta(cuDoubleComplex *a,cuDoubleComplex *b,cuDoubleComplex *c)
{
    
		int j=threadIdx.x+blockIdx.x*blockDim.x;
 

		cuDoubleComplex QsubJ = make_cuDoubleComplex(1,0);
		cuDoubleComplex mo=make_cuDoubleComplex(-1,0);
	
		for(int i=0;i < nsize;i++) { 
			
			if(i != j)
	  	{ 
					cuDoubleComplex b1=cuCsub(b[j],b[i]);
					QsubJ =cuCmul(QsubJ,b1);
			}
		} 

		cuDoubleComplex fz =make_cuDoubleComplex(1,0);
		for(int k = nsize-1;k >= 0;k--)
	 	{
			//printf("a[%d] = %0.10f + %0.10f*I\n",k,cuCreal(a[k]),cuCimag(a[k]));  
			cuDoubleComplex a1=cuCmul(fz,b[j]);
			fz = cuCadd(a1, a[k]);
		}
		c[j]=cuCdiv(cuCmul(mo,fz),QsubJ);         
}

//--------------------------------------------------GPU Function---------------------------------------

//----------------------------------------------------Main---------------------------------

int main() {
	
  complex<double> cList[coff_size];
 	complex<double> z; 
 	double x,y; //x for real and y for imaginary parts of the coefficient
 	int n=0; //n is number degree of polynomial



 //------Read Coefficients------------------------------------------------

	n=120;
	for(int i=0;i<n;i++ )
	{
			cList[i]=complex<double>(i+1,i+1);
	}

	
	nsize=n;

	cList[n] = complex<double>(1,0); //Store in cList[]

	if(n>=threads)
	durand_kerner(cList,n);
	else
	printf("No of Threads> No of Blocks,hence  program terminated");
	
	
}

//----------------------------------------------------Main----------------------------------------------

//----------------------------------------------------DK Function---------------------------------

void durand_kerner(complex<double> cList[],int n) {

	R = 1 + max_cof(cList,n);  //End Equation 5
	float time = 0,total=0;
	calc_theta(n);
	int k=0;
    cudaEvent_t start, stop;
		float elapsedTime;
    cuDoubleComplex *d_a, *d_b,*d_c;
    int size = n*sizeof(cList[0]);

		for(int j=0;j<n;j++)
				{
						z[j]=z[j]+deltaZ[j];
				}


    cudaMalloc((void **)&d_a, size);
    cudaMemcpy(d_a, &cList2, size, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_b, size);
		cudaMalloc((void **)&d_c, size);
		
		cudaEventCreate(&start);
    cudaEventRecord(start,0);

		for(int i=0;i<max_iter;i++)
		{	
				k+=1;
		 		deltaZMax=0;
				cudaMemcpy(d_b, &z, size, cudaMemcpyHostToDevice);
				calc_delta<<<n/threads + 1 ,threads>>>(d_a,d_b,d_c);
				cudaDeviceSynchronize();
				cudaMemcpy(&deltaZ, d_c, size, cudaMemcpyDeviceToHost);
			
				for(int j=0;j<n;j++)
				{
						z[j]=z[j]+deltaZ[j];
						if(abs(deltaZ[j]) > deltaZMax)
						{
							deltaZMax = abs(deltaZ[j]);
						}
				}
			if(deltaZMax <= epsilon)
			{ 
					break;   
			}

		}
		cudaEventCreate(&stop);
    cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start,stop);
    printf("No of Threads=%d\nNo of iterations=%d\nElapsed time (in seconds): %f\n" ,threads,k,elapsedTime/1000);
		
   printz(cList,n);
	 printfile(cList,n,k,elapsedTime);
      
}

//----------------------------------------------------DK Function----------------------------------------

//----------------------------------------------------Auxiliary Function---------------------------------

void calc_theta(int n) { 
	for(int j=0;j < n;j++) { 
        z[j]=complex<double> (cos(  j*((2*M_PI)/n) )*R,sin(  j*((2*M_PI)/n) )*R);
	} 

}

double max_cof(complex<double> cList[],int n)
{
	double r;
	for(int j=0;j < n;j++) {
			cList2[j]=cList[j];
		if(abs(cList[j]) > R) { 
			r = abs(cList[j]);
		}
	} 	
	return r;
}

void printz(complex<double> cList[],int n)
{
		printf("Final Output:(Note: if the roots repeat then there exist less than n-1 roots for the equation)\n");
		for(int i=0;i < n;i++) {  
                	printf("z[%d] = %0.10f + %0.10f*I\n",i,real(z[i]),imag(z[i]));
                fflush(stdout);
        	}
}

void printfile(complex<double> cList[],int n,int k,float st)
{		
		FILE *fp;  
   		fp = fopen("project_roots.txt", "w");
		fprintf(fp,"Durand Kerner Serial Algorithm:\n");
		fprintf(fp,"Max Iteration=%d\n",k);	
		fprintf(fp,"Time Taken=%f\n",st);
		fprintf(fp,"Final Output:(Note: if the roots repeat then there exist less than n-1 roots for the equation)\n");
		for(int i=0;i < n;i++) {  
                	fprintf(fp,"z[%d] = %0.10f + %0.10f*I\n",i,real(z[i]),imag(z[i]));
                fflush(stdout);
        	}
		fclose(fp);
}
//----------------------------------------------------Auxiliary Function---------------------------------

