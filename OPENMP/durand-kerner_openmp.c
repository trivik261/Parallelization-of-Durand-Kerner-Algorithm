#include <stdio.h>
#include <math.h>
#include <complex.h>
#include<omp.h>
#include<unistd.h>
#include<string.h>
#include <stdlib.h>

#define M_PI 3.14159265358979323846
#define coff_size 500

double R=0; 
double complex z[coff_size]; 
double complex z1[coff_size]; 
double complex deltaZ[coff_size]; 
double deltaZMax;
double epsilon = 1e-6;
double complex fz,t,te,fz_temp; 
int max_iter = 50;


//----------------------Function Prototypes-------------------------------
void durand_kerner(); //Prototypes
void calc_theta();
double max_cof();
void printz();
void update_z();
void update_fz();
void printfile(double complex cList[],int n,int k,float st,int threads);

//----------------------Function Prototypes-------------------------------


int main() {

	double complex cList[coff_size];  	//List of coefficients
	double complex z; 
	double x,y; 						//x for real and y for imaginary parts of the coefficient
	int n=0; 							//n is number degree of polynomial
	
	
	while(scanf("%lf %lf",&x,&y) == 2)  
	{ 									//Read coefficients from stdin
		cList[n] = (x + y*I);
		n++;
	}
	x = 1;  							//Cn = 1, because the equation has to be normalized
	y = 0;
	z = (x + y*I);
	cList[n] = z; 						//Store in cList[]

	omp_set_num_threads(10);
	durand_kerner(cList,n,8);

}

//----------------------------------Function Definition-------------------------------


void durand_kerner(double complex cList[],int n,int threads) {
	float st,total=0;
	
	R = 1 + max_cof(cList,n); 			
	int a,i,j,k,l,m;
	
	for(a=0;a < n;a++) 
	{
		z[a] = ( cos( a*((2*M_PI)/n) ) + (I*sin( a*((2*M_PI)/n) )) )*R;
	}

	for(k=1;k <= max_iter;k++)
	{ 
		sleep(0.1);
		deltaZMax = 0; 
		st=omp_get_wtime();

		double complex QsubJ,fz_temp;
		#pragma omp parallel private (i,j,l,QsubJ,fz_temp,m) shared (z,deltaZ,n,cList)
		{
			#pragma omp for 
			for( j=0;j < n;j++)
			{ 
					QsubJ = 1;
					for(i=0;i < n;i++) { 
						if(i != j) { 
							QsubJ*= (z[j]-z[i]);
						}
					} 

					fz_temp = 1; 
					te=z[j];
					
					for(l = n-1;l >= 0;l--) {
						t=fz_temp*te;
						fz_temp =t+ cList[l];
					}
					

					deltaZ[j] = (-fz_temp/QsubJ);
			}
			#pragma omp barrier

		}
		#pragma omp barrier

		for(int m=0;m<n;m++)
		{	
			z[m]+=deltaZ[m];
			if(cabs(deltaZ[m]) > deltaZMax)
			{
				deltaZMax = cabs(deltaZ[m]);
			}
		}
		
		printf("Zmax=%f %d\n",deltaZMax,k);
		if(deltaZMax <= epsilon) { 
			break;   
		}

	}
	st=omp_get_wtime()-st;
	printfile(cList,n,k,st,threads);
	printz(cList,n);


}


double max_cof(double complex cList[],int n)
{
	double r;
	for(int j=0;j < n;j++) {
		if(cabs(cList[j]) > R) { 
			r = cabs(cList[j]);
		}
	} 
	return r;
}

void printz(double complex cList[],int n)
{
		printf("Final Output:(Note: if the roots repeat then there exist less than n-1 roots for the equation)\n");
		for(int i=0;i < n;i++) {  
                	printf("z[%d] = %0.10f + %0.10f*I\n",i,creal(z[i]),cimag(z[i]));
                fflush(stdout);
        	}
}
void printfile(double complex cList[],int n,int k,float st,int threads)
{		
		FILE *fp;  
   		fp = fopen("openmp_project_roots.txt", "w");
		fprintf(fp,"Durand Kerner OpenMP Algorithm:\n");
		fprintf(fp,"Max Iteration=%d\n",k);	
		fprintf(fp,"Time Taken=%f\n",st);
		fprintf(fp,"Final Output:(Note: if the roots repeat then there exist less than n-1 roots for the equation)\n");
		for(int i=0;i < n;i++) {  
                	fprintf(fp,"z[%d] = %0.10f + %0.10f*I\n",i,creal(z[i]),cimag(z[i]));
                fflush(stdout);
        	}
		fclose(fp);
		fp=fopen("openmp_time.txt", "a");
		fprintf(fp,"Thread=%d\tTime Taken=%f\n",threads,st);
		fclose(fp);
}

