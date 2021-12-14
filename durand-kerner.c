#include <stdio.h>
#include <math.h>
#include <complex.h>
#include<omp.h>

#define M_PI 3.14159265358979323846
#define coff_size 800

double R=0; 
double complex z[coff_size]; 
double complex deltaZ[coff_size]; 
double deltaZMax;
double epsilon = 1e-6;
double complex QsubJ,fz; 
int max_iter = 40;


//----------------------Function Prototypes-------------------------------
void durand_kerner(); //Prototypes
void calc_theta();
double max_cof();
void printz();
void update_z();
void update_fz();
void printfile(double complex cList[],int n,int k,float st);


int main() {

	double complex cList[coff_size];  //List of coefficients
	double complex z; 
	double x,y; //x for real and y for imaginary parts of the coefficient
	int n=0; //n is number degree of polynomial
	

//------Read Coefficients------------------------------------------------
	printf("Enter coefficients and enter any char other than number when done:\n");	
	while(scanf("%lf %lf",&x,&y) == 2)  { //Read coefficients from stdin
		cList[n] = (x + y*I);
		n++;
	}
	x = 1;  //Cn = 1, because the equation has to be normalized
	y = 0;
	z = (x + y*I);
	cList[n] = z; //Store in cList[]
	
	
	durand_kerner(cList,n);
	
	
}
//----------------------------------Function Definition-------------------------------


void durand_kerner(double complex cList[],int n) {
	float st,total;
	
	R = 1 + max_cof(cList,n);  //End Equation 5

	calc_theta(n);
	int k;
	st=omp_get_wtime();
	for(k=1;k <= max_iter;k++) { 
		
		

		deltaZMax = 0; 

		update_fz(cList,n,k); 
		
		if(deltaZMax <= epsilon) { 
			break;   
		}
		printf("Zmax=%f %d\n",deltaZMax,k);
	}
	st=omp_get_wtime()-st;
	
	printfile(cList,n,k,st);
	printf("Time Taken=%f\n",st);
	printz(cList,n);

}

void calc_theta(int n) { 
	for(int j=0;j < n;j++) { 
		z[j] = ( cos( j*((2*M_PI)/n) ) + (I*sin( j*((2*M_PI)/n) )) )*R;
	} 

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

void update_fz(double complex cList[],int n,int o)
{
	for(int j=0;j < n;j++) { 
			
			QsubJ = 1; 
			for(int i=0;i < n;i++) { 
				if(i != j) { 
					QsubJ = (z[j]-z[i])*QsubJ;
				}
			} 
			fz = 1; 
			for(int k = n-1;k >= 0;k--) {
				fz = fz*z[j] + cList[k];
                
			}

			deltaZ[j] = (-fz/QsubJ);
			
			}

			for(int j=0;j<n;j++)
			{
				z[j] = z[j] + deltaZ[j];
			if(cabs(deltaZ[j]) > deltaZMax) {
				deltaZMax = cabs(deltaZ[j]);
			}
		} 
}

void printfile(double complex cList[],int n,int k,float st)
{		
		FILE *fp;  
   		fp = fopen("project_roots.txt", "w");
		fprintf(fp,"Durand Kerner Serial Algorithm:\n");
		fprintf(fp,"Max Iteration=%d\n",k);	
		fprintf(fp,"Time Taken=%f\n",st);
		fprintf(fp,"Final Output:(Note: if the roots repeat then there exist less than n-1 roots for the equation)\n");
		for(int i=0;i < n;i++) {  
                	fprintf(fp,"z[%d] = %0.10f + %0.10f*I\n",i,creal(z[i]),cimag(z[i]));
                fflush(stdout);
        	}
		fclose(fp);
}
