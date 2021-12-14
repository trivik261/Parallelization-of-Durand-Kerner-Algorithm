#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <mpi.h>
#include <stdlib.h>

#define M_PI 3.14159265358979323846
#define coff_size 20

#define MASTER 0

#define FROM_MASTER 1

#define FROM_WORKER 2

double R=0; 
double complex z[coff_size]; 
double complex cList[coff_size];
double complex deltaZ[coff_size]; 
double deltaZMax;
double epsilon = 1e-6;
double complex QsubJ,fz; 
int max_iter = 500;


//----------------------Function Prototypes-------------------------------
void durand_kerner(); //Prototypes
void calc_theta();
double max_cof();
void hello();
void printz();
void update_z();
//void update_fz();
void printfile(double complex cList[],int n,int k,float st);


int main(int argc, char *argv[]) {

	  //List of coefficients
	double complex z; 
	double x,y; //x for real and y for imaginary parts of the coefficient
	int n=0; //n is number degree of polynomial
	

//------Read Coefficients------------------------------------------------
	// printf("Enter coefficients and enter any char other than number when done:\n");	
	// // while(scanf("%lf %lf",&x,&y) == 2)  { //Read coefficients from stdin
	// // 	cList[n] = (x + y*I);
	// // 	n++;
	// // }
	// for(int i=0;i<10;i++)
	// {
	// 	cList[i] = (i+1 + (i+1)*I);
	// }
	// n=10;
	// x = 1;  //Cn = 1, because the equation has to be normalized
	// y = 0;
	// z = (x + y*I);
	// cList[n] = z; //Store in cList[]
	// //
	
	durand_kerner(n,argc, argv);
	//
	
    return 0;
}
//----------------------------------Function Definition-------------------------------
void hello()
{
	printf("Hello=");
}

void durand_kerner(int n,int argc, char *argv[]) {

    //----------------------------------MPI-------------------------------
    int no_tasks, taskid, no_workers, source, workers, mtype, no_elements_iter, no_elements, no_elements_left, index, i, j, k, rc;
	double start,end;

    MPI_Status status;
	MPI_Request request;
    MPI_Init(&argc, &argv);
	
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &no_tasks);
    if (no_tasks < 2)
    {
        printf("Available Processors =%d\n",no_tasks);
        printf("Program is Terminated since there are less than 2 threads\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(1);
    }

	char pro_name[MPI_MAX_PROCESSOR_NAME];
	int length;
	MPI_Get_processor_name(pro_name,&length);

	printf("Processor name = %s, rank %d out of %d processors\n",pro_name,taskid,no_tasks);
    no_workers = no_tasks - 1;
    //----------------------------------MPI-------------------------------

    if(taskid==0)
    {
		for(int i=0;i<10;i++)
		{
			cList[i] = (i+1 + (i+1)*I);
			n++;
		}
		cList[n]=(1 + 0*I);
		//printf("%d=\n",n);
		start=MPI_Wtime();
        float st,total;
	
        R = 1 + max_cof(cList,n);  //End Equation 5

        calc_theta(n);
		
        int k;
		index=0;

		max_iter=2;
		MPI_Send(&max_iter, 1, MPI_INT, 1, 1, MPI_COMM_WORLD);
		MPI_Send(&n, 1, MPI_INT, 1,1, MPI_COMM_WORLD);
		MPI_Send(&cList, n, MPI_DOUBLE_COMPLEX, 1,1, MPI_COMM_WORLD);

        for(k=1;k <= max_iter;k++) 
		{ 
            
				deltaZMax = 0; 

			no_elements = n / no_workers;
			no_elements_left = n % no_workers;
			index = 0;
			mtype = FROM_MASTER;
			
			for (workers = 1; workers <= no_workers; workers++)
			{
				
				no_elements_iter = (workers <= no_elements_left) ? no_elements + 1 : no_elements;
				
				MPI_Send(&index, 1, MPI_INT, workers, mtype, MPI_COMM_WORLD);
				printf("k=%d %d",k,no_elements_iter);
				MPI_Send(&no_elements_iter, 1, MPI_INT, workers, mtype, MPI_COMM_WORLD);
				MPI_Send(&z, n, MPI_DOUBLE_COMPLEX, workers, mtype, MPI_COMM_WORLD);
				
				
				index += no_elements_iter;
				
			}

			mtype = FROM_WORKER;
			for (i = 1; i <= no_workers; i++)
			{
				source = i;
				
				MPI_Recv(&index, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
				MPI_Recv(&no_elements_iter, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
				MPI_Recv(&deltaZ[index], no_elements_iter, MPI_DOUBLE_COMPLEX, source, mtype, MPI_COMM_WORLD, &status);
			}

			for(int j=0;j<n;j++)
			{
				z[j] = z[j] + deltaZ[j];
				//printf("Iter=%d %d \n",j,k);
				//	printf("z[%d] = %0.10f + %0.10f*I\n",j,creal(z[j]),cimag(z[j]));
				if(cabs(deltaZ[j]) > deltaZMax) {
					deltaZMax = cabs(deltaZ[j]);
				}
			}

				if(deltaZMax <= epsilon) { 
					printf("Hello");
					break;   
				}
				printf("Zmax=%f %d; ",deltaZMax,k);
				MPI_Wait(&request, MPI_STATUS_IGNORE);
				MPI_Barrier(MPI_COMM_WORLD);
        }
        st=MPI_Wtime()-start;
        printf("Max Iteration=%d\n",k);	
        printfile(cList,n,k,st);
        printf("Time Taken=%f\n",st);
		
    }
    if(taskid>0)
    {
        //Send z,cList full
        //Receive deltaZ
		mtype = FROM_MASTER;
		//int max_iter2;
		MPI_Recv(&max_iter, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
		printf("Iter=%d\n",max_iter);
		MPI_Recv(&n, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
		
		MPI_Recv(&cList, n, MPI_DOUBLE_COMPLEX, MASTER, mtype, MPI_COMM_WORLD, &status);
		
		for(int l=0;l<3;l++)
		{
			
			MPI_Recv(&index, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
			printf("l=%d %d\n",l,index);
			MPI_Recv(&no_elements_iter, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
			
			MPI_Recv(&z, n, MPI_DOUBLE_COMPLEX, MASTER, mtype, MPI_COMM_WORLD, &status);
			


			printf("z[%d] = %0.10f + %0.10f*I\n",0,creal(cList[0]),cimag(cList[0]));
			//printf("Helllo=z[%d] = %0.10f + %0.10f*I\n",0,creal(z[0]),cimag(z[0]));
			for(int j=0;j < no_elements_iter;j++) { 
				
				QsubJ = 1; 
				for(int i=0;i < n;i++) { 
					if(i != j) { 
						QsubJ = (z[j]-z[i])*QsubJ;
					}
				} 
				
				fz = 1; 
				for(int k = n-1;k >= 0;k--) {
					//printf("z[%d] = %0.10f + %0.10f*I\n",k,creal(cList[0]),cimag(cList[0]));
					//fz = fz*z[j] + cList[k];
					fz = fz*z[j] ;
			}
			
			deltaZ[j] = (-fz/QsubJ);
			}
			

			mtype = FROM_WORKER;

			MPI_Send(&index, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);

			MPI_Send(&no_elements_iter, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);

			MPI_Send(&deltaZ, no_elements_iter, MPI_DOUBLE_COMPLEX, MASTER, mtype, MPI_COMM_WORLD);
			MPI_Wait(&request, MPI_STATUS_IGNORE);
			MPI_Barrier(MPI_COMM_WORLD);
		}
		


	}
	MPI_Finalize();
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
/*
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
				//fz=cList[k]/(1-z[j]);
                
			}

			deltaZ[j] = (-fz/QsubJ);
			z[j] = z[j] + deltaZ[j];
			printf("Iter=%d %d \n",j,o);
				printf("z[%d] = %0.10f + %0.10f*I\n",j,creal(z[j]),cimag(z[j]));
			if(cabs(deltaZ[j]) > deltaZMax) {
				deltaZMax = cabs(deltaZ[j]);
			}
		} 
		printf("\n");
}
*/
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
