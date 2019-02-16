// Author : Philippe Martin

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <getopt.h>
#include <sys/time.h>

#define min(a,b) (a<=b?a:b)
#define max(a,b) (a>=b?a:b)

///// GPU kernels /////

	// signal creations
 __global__
 void device_create_signal(int N, float freq1, float freq2, float *signal) 
 {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
 	if (i < N)
 		signal[i] = sinf(2*M_PI*freq1*i)+0.5*sinf(2*M_PI*freq2*i);
 }
 
 __global__
void device_create_signal_box(int N, int s, float *signal_box)
 {
 	int i = blockIdx.x*blockDim.x + threadIdx.x;
 	if (i < N)
 		signal_box[i] = 0; 
	if (i <= s)
		signal_box[i] = 1; 
 }

 __global__
 void device_create_signal_gaussian(int N, int s, float *signal_gaussien) 
 {
 	int i = blockIdx.x*blockDim.x + threadIdx.x;
 	if (i < N)
 		signal_gaussien[i] = exp(-(powf(i,2))/(2*powf(s,2)))/(s*sqrtf(2*M_PI)); 
 }

	// filter kernels

 __global__
 void device_box_filter(int N, int s, float *signal,  float *signal_box, float *output)
 {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
 	if (x < N)
 	{	
		output[x] = 0;
		float sum = 0.f;
 		for ( int y = x-s; y<= min(N,x+s); y++)
		{
			sum += signal_box[labs(x-y)];
                	output[x] += signal[labs(y)]*signal_box[labs(x-y)];
		}
		output[x] /= sum ;
 	}
 }
 
 __global__
 void device_gaussian_filter(int N, int s, float *signal, float *signal_gaussian, float *output)
 {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	if (x < N)
	{
		 output[x] = 0;
		float sum = 0.f;
 		for ( int y = x-s; y<= min(N,x+s); y++)
		{
			sum += signal_gaussian[labs(x-y)];
                	output[x] += signal[labs(y)]*signal_gaussian[labs(x-y)];
		}
		output[x] /= sum ;
	}
 }
 
 
///// CPU kernels /////

float* host_box_filter(int N, int s, float *signal, float *signal_box)
{
    float *filtered_signal = (float*)malloc(N*sizeof(float)); ;
    for (int x = 0; x < N; x++)
    {
        float sum = 0 ;
        for (int y = x-s; y <= x+s ; y++)
        {
            if (labs(y) < N)
            {
                sum += signal_box[labs(x-y)];
                filtered_signal[x] += signal[labs(y)]*signal_box[labs(x-y)];
            }
        }
        filtered_signal[x] /= sum ;
        
        
    }
    return filtered_signal;
}


float* host_gaussian_filter(int N, int s, float *signal, float *signal_gaussien)
{
    float *filtered_signal = (float*)malloc(N*sizeof(float)) ;
    for (int x = 0; x < N; x++)
    {
        float sum = 0 ;
        for (int y = x-s ; y <= x+s; y++)
        {
            if (labs(y)< N)
            {
                sum += signal_gaussien[labs(x-y)];
                filtered_signal[x] += signal[labs(y)]*signal_gaussien[labs(x-y)];
            }
        }
        filtered_signal[x] /= sum ;

    }
    return filtered_signal;
}

// Main for questions < g

int main(int argc, char *argv[])
{
    struct timespec start, stop;
    
    if (argc == 7)
    {
	int N = pow(10,atoi(argv[1]));
        float freq1 = atof(argv[2]);
        float freq2 = atof(argv[3]);
        int s = atoi(argv[4]);
        char *filter = argv[5];
        char *version = argv[6];

        float *signal, *d_signal, *signal_box, *d_signal_box, *signal_gaussien, *d_signal_gaussien;
	float *output, *d_output;
        signal = (float*)malloc(N*sizeof(float));
        signal_box = (float*)malloc(N*sizeof(float));
        signal_gaussien = (float*)malloc(N*sizeof(float));
	output = (float*)malloc(N*sizeof(float));
        
	///// GPU version /////

        if (version[0] == 'G') 
        {
            
            // Moving signals to GPU
            cudaMalloc(&d_signal, N*sizeof(float));
            cudaMalloc(&d_output, N*sizeof(float));
            // Creating signal :
            
            cudaMemcpy(d_signal, signal, N*sizeof(float), cudaMemcpyHostToDevice);
            device_create_signal<<<(N+255)/256, 256>>>(N, freq1, freq2, d_signal);

            if (filter[0] == 'B')
            {
                clock_gettime(CLOCK_MONOTONIC_RAW, &start);
             
		
                // Moving gaussian signal to GPU
                cudaMalloc(&d_signal_box, N*sizeof(float));
                cudaMemcpy(d_signal_box, signal_box, N*sizeof(float), cudaMemcpyHostToDevice);
             
                // Creating Box filter and applying convolution
            	device_create_signal_box<<<(N+255)/256, 256>>>(N, s, d_signal_box);
                device_box_filter<<<(N+255)/256, 256>>>(N, s, d_signal, d_signal_box, d_output);
                
                // Managing memories
                cudaMemcpy(signal, d_signal, N*sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(output, d_output, N*sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(signal_box, d_signal_box, N*sizeof(float), cudaMemcpyDeviceToHost);
                cudaFree(d_signal_box);
                
                // Measure time
                clock_gettime(CLOCK_MONOTONIC_RAW, &stop);
                printf("temps Box GPU=%lu milli secondes\n",(stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_nsec - start.tv_nsec) / 1000);
            }
            else
            {
                clock_gettime(CLOCK_MONOTONIC_RAW, &start);
             
                // Moving gaussian signal to GPU
                cudaMalloc(&d_signal_gaussien, N*sizeof(float));
                cudaMemcpy(d_signal_gaussien, signal_gaussien, N*sizeof(float), cudaMemcpyHostToDevice);
                
                // Creating Gaussian filter and applying convolution
                device_create_signal_gaussian<<<(N+255)/256, 256>>>(N, s, d_signal_gaussien);
                device_gaussian_filter<<<(N+255)/256, 256>>>(N, s, d_signal, d_signal_gaussien,d_output);
                
                // Managing memories
                cudaMemcpy(signal, d_signal, N*sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(output, d_output, N*sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(signal_gaussien, d_signal_gaussien, N*sizeof(float), cudaMemcpyDeviceToHost);
                cudaFree(d_signal_gaussien);
	
                // Measure time
                clock_gettime(CLOCK_MONOTONIC_RAW, &stop);
                printf("temps Gausss GPU=%lu milli secondes\n",(stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_nsec - start.tv_nsec) / 1000);
            }
            cudaFree(d_signal);
            cudaFree(d_output);
      
             
        }



	///// CPU version /////


        else 
        {
            // Creating signals :
            for (int i = 0; i < N; i++)
            {
                signal[i] = sin(2*M_PI*freq1*i)+0.5*sin(2*M_PI*freq2*i);
                signal_gaussien[i] = exp(-(pow(i,2))/(2*pow(s,2)))/(s*sqrt(2*M_PI));
                if (i <= s)
                    signal_box[i] = 1;
                else
                    signal_box[i] = 0;
                
            }
            printf("signals built \n");
	    // Box filter
            if (filter[0] == 'B')
            {
                clock_gettime(CLOCK_MONOTONIC_RAW, &start);
                
                output = host_box_filter(N, s, signal, signal_box);
                
                clock_gettime(CLOCK_MONOTONIC_RAW, &stop);
                printf("temps=%lu milli secondes\n",(stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_nsec - start.tv_nsec) / 1000);
            }
		
	    // Gaussian filter
            else
            {
                clock_gettime(CLOCK_MONOTONIC_RAW, &start);
                
                output = host_gaussian_filter(N, s, signal, signal_gaussien);
                
                clock_gettime(CLOCK_MONOTONIC_RAW, &stop);
                printf("temps=%lu milli secondes\n",(stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_nsec - start.tv_nsec) / 1000);
        
            }
        }

	//// Creating signal.data file ////
        
        FILE *fichier = fopen("/cal/homes/imatp3/Documents/signal.data","w+");
        for (int i = 0; i < min(200,N); i++)
        {
            fprintf(fichier, "%f", output[i]);
            fprintf(fichier,"\n");
        }

        printf("signal.data file was updated \n");
        fclose(fichier);
        free(signal);
	free(output);
        free(signal_gaussien);
        
        return 0;
    }
     else // wrong number of arguments passed
     {
         printf("Nb of arguments should be 6, format (power_of_10, freq1,freq2,s,filtre ={Gauss,Box}, version = {GPU,CPU}");
         return 1;
     }
}



/*

 // Question g : There might be some issues because the previous main was changed afterwards. Please consider the previous one.




int main(int argc, char *argv[])
{
    struct timespec start, stop;
    
    if (argc == 5)
    {
        float freq1 = atof(argv[1]);
        float freq2 = atof(argv[2]);
        int s = atoi(argv[3]);
	int k = atoi(argv[4]);
	

	//GPU version

        for (int j = 1; j <=k; j++)
	{
		printf("%d \n",j);
		int N = 20*j;

        	float *signal, *d_signal, *signal_box, *d_signal_box, *signal_gaussien, *d_signal_gaussien;
       		float *output, *d_output;
		signal = (float*)malloc(N*sizeof(float));
        	signal_box = (float*)malloc(N*sizeof(float));
        	signal_gaussien = (float*)malloc(N*sizeof(float));
        	output = (float*)malloc(N*sizeof(float));
		// GPU Box
            
            	// Moving signal to GPU
            	cudaMalloc(&d_signal, N*sizeof(float));

            	// Creating signal :
            
            	cudaMemcpy(d_signal, signal, N*sizeof(float), cudaMemcpyHostToDevice);
            	device_create_signal<<<(N+255)/256, 256>>>(N, freq1, freq2, d_signal);
             
		
                // Moving gaussian signal to GPU
                cudaMalloc(&d_signal_box, N*sizeof(float));
                cudaMemcpy(d_signal_box, signal_box, N*sizeof(float), cudaMemcpyHostToDevice);
             
                // Creating Box filter and applying convolution
            	device_create_signal_box<<<(N+255)/256, 256>>>(N, s, d_signal_box);

		clock_gettime(CLOCK_MONOTONIC_RAW, &start);
                device_box_filter<<<(N+255)/256, 256>>>(N, s, d_signal, d_signal_box);
                clock_gettime(CLOCK_MONOTONIC_RAW, &stop);

		// Managing memories
		cudaMemcpy(signal, d_signal, N*sizeof(float), cudaMemcpyDeviceToHost);     
		cudaMemcpy(output, d_output, N*sizeof(float), cudaMemcpyDeviceToHost);	
		cudaMemcpy(signal_box, d_signal_box, N*sizeof(float), cudaMemcpyDeviceToHost);
		cudaFree(d_signal_box);

		// Measure time
                printf("temps Box GPU for k= %d : = %lu milli secondes\n",j , (stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_nsec - start.tv_nsec) / 1000);

	// GPU Gauss           
		signal = (float*)malloc(N*sizeof(float));

             
                // Moving gaussian signal to GPU
                cudaMalloc(&d_signal_gaussien, N*sizeof(float));
                cudaMemcpy(d_signal_gaussien, signal_gaussien, N*sizeof(float), cudaMemcpyHostToDevice);
                
                // Creating Gaussian filter and applying convolution
		device_create_signal_gaussian<<<(N+255)/256, 256>>>(N, s, d_signal_gaussien);
		clock_gettime(CLOCK_MONOTONIC_RAW, &start);
                device_gaussian_filter<<<(N+255)/256, 256>>>(N, s, d_signal, d_signal_gaussien);
                clock_gettime(CLOCK_MONOTONIC_RAW, &stop);

		// Managing memories		
		cudaMemcpy(signal, d_signal, N*sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(output, d_output, N*sizeof(float), cudaMemcpyDeviceToHost);
             	cudaMemcpy(signal_gaussien, d_signal_gaussien, N*sizeof(float), cudaMemcpyDeviceToHost);
		cudaFree(d_signal_gaussien);
	
		// Measure time
                clock_gettime(CLOCK_MONOTONIC_RAW, &stop);
                printf("temps Gauss GPU for k= %d : =%lu milli secondes\n",j ,(stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_nsec - start.tv_nsec) / 1000);
            
            cudaFree(d_signal);




	// CPU version

            // Creating signals :
            for (int i = 0; i < N; i++)
            	{
                signal[i] = sin(2*M_PI*freq1/N*i)+sin(2*M_PI*freq2/N*i);
                if (i < s)
                    signal_box[i] = 1;
                else
                    signal_box[i] = 0;
		}
  
	    // Box filter
                clock_gettime(CLOCK_MONOTONIC_RAW, &start);

                output = host_box_filter(N, s, signal, signal_box);
                
                clock_gettime(CLOCK_MONOTONIC_RAW, &stop);
                printf("temps Box CPU for k= %d : = %lu milli secondes\n",j, (stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_nsec - start.tv_nsec) / 1000);
            
		
	    // Creating signals :
            for (int i = 0; i < N; i++)
            	{
                signal[i] = sin(2*M_PI*freq1/N*i)+sin(2*M_PI*freq2/N*i);
                signal_gaussien[i] = exp(-(pow(i,2))/(2*pow(s,2)))/(s*sqrt(2*M_PI));
		}
  
	    // Gaussian filter

                clock_gettime(CLOCK_MONOTONIC_RAW, &start);
                
                output = host_gaussian_filter(N, s, signal, signal_gaussien);
                
                clock_gettime(CLOCK_MONOTONIC_RAW, &stop);
                printf("temps Gauss CPU for k= %d : %lu milli secondes\n", j, (stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_nsec - start.tv_nsec) / 1000); 
        
        free(signal);
        free(signal_gaussien);
	free(signal_box);
	}

	return 0;
    }
     else // wrong number of arguments passed
     {
         printf("Nb of arguments should be 6");
         return 1;
     }
}
*/


