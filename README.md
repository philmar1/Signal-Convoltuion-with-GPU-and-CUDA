# Signal-Convoltuion-with-GPU-and-CUDA

This is a current project that aims to build a GPU version of fitering signals and images. The programmation language is C and the whole code was written by myself. The image filtering part will be added soon. 

In order to run the code, you must possess a GPU and have cuda installed. You'll also need gnuplot to display the graphs.

- The code filters a signal of the form (sin(2pi\*freq1\*i) + 0.5(sin(2pi\*freq1\*i) with either a Box filter or a Gaussian filter of support s using a symetry for the signals to deal with out of bounds
- The length of the signal is 10^N
- Filter is either Box or Gauss
- Version is either CPU or GPU


If you do, please do the following steps :
- go to the code and change the line 253 to set the direction of your repository
- change the extention of signal.c to signal.cu
- run the command : nvcc -lm -o signal signal.cu (compilation)
- run the command : ./signal N freq1 freq2 s Filter Version
- run the command : gnuplot script_signal.gnuplot signal.data

The question g aims to compare the speed of the CPU and GPU version with respect to a length of the input signal of 20*k, 0<k<31. We observed a 30 times speed boost.
