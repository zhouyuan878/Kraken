#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <cuda.h>
#include <cfloat>
#include <sys/time.h>

long long start_timer() {
struct timeval tv;
gettimeofday(&tv, NULL);
return tv.tv_sec * 1000000 + tv.tv_usec;
}

long long stop_timer(long long start_time, char *name) {
struct timeval tv;
gettimeofday(&tv, NULL);
long long end_time = tv.tv_sec * 1000000 + tv.tv_usec;
printf("%s: %.5f sec\n", name, ((float) (end_time - start_time)) / (1000 * 1000));
return end_time - start_time;
}

int main(void)
{

FILE    *infile;
char    *buffer;
long    numbytes;
char*   devBuffer;

infile = fopen("database.kdb", "r");

if(infile == NULL)
return 1;

fseek(infile, 0L, SEEK_END);
numbytes = ftell(infile);
printf("%lu",numbytes);

fseek(infile, 0L, SEEK_SET);

buffer = (char*)calloc(numbytes, sizeof(char));

if(buffer == NULL)
return 1;

long long memory_start_time = start_timer();

if (cudaMalloc((void **) &devBuffer, numbytes) != cudaSuccess)
printf("Error allocating GPU memory");

fread(buffer, sizeof(char), numbytes, infile);
cudaMemcpy(devBuffer, buffer, numbytes, cudaMemcpyHostToDevice);
stop_timer(memory_start_time, "\n\t  Memory Transfer to GPU");
fclose(infile);

free(buffer);
cudaFree(devBuffer);

return 0;
}
